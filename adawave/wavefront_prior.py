"""Self-supervised wavefront-prior denoising.

Each point's denoised position is parameterised by displacements along
its estimated local basis:

    x_i = x_noisy_i + alpha_i * n_i + (feat_i * beta_i) * b_i

with ``n_i`` the unit normal, ``b_i = n_i x p_i`` the binormal (across-
edge direction in the tangent plane), ``p_i`` the principal singular
direction, and ``feat_i`` a geometric feature gate. The third basis
vector ``p_i`` (along the edge) is intentionally *unconstrained* — that
is the freedom needed to slide a noisy edge sample onto the true edge.

Energy minimised
----------------
    E = sum_i alpha_i^2 + feat_i^2 beta_i^2                 (data fidelity)
      + lam_n sum_{(i,j)} w_ij ((x_i - x_j) . n_i)^2        (normal channel)
      + lam_b sum_{(i,j)} w_ij feat_i feat_j ((x_i - x_j) . b_i)^2
                                                            (binormal channel)

For a flat region (``feat_i = 0``) only the normal channel is active and
the system reduces to anisotropic graph Laplacian smoothing of the
height field — i.e. classical surface fairing in normal direction.
On feature points the binormal channel adds a "do not jump across the
edge" constraint, while motion along the edge stays free.

Theoretical view
----------------
Letting ``alpha`` and ``beta`` be continuous fields, the continuum limit
recovers anisotropic mean-curvature flow restricted to the normal
direction plus a 1-D flow of the singular set along its tangent.
"""

from __future__ import annotations

import numpy as np

from .neighbors import NeighborQuery
from .geometry import estimate_local_frames, filter_neighbors_by_normal
from .patch import build_density_patches
from .directional import compute_directional_responses
from .descriptors import compute_descriptors


# ---------------------------------------------------------------------------
# Descriptor estimation
# ---------------------------------------------------------------------------
def _estimate_descriptors(X: np.ndarray, k: int, n_dirs: int, grid_size: int,
                          normal_thresh: float = 0.3):
    nq = NeighborQuery(X)
    dists, indices = nq.query_knn(k)
    frames = estimate_local_frames(X, indices)
    nq_w = filter_neighbors_by_normal(X, indices, dists,
                                      frames["normals"],
                                      normal_thresh=normal_thresh,
                                      residual_sigma_factor=1.5)
    patches = build_density_patches(X, indices,
                                    frames["normals"], frames["tangent1"],
                                    frames["tangent2"], grid_size=grid_size,
                                    neighbor_weights=nq_w)
    responses = compute_directional_responses(patches, n_dirs=n_dirs)
    desc = compute_descriptors(responses, n_dirs=n_dirs)
    return {
        "indices": indices,
        "dists": dists,
        "frames": frames,
        "neighbor_quality": nq_w,
        "strength": desc["singularity_strength"],
        "principal_direction": desc["principal_direction"],
    }


def _feature_gate(eigenvalues: np.ndarray, descriptor_strength: np.ndarray,
                  sv_floor: float = 0.02, sv_high_q: float = 0.95) -> np.ndarray:
    sv = eigenvalues[:, 2] / (eigenvalues.sum(axis=1) + 1e-12)
    sv_high = float(np.quantile(sv, sv_high_q))
    denom = max(sv_high - sv_floor, 1e-6)
    geom = np.clip((sv - sv_floor) / denom, 0.0, 1.0)
    return geom * descriptor_strength


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------
def denoise_wavefront_prior(X_noisy: np.ndarray,
                            k: int = 12,
                            n_dirs: int = 8,
                            grid_size: int = 12,
                            lam_n: float = 5.0,
                            lam_b: float = 5.0,
                            data_weight: float = 1.0,
                            feature_anchor: float = 0.0,
                            n_iters: int = 100,
                            step: float = 0.30,
                            momentum: float = 0.85,
                            max_shift_factor: float = 1.5,
                            refresh_every: int = 0,
                            normal_thresh: float = 0.3,
                            verbose: bool = False) -> dict:
    """Self-supervised wavefront-prior denoising.

    Larger ``lam_n / data_weight`` => stronger smoothing in the normal direction.
    """
    X_noisy = np.ascontiguousarray(X_noisy, dtype=np.float64)
    desc = _estimate_descriptors(X_noisy, k, n_dirs, grid_size, normal_thresh=normal_thresh)
    spacing = float(np.median(desc["dists"][:, 0]))

    N = len(X_noisy)
    indices = desc["indices"]
    w_edge = desc["neighbor_quality"]

    def _setup(desc):
        n = desc["frames"]["normals"]
        t1 = desc["frames"]["tangent1"]
        t2 = np.cross(n, t1)
        t2 /= (np.linalg.norm(t2, axis=1, keepdims=True) + 1e-12)
        cs = np.cos(desc["principal_direction"])[:, None]
        sn = np.sin(desc["principal_direction"])[:, None]
        p = cs * t1 + sn * t2
        b = np.cross(n, p)
        b /= (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
        feat = _feature_gate(desc["frames"]["eigenvalues"], desc["strength"])
        return n, b, feat

    n_vec, b_vec, feat = _setup(desc)

    # precompute the constant edge offsets c^n_ij, c^b_ij based on X_noisy
    def _edge_constants(X_ref):
        diff = X_ref[indices] - X_ref[:, None, :]                  # (N, k, 3) = x_j - x_i
        c_n = -(diff * n_vec[:, None, :]).sum(-1)                  # (x_i - x_j) . n_i
        c_b = -(diff * b_vec[:, None, :]).sum(-1)                  # (x_i - x_j) . b_i
        return c_n, c_b

    c_n, c_b = _edge_constants(X_noisy)
    n_dot = (n_vec[indices] * n_vec[:, None, :]).sum(-1)            # (N, k)
    b_dot = (b_vec[indices] * b_vec[:, None, :]).sum(-1)
    nj_dot_bi = (n_vec[indices] * b_vec[:, None, :]).sum(-1)
    bj_dot_ni = (b_vec[indices] * n_vec[:, None, :]).sum(-1)

    alpha = np.zeros(N)
    beta = np.zeros(N)
    v_a = np.zeros(N)
    v_b = np.zeros(N)
    energies = []
    step_abs = step * spacing

    for it in range(n_iters):
        if refresh_every and it > 0 and it % refresh_every == 0:
            X_cur = X_noisy + alpha[:, None] * n_vec + (feat[:, None] * beta[:, None]) * b_vec
            desc = _estimate_descriptors(X_cur, k, n_dirs, grid_size, normal_thresh=normal_thresh)
            indices = desc["indices"]
            w_edge = desc["neighbor_quality"]
            n_vec, b_vec, feat = _setup(desc)
            c_n, c_b = _edge_constants(X_cur)
            n_dot = (n_vec[indices] * n_vec[:, None, :]).sum(-1)
            b_dot = (b_vec[indices] * b_vec[:, None, :]).sum(-1)
            nj_dot_bi = (n_vec[indices] * b_vec[:, None, :]).sum(-1)
            bj_dot_ni = (b_vec[indices] * n_vec[:, None, :]).sum(-1)
            # alpha/beta now describe displacements from the *new* baseline
            alpha = np.zeros(N)
            beta = np.zeros(N)
            v_a *= 0.0
            v_b *= 0.0

        beta_eff = feat * beta                                       # (N,)
        # per-point data weight: feature points get extra anchor to original
        dw = data_weight * (1.0 + feature_anchor * feat)             # (N,)

        # residuals (x_i - x_j) . n_i  and  . b_i
        r_n = (c_n
               + alpha[:, None]                                      # +alpha_i
               - alpha[indices] * n_dot                              # -alpha_j (n_j.n_i)
               - beta_eff[indices] * bj_dot_ni)                      # -beta_eff_j (b_j.n_i)
        r_b = (c_b
               + beta_eff[:, None]                                   # +feat_i*beta_i
               - beta_eff[indices] * b_dot                           # -beta_eff_j (b_j.b_i)
               - alpha[indices] * nj_dot_bi)                         # -alpha_j (n_j.b_i)

        ff = feat[:, None] * feat[indices]                           # (N, k) - mask for b-channel

        # energies
        E_data = float((dw * (alpha * alpha + beta_eff * beta_eff)).sum())
        E_n = float((w_edge * r_n * r_n).sum())
        E_b = float((w_edge * ff * r_b * r_b).sum())
        total = E_data + lam_n * E_n + lam_b * E_b
        energies.append(total)

        # ---- gradient of E w.r.t. alpha ----
        # d r_n_ij / d alpha_i = 1
        # d r_n_ij / d alpha_j = -(n_j.n_i)
        # d r_b_ij / d alpha_i = 0
        # d r_b_ij / d alpha_j = -(n_j.b_i)
        ga_self = 2.0 * (w_edge * r_n).sum(-1)                       # from i side
        ga_nbr_n = -2.0 * (w_edge * r_n * n_dot)                     # (N, k) to scatter to j
        ga_nbr_b = -2.0 * (w_edge * ff * r_b * nj_dot_bi)            # (N, k) to scatter to j
        ga = 2.0 * dw * alpha + lam_n * ga_self
        np.add.at(ga, indices.ravel(),
                  (lam_n * ga_nbr_n + lam_b * ga_nbr_b).ravel())

        # ---- gradient of E w.r.t. beta ----
        # d r_b_ij / d beta_i = feat_i
        # d r_b_ij / d beta_j = -feat_j (b_j.b_i)
        # d r_n_ij / d beta_i = 0
        # d r_n_ij / d beta_j = -feat_j (b_j.n_i)
        gb_self = 2.0 * feat * (w_edge * ff * r_b).sum(-1)
        gb_nbr_b = -2.0 * (w_edge * ff * r_b * feat[indices] * b_dot)
        gb_nbr_n = -2.0 * (w_edge * r_n * feat[indices] * bj_dot_ni)
        gb = 2.0 * dw * feat * feat * beta + lam_b * gb_self
        np.add.at(gb, indices.ravel(),
                  (lam_b * gb_nbr_b + lam_n * gb_nbr_n).ravel())

        # normalised gradient step
        gmax_a = np.max(np.abs(ga)) + 1e-12
        gmax_b = np.max(np.abs(gb)) + 1e-12
        v_a = momentum * v_a - step_abs * (ga / gmax_a)
        v_b = momentum * v_b - step_abs * (gb / gmax_b)
        alpha = alpha + v_a
        beta = beta + v_b

        cap = max_shift_factor * spacing
        alpha = np.clip(alpha, -cap, cap)
        beta = np.clip(beta, -cap, cap)

        if verbose and (it % max(1, n_iters // 10) == 0 or it == n_iters - 1):
            print(f"  iter {it:3d}  total={total:.3e}  "
                  f"E_data={E_data:.3e}  E_n={E_n:.3e}  E_b={E_b:.3e}  "
                  f"|alpha|={np.abs(alpha).mean():.4f}  |beta|={np.abs(beta).mean():.4f}")

    X_out = X_noisy + alpha[:, None] * n_vec + (feat * beta)[:, None] * b_vec
    return {
        "xyz": X_out,
        "alpha": alpha,
        "beta": beta,
        "feature_gate": feat,
        "energy_trace": np.array(energies),
        "strength": desc["strength"],
    }
