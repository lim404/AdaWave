"""Wavefront-Prior v2: noise-and-structure adaptive restoration with an
exact structured linear solve.

Upgrades over ``wavefront_prior.py`` (v1):

1.  Noise-and-structure joint adaptivity (no per-dataset switches):
      - per-point noise scale  sigma_i = 1.4826 * MAD_j( n_i . (p_j - p_i) )
      - consensus noise floor  sigma_g = median of sigma_i over the flattest
        half of the cloud (surface-variation below median), i.e. a
        structure-free noise estimate;
      - structure confidence   c_i = excess_i * strength_i, where
        excess_i = clip( (sigma_i^2 - sigma_g^2) / (sigma_i^2 + sigma_g^2) )
        is a dimensionless measure of local spread *beyond* the noise floor
        (0 on noise-only planes, -> 1 on true singular structure), and
        strength_i is the directional-singularity descriptor response;
      - the data / regularizer balance scales with the *relative* noise
        level  (sigma_g / ell_g)^2  (ell_g = median point spacing), so the
        same global configuration smooths aggressively on high-relative-
        noise clouds (CAD benchmarks) and conservatively on sparse ALS;
      - displacement cap is  cap_sigma * sigma_g  (noise units), not a
        fixed fraction of spacing.

2.  The convex quadratic energy is minimised *exactly* by a matrix-free
    Jacobi-preconditioned conjugate-gradient solve of  H theta = -q.
    No learning rate, no momentum, no iteration-count tuning.

Energy (theta = (alpha, beta)):

    E = sum_i d_i ( alpha_i^2 + (c_i beta_i)^2 )
      + s2 * [ lam_n sum_ij w_ij r_n_ij^2  +  lam_b sum_ij w_ij c_i c_j r_b_ij^2 ]

    d_i  = 1 + anchor * c_i
    s2   = (sigma_g / ell_g)^2          (relative-noise balance)
    r_n_ij = (x_i - x_j) . n_i          (normal channel)
    r_b_ij = (x_i - x_j) . b_i          (binormal, cross-edge channel)

with  x_i = x_noisy_i + alpha_i n_i + (c_i beta_i) b_i.  The along-edge
direction p_i stays unconstrained, as in v1.
"""

from __future__ import annotations

import numpy as np

from .neighbors import NeighborQuery
from .geometry import estimate_local_frames, filter_neighbors_by_normal
from .patch import build_density_patches
from .directional import compute_directional_responses
from .descriptors import compute_descriptors


# ---------------------------------------------------------------------------
# Descriptor estimation (same pipeline as v1)
# ---------------------------------------------------------------------------
def _estimate_descriptors(X: np.ndarray, k: int, n_dirs: int, grid_size: int,
                          normal_thresh: float):
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


# ---------------------------------------------------------------------------
# Noise and structure estimation
# ---------------------------------------------------------------------------
def estimate_noise_and_structure(X: np.ndarray, indices: np.ndarray,
                                 dists: np.ndarray, frames: dict,
                                 strength: np.ndarray,
                                 struct_stat: str = "q75") -> dict:
    """Per-point noise scale, consensus noise floor and structure confidence.

    All quantities are label-free and dimensionless where they need to be;
    nothing depends on the dataset identity or a known noise sigma.
    """
    normals = frames["normals"]
    diff = X[indices] - X[:, None, :]                      # (N, k, 3)
    res = (diff * normals[:, None, :]).sum(-1)             # (N, k) plane residuals

    med = np.median(res, axis=1, keepdims=True)
    absdev = np.abs(res - med)
    sigma_pt = 1.4826 * np.median(absdev, axis=1)      # (N,) robust spread (MAD)
    # structure-detection spread: 75th percentile of absolute deviations.
    # MAD has a 50% breakdown point and misses one-sided contamination
    # (e.g. a roof-edge point with a third of its neighbours on the ground);
    # the Q75 statistic fires already at ~25% cross-surface contamination.
    # (struct_stat='mad' is the ablation variant.)
    if struct_stat == "q75":
        spread_pt = np.quantile(absdev, 0.75, axis=1)
    else:
        spread_pt = 1.4826 * np.median(absdev, axis=1)

    # local spacing: median neighbour distance per point, global median spacing
    ell_pt = np.median(dists, axis=1)
    ell_g = float(np.median(dists[:, 0]))

    # consensus noise floor: median sigma over the flattest half of the cloud
    ev = frames["eigenvalues"]                             # (N, 3) descending
    sv = ev[:, 2] / (ev.sum(axis=1) + 1e-12)               # surface variation
    flat = sv <= np.median(sv)
    sigma_raw = float(np.median(sigma_pt[flat])) + 1e-12

    # first-order shrinkage correction: the kNN plane fit absorbs part of the
    # noise and the flattest-half selection biases the median low; the ratio
    # sigma_raw / sigma is geometry-independent and depends only on the
    # relative noise level (calibrated on synthetic plane/ridge/sphere)
    rel_raw = sigma_raw / (ell_g + 1e-12)
    shrink = max(0.77 - 0.38 * min(rel_raw, 1.0), 0.40)
    sigma_g = sigma_raw / shrink

    # structure confidence: spread significantly beyond the noise floor,
    # gated by the directional-singularity descriptor.  tau > 1 suppresses
    # sampling-variance false positives on noise-only planes.  The floor for
    # the Q75 spread uses the same estimator so its calibration cancels.
    tau = 1.5
    spread_floor = float(np.median(spread_pt[flat])) + 1e-12
    s2p = spread_pt ** 2
    s2g = (tau * spread_floor) ** 2
    excess = np.clip((s2p - s2g) / (s2p + s2g + 1e-24), 0.0, 1.0)
    # directional (binormal) channel additionally requires a reliable
    # principal direction, hence the descriptor-strength gate
    c_str = excess * strength

    return {
        "sigma_pt": sigma_pt,
        "sigma_g": sigma_g,
        "ell_pt": ell_pt,
        "ell_g": ell_g,
        "rel_noise": sigma_g / (ell_g + 1e-12),
        "surface_variation": sv,
        "excess": excess,
        "c_str": c_str,
    }


# ---------------------------------------------------------------------------
# Matrix-free quadratic solve
# ---------------------------------------------------------------------------
def _solve_pcg(matvec, q, diag, tol=1e-6, maxiter=500):
    """Solve H theta = -q by Jacobi-preconditioned CG. Returns (theta, info)."""
    b = -q
    x = np.zeros_like(b)
    r = b.copy()                       # r = b - H x, x = 0
    Minv = 1.0 / np.maximum(diag, 1e-30)
    z = Minv * r
    p = z.copy()
    rz = float(r @ z)
    b_norm = float(np.linalg.norm(b)) + 1e-30
    n_iter = 0
    for n_iter in range(1, maxiter + 1):
        Hp = matvec(p)
        pHp = float(p @ Hp)
        if pHp <= 0:
            break                      # numerical safeguard; H is SPD by design
        a = rz / pHp
        x += a * p
        r -= a * Hp
        if np.linalg.norm(r) / b_norm < tol:
            break
        z = Minv * r
        rz_new = float(r @ z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, {"iters": n_iter, "rel_res": float(np.linalg.norm(r)) / b_norm}


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------
def denoise_wavefront_prior_v2(X_noisy: np.ndarray,
                               k: int = 12,
                               n_dirs: int = 8,
                               grid_size: int = 12,
                               lam_n: float = 12.5,
                               lam_b: float = 25.0,
                               anchor: float = 1000.0,
                               anchor_mode: str = "const",
                               cap_sigma: float = 2.0,
                               prop_decay: float = 0.75,
                               normal_thresh: float = 0.7,
                               n_outer: int = 1,
                               cg_tol: float = 1e-6,
                               cg_maxiter: int = 500,
                               struct_stat: str = "q75",
                               beta_channel: bool = False,
                               s2_override: float | None = None,
                               cap_mode: str = "sigma",
                               use_geom_weights="aniso",
                               solver: str = "pcg",
                               verbose: bool = False) -> dict:
    """Noise-and-structure adaptive wavefront restoration (v2).

    One global configuration is intended for all datasets, densities and
    noise regimes; the per-point adaptivity is estimated from the input.
    """
    X_cur = np.ascontiguousarray(X_noisy, dtype=np.float64)
    info_all = []

    for outer in range(max(1, n_outer)):
        desc = _estimate_descriptors(X_cur, k, n_dirs, grid_size, normal_thresh)
        indices = desc["indices"]
        w_edge = desc["neighbor_quality"]
        frames = desc["frames"]

        ns = estimate_noise_and_structure(X_cur, indices, desc["dists"],
                                          frames, desc["strength"],
                                          struct_stat=struct_stat)
        # beta_channel=False (frozen default): the displacement is alpha-only
        # (x = x_noisy + alpha n); the wavefront direction acts through the
        # anisotropic graph weights instead of a binormal variable.  The
        # freeze experiments showed the binormal channel is inert (results
        # identical to <=0.01pp with lam_b in [0,400] and a free beta cap).
        c = ns["c_str"] if beta_channel else np.zeros(len(X_cur))
        ex = ns["excess"]
        sigma_g = ns["sigma_g"]
        # relative-noise balance (s2_override is the fixed-lambda ablation)
        s2 = ns["rel_noise"] ** 2 if s2_override is None else s2_override

        # structure-proximity buffer: asymmetric max-propagation of the
        # noise-excess so that smooth points *adjacent* to a singularity
        # also reduce their smoothing step (prevents cross-boundary drag)
        ex_p = ex.copy()
        for _ in range(2):
            ex_p = np.maximum(ex_p, prop_decay * ex_p[indices].max(axis=1))

        # local basis
        n_vec = frames["normals"]
        t1 = frames["tangent1"]
        t2 = np.cross(n_vec, t1)
        t2 /= (np.linalg.norm(t2, axis=1, keepdims=True) + 1e-12)
        cs = np.cos(desc["principal_direction"])[:, None]
        sn = np.sin(desc["principal_direction"])[:, None]
        p_vec = cs * t1 + sn * t2
        b_vec = np.cross(n_vec, p_vec)
        b_vec /= (np.linalg.norm(b_vec, axis=1, keepdims=True) + 1e-12)

        # geometry-consistent smoothing weights: an edge that touches a
        # structural point must not pull mass across the singularity.  The
        # quadratic per-side decay effectively decouples high-excess regions
        # (structure or surface-free scatter such as vegetation) from the
        # smoothing graph: what the local-surface model cannot represent is
        # left untouched rather than dragged towards a fictitious surface.
        if use_geom_weights == "aniso":
            # wavefront-anisotropic decoupling: at singular points, keep the
            # smoothing pull from neighbours ALONG the estimated wavefront
            # direction p_i (1-D diffusion along the singular fibre) and
            # suppress the cross-edge pull; the direction is trusted in
            # proportion to the descriptor strength, otherwise the factor
            # falls back to the isotropic (1-ex_p)^2 decay.
            diffn = X_cur[indices] - X_cur[:, None, :]
            diffn = diffn / (np.linalg.norm(diffn, axis=-1, keepdims=True) + 1e-12)
            rel = desc["strength"]
            along_i = (diffn * p_vec[:, None, :]).sum(-1)
            fac_i = (1.0 - ex_p[:, None]
                     * (1.0 - rel[:, None] * along_i ** 2)) ** 2
            along_j = (diffn * p_vec[indices]).sum(-1)
            fac_j = (1.0 - ex_p[indices]
                     * (1.0 - rel[indices] * along_j ** 2)) ** 2
            w_edge = w_edge * fac_i * fac_j
        elif use_geom_weights:
            w_edge = (w_edge * ((1.0 - ex_p) ** 2)[:, None]
                      * (1.0 - ex_p[indices]) ** 2)

        # constant edge offsets and basis couplings
        diff = X_cur[indices] - X_cur[:, None, :]
        c_n = -(diff * n_vec[:, None, :]).sum(-1)       # (x_i - x_j) . n_i
        c_b = -(diff * b_vec[:, None, :]).sum(-1)
        n_dot = (n_vec[indices] * n_vec[:, None, :]).sum(-1)
        b_dot = (b_vec[indices] * b_vec[:, None, :]).sum(-1)
        nj_dot_bi = (n_vec[indices] * b_vec[:, None, :]).sum(-1)
        bj_dot_ni = (b_vec[indices] * n_vec[:, None, :]).sum(-1)

        # adaptive weights: the anchor uses the *squared* noise-excess so
        # that sampling-variance residue on planes (~0.05) is negligible
        # while true singular structure (~0.5) is strongly pinned.
        # anchor_mode='auto' is the parameter-free form 1 + e^2/((1-e)^2+eps):
        # negligible at mid excess (cap and graph weights already protect
        # there) and diverging as e -> 1 (hard pin exactly where the local
        # surface model has no explanatory power).
        if anchor_mode == "auto":
            dw = 1.0 + ex ** 2 / ((1.0 - ex) ** 2 + 1e-2)
        else:
            dw = 1.0 + anchor * ex ** 2                  # (N,) data weight
        wn = s2 * lam_n * w_edge                         # (N, k) normal channel
        ff = c[:, None] * c[indices]
        wb = s2 * lam_b * w_edge * ff                    # (N, k) binormal channel
        ridge = 1e-8 + 1e-6 * float(dw.mean())           # keeps beta block SPD

        N = len(X_cur)

        def grad(theta):
            """Gradient of E at theta = [alpha; beta]; linear in theta."""
            alpha = theta[:N]
            beta = theta[N:]
            beta_eff = c * beta
            r_n = (c_n + alpha[:, None]
                   - alpha[indices] * n_dot
                   - beta_eff[indices] * bj_dot_ni)
            r_b = (c_b + beta_eff[:, None]
                   - beta_eff[indices] * b_dot
                   - alpha[indices] * nj_dot_bi)
            ga = 2.0 * dw * alpha + 2.0 * (wn * r_n).sum(-1)
            np.add.at(ga, indices.ravel(),
                      (-2.0 * (wn * r_n * n_dot + wb * r_b * nj_dot_bi)).ravel())
            gb = (2.0 * dw * c * c * beta + 2.0 * ridge * beta
                  + 2.0 * c * (wb * r_b).sum(-1))
            np.add.at(gb, indices.ravel(),
                      (-2.0 * (wb * r_b * c[indices] * b_dot
                               + wn * r_n * c[indices] * bj_dot_ni)).ravel())
            return np.concatenate([ga, gb])

        q = grad(np.zeros(2 * N))

        def matvec(v):
            return grad(v) - q

        # analytic Jacobi diagonal of H
        diag_a = 2.0 * dw + 2.0 * wn.sum(-1)
        np.add.at(diag_a, indices.ravel(),
                  (2.0 * (wn * n_dot ** 2 + wb * nj_dot_bi ** 2)).ravel())
        diag_b = 2.0 * dw * c * c + 2.0 * ridge + 2.0 * c * c * wb.sum(-1)
        cj2 = c[indices] ** 2
        np.add.at(diag_b, indices.ravel(),
                  (2.0 * (wb * cj2 * b_dot ** 2
                          + wn * cj2 * bj_dot_ni ** 2)).ravel())
        diag = np.concatenate([diag_a, diag_b])

        if solver == "pcg":
            theta, cg_info = _solve_pcg(matvec, q, diag, tol=cg_tol,
                                        maxiter=cg_maxiter)
        else:
            # ablation: v1-style momentum gradient descent on the same energy
            theta = np.zeros(2 * N)
            vel = np.zeros(2 * N)
            step_abs = 0.05 * ns["ell_g"]
            for _ in range(80):
                g = grad(theta)
                gmax = np.max(np.abs(g)) + 1e-12
                vel = 0.5 * vel - step_abs * (g / gmax)
                theta = theta + vel
            g = grad(theta)
            cg_info = {"iters": 80,
                       "rel_res": float(np.linalg.norm(g)
                                        / (np.linalg.norm(q) + 1e-30))}
        alpha = theta[:N]
        beta = theta[N:]

        # adaptive displacement cap in noise units, tightened on structure:
        # smooth points may move up to cap_sigma * sigma_g, singular points
        # are progressively restricted (cap_mode='spacing' is the v1-style
        # fixed-fraction-of-spacing ablation)
        if cap_mode == "sigma":
            cap = cap_sigma * sigma_g * (1.0 - ex_p) ** 3
        else:
            cap = 0.15 * ns["ell_g"] * np.ones(N)
        alpha = np.clip(alpha, -cap, cap)
        # the binormal displacement is the *structure-directed* channel: it
        # is already gated by c = ex * strength (only fires on singular
        # points with a reliable wavefront direction), so it keeps the plain
        # noise-unit cap instead of the structure-tightened one
        cap_b = cap_sigma * sigma_g
        beta_eff = np.clip(c * beta, -cap_b, cap_b)

        X_cur = X_cur + alpha[:, None] * n_vec + beta_eff[:, None] * b_vec
        disp = np.abs(alpha) / (sigma_g + 1e-30)
        info_all.append({"cg": cg_info, "sigma_g": sigma_g,
                         "rel_noise": ns["rel_noise"],
                         "mean_c": float(c.mean()),
                         "mean_shift": float(np.abs(alpha).mean()),
                         "disp_p95": float(np.percentile(disp, 95)),
                         "disp_p99": float(np.percentile(disp, 99)),
                         "disp_max": float(disp.max()),
                         "beta_frac": float((np.abs(beta_eff)
                                             > 0.01 * sigma_g).mean()),
                         "beta_max": float(np.abs(beta_eff).max()
                                           / (sigma_g + 1e-30))})
        if verbose:
            print(f"  outer {outer}: sigma_g={sigma_g*100:.2f}cm  "
                  f"rel_noise={ns['rel_noise']:.3f}  s2={s2:.4f}  "
                  f"mean_c={c.mean():.3f}  cg_iters={cg_info['iters']}  "
                  f"rel_res={cg_info['rel_res']:.2e}  "
                  f"|alpha|={np.abs(alpha).mean()*100:.2f}cm")

    return {
        "xyz": X_cur,
        "alpha": alpha,
        "beta": beta,
        "feature_gate": c,
        "sigma_g": sigma_g,
        "rel_noise": info_all[-1]["rel_noise"],
        "info": info_all,
    }
