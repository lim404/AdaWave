"""Vectorised drop-in implementations of the geometry stage.

Same mathematics as `geometry.py` (batched np.linalg.eigh over the
covariance stack; broadcast neighbour filtering); intended to be
numerically equivalent up to floating-point reduction order. Enable via
`fast_geometry.patch()` AFTER the equivalence test
(`test_fast_equivalence.py`) passes on your platform. The frozen
reference implementation remains the default.
"""
from __future__ import annotations

import numpy as np


def estimate_local_frames_fast(xyz: np.ndarray, neighbor_indices: np.ndarray) -> dict:
    nbr = xyz[neighbor_indices]                       # (N, k, 3)
    mean = nbr.mean(axis=1, keepdims=True)
    d = nbr - mean
    k = nbr.shape[1]
    cov = np.einsum("nki,nkj->nij", d, d) / (k - 1)   # matches np.cov ddof=1
    eigvals, eigvecs = np.linalg.eigh(cov)            # ascending
    # descending order, as in the loop implementation
    order = np.argsort(eigvals, axis=1)[:, ::-1]
    eigvals = np.take_along_axis(eigvals, order, axis=1)
    eigvecs = np.take_along_axis(eigvecs, order[:, None, :], axis=2)
    tangent1 = eigvecs[:, :, 0]
    tangent2 = eigvecs[:, :, 1]
    normals = eigvecs[:, :, 2].copy()
    flip = normals[:, 2] < 0
    normals[flip] *= -1
    return {"normals": normals, "tangent1": tangent1,
            "tangent2": tangent2, "eigenvalues": eigvals}

def filter_neighbors_by_normal_fast(xyz, neighbor_indices, neighbor_dists,
                                    normals, normal_thresh: float = 0.3,
                                    residual_sigma_factor: float = 2.0):
    nbr_normals = normals[neighbor_indices]                    # (N, k, 3)
    cos_sim = np.abs(np.einsum("nkj,nj->nk", nbr_normals, normals))
    w_normal = np.clip((cos_sim - normal_thresh) / (1.0 - normal_thresh), 0, 1)
    diff = xyz[neighbor_indices] - xyz[:, None, :]
    residuals = np.abs(np.einsum("nkj,nj->nk", diff, normals))
    sigma_r = residual_sigma_factor * np.median(neighbor_dists, axis=1)
    sigma_r = np.where(sigma_r < 1e-12, 1.0, sigma_r)
    w_residual = np.exp(-0.5 * (residuals / sigma_r[:, None]) ** 2)
    return w_normal * w_residual

def compute_directional_responses_fast(patches: np.ndarray, n_dirs: int = 8,
                                       scales=None) -> np.ndarray:
    """Batched version of directional.compute_directional_responses:
    identical scipy convolution applied over the whole patch stack."""
    from scipy.ndimage import convolve
    from adawave.directional import _make_directional_kernels
    if scales is None:
        scales = [5, 9]
    n = len(patches)
    responses = np.zeros((n, len(scales), n_dirs), dtype=np.float64)
    for si, ks in enumerate(scales):
        kernels = _make_directional_kernels(n_dirs, ks)
        for di, kernel in enumerate(kernels):
            kk = kernel if ks <= patches.shape[1] else kernel[:patches.shape[1],
                                                             :patches.shape[2]]
            resp = convolve(patches, kk[None, :, :], mode="constant", cval=0.0)
            responses[:, si, di] = np.abs(resp).sum(axis=(1, 2))
    return responses

def build_density_patches_fast(xyz, neighbor_indices, normals, tangent1,
                               tangent2, grid_size: int = 12,
                               sigma: float = 1.0, neighbor_weights=None):
    """Vectorised scatter version of patch.build_density_patches
    (pure Gaussian density accumulation, max-normalised)."""
    n, k = neighbor_indices.shape
    half = grid_size // 2
    diff = xyz[neighbor_indices] - xyz[:, None, :]              # (N,k,3)
    u = np.einsum("nkj,nj->nk", diff, tangent1)
    v = np.einsum("nkj,nj->nk", diff, tangent2)
    scale = np.median(np.sqrt(u * u + v * v), axis=1)
    scale = np.where(scale < 1e-12, 1.0, scale)[:, None]
    ug = u / scale * (half - 1) + half
    vg = v / scale * (half - 1) + half
    nw = np.ones((n, k)) if neighbor_weights is None else neighbor_weights
    active = nw >= 0.01

    gi0 = np.floor(ug).astype(np.int64)
    gj0 = np.floor(vg).astype(np.int64)
    gi_lo = np.maximum(gi0 - 1, 0)
    gi_hi = np.minimum(np.ceil(ug).astype(np.int64) + 1, grid_size - 1)
    gj_lo = np.maximum(gj0 - 1, 0)
    gj_hi = np.minimum(np.ceil(vg).astype(np.int64) + 1, grid_size - 1)

    patches = np.zeros((n, grid_size, grid_size))
    ii = np.arange(n)[:, None]
    for dr in range(-1, 3):
        for dc in range(-1, 3):
            r = gi0 + dr
            c = gj0 + dc
            ok = (active & (r >= gi_lo) & (r <= gi_hi)
                  & (c >= gj_lo) & (c <= gj_hi))
            d2 = (r - ug) ** 2 + (c - vg) ** 2
            w = np.where(ok, nw * np.exp(-d2 / (2 * sigma ** 2)), 0.0)
            ri = np.clip(r, 0, grid_size - 1)
            ci = np.clip(c, 0, grid_size - 1)
            flat = (ii * grid_size + ri) * grid_size + ci
            np.add.at(patches.reshape(-1), flat.ravel(), w.ravel())
    pmax = patches.max(axis=(1, 2), keepdims=True)
    patches = np.where(pmax > 1e-12, patches / np.where(pmax > 1e-12, pmax, 1.0),
                       patches)
    return patches

def patch():
    """Swap the fast implementations into the already-imported modules."""
    import adawave.geometry as g
    import adawave.patch as p
    import adawave.directional as dd
    import adawave.restoration as w2
    g.estimate_local_frames = estimate_local_frames_fast
    g.filter_neighbors_by_normal = filter_neighbors_by_normal_fast
    p.build_density_patches = build_density_patches_fast
    dd.compute_directional_responses = compute_directional_responses_fast
    w2.estimate_local_frames = estimate_local_frames_fast
    w2.filter_neighbors_by_normal = filter_neighbors_by_normal_fast
    w2.build_density_patches = build_density_patches_fast
    w2.compute_directional_responses = compute_directional_responses_fast
