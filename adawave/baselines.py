"""Baseline denoising methods for comparison.

Implements classical and advanced point cloud denoising baselines:
  - MLS  (Moving Least Squares — normal-guided projection)
  - WLOP (Weighted Locally Optimal Projection)
  - GLR  (Graph Laplacian Regularization)
  - Jet  (Local quadratic surface fitting)
  - CLOP (Edge-aware bilateral normal filtering)
"""

import numpy as np
from scipy.spatial import cKDTree


# ========================================================================
# Helper: estimate PCA normals
# ========================================================================

def _estimate_normals(xyz, indices):
    """PCA-based normal estimation."""
    n = len(xyz)
    normals = np.zeros((n, 3))
    for i in range(n):
        nbr = xyz[indices[i]]
        cov = np.cov(nbr.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normals[i] = eigvecs[:, 0]  # smallest eigenvalue
    flip = normals[:, 2] < 0
    normals[flip] *= -1
    return normals


# ========================================================================
# MLS — Moving Least Squares
# ========================================================================

def denoise_mls(xyz: np.ndarray, k: int = 12, n_iters: int = 2,
                step: float = 0.5) -> np.ndarray:
    """Moving Least Squares denoising (normal-guided).

    Fits a weighted local plane to neighbors and projects the point
    along the normal direction onto that plane.

    References:
        Alexa et al. "Computing and rendering point set surfaces." (2003)
        Levin, D. "Mesh-independent surface interpolation." (2004)
    """
    current = xyz.copy()
    for _ in range(n_iters):
        tree = cKDTree(current)
        dists, indices = tree.query(current, k=k + 1)
        dists = dists[:, 1:]
        indices = indices[:, 1:]

        normals = _estimate_normals(current, indices)
        median_spacing = np.median(dists[:, 0])

        new_xyz = current.copy()
        for i in range(len(current)):
            nbr = current[indices[i]]
            d = dists[i]
            sigma = np.median(d)
            if sigma < 1e-12:
                continue

            w = np.exp(-0.5 * (d / sigma) ** 2)
            w_sum = w.sum()
            centroid = (w[:, None] * nbr).sum(axis=0) / w_sum

            # Project along normal only
            offset = current[i] - centroid
            proj_dist = np.dot(offset, normals[i])
            proj_dist = np.clip(proj_dist, -0.5 * median_spacing,
                                0.5 * median_spacing)
            new_xyz[i] -= step * proj_dist * normals[i]

        current = new_xyz
    return current


# ========================================================================
# WLOP — Weighted Locally Optimal Projection
# ========================================================================

def denoise_wlop(xyz: np.ndarray, k: int = 20, n_iters: int = 2,
                 h_factor: float = 4.0) -> np.ndarray:
    """Normal-guided Locally Optimal Projection denoising.

    Projects points toward the locally optimal position along surface
    normals using density-weighted attraction. Adapted from WLOP for
    denoising rather than consolidation.

    References:
        Huang et al. "Consolidation of unorganized point clouds for
        surface reconstruction." (SIGGRAPH Asia 2009)
    """
    n = len(xyz)
    current = xyz.copy()

    tree_input = cKDTree(xyz)
    nn_dists, _ = tree_input.query(xyz, k=2)
    h = h_factor * np.median(nn_dists[:, 1])

    def wendland(r, h_val):
        t = np.clip(r / h_val, 0, 1)
        return (1 - t) ** 4 * (4 * t + 1)

    # Local density for input weighting
    d_in, _ = tree_input.query(xyz, k=min(k, n))
    v_input = 1.0 / (np.mean(d_in[:, 1:], axis=1) + 1e-12)
    v_input /= v_input.mean()

    for _ in range(n_iters):
        tree = cKDTree(current)
        dists, indices = tree.query(current, k=k + 1)
        dists = dists[:, 1:]
        indices = indices[:, 1:]

        normals = _estimate_normals(current, indices)
        median_spacing = np.median(dists[:, 0])

        new_xyz = current.copy()
        for i in range(n):
            # Density-weighted attraction toward input neighbors
            d_qi, idx_qi = tree_input.query(current[i], k=min(k, n))
            nbr_pts = xyz[idx_qi]
            w = wendland(d_qi, h) / (v_input[idx_qi] + 1e-12)
            w_sum = w.sum()
            if w_sum < 1e-12:
                continue

            # Weighted centroid
            centroid = (w[:, None] * nbr_pts).sum(axis=0) / w_sum

            # Project displacement along normal
            disp = centroid - current[i]
            shift = np.dot(disp, normals[i])
            shift = np.clip(shift, -0.5 * median_spacing,
                            0.5 * median_spacing)
            new_xyz[i] += shift * normals[i]

        current = new_xyz
    return current


# ========================================================================
# GLR — Graph Laplacian Regularization
# ========================================================================

def denoise_glr(xyz: np.ndarray, k: int = 12, mu: float = 0.5,
                n_iters: int = 3) -> np.ndarray:
    """Normal-guided Graph Laplacian Regularization denoising.

    Iterative Laplacian smoothing projected along local normals,
    preventing cross-surface mixing in multi-class LiDAR scenes.

    References:
        Zeng et al. "3D point cloud denoising using graph Laplacian
        regularization." (IEEE TIP 2020)
    """
    current = xyz.copy()
    for _ in range(n_iters):
        tree = cKDTree(current)
        dists, indices = tree.query(current, k=k + 1)
        dists = dists[:, 1:]
        indices = indices[:, 1:]

        n_pts = len(current)
        normals = _estimate_normals(current, indices)
        median_spacing = np.median(dists[:, 0])
        global_sigma = np.median(dists) * 1.5

        new_xyz = current.copy()
        for i in range(n_pts):
            nbr = current[indices[i]]
            d = dists[i]
            w = np.exp(-0.5 * (d / global_sigma) ** 2)

            # Compute Laplacian displacement along normal
            diff = nbr - current[i]
            normal_proj = diff @ normals[i]

            w_sum = w.sum()
            if w_sum < 1e-12:
                continue

            # Weighted average normal displacement (graph Laplacian update)
            lap_shift = np.dot(w, normal_proj) / w_sum
            lap_shift = np.clip(lap_shift, -0.5 * median_spacing,
                                0.5 * median_spacing)
            new_xyz[i] += mu * lap_shift * normals[i]

        current = new_xyz
    return current


# ========================================================================
# Jet fitting — Local quadratic surface fitting
# ========================================================================

def denoise_jet(xyz: np.ndarray, k: int = 20, n_iters: int = 2,
                step: float = 0.5) -> np.ndarray:
    """Jet fitting denoising.

    For each point, fits a local quadratic surface (osculating jet) and
    projects the point along the normal onto the surface.

    References:
        Cazals & Pouget. "Estimating differential quantities using
        polynomial fitting of osculating jets." (SGP 2003)
    """
    current = xyz.copy()
    for _ in range(n_iters):
        tree = cKDTree(current)
        dists, indices = tree.query(current, k=k + 1)
        dists = dists[:, 1:]
        indices = indices[:, 1:]

        normals = _estimate_normals(current, indices)
        median_spacing = np.median(dists[:, 0])

        new_xyz = current.copy()
        for i in range(len(current)):
            nbr = current[indices[i]]
            d = dists[i]
            sigma = np.median(d)
            if sigma < 1e-12:
                continue

            # Local frame
            diff_c = nbr - nbr.mean(axis=0)
            cov = diff_c.T @ diff_c / len(nbr)
            eigvals, eigvecs = np.linalg.eigh(cov)
            idx_sort = np.argsort(eigvals)[::-1]
            eigvecs = eigvecs[:, idx_sort]
            t1, t2, normal = eigvecs[:, 0], eigvecs[:, 1], eigvecs[:, 2]
            if normal[2] < 0:
                normal = -normal

            # Project into local frame
            local_diff = nbr - current[i]
            u = local_diff @ t1
            v = local_diff @ t2
            h = local_diff @ normal

            w = np.exp(-0.5 * (d / sigma) ** 2)

            # Quadratic fit: h = a*u^2 + b*u*v + c*v^2 + d*u + e*v + f
            A_mat = np.column_stack([
                u ** 2, u * v, v ** 2, u, v, np.ones(len(u))
            ])
            W_sqrt = np.sqrt(w)
            AW = A_mat * W_sqrt[:, None]
            hW = h * W_sqrt
            try:
                coeffs, _, _, _ = np.linalg.lstsq(AW, hW, rcond=None)
                h_fit = coeffs[5]  # height at origin
            except np.linalg.LinAlgError:
                continue

            # Project along PCA normal with step control
            h_fit = np.clip(h_fit, -0.5 * median_spacing,
                            0.5 * median_spacing)
            new_xyz[i] += step * h_fit * normals[i]

        current = new_xyz
    return current


# ========================================================================
# CLOP — Edge-Aware Bilateral Normal Filtering
# ========================================================================

def denoise_clop(xyz: np.ndarray, k: int = 12, n_iters: int = 2,
                 edge_sigma: float = 0.3) -> np.ndarray:
    """Edge-aware point cloud denoising (bilateral normal filtering).

    First bilaterally filters normals to preserve sharp features, then
    updates vertex positions along the filtered normals.

    References:
        Zheng et al. "Bilateral normal filtering for mesh denoising."
        (IEEE TVCG 2011), adapted to point clouds.
        Huang et al. "Edge-aware point set resampling." (TOG 2013)
    """
    current = xyz.copy()

    for _ in range(n_iters):
        tree = cKDTree(current)
        dists, indices = tree.query(current, k=k + 1)
        dists = dists[:, 1:]
        indices = indices[:, 1:]
        n = len(current)

        normals = _estimate_normals(current, indices)

        # Step 1: Bilateral normal filtering
        filtered_normals = normals.copy()
        for i in range(n):
            nbr_normals = normals[indices[i]]
            d = dists[i]
            sigma_s = np.median(d)
            if sigma_s < 1e-12:
                continue

            w_s = np.exp(-0.5 * (d / sigma_s) ** 2)
            cos_sim = nbr_normals @ normals[i]
            w_n = np.exp(-0.5 * ((1 - np.abs(cos_sim)) / edge_sigma) ** 2)

            w = w_s * w_n
            w_sum = w.sum()
            if w_sum < 1e-12:
                continue

            avg_normal = (w[:, None] * nbr_normals).sum(axis=0) / w_sum
            norm = np.linalg.norm(avg_normal)
            if norm > 1e-12:
                filtered_normals[i] = avg_normal / norm

        # Step 2: Vertex update along filtered normals
        new_xyz = current.copy()
        median_spacing = np.median(dists[:, 0])
        for i in range(n):
            nbr = current[indices[i]]
            diff = nbr - current[i]
            d = dists[i]
            sigma_s = np.median(d)
            if sigma_s < 1e-12:
                continue

            n_vec = filtered_normals[i]
            proj = diff @ n_vec
            sigma_r = 0.3 * sigma_s
            w = np.exp(-0.5 * (d / sigma_s) ** 2) * \
                np.exp(-0.5 * (proj / sigma_r) ** 2)
            w_sum = w.sum()
            if w_sum < 1e-12:
                continue
            shift = np.dot(w, proj) / w_sum
            shift = np.clip(shift, -0.5 * median_spacing,
                            0.5 * median_spacing)
            new_xyz[i] += 0.2 * shift * n_vec

        current = new_xyz

    return current
