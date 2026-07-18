"""Module 3: Local geometry estimation.

PCA-based normal estimation and local tangent plane construction.
Includes normal-consistent neighbor filtering to reduce cross-surface contamination.
"""

import numpy as np


def estimate_local_frames(xyz: np.ndarray, neighbor_indices: np.ndarray
                          ) -> dict:
    """Estimate local coordinate frames via PCA for each point.

    Parameters
    ----------
    xyz : (N, 3) point positions.
    neighbor_indices : (N, k) indices of k nearest neighbors.

    Returns
    -------
    dict with:
        'normals'    : (N, 3)   local normal direction (smallest eigenvalue)
        'tangent1'   : (N, 3)   first tangent direction (largest eigenvalue)
        'tangent2'   : (N, 3)   second tangent direction
        'eigenvalues': (N, 3)   sorted eigenvalues (descending)
    """
    n = len(xyz)
    normals = np.zeros((n, 3), dtype=np.float64)
    tangent1 = np.zeros((n, 3), dtype=np.float64)
    tangent2 = np.zeros((n, 3), dtype=np.float64)
    eigenvalues = np.zeros((n, 3), dtype=np.float64)

    for i in range(n):
        nbr = xyz[neighbor_indices[i]]
        cov = np.cov(nbr.T)
        eigvals, eigvecs = np.linalg.eigh(cov)

        # eigh returns ascending order; reverse to descending
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        tangent1[i] = eigvecs[:, 0]
        tangent2[i] = eigvecs[:, 1]
        normals[i] = eigvecs[:, 2]
        eigenvalues[i] = eigvals

    # Consistent normal orientation: make normals point "upward" (positive Z)
    flip = normals[:, 2] < 0
    normals[flip] *= -1

    return {
        "normals": normals,
        "tangent1": tangent1,
        "tangent2": tangent2,
        "eigenvalues": eigenvalues,
    }


def filter_neighbors_by_normal(xyz: np.ndarray, neighbor_indices: np.ndarray,
                               neighbor_dists: np.ndarray,
                               normals: np.ndarray,
                               normal_thresh: float = 0.3,
                               residual_sigma_factor: float = 2.0
                               ) -> np.ndarray:
    """Filter kNN neighbors by normal consistency and plane residual.

    For each point, compute a weight in [0, 1] for each neighbor based on:
    1. Normal similarity: cos angle between point normal and neighbor normal.
    2. Plane residual: how far the neighbor lies from the point's tangent plane.

    Parameters
    ----------
    xyz : (N, 3)
    neighbor_indices : (N, k)
    neighbor_dists : (N, k)
    normals : (N, 3)
    normal_thresh : float, minimum normal dot product to keep full weight.
    residual_sigma_factor : float, plane residual sigma as fraction of median dist.

    Returns
    -------
    weights : (N, k) neighbor quality weights in [0, 1].
    """
    n, k = neighbor_indices.shape
    weights = np.ones((n, k), dtype=np.float64)

    for i in range(n):
        ni = normals[i]
        nbr_idx = neighbor_indices[i]
        nbr_pts = xyz[nbr_idx]

        # 1. Normal similarity weight
        nbr_normals = normals[nbr_idx]  # (k, 3)
        cos_sim = np.abs(nbr_normals @ ni)  # (k,), abs because normal sign may differ
        # Soft threshold: full weight above normal_thresh, decays below
        w_normal = np.clip((cos_sim - normal_thresh) / (1.0 - normal_thresh), 0, 1)

        # 2. Plane residual weight
        diff = nbr_pts - xyz[i]  # (k, 3)
        residuals = np.abs(diff @ ni)  # (k,) distance to tangent plane
        sigma_r = residual_sigma_factor * np.median(neighbor_dists[i])
        if sigma_r < 1e-12:
            sigma_r = 1.0
        w_residual = np.exp(-0.5 * (residuals / sigma_r) ** 2)

        weights[i] = w_normal * w_residual

    return weights
