"""Module 4: Local patch construction.

Project neighborhood points onto local tangent plane and build 2D patches.
Supports density patch and height patch modes.
"""

import numpy as np


def build_density_patches(xyz: np.ndarray, neighbor_indices: np.ndarray,
                          normals: np.ndarray, tangent1: np.ndarray,
                          tangent2: np.ndarray, grid_size: int = 16,
                          sigma: float = 1.0,
                          neighbor_weights: np.ndarray = None) -> np.ndarray:
    """Build 2D density patches for all points.

    For each point, project its neighbors onto the local tangent plane,
    then accumulate a Gaussian-weighted density on a 2D grid.

    Parameters
    ----------
    xyz : (N, 3) point positions.
    neighbor_indices : (N, k) neighbor indices.
    normals : (N, 3) local normals (unused here but kept for API consistency).
    tangent1 : (N, 3) first tangent direction.
    tangent2 : (N, 3) second tangent direction.
    grid_size : int, size of the square patch grid.
    sigma : float, Gaussian kernel width in grid units.
    neighbor_weights : (N, k) optional per-neighbor quality weights.

    Returns
    -------
    patches : (N, grid_size, grid_size) density patches.
    """
    n = len(xyz)
    k = neighbor_indices.shape[1]
    half = grid_size // 2
    patches = np.zeros((n, grid_size, grid_size), dtype=np.float64)

    for i in range(n):
        center = xyz[i]
        nbr = xyz[neighbor_indices[i]]
        diff = nbr - center

        u = diff @ tangent1[i]
        v = diff @ tangent2[i]

        dists = np.sqrt(u ** 2 + v ** 2)
        scale = np.median(dists)
        if scale < 1e-12:
            scale = 1.0

        ug = u / scale * (half - 1) + half
        vg = v / scale * (half - 1) + half

        for j in range(k):
            # Apply neighbor quality weight if available
            nw = neighbor_weights[i, j] if neighbor_weights is not None else 1.0
            if nw < 0.01:
                continue

            gi = ug[j]
            gj = vg[j]
            gi_lo = max(int(np.floor(gi)) - 1, 0)
            gi_hi = min(int(np.ceil(gi)) + 1, grid_size - 1)
            gj_lo = max(int(np.floor(gj)) - 1, 0)
            gj_hi = min(int(np.ceil(gj)) + 1, grid_size - 1)

            for r in range(gi_lo, gi_hi + 1):
                for c in range(gj_lo, gj_hi + 1):
                    d2 = (r - gi) ** 2 + (c - gj) ** 2
                    patches[i, r, c] += nw * np.exp(-d2 / (2 * sigma ** 2))

        pmax = patches[i].max()
        if pmax > 1e-12:
            patches[i] /= pmax

    return patches


def build_height_patches(xyz: np.ndarray, neighbor_indices: np.ndarray,
                         normals: np.ndarray, tangent1: np.ndarray,
                         tangent2: np.ndarray, grid_size: int = 16,
                         sigma: float = 1.0,
                         neighbor_weights: np.ndarray = None) -> np.ndarray:
    """Build 2D height patches for all points.

    Like density patches, but records the signed height (normal projection)
    relative to the local tangent plane, rather than point density.
    More sensitive to surface geometry variations like roof edges and facades.

    Parameters
    ----------
    xyz : (N, 3) point positions.
    neighbor_indices : (N, k) neighbor indices.
    normals : (N, 3) local normals.
    tangent1 : (N, 3) first tangent direction.
    tangent2 : (N, 3) second tangent direction.
    grid_size : int, size of the square patch grid.
    sigma : float, Gaussian kernel width in grid units.
    neighbor_weights : (N, k) optional per-neighbor quality weights.

    Returns
    -------
    patches : (N, grid_size, grid_size) height patches.
    """
    n = len(xyz)
    k = neighbor_indices.shape[1]
    half = grid_size // 2
    patches = np.zeros((n, grid_size, grid_size), dtype=np.float64)
    weight_acc = np.zeros((n, grid_size, grid_size), dtype=np.float64)

    for i in range(n):
        center = xyz[i]
        nbr = xyz[neighbor_indices[i]]
        diff = nbr - center

        u = diff @ tangent1[i]
        v = diff @ tangent2[i]
        h = diff @ normals[i]  # signed height above tangent plane

        dists = np.sqrt(u ** 2 + v ** 2)
        scale = np.median(dists)
        if scale < 1e-12:
            scale = 1.0

        ug = u / scale * (half - 1) + half
        vg = v / scale * (half - 1) + half

        for j in range(k):
            nw = neighbor_weights[i, j] if neighbor_weights is not None else 1.0
            if nw < 0.01:
                continue

            gi = ug[j]
            gj = vg[j]
            gi_lo = max(int(np.floor(gi)) - 1, 0)
            gi_hi = min(int(np.ceil(gi)) + 1, grid_size - 1)
            gj_lo = max(int(np.floor(gj)) - 1, 0)
            gj_hi = min(int(np.ceil(gj)) + 1, grid_size - 1)

            for r in range(gi_lo, gi_hi + 1):
                for c in range(gj_lo, gj_hi + 1):
                    d2 = (r - gi) ** 2 + (c - gj) ** 2
                    w = nw * np.exp(-d2 / (2 * sigma ** 2))
                    patches[i, r, c] += w * h[j]
                    weight_acc[i, r, c] += w

        # Normalize: weighted average height per cell
        valid = weight_acc[i] > 1e-12
        patches[i, valid] /= weight_acc[i, valid]

        # Normalize to [-1, 1] range
        amax = np.abs(patches[i]).max()
        if amax > 1e-12:
            patches[i] /= amax

    return patches
