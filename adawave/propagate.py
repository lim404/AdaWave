"""Module 6b: Wavefront-inspired structural propagation.

Propagates singularity strength along geometrically and directionally
consistent neighborhoods. The propagated field is used as a continuous
structure-proximity weight in the denoising step, creating smooth
"buffer zones" around detected features without changing the binary
feature classification.
"""

import numpy as np


def propagate_singularity(xyz: np.ndarray, neighbor_indices: np.ndarray,
                          normals: np.ndarray, tangent1: np.ndarray,
                          singularity_strength: np.ndarray,
                          principal_direction: np.ndarray,
                          neighbor_quality: np.ndarray = None,
                          lam: float = 0.4, n_rounds: int = 2,
                          alpha: float = 2.0, beta: float = 2.0,
                          gamma: float = 2.0,
                          sigma_d_factor: float = 2.0) -> np.ndarray:
    """Propagate singularity strength via asymmetric max-propagation.

    Each point can only be boosted (never lowered) by neighbors with
    higher singularity strength, subject to geometric and directional
    consistency constraints.

    Parameters
    ----------
    xyz : (N, 3) point positions.
    neighbor_indices : (N, k) neighbor indices.
    normals : (N, 3) local normals.
    tangent1 : (N, 3) first tangent direction.
    singularity_strength : (N,) original singularity strength in [0, 1].
    principal_direction : (N,) principal angle in [0, pi).
    neighbor_quality : (N, k) optional normal-consistency weights.
    lam : float, propagation strength.
    n_rounds : int, number of propagation rounds.
    alpha : float, exponent for normal similarity term.
    beta : float, exponent for principal direction consistency term.
    gamma : float, exponent for directional alignment term.
    sigma_d_factor : float, spatial sigma as multiple of median neighbor distance.

    Returns
    -------
    propagated : (N,) propagated singularity strength in [0, 1].
    """
    n, k = neighbor_indices.shape
    current = singularity_strength.copy()

    # Precompute principal direction 3D vectors
    t2 = np.cross(normals, tangent1)
    t2_norms = np.linalg.norm(t2, axis=1, keepdims=True)
    t2 = np.where(t2_norms > 1e-12, t2 / t2_norms, 0.0)

    cos_theta = np.cos(principal_direction)
    sin_theta = np.sin(principal_direction)
    principal_3d = cos_theta[:, None] * tangent1 + sin_theta[:, None] * t2

    # Spatial sigma
    nn1_dists = np.linalg.norm(xyz[neighbor_indices[:, 0]] - xyz, axis=1)
    sigma_d = sigma_d_factor * np.median(nn1_dists)
    sigma_d_sq = sigma_d ** 2

    for _ in range(n_rounds):
        new_strength = current.copy()

        for i in range(n):
            nbr_idx = neighbor_indices[i]
            nbr_strength = current[nbr_idx]

            # Only consider neighbors with higher strength
            higher_mask = nbr_strength > current[i]
            if not higher_mask.any():
                continue

            nbr_pts = xyz[nbr_idx]
            diff = nbr_pts - xyz[i]
            dists_sq = np.sum(diff ** 2, axis=1)

            w_dist = np.exp(-dists_sq / sigma_d_sq)
            cos_normal = np.abs(normals[nbr_idx] @ normals[i])
            w_normal = np.power(np.maximum(cos_normal, 0.0), alpha)

            angle_diff = principal_direction[nbr_idx] - principal_direction[i]
            cos_angle = np.cos(angle_diff)
            w_angle = np.power(np.maximum(cos_angle, 0.0), beta)

            dists = np.sqrt(dists_sq)
            d_hat = diff / (dists[:, None] + 1e-12)
            align = np.abs(d_hat @ principal_3d[i])
            w_align = np.power(np.maximum(align, 0.0), gamma)

            w = w_dist * w_normal * w_angle * w_align
            if neighbor_quality is not None:
                w *= neighbor_quality[i]
            w *= higher_mask.astype(np.float64)

            w_sum = w.sum()
            if w_sum < 1e-12:
                continue

            w_norm = w / w_sum
            propagated_val = np.dot(w_norm, nbr_strength)
            boosted = (1.0 - lam) * current[i] + lam * propagated_val
            new_strength[i] = max(current[i], boosted)

        current = new_strength

    np.clip(current, 0.0, 1.0, out=current)
    return current
