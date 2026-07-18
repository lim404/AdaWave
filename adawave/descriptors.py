"""Module 6: Structure descriptor computation.

Computes singularity strength, principal direction, and directional entropy
from directional filter responses.
"""

import numpy as np


def compute_descriptors(responses: np.ndarray, n_dirs: int = 8
                        ) -> dict:
    """Compute three core descriptors from directional responses.

    Parameters
    ----------
    responses : (N, n_scales, n_dirs) directional response magnitudes.
    n_dirs : number of discrete directions.

    Returns
    -------
    dict with:
        'singularity_strength' : (N,) max response across scales and directions.
        'principal_direction'  : (N,) angle in [0, pi) of strongest response.
        'directional_entropy'  : (N,) entropy of directional response distribution.
        'is_feature'           : (N,) bool mask for feature points.
    """
    n = len(responses)
    angles = np.linspace(0, np.pi, n_dirs, endpoint=False)

    # Aggregate across scales: take max over scales for each direction
    # responses shape: (N, n_scales, n_dirs)
    resp_max_scale = responses.max(axis=1)  # (N, n_dirs)

    # 1. Singularity strength: max response across all directions
    singularity_strength = resp_max_scale.max(axis=1)  # (N,)

    # 2. Principal direction: angle of max response
    best_dir_idx = resp_max_scale.argmax(axis=1)  # (N,)
    principal_direction = angles[best_dir_idx]  # (N,)

    # 3. Directional entropy
    directional_entropy = np.zeros(n, dtype=np.float64)
    for i in range(n):
        p = resp_max_scale[i].copy()
        p_sum = p.sum()
        if p_sum > 1e-12:
            p /= p_sum
            # Avoid log(0)
            p_clipped = np.clip(p, 1e-12, None)
            directional_entropy[i] = -np.sum(p * np.log(p_clipped))
        else:
            directional_entropy[i] = 0.0

    # Normalize singularity strength to [0, 1]
    ss_max = singularity_strength.max()
    if ss_max > 1e-12:
        singularity_strength /= ss_max

    # Classify feature points:
    # Use top 5% by singularity strength — conservative to avoid under-smoothing
    # the many false-positive interior points that a lower threshold would include.
    ss_threshold = np.percentile(singularity_strength, 95)
    is_feature = singularity_strength > ss_threshold

    return {
        "singularity_strength": singularity_strength,
        "principal_direction": principal_direction,
        "directional_entropy": directional_entropy,
        "is_feature": is_feature,
    }
