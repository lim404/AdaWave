"""Module 5: Directional response analysis.

Multi-direction, multi-scale filtering on 2D density patches.
Wavefront-inspired directional singularity proxy.
"""

import numpy as np
from scipy.ndimage import convolve


def _make_directional_kernels(n_dirs: int = 8, kernel_size: int = 5
                              ) -> list[np.ndarray]:
    """Create directional derivative kernels at evenly spaced angles.

    Each kernel approximates a first-order directional derivative filter.

    Parameters
    ----------
    n_dirs : number of discrete directions.
    kernel_size : spatial extent of each kernel (odd number).

    Returns
    -------
    List of (kernel_size, kernel_size) kernels.
    """
    half = kernel_size // 2
    kernels = []
    angles = np.linspace(0, np.pi, n_dirs, endpoint=False)

    for theta in angles:
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        for r in range(-half, half + 1):
            for c in range(-half, half + 1):
                # Project (r, c) onto direction theta
                proj = r * cos_t + c * sin_t
                perp = -r * sin_t + c * cos_t
                # Directional derivative: weight by projection, Gaussian in perp
                sigma_par = half / 2.0
                sigma_perp = half / 3.0
                g = np.exp(-perp ** 2 / (2 * sigma_perp ** 2))
                kernel[r + half, c + half] = proj * g / (sigma_par ** 2)

        # Zero-mean
        kernel -= kernel.mean()
        # Normalize energy
        norm = np.sqrt((kernel ** 2).sum())
        if norm > 1e-12:
            kernel /= norm
        kernels.append(kernel)

    return kernels


def compute_directional_responses(patches: np.ndarray, n_dirs: int = 8,
                                  scales: list[int] | None = None
                                  ) -> np.ndarray:
    """Compute directional filter responses on each patch.

    Parameters
    ----------
    patches : (N, H, W) density patches.
    n_dirs : number of discrete directions.
    scales : list of kernel sizes for multi-scale analysis.
             Default: [5, 9] (two scales).

    Returns
    -------
    responses : (N, n_scales, n_dirs) absolute filter response magnitudes.
    """
    if scales is None:
        scales = [5, 9]

    n = len(patches)
    n_scales = len(scales)
    responses = np.zeros((n, n_scales, n_dirs), dtype=np.float64)

    for si, ks in enumerate(scales):
        kernels = _make_directional_kernels(n_dirs, ks)
        for i in range(n):
            patch = patches[i]
            for di, kernel in enumerate(kernels):
                # Pad kernel to patch size if needed or crop
                if ks <= patch.shape[0]:
                    resp = convolve(patch, kernel, mode='constant', cval=0.0)
                    responses[i, si, di] = np.abs(resp).sum()
                else:
                    # Kernel larger than patch: use center portion
                    resp = convolve(patch, kernel[:patch.shape[0],
                                                  :patch.shape[1]],
                                    mode='constant', cval=0.0)
                    responses[i, si, di] = np.abs(resp).sum()

    return responses
