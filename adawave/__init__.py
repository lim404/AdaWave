"""AdaWave: self-calibrated selective restoration of irregular point sets.

The default implementation is the FROZEN reference (loop-based geometry
stages); `fast_geometry.patch()` swaps in the vectorised, numerically
equivalent implementations (see tests/test_fast_equivalence.py).
"""
from .restoration import denoise_wavefront_prior_v2 as restore
from .calibration import estimate_noise_and_structure
from .outlier import calibrated_outlier_score
