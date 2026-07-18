"""Theorem 1: global rescaling scales sigma_g and displacements, leaves
gating invariant."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from adawave.restoration import denoise_wavefront_prior_v2


def test_scale_equivariance():
    rng = np.random.default_rng(1)
    X = rng.uniform(-1.5, 1.5, (4000, 2))
    X = np.column_stack([X[:, 0], X[:, 1], -0.5 * np.abs(X[:, 0])]) \
        + rng.normal(0, 0.02, (4000, 3))
    s = 37.0
    o1 = denoise_wavefront_prior_v2(X)
    o2 = denoise_wavefront_prior_v2(s * X)
    assert abs(o2["sigma_g"] / o1["sigma_g"] - s) < 1e-6 * s
    assert abs(o2["rel_noise"] - o1["rel_noise"]) < 1e-9
    d1 = o1["xyz"] - X
    d2 = o2["xyz"] - s * X
    assert np.abs(d2 - s * d1).max() < 1e-6 * s


if __name__ == "__main__":
    test_scale_equivariance()
    print("PASS")
