"""Theorem 5 (deterministic clause): |dx_i| <= c_sigma sigma_g (1-e^p)^3."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from adawave.restoration import (denoise_wavefront_prior_v2,
                                 _estimate_descriptors,
                                 estimate_noise_and_structure)


def test_certificate():
    rng = np.random.default_rng(2)
    X = rng.uniform(-1.5, 1.5, (5000, 2))
    X = np.column_stack([X[:, 0], X[:, 1], -np.abs(X[:, 0])]) \
        + rng.normal(0, 0.03, (5000, 3))
    out = denoise_wavefront_prior_v2(X)
    d = _estimate_descriptors(X, 12, 8, 12, 0.7)
    ns = estimate_noise_and_structure(X, d["indices"], d["dists"],
                                      d["frames"], d["strength"])
    ex_p = ns["excess"].copy()
    for _ in range(2):
        ex_p = np.maximum(ex_p, 0.75 * ex_p[d["indices"]].max(axis=1))
    cap = 2.0 * out["sigma_g"] * (1.0 - ex_p) ** 3
    disp = np.linalg.norm(out["xyz"] - X, axis=1)
    assert np.all(disp <= cap + 1e-12)


if __name__ == "__main__":
    test_certificate()
    print("PASS")
