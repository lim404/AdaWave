"""Theorem 3: false-activation rate on a noise-only plane lies in the
selection-aware theory's predicted range (P(e>0.3) ~ 1.3% at k=12)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from adawave.restoration import (_estimate_descriptors,
                                 estimate_noise_and_structure)


def test_null_calibration():
    rates = []
    for seed in [7, 19, 101]:
        rng = np.random.default_rng(seed)
        xy = rng.uniform(-1.5, 1.5, (6000, 2))
        X = np.column_stack([xy, np.zeros(6000)]) \
            + rng.normal(0, 0.01, (6000, 3))
        d = _estimate_descriptors(X, 12, 8, 12, 0.7)
        ns = estimate_noise_and_structure(X, d["indices"], d["dists"],
                                          d["frames"], d["strength"])
        rates.append(float((ns["excess"] > 0.3).mean()))
    rate = float(np.mean(rates))
    assert 0.004 <= rate <= 0.025, rate    # theory: ~0.013


if __name__ == "__main__":
    test_null_calibration()
    print("PASS")
