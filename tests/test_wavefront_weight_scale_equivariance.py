"""Eq.(6) weight-level scale equivariance: w^wf(sx) == w^wf(x)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from adawave.restoration import _estimate_descriptors, estimate_noise_and_structure


def wf_weights(X):
    d = _estimate_descriptors(X, 12, 8, 12, 0.7)
    ns = estimate_noise_and_structure(X, d["indices"], d["dists"],
                                      d["frames"], d["strength"])
    ex_p = ns["excess"].copy()
    for _ in range(2):
        ex_p = np.maximum(ex_p, 0.75 * ex_p[d["indices"]].max(axis=1))
    fr = d["frames"]
    t2 = np.cross(fr["normals"], fr["tangent1"])
    t2 /= (np.linalg.norm(t2, axis=1, keepdims=True) + 1e-12)
    p = (np.cos(d["principal_direction"])[:, None] * fr["tangent1"]
         + np.sin(d["principal_direction"])[:, None] * t2)
    diffn = X[d["indices"]] - X[:, None, :]
    diffn = diffn / (np.linalg.norm(diffn, axis=-1, keepdims=True) + 1e-12)
    r = d["strength"]
    fi = (1 - ex_p[:, None] * (1 - r[:, None] * ((diffn*p[:, None, :]).sum(-1))**2))**2
    fj = (1 - ex_p[d["indices"]] * (1 - r[d["indices"]] * ((diffn*p[d["indices"]]).sum(-1))**2))**2
    return d["neighbor_quality"] * fi * fj


def test_weight_scale_equivariance():
    rng = np.random.default_rng(3)
    xy = rng.uniform(-1.5, 1.5, (3000, 2))
    X = np.column_stack([xy[:, 0], xy[:, 1], -0.5*np.abs(xy[:, 0])]) \
        + rng.normal(0, 0.02, (3000, 3))
    w1 = wf_weights(X)
    w2 = wf_weights(37.0 * X)
    assert np.abs(w1 - w2).max() < 1e-9, np.abs(w1 - w2).max()


if __name__ == "__main__":
    test_wavefront = test_weight_scale_equivariance
    test_weight_scale_equivariance()
    print("PASS")
