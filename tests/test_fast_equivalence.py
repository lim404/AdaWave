"""Equivalence of the vectorised implementation vs the frozen reference."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from adawave.neighbors import NeighborQuery
from adawave.geometry import estimate_local_frames, filter_neighbors_by_normal
from adawave.patch import build_density_patches
from adawave.directional import compute_directional_responses
from adawave.fast_geometry import (estimate_local_frames_fast,
                                   filter_neighbors_by_normal_fast,
                                   build_density_patches_fast,
                                   compute_directional_responses_fast)
from adawave.restoration import denoise_wavefront_prior_v2


def test_full_equivalence():
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, (6000, 2))
    X = np.column_stack([X, 0.3 * np.sin(X[:, 0] * 2)]) \
        + rng.normal(0, 0.02, (6000, 3))
    nq = NeighborQuery(X)
    d, idx = nq.query_knn(12)
    ref = estimate_local_frames(X, idx)
    fast = estimate_local_frames_fast(X, idx)
    assert np.allclose(ref["eigenvalues"], fast["eigenvalues"], atol=1e-12)
    assert np.all(np.abs((ref["normals"] * fast["normals"]).sum(1)) > 1 - 1e-10)
    wr = filter_neighbors_by_normal(X, idx, d, ref["normals"], 0.7, 1.5)
    wf = filter_neighbors_by_normal_fast(X, idx, d, ref["normals"], 0.7, 1.5)
    assert np.allclose(wr, wf, atol=1e-12)
    P1 = build_density_patches(X, idx, ref["normals"], ref["tangent1"],
                               ref["tangent2"], grid_size=12, neighbor_weights=wr)
    P2 = build_density_patches_fast(X, idx, ref["normals"], ref["tangent1"],
                                    ref["tangent2"], grid_size=12,
                                    neighbor_weights=wr)
    assert np.abs(P1 - P2).max() < 1e-10
    assert np.abs(compute_directional_responses(P1, 8)
                  - compute_directional_responses_fast(P1, 8)).max() < 1e-9
    out_ref = denoise_wavefront_prior_v2(X)["xyz"]
    import adawave.fast_geometry as fg
    fg.patch()
    out_fast = denoise_wavefront_prior_v2(X)["xyz"]
    assert np.abs(out_ref - out_fast).max() < 1e-9


if __name__ == "__main__":
    test_full_equivalence()
    print("PASS")
