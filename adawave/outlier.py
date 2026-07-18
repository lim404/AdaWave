"""Task 2: calibrated outlier score, o_i = med_j |n_j.(x_i-x_j)| / sigma_g.
Fixed threshold t_o = 4 corresponds to a <=1e-4 per-inlier false-rejection
budget under the null (see the paper, Sec. Q4)."""
import numpy as np
from .neighbors import NeighborQuery
from .geometry import estimate_local_frames
from .restoration import estimate_noise_and_structure

T_DEFAULT = 4.0


def calibrated_outlier_score(X, k: int = 12):
    nq = NeighborQuery(X)
    dists, idx = nq.query_knn(k)
    frames = estimate_local_frames(X, idx)
    ns = estimate_noise_and_structure(X, idx, dists, frames, np.zeros(len(X)))
    nrm = frames["normals"]
    diff = X[:, None, :] - X[idx]
    res = np.abs((diff * nrm[idx]).sum(-1))
    return np.median(res, axis=1) / (ns["sigma_g"] + 1e-12)


def remove_outliers(X, k: int = 12, threshold: float = T_DEFAULT):
    keep = calibrated_outlier_score(X, k) <= threshold
    return X[keep], keep
