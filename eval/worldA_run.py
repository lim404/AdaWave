"""World A runner — executes the pre-registered spec (worldA_spec.py) ONCE.

Point-to-surface RMSE against a 200k dense reference sample of each shape,
split into near-singular (<= 3*ell of singular set) and smooth regions.
Writes worldA_results.csv.
"""
import sys, csv, time
sys.path.insert(0, ".")
import numpy as np
from scipy.spatial import cKDTree

from worldA_spec import (SHAPES, NOISE_VARIANTS, SEEDS, SIGMAS, ELL,
                         N_PTS, N_REF)
from eval_paper import baseline_bilateral_normal
from adawave.baselines import denoise_mls, denoise_glr
from adawave.wavefront_prior import denoise_wavefront_prior
from adawave.restoration import denoise_wavefront_prior_v2

METHODS = [
    ("Bilateral", lambda x: baseline_bilateral_normal(x, k=12, n_iters=2, weight=0.2)),
    ("MLS",       lambda x: denoise_mls(x, k=12, n_iters=2)),
    ("GLR",       lambda x: denoise_glr(x, k=12)),
    ("Prior-v1",  lambda x: denoise_wavefront_prior(
        x, k=12, n_dirs=8, grid_size=12, lam_n=1.0, lam_b=2.0, n_iters=80,
        step=0.05, momentum=0.5, max_shift_factor=0.15, normal_thresh=0.7,
        feature_anchor=100.0)["xyz"]),
    ("Prior-v2",  lambda x: denoise_wavefront_prior_v2(x)["xyz"]),
]


def add_noise(pts, sigma, kind, rng):
    if kind == "gauss":
        return pts + rng.normal(0, sigma, pts.shape)
    if kind == "laplace":
        return pts + rng.laplace(0, sigma / np.sqrt(2.0), pts.shape)
    if kind == "outlier":
        noisy = pts + rng.normal(0, sigma, pts.shape)
        m = rng.uniform(0, 1, len(pts)) < 0.05
        noisy[m] += rng.uniform(-10 * ELL, 10 * ELL, (int(m.sum()), 3))
        return noisy
    raise ValueError(kind)


def p2s_rmse(pts, ref_tree):
    d, _ = ref_tree.query(pts, k=1)
    return float(np.sqrt(np.mean(d ** 2)))


def imp(noi, den):
    return (1 - den / noi) * 100 if noi > 1e-15 else float("nan")


rows = []
for sname, sampler in SHAPES:
    rng_ref = np.random.default_rng(0)
    ref, singdist = sampler(N_REF, rng_ref)
    ref_tree = cKDTree(ref)
    configs = [(s, "gauss") for s in SIGMAS]
    configs += [(1.0 * ELL, kind) for (vs, kind) in NOISE_VARIANTS if vs == sname]
    for sigma, kind in configs:
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            clean, _ = sampler(N_PTS, rng)
            noisy = add_noise(clean, sigma, kind, rng)
            near = None
            if singdist is not None:
                near = singdist(clean) <= 3 * ELL
            noi_all = p2s_rmse(noisy, ref_tree)
            noi_near = p2s_rmse(noisy[near], ref_tree) if near is not None and near.sum() > 20 else None
            noi_sm = p2s_rmse(noisy[~near], ref_tree) if near is not None else noi_all
            for mname, fn in METHODS:
                t0 = time.time()
                d = fn(noisy)
                if len(d) != len(noisy):
                    pass  # all listed methods preserve count
                r = dict(shape=sname, sigma=round(sigma / ELL, 2), noise=kind,
                         seed=seed, method=mname,
                         overall=imp(noi_all, p2s_rmse(d, ref_tree)),
                         near=imp(noi_near, p2s_rmse(d[near], ref_tree)) if noi_near else float("nan"),
                         smooth=imp(noi_sm, p2s_rmse(d[~near] if near is not None else d, ref_tree)),
                         time=time.time() - t0)
                rows.append(r)
            print(f"{sname:<12} s={sigma/ELL:.1f}ell {kind:<7} seed={seed} done",
                  flush=True)

with open("worldA_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

# summary: mean over seeds, per shape x sigma(gauss), per method
print("\n===== World A summary (gauss noise, mean over 3 seeds) =====")
hdr = f"{'shape':<12} {'s/ell':>5} " + "".join(f"{m:>22}" for m, _ in METHODS)
print(hdr + "   (overall | near)")
for sname, _ in SHAPES:
    for sigma in SIGMAS:
        line = f"{sname:<12} {sigma/ELL:>5.1f} "
        for mname, _ in METHODS:
            sel = [r for r in rows if r["shape"] == sname and r["method"] == mname
                   and r["noise"] == "gauss" and abs(r["sigma"] - sigma / ELL) < 1e-6]
            ov = np.mean([r["overall"] for r in sel])
            nr = np.nanmean([r["near"] for r in sel]) if sel else float("nan")
            line += f" {ov:+7.1f}|{nr:+7.1f}      "
        print(line, flush=True)
