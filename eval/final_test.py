"""FINAL TEST — classical methods (one shot; pre-registered protocol).

Writes final_test_als.csv / final_test_mn.csv and prints per-sigma
summaries plus paired Wilcoxon tests of Prior-v2 against every baseline.
"""
import sys, csv, time
sys.path.insert(0, ".")
import numpy as np
from scipy import stats
from scipy.spatial import cKDTree

from final_test_data import als_samples, mn_samples
from eval_datasets import z_rmse, find_boundary_points
from eval_paper import baseline_bilateral_normal
from adawave.baselines import (denoise_mls, denoise_wlop, denoise_glr,
                                 denoise_jet, denoise_clop)
from adawave.wavefront_prior import denoise_wavefront_prior
from adawave.restoration import denoise_wavefront_prior_v2

METHODS = [
    ("Bilateral", lambda x: baseline_bilateral_normal(x, k=12, n_iters=2, weight=0.2)),
    ("MLS",       lambda x: denoise_mls(x, k=12, n_iters=2)),
    ("WLOP",      lambda x: denoise_wlop(x, k=20, n_iters=2)),
    ("Jet",       lambda x: denoise_jet(x, k=20, n_iters=2)),
    ("GLR",       lambda x: denoise_glr(x, k=12)),
    ("CLOP",      lambda x: denoise_clop(x, k=12, n_iters=2)),
    ("Prior-v1",  lambda x: denoise_wavefront_prior(
        x, k=12, n_dirs=8, grid_size=12, lam_n=1.0, lam_b=2.0, n_iters=80,
        step=0.05, momentum=0.5, max_shift_factor=0.15, normal_thresh=0.7,
        feature_anchor=100.0)["xyz"]),
    ("Prior-v2",  lambda x: denoise_wavefront_prior_v2(x)["xyz"]),
]


def imp(noi, den):
    return (1 - den / noi) * 100 if noi > 1e-15 else float("nan")


rows = []
for (name, sigma, seed, clean, noisy, lab, pl, el) in als_samples():
    bd = find_boundary_points(clean, lab, k=12)
    pm, em = np.isin(lab, pl), np.isin(lab, el)
    for mname, fn in METHODS:
        t0 = time.time()
        d = fn(noisy)
        if len(d) != len(clean):
            _, nn = cKDTree(d).query(clean, k=1)
            d = d[nn]
        rows.append(dict(sample=name, sigma=sigma, seed=seed, method=mname,
                         Z=imp(z_rmse(noisy, clean), z_rmse(d, clean)),
                         Planar=imp(z_rmse(noisy[pm], clean[pm]), z_rmse(d[pm], clean[pm])),
                         Edge=imp(z_rmse(noisy[em], clean[em]), z_rmse(d[em], clean[em])) if em.sum() > 10 else float("nan"),
                         Bdry=imp(z_rmse(noisy[bd], clean[bd]), z_rmse(d[bd], clean[bd])),
                         time=time.time() - t0))
    print(f"{name} s={sigma} seed={seed} done", flush=True)

with open("final_test_als.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)


def p2p(a, b):
    return np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1)))


def cd(a, b):
    ta, tb = cKDTree(a), cKDTree(b)
    return 0.5 * (np.mean(ta.query(b, k=1)[0] ** 2) + np.mean(tb.query(a, k=1)[0] ** 2))


mn_rows = []
for (cat, seed, pts, noisy) in mn_samples():
    for mname, fn in METHODS:
        d = fn(noisy)
        if len(d) != len(pts):
            _, nn = cKDTree(d).query(pts, k=1)
            dr = d[nn]
        else:
            dr = d
        mn_rows.append(dict(sample=cat, seed=seed, method=mname,
                            RMSE=imp(p2p(noisy, pts), p2p(dr, pts)),
                            CD=imp(cd(noisy, pts), cd(d, pts))))
    print(f"MN {cat} seed={seed} done", flush=True)

with open("final_test_mn.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(mn_rows[0].keys()))
    w.writeheader(); w.writerows(mn_rows)

# ------------------------------------------------------------- summaries
print("\n===== FINAL TEST — ALS (mean ± std over 8 windows x 3 seeds) =====")
for sigma in sorted({r["sigma"] for r in rows}):
    print(f"--- sigma = {int(sigma*100)} cm ---")
    for mname, _ in METHODS:
        sel = [r for r in rows if r["method"] == mname and r["sigma"] == sigma]
        line = f"{mname:<10}"
        for key in ["Z", "Planar", "Edge", "Bdry"]:
            v = np.array([r[key] for r in sel], dtype=float)
            v = v[~np.isnan(v)]
            line += f" {key}={v.mean():+6.2f}±{v.std():5.2f}"
        print(line)

print("\n===== FINAL TEST — ModelNet40 (8 categories x 3 seeds) =====")
for mname, _ in METHODS:
    sel = [r for r in mn_rows if r["method"] == mname]
    R = np.array([r["RMSE"] for r in sel]); C = np.array([r["CD"] for r in sel])
    print(f"{mname:<10} RMSE={R.mean():+6.2f}±{R.std():4.2f} CD={C.mean():+6.2f}±{C.std():5.2f}")

print("\n===== Paired Wilcoxon: Prior-v2 minus baseline (ALS Z, all sigmas) =====")
keys = sorted({(r["sample"], r["sigma"], r["seed"]) for r in rows})
def vec(m, metric="Z"):
    d = {(r["sample"], r["sigma"], r["seed"]): r[metric]
         for r in rows if r["method"] == m}
    return np.array([d[k] for k in keys])
v2 = vec("Prior-v2")
for mname, _ in METHODS[:-1]:
    dvec = v2 - vec(mname)
    w = stats.wilcoxon(dvec, alternative="greater")
    print(f"vs {mname:<10} mean diff={dvec.mean():+6.2f}pp  p={w.pvalue:.2e}  "
          f"W/L={int((dvec>0).sum())}/{int((dvec<0).sum())} of {len(dvec)}")
