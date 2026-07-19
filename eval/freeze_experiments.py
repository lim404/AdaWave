"""Freeze-decision experiments (development set ONLY).

Decides, per reviewer-grade criteria:
  A. anchor: const-1000 vs parameter-free 'auto' vs none — judged on tail
     risk (p95/p99/max displacement in sigma units), worst-crop Edge/Bdry,
     PCG convergence, across sigma in {2,5,10} cm and CAD.
  B. beta channel: lam_b=25 vs 0 under the aniso default — if
     indistinguishable everywhere AND beta usage is negligible, the beta
     variable is deleted from the frozen formulation.

Writes freeze_experiments.csv.
"""
import sys, csv
sys.path.insert(0, ".")
import paths
import numpy as np
import trimesh
from scipy.spatial import cKDTree

from eval_datasets import (VAIHINGEN_CROPS, DALES_CROPS,
                           prepare_vaihingen_crop, prepare_dales_crop,
                           z_rmse, find_boundary_points)
from adawave.restoration import denoise_wavefront_prior_v2

VARIANTS = [
    ("anchor-const", dict(anchor=1000.0, anchor_mode="const")),
    ("anchor-auto",  dict(anchor_mode="auto")),
    ("anchor-none",  dict(anchor=0.0, anchor_mode="const")),
    ("beta-off",     dict(anchor=1000.0, anchor_mode="const", lam_b=0.0)),
]


def imp(noi, den):
    return (1 - den / noi) * 100 if noi > 1e-15 else float("nan")


# preload ALS crops for all sigmas
als = {}
for sigma in [0.02, 0.05, 0.10]:
    sets = []
    for crop_def, kind in ([(c, "vai") for c in VAIHINGEN_CROPS]
                           + [(c, "dal") for c in DALES_CROPS]):
        if kind == "vai":
            clean, noisy, labels = prepare_vaihingen_crop(crop_def, noise_std=sigma, seed=42)
            pl, el = [2, 5], [4, 6]
        else:
            clean, noisy, labels = prepare_dales_crop(crop_def, noise_std=sigma, seed=42)
            pl, el = [1, 8], [6, 7]
        bd = find_boundary_points(clean, labels, k=12)
        sets.append((crop_def[0], clean, noisy, np.isin(labels, pl),
                     np.isin(labels, el), bd))
    als[sigma] = sets

MN = []
rng = np.random.default_rng(42)
for name, rel in [("airplane", "airplane_0627"), ("chair", "chair_0890"),
                  ("car", "car_0198"), ("table", "table_0398"),
                  ("guitar", "guitar_0157"), ("cone", "cone_0173"),
                  ("bottle", "bottle_0344"), ("piano", "piano_0240")]:
    mesh = trimesh.load(paths.require(
        paths.modelnet40_root() / name / "test" / f"{rel}.off", "ModelNet40 mesh"))
    pts, _ = trimesh.sample.sample_surface(mesh, 5000)
    pts = np.asarray(pts, dtype=np.float64)
    pts /= np.linalg.norm(pts.max(0) - pts.min(0))
    MN.append((name, pts, pts + rng.normal(0, 0.015, pts.shape)))


def p2p(a, b):
    return np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1)))


def cd(a, b):
    ta, tb = cKDTree(a), cKDTree(b)
    return 0.5 * (np.mean(ta.query(b, k=1)[0] ** 2) + np.mean(tb.query(a, k=1)[0] ** 2))


rows = []
for vname, kw in VARIANTS:
    for sigma, sets in als.items():
        Z, E, B, p95, p99, mx, cg, bfrac = [], [], [], [], [], [], [], []
        for cname, clean, noisy, pm, em, bd in sets:
            out = denoise_wavefront_prior_v2(noisy, **kw)
            d = out["xyz"]
            inf = out["info"][-1]
            Z.append(imp(z_rmse(noisy, clean), z_rmse(d, clean)))
            if em.sum() > 10:
                E.append(imp(z_rmse(noisy[em], clean[em]), z_rmse(d[em], clean[em])))
            B.append(imp(z_rmse(noisy[bd], clean[bd]), z_rmse(d[bd], clean[bd])))
            p95.append(inf["disp_p95"]); p99.append(inf["disp_p99"])
            mx.append(inf["disp_max"]); cg.append(inf["cg"]["iters"])
            bfrac.append(inf["beta_frac"])
        r = dict(variant=vname, domain=f"ALS-{int(sigma*100)}cm",
                 Z=np.mean(Z), Edge=np.mean(E), Bdry=np.mean(B),
                 Edge_worst=np.min(E), Bdry_worst=np.min(B),
                 disp_p95=np.mean(p95), disp_p99=np.mean(p99),
                 disp_max=np.max(mx), cg_iters=np.mean(cg),
                 beta_frac=np.mean(bfrac))
        rows.append(r)
        print(f"{vname:<13} {r['domain']:<9} Z={r['Z']:+6.2f} Edge={r['Edge']:+6.2f} "
              f"(worst {r['Edge_worst']:+6.2f}) Bdry={r['Bdry']:+6.2f} "
              f"(worst {r['Bdry_worst']:+6.2f}) p99={r['disp_p99']:4.2f}s "
              f"max={r['disp_max']:4.2f}s cg={r['cg_iters']:.0f} "
              f"betaf={r['beta_frac']:.3f}", flush=True)
    # CAD
    R, C, p99c, mxc = [], [], [], []
    for name, pts, noisy in MN:
        out = denoise_wavefront_prior_v2(noisy, **kw)
        d = out["xyz"]
        inf = out["info"][-1]
        R.append(imp(p2p(noisy, pts), p2p(d, pts)))
        C.append(imp(cd(noisy, pts), cd(d, pts)))
        p99c.append(inf["disp_p99"]); mxc.append(inf["disp_max"])
    r = dict(variant=vname, domain="MN40", Z=np.mean(R), Edge=np.mean(C),
             Bdry=float("nan"), Edge_worst=np.min(C), Bdry_worst=float("nan"),
             disp_p95=float("nan"), disp_p99=np.mean(p99c),
             disp_max=np.max(mxc), cg_iters=float("nan"),
             beta_frac=float("nan"))
    rows.append(r)
    print(f"{vname:<13} MN40      RMSE={r['Z']:+6.2f} CD={r['Edge']:+6.2f} "
          f"(worst {r['Edge_worst']:+6.2f}) p99={r['disp_p99']:4.2f}s "
          f"max={r['disp_max']:4.2f}s", flush=True)

with open("freeze_experiments.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("saved freeze_experiments.csv")
