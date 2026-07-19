"""Ablation suite for AdaWave (plan §9).

Variants (each toggles ONE mechanism):
  full          complete v2
  fixed-lam     s2_override=0.08 (ALS-tuned fixed balance; no noise adaptivity)
  no-anchor     anchor=0
  no-buffer     prop_decay=0 (no structure-proximity propagation)
  no-geomw      use_geom_weights=False
  mad-struct    struct_stat='mad' (50%-breakdown detector)
  fixed-cap     cap_mode='spacing' (v1-style 0.15*spacing cap)
  no-binormal   lam_b=0
  gd-solver     v1-style momentum GD instead of PCG

Reports 8-crop ALS means (Z/Planar/Edge/Bdry) and 8-shape ModelNet40 means
(RMSE/CD). Writes ablation_v2.csv.
"""
import sys, time, csv
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
    ("full",        {}),
    ("fixed-lam",   {"s2_override": 0.08}),
    ("no-anchor",   {"anchor": 0.0}),
    ("no-buffer",   {"prop_decay": 0.0}),
    ("no-geomw",    {"use_geom_weights": False}),
    ("mad-struct",  {"struct_stat": "mad"}),
    ("fixed-cap",   {"cap_mode": "spacing"}),
    ("no-binormal", {"lam_b": 0.0}),
    ("gd-solver",   {"solver": "gd"}),
]


def imp(noi, den):
    return (1 - den / noi) * 100 if noi > 1e-15 else float("nan")


# ---- load all data once
als_data = []
for crop_def, kind in ([(c, "vai") for c in VAIHINGEN_CROPS]
                       + [(c, "dal") for c in DALES_CROPS]):
    if kind == "vai":
        clean, noisy, labels = prepare_vaihingen_crop(crop_def, noise_std=0.05, seed=42)
        pl, el = [2, 5], [4, 6]
    else:
        clean, noisy, labels = prepare_dales_crop(crop_def, noise_std=0.05, seed=42)
        pl, el = [1, 8], [6, 7]
    bd = find_boundary_points(clean, labels, k=12)
    als_data.append((crop_def[0], clean, noisy,
                     np.isin(labels, pl), np.isin(labels, el), bd))

MN_SAMPLES = [("airplane", "test/airplane_0627.off"), ("chair", "test/chair_0890.off"),
              ("car", "test/car_0198.off"), ("table", "test/table_0398.off"),
              ("guitar", "test/guitar_0157.off"), ("cone", "test/cone_0173.off"),
              ("bottle", "test/bottle_0344.off"), ("piano", "test/piano_0240.off")]
mn_data = []
rng = np.random.default_rng(42)
for name, rel in MN_SAMPLES:
    mesh = trimesh.load(paths.require(paths.modelnet40_root() / name / rel, "ModelNet40 mesh"))
    pts, _ = trimesh.sample.sample_surface(mesh, 5000)
    pts = np.asarray(pts, dtype=np.float64)
    pts /= np.linalg.norm(pts.max(0) - pts.min(0))
    mn_data.append((name, pts, pts + rng.normal(0, 0.015, pts.shape)))


def p2p(a, b):
    return np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1)))


def cd(a, b):
    ta, tb = cKDTree(a), cKDTree(b)
    return 0.5 * (np.mean(ta.query(b, k=1)[0] ** 2) + np.mean(tb.query(a, k=1)[0] ** 2))


rows = []
for vname, kw in VARIANTS:
    t0 = time.time()
    Z, P, E, B = [], [], [], []
    for cname, clean, noisy, pm, em, bd in als_data:
        d = denoise_wavefront_prior_v2(noisy, **kw)["xyz"]
        Z.append(imp(z_rmse(noisy, clean), z_rmse(d, clean)))
        P.append(imp(z_rmse(noisy[pm], clean[pm]), z_rmse(d[pm], clean[pm])))
        if em.sum() > 10:
            E.append(imp(z_rmse(noisy[em], clean[em]), z_rmse(d[em], clean[em])))
        B.append(imp(z_rmse(noisy[bd], clean[bd]), z_rmse(d[bd], clean[bd])))
    R, C = [], []
    for sname, pts, noisy in mn_data:
        d = denoise_wavefront_prior_v2(noisy, **kw)["xyz"]
        R.append(imp(p2p(noisy, pts), p2p(d, pts)))
        C.append(imp(cd(noisy, pts), cd(d, pts)))
    r = dict(variant=vname,
             als_Z=np.mean(Z), als_Planar=np.mean(P), als_Edge=np.mean(E),
             als_Bdry=np.mean(B), mn_RMSE=np.mean(R), mn_CD=np.mean(C),
             time=time.time() - t0)
    rows.append(r)
    print(f"{vname:<12} ALS: Z={r['als_Z']:+6.2f} Plan={r['als_Planar']:+6.2f} "
          f"Edge={r['als_Edge']:+6.2f} Bdry={r['als_Bdry']:+6.2f} | "
          f"MN: RMSE={r['mn_RMSE']:+6.2f} CD={r['mn_CD']:+6.2f} "
          f"({r['time']:.0f}s)", flush=True)

with open("ablation_v2.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("saved ablation_v2.csv")
