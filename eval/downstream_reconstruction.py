"""Downstream task 2: surface reconstruction after denoising.

Protocol (dev shapes): 8 ModelNet40 dev shapes, sigma = 1.5% diag (seed 42).
Screened Poisson (pymeshlab, depth 8) is applied with IDENTICAL settings to
the noisy cloud and to each denoised cloud; the reference is 100k samples
of the ground-truth mesh. Metrics on 100k samples of every reconstructed
mesh: accuracy (recon->GT), completeness (GT->recon), Chamfer = mean of
both, and normal consistency (dot of recon sample normal vs nearest GT
sample normal). Improvement % vs the reconstruction from the NOISY cloud.
Writes downstream_reconstruction.csv.
"""
import paths
import sys, csv
sys.path.insert(0, ".")
import numpy as np
import trimesh
import pymeshlab
from scipy.spatial import cKDTree

from eval_paper import baseline_bilateral_normal
from adawave.baselines import denoise_mls, denoise_glr
from adawave.restoration import denoise_wavefront_prior_v2

METHODS = [
    ("Noisy",     lambda x: x),
    ("Bilateral", lambda x: baseline_bilateral_normal(x, k=12, n_iters=2, weight=0.2)),
    ("MLS",       lambda x: denoise_mls(x, k=12, n_iters=2)),
    ("GLR",       lambda x: denoise_glr(x, k=12)),
    ("Prior-v2",  lambda x: denoise_wavefront_prior_v2(x)["xyz"]),
]
SAMPLES = [("airplane", "airplane_0627"), ("chair", "chair_0890"),
           ("car", "car_0198"), ("table", "table_0398"),
           ("guitar", "guitar_0157"), ("cone", "cone_0173"),
           ("bottle", "bottle_0344"), ("piano", "piano_0240")]


def poisson_recon(pts):
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(pts, dtype=np.float64)))
    ms.compute_normal_for_point_clouds(k=12)
    ms.generate_surface_reconstruction_screened_poisson(depth=8)
    m = ms.current_mesh()
    return trimesh.Trimesh(vertices=m.vertex_matrix(),
                           faces=m.face_matrix(), process=False)


rows = []
rng = np.random.default_rng(42)
for name, rel in SAMPLES:
    gt_mesh = trimesh.load(paths.require(
        paths.modelnet40_root() / name / "test" / f"{rel}.off", "ModelNet40 mesh"))
    pts, _ = trimesh.sample.sample_surface(gt_mesh, 5000)
    pts = np.asarray(pts, dtype=np.float64)
    scale = np.linalg.norm(pts.max(0) - pts.min(0))
    pts /= scale
    gt_mesh = gt_mesh.copy()
    gt_mesh.apply_scale(1.0 / scale)
    gt_s, gt_fi = trimesh.sample.sample_surface(gt_mesh, 100_000)
    gt_s = np.asarray(gt_s)
    gt_n = np.asarray(gt_mesh.face_normals[gt_fi])
    gt_tree = cKDTree(gt_s)
    noisy = pts + rng.normal(0, 0.015, pts.shape)
    base = {}
    for mname, fn in METHODS:
        den = fn(noisy)
        try:
            rmesh = poisson_recon(den)
            rs, rfi = trimesh.sample.sample_surface(rmesh, 100_000)
            rs = np.asarray(rs)
            rn = np.asarray(rmesh.face_normals[rfi])
        except Exception as e:
            print(f"{name} {mname}: recon failed {e}")
            continue
        acc_d, acc_i = gt_tree.query(rs, k=1)
        comp_d, _ = cKDTree(rs).query(gt_s, k=1)
        acc = float(np.mean(acc_d))
        comp = float(np.mean(comp_d))
        chamfer = 0.5 * (acc + comp)
        nc = float(np.mean(np.abs(np.sum(rn * gt_n[acc_i], axis=1))))
        r = dict(shape=name, method=mname, acc=acc * 1000, comp=comp * 1000,
                 chamfer=chamfer * 1000, normal_cons=nc)
        rows.append(r)
        print(f"{name:<9} {mname:<10} acc={acc*1000:6.3f} comp={comp*1000:6.3f} "
              f"CD={chamfer*1000:6.3f} (x1e-3) NC={nc:.4f}", flush=True)

with open("downstream_reconstruction.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

print("\n===== RECON SUMMARY (8-shape mean; improvement vs recon-from-noisy) =====")
for mname, _ in METHODS:
    sel = [r for r in rows if r["method"] == mname]
    if not sel:
        continue
    ch = np.mean([r["chamfer"] for r in sel])
    nc = np.mean([r["normal_cons"] for r in sel])
    if mname == "Noisy":
        base_ch, base_nc = ch, nc
        print(f"{mname:<10} CD={ch:6.3f}  NC={nc:.4f}  (baseline)")
    else:
        print(f"{mname:<10} CD={ch:6.3f} ({(1-ch/base_ch)*100:+5.1f}%)  "
              f"NC={nc:.4f} ({(nc-base_nc):+.4f})")
