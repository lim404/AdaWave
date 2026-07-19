"""ScanNet++ downstream task 1: normal estimation after denoising.

Protocol (registered): iPhone multi-frame fused crops (the regime with
genuine geometric noise), 12 scenes. PCA(k=12) normals are estimated on
the noisy and on each denoised cloud with the SAME estimator; ground truth
is the face normal of the nearest of 400k mesh samples. Regions are
defined from the REFERENCE mesh only (method-blind): a mesh sample is
'planar' if the mean angular dispersion of normals within 5 cm is < 10 deg,
'edge' if > 25 deg. Reported: mean / median angular error, % < 5/10/20 deg,
overall and per region. Writes worldB_normals.csv.
"""
import paths
import os, sys, csv, glob, json, struct, time
sys.path.insert(0, ".")
import numpy as np
import trimesh
import lz4.block
from scipy.spatial import cKDTree

from eval_paper import baseline_bilateral_normal
from adawave.baselines import denoise_mls, denoise_glr
from adawave.restoration import denoise_wavefront_prior_v2
from adawave.neighbors import NeighborQuery
from adawave.geometry import estimate_local_frames

ROOT = str(paths.scannetpp_root())
SEED = 7
N_CROP = 10_000

METHODS = [
    ("Noisy",     lambda x: x),
    ("Bilateral", lambda x: baseline_bilateral_normal(x, k=12, n_iters=2, weight=0.2)),
    ("MLS",       lambda x: denoise_mls(x, k=12, n_iters=2)),
    ("GLR",       lambda x: denoise_glr(x, k=12)),
    ("Prior-v2",  lambda x: denoise_wavefront_prior_v2(x)["xyz"]),
]


def q2R(qw, qx, qy, qz):
    return np.array([[1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
                     [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
                     [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)]])


def pca_normals(pts, k=12):
    nq = NeighborQuery(pts)
    _, idx = nq.query_knn(k)
    return estimate_local_frames(pts, idx)["normals"]


def ang_err(n_est, n_gt):
    cos = np.abs(np.sum(n_est * n_gt, axis=1)).clip(0, 1)
    return np.degrees(np.arccos(cos))


rows = []
for scene_dir in sorted(glob.glob(os.path.join(ROOT, "*"))):
    sid = os.path.basename(scene_dir)
    mesh_path = os.path.join(scene_dir, "scans", "mesh_aligned_0.05.ply")
    dep_path = os.path.join(scene_dir, "iphone", "depth.bin")
    col_path = os.path.join(scene_dir, "iphone", "colmap", "images.txt")
    tj_path = os.path.join(scene_dir, "iphone", "nerfstudio", "transforms.json")
    if not all(os.path.exists(p) for p in (mesh_path, dep_path, col_path, tj_path)):
        continue
    t0 = time.time()
    mesh = trimesh.load(mesh_path)
    ref, fidx = trimesh.sample.sample_surface(mesh, 400_000)
    ref = np.asarray(ref, dtype=np.float64)
    ref_n = np.asarray(mesh.face_normals[fidx], dtype=np.float64)
    ref_tree = cKDTree(ref)
    # method-blind region labels on the reference
    sub = np.arange(0, len(ref), 8)
    nb = ref_tree.query_ball_point(ref[sub], r=0.05)
    disp = np.array([np.degrees(np.arccos(
        np.abs(ref_n[i_list] @ ref_n[s]).clip(0, 1)).mean()) if len(i_list) > 3
        else 0.0 for s, i_list in zip(sub, nb)])
    region = np.zeros(len(ref), dtype=int)          # 0 = mid
    reg_sub = np.where(disp < 10, 1, np.where(disp > 25, 2, 0))
    region_tree = cKDTree(ref[sub])
    del mesh

    # ---- rebuild the fused iPhone crop (same as worldB_iphone.py)
    raw = open(dep_path, "rb").read()
    offs, off = [], 0
    while off < len(raw):
        n = struct.unpack("<I", raw[off:off+4])[0]
        offs.append((off + 4, n)); off += 4 + n
    imgs = []
    with open(col_path) as f:
        for l in f:
            l = l.strip()
            if not l or l.startswith("#"):
                continue
            p = l.split()
            if len(p) >= 10 and p[9].startswith("frame_"):
                imgs.append((list(map(float, p[1:8])), p[9]))
    tj = json.load(open(tj_path))
    sx, sy = 256 / tj["w"], 192 / tj["h"]
    fxd, fyd = tj["fl_x"] * sx, tj["fl_y"] * sy
    cxd, cyd = tj["cx"] * sx, tj["cy"] * sy
    u, v = np.meshgrid(np.arange(256), np.arange(192))
    pts_all = []
    for qt, name in imgs[::5][:40]:
        qw, qx, qy, qz, tx, ty, tz = qt
        idx = int(name.split("_")[1].split(".")[0])
        if idx >= len(offs):
            continue
        o, n = offs[idx]
        d = np.frombuffer(lz4.block.decompress(raw[o:o+n], uncompressed_size=98304),
                          dtype=np.uint16).reshape(192, 256).astype(np.float64) / 1000.0
        m = (d > 0.2) & (d < 5.0)
        cam = np.stack([(u[m] - cxd) / fxd * d[m],
                        (v[m] - cyd) / fyd * d[m], d[m]], axis=1)[::8]
        pts_all.append((cam - np.array([tx, ty, tz])) @ q2R(qw, qx, qy, qz))
    del raw
    if not pts_all:
        continue
    pts = np.vstack(pts_all)
    pts = pts[np.isfinite(pts).all(axis=1)]
    cx_, cy_ = np.median(pts[:, 0]), np.median(pts[:, 1])
    half = 1.0
    while True:
        m = ((np.abs(pts[:, 0] - cx_) <= half) & (np.abs(pts[:, 1] - cy_) <= half))
        if m.sum() >= 8000 or half > 16:
            break
        half *= 2.0
    crop = pts[m]
    rng = np.random.default_rng(SEED)
    if len(crop) > N_CROP:
        idx = rng.choice(len(crop), N_CROP, replace=False)
        idx.sort()
        crop = crop[idx]

    for mname, fn in METHODS:
        den = fn(crop)
        n_est = pca_normals(den)
        _, nn = ref_tree.query(den, k=1)
        errs = ang_err(n_est, ref_n[nn])
        _, rs = region_tree.query(den, k=1)
        reg = reg_sub[rs]
        r = dict(scene=sid, method=mname,
                 mean=float(errs.mean()), med=float(np.median(errs)),
                 lt5=float((errs < 5).mean()) * 100,
                 lt10=float((errs < 10).mean()) * 100,
                 lt20=float((errs < 20).mean()) * 100,
                 mean_planar=float(errs[reg == 1].mean()) if (reg == 1).sum() > 20 else float("nan"),
                 mean_edge=float(errs[reg == 2].mean()) if (reg == 2).sum() > 20 else float("nan"))
        rows.append(r)
        print(f"{sid} {mname:<10} mean={r['mean']:5.2f} med={r['med']:5.2f} "
              f"<10deg={r['lt10']:5.1f}% planar={r['mean_planar']:5.2f} "
              f"edge={r['mean_edge']:5.2f}", flush=True)
    print(f"{sid} done ({time.time()-t0:.0f}s)", flush=True)

with open("worldB_normals.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

print("\n===== NORMAL ESTIMATION SUMMARY (12-scene mean) =====")
for mname, _ in METHODS:
    sel = [r for r in rows if r["method"] == mname]
    if not sel:
        continue
    print(f"{mname:<10} mean={np.mean([r['mean'] for r in sel]):5.2f} "
          f"med={np.mean([r['med'] for r in sel]):5.2f} "
          f"<10deg={np.mean([r['lt10'] for r in sel]):5.1f}% "
          f"planar={np.nanmean([r['mean_planar'] for r in sel]):5.2f} "
          f"edge={np.nanmean([r['mean_edge'] for r in sel]):5.2f}")
