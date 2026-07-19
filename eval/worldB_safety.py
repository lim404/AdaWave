"""World B safety battery — expanded protocol (registered addendum).

Per ScanNet++ scene (all 12): laser crop, iPhone multi-frame fused crop,
and 10 evenly-spaced single iPhone depth frames. For every sample and
method we record:
  - p2s RMSE / MAE improvement % vs the laser-mesh reference
  - displacement median / p95 (absolute, /sigma_g, /local spacing)
  - v2's estimated sigma_g and the crop's median NN spacing
Summary reports per regime and method:
  - no-harm rate  R = mean[ dRMSE >= -eps ], eps = 0.2 %
  - worst-case    min dRMSE
  - displacement medians (proving abstention is near-zero update, not luck)
Writes worldB_safety.csv.
"""
import paths
import os, sys, csv, glob, json, struct, time
sys.path.insert(0, ".")
import numpy as np
import trimesh
import lz4.block
from plyfile import PlyData
from scipy.spatial import cKDTree

from eval_paper import baseline_bilateral_normal
from adawave.baselines import denoise_mls, denoise_glr
from adawave.restoration import denoise_wavefront_prior_v2

ROOT = str(paths.scannetpp_root())
SEED = 7
N_CROP = 10_000
N_FRAMES = 10
EPS = 0.2

METHODS = [
    ("Bilateral", lambda x: baseline_bilateral_normal(x, k=12, n_iters=2, weight=0.2)),
    ("MLS",       lambda x: denoise_mls(x, k=12, n_iters=2)),
    ("GLR",       lambda x: denoise_glr(x, k=12)),
    ("Prior-v2",  None),   # handled specially to capture sigma_g
]


def q2R(qw, qx, qy, qz):
    return np.array([[1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
                     [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
                     [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)]])


def imp(a, b):
    return (1 - b / a) * 100 if a > 1e-15 else float("nan")


def spacing_of(pts):
    t = cKDTree(pts)
    d, _ = t.query(pts[:: max(1, len(pts) // 2000)], k=2)
    return float(np.median(d[:, 1]))


def eval_sample(regime, sid, tag, pts, ref_tree, rows):
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 3000:
        return
    ell = spacing_of(pts)
    dn, _ = ref_tree.query(pts, k=1)
    rmse_n, mae_n = float(np.sqrt((dn**2).mean())), float(dn.mean())
    out2 = denoise_wavefront_prior_v2(pts)
    sigma_g = out2["sigma_g"]
    for mname, fn in METHODS:
        den = out2["xyz"] if mname == "Prior-v2" else fn(pts)
        dd, _ = ref_tree.query(den, k=1)
        disp = np.linalg.norm(den - pts, axis=1)
        rows.append(dict(
            regime=regime, scene=sid, tag=tag, method=mname, n=len(pts),
            raw_rmse_cm=rmse_n * 100,
            dRMSE=imp(rmse_n, float(np.sqrt((dd**2).mean()))),
            dMAE=imp(mae_n, float(dd.mean())),
            disp_med_mm=float(np.median(disp)) * 1000,
            disp_p95_mm=float(np.percentile(disp, 95)) * 1000,
            disp_med_sig=float(np.median(disp)) / (sigma_g + 1e-12),
            disp_med_ell=float(np.median(disp)) / (ell + 1e-12),
            sigma_g_mm=sigma_g * 1000, ell_mm=ell * 1000))


rows = []
for scene_dir in sorted(glob.glob(os.path.join(ROOT, "*"))):
    sid = os.path.basename(scene_dir)
    mesh_path = os.path.join(scene_dir, "scans", "mesh_aligned_0.05.ply")
    dep_path = os.path.join(scene_dir, "iphone", "depth.bin")
    col_path = os.path.join(scene_dir, "iphone", "colmap", "images.txt")
    tj_path = os.path.join(scene_dir, "iphone", "nerfstudio", "transforms.json")
    pc_path = os.path.join(scene_dir, "scans", "pc_aligned.ply")
    if not all(os.path.exists(p) for p in (mesh_path, dep_path, col_path, tj_path, pc_path)):
        print(f"{sid}: missing, skip", flush=True)
        continue
    t0 = time.time()
    mesh = trimesh.load(mesh_path)
    ref, _ = trimesh.sample.sample_surface(mesh, 400_000)
    ref_tree = cKDTree(np.asarray(ref, dtype=np.float64))
    del mesh, ref
    rng = np.random.default_rng(SEED)

    # ---- laser crop
    v = PlyData.read(pc_path).elements[0]
    xyz = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    cx_, cy_ = np.median(xyz[:, 0]), np.median(xyz[:, 1])
    half = 1.0
    while True:
        m = ((np.abs(xyz[:, 0] - cx_) <= half) & (np.abs(xyz[:, 1] - cy_) <= half))
        if m.sum() >= 8000 or half > 16:
            break
        half *= 2.0
    crop = xyz[m]
    del xyz
    if len(crop) > N_CROP:
        idx = rng.choice(len(crop), N_CROP, replace=False)
        idx.sort()
        crop = crop[idx]
    eval_sample("laser", sid, "crop", crop, ref_tree, rows)

    # ---- iPhone: index depth + poses
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

    def frame_pts(qt, name, sub):
        qw, qx, qy, qz, tx, ty, tz = qt
        idx = int(name.split("_")[1].split(".")[0])
        if idx >= len(offs):
            return None
        o, n = offs[idx]
        d = np.frombuffer(lz4.block.decompress(raw[o:o+n], uncompressed_size=98304),
                          dtype=np.uint16).reshape(192, 256).astype(np.float64) / 1000.0
        m = (d > 0.2) & (d < 5.0)
        if m.sum() < 1000:
            return None
        cam = np.stack([(u[m] - cxd) / fxd * d[m],
                        (v[m] - cyd) / fyd * d[m], d[m]], axis=1)[::sub]
        return (cam - np.array([tx, ty, tz])) @ q2R(qw, qx, qy, qz)

    # fused crop (40 frames, as B1)
    pts_all = [p for p in (frame_pts(qt, nm, 8) for qt, nm in imgs[::5][:40])
               if p is not None]
    if pts_all:
        pts = np.vstack(pts_all)
        pts = pts[np.isfinite(pts).all(axis=1)]
        cx2, cy2 = np.median(pts[:, 0]), np.median(pts[:, 1])
        half2 = 1.0
        while True:
            m = ((np.abs(pts[:, 0] - cx2) <= half2) & (np.abs(pts[:, 1] - cy2) <= half2))
            if m.sum() >= 8000 or half2 > 16:
                break
            half2 *= 2.0
        fcrop = pts[m]
        if len(fcrop) > N_CROP:
            idx = rng.choice(len(fcrop), N_CROP, replace=False)
            idx.sort()
            fcrop = fcrop[idx]
        eval_sample("iphone-fused", sid, "crop", fcrop, ref_tree, rows)

    # single frames (10 evenly spaced)
    stride = max(1, len(imgs) // N_FRAMES)
    for (qt, nm) in imgs[::stride][:N_FRAMES]:
        p = frame_pts(qt, nm, 4)
        if p is None:
            continue
        eval_sample("iphone-frame", sid, nm, p, ref_tree, rows)

    del raw
    print(f"{sid} done ({time.time()-t0:.0f}s, samples so far {len(rows)})", flush=True)

with open("worldB_safety.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

print("\n===== SAFETY SUMMARY (eps = 0.2%) =====")
for regime in ["laser", "iphone-fused", "iphone-frame"]:
    print(f"--- {regime} ---")
    for mname, _ in METHODS:
        sel = [r for r in rows if r["regime"] == regime and r["method"] == mname]
        if not sel:
            continue
        dr = np.array([r["dRMSE"] for r in sel])
        noharm = float((dr >= -EPS).mean()) * 100
        print(f"{mname:<10} n={len(sel):>3} mean dRMSE={dr.mean():+6.2f}% "
              f"worst={dr.min():+7.2f}%  no-harm={noharm:5.1f}%  "
              f"disp_med={np.median([r['disp_med_mm'] for r in sel]):6.2f}mm "
              f"(={np.median([r['disp_med_sig'] for r in sel]):.2f} sigma, "
              f"{np.median([r['disp_med_ell'] for r in sel]):.2f} ell)")
