"""
Comprehensive multi-dataset evaluation for TGRS submission.

Dataset 1: ISPRS Vaihingen (ALS, urban, 9 classes) — 4 diverse crops
Dataset 2: DALES (ALS, large-scale, 8 classes) — 4 crops from different tiles
Dataset 3: Waymo (automotive LiDAR, 64-beam) — 4 frames from different scenes

All datasets × 7 baselines + ours, with extended metrics.
"""

import paths
import numpy as np
import time
import sys
import os

sys.path.insert(0, ".")

from adawave.baselines import (denoise_mls, denoise_wlop, denoise_glr,
                                  denoise_jet, denoise_clop)

# =========================================================================
# Shared metrics
# =========================================================================

def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


def z_rmse(d, c):
    return rmse(d[:, 2], c[:, 2])


def xyz_rmse(d, c):
    return rmse(d, c)


def chamfer_distance(a, b):
    from scipy.spatial import cKDTree
    d_ab, _ = cKDTree(b).query(a, k=1)
    d_ba, _ = cKDTree(a).query(b, k=1)
    return (np.mean(d_ab ** 2) + np.mean(d_ba ** 2)) / 2


def normal_consistency(denoised, clean, k=12):
    from scipy.spatial import cKDTree

    def _normals(pts):
        tree = cKDTree(pts)
        _, idx = tree.query(pts, k=k + 1)
        idx = idx[:, 1:]
        normals = np.zeros_like(pts)
        for i in range(len(pts)):
            cov = np.cov((pts[idx[i]] - pts[i]).T)
            _, vecs = np.linalg.eigh(cov)
            normals[i] = vecs[:, 0]
        return normals

    nc = _normals(clean)
    nd = _normals(denoised)
    return np.mean(np.abs(np.sum(nc * nd, axis=1)))


def find_boundary_points(xyz, labels, k=12, mixed_ratio=0.3):
    from scipy.spatial import cKDTree
    tree = cKDTree(xyz)
    _, indices = tree.query(xyz, k=k + 1)
    indices = indices[:, 1:]
    boundary = np.zeros(len(xyz), dtype=bool)
    for i in range(len(xyz)):
        same = (labels[indices[i]] == labels[i]).mean()
        if same < (1.0 - mixed_ratio):
            boundary[i] = True
    return boundary


def pct(noisy_val, denoised_val):
    return (1 - denoised_val / noisy_val) * 100


# =========================================================================
# Method definitions (shared across all datasets)
# =========================================================================

def bilateral(x):
    from adawave.neighbors import NeighborQuery
    from adawave.geometry import estimate_local_frames
    xyz = x.copy()
    for _ in range(2):
        nq = NeighborQuery(xyz)
        dists, indices = nq.query_knn(12)
        frames = estimate_local_frames(xyz, indices)
        normals = frames["normals"]
        median_spacing = np.median(dists[:, 0])
        xyz_new = xyz.copy()
        for i in range(len(xyz)):
            diff = xyz[indices[i]] - xyz[i]
            d = np.linalg.norm(diff, axis=1)
            np_proj = diff @ normals[i]
            sigma_s = np.median(d)
            if sigma_s < 1e-12:
                continue
            sigma_r = 0.3 * sigma_s
            w = np.exp(-0.5 * (d / sigma_s) ** 2) * np.exp(-0.5 * (np_proj / sigma_r) ** 2)
            ws = w.sum()
            if ws < 1e-12:
                continue
            shift = np.clip(np.dot(w, np_proj) / ws, -0.5 * median_spacing, 0.5 * median_spacing)
            xyz_new[i] += 0.2 * shift * normals[i]
        xyz = xyz_new
    return xyz


def ours_full(x):
    from adawave.neighbors import NeighborQuery
    from adawave.denoise import iterative_denoise
    nq = NeighborQuery(x.copy())
    result = iterative_denoise(
        x.copy(), nq, k=12, n_iters=2, n_dirs=8, grid_size=12,
        smooth_weight=0.2, feature_weight=0.12,
        patch_type="density", verbose=False)
    return result["xyz"]


METHODS = [
    ("Bilateral",   bilateral),
    ("MLS",         lambda x: denoise_mls(x, k=12, n_iters=2)),
    ("Jet",         lambda x: denoise_jet(x, k=20, n_iters=2)),
    ("CLOP",        lambda x: denoise_clop(x, k=12, n_iters=2)),
    ("GLR",         lambda x: denoise_glr(x, k=12, mu=0.5)),
    ("WLOP",        lambda x: denoise_wlop(x, k=20, n_iters=2)),
    ("Ours",        ours_full),
]


def preprocess_shared(noisy_xyz):
    """Shared preprocessing: dedup + SOR."""
    from adawave.preprocess import remove_duplicates, statistical_outlier_removal
    dup_mask = remove_duplicates(noisy_xyz)
    sor_mask = statistical_outlier_removal(noisy_xyz[dup_mask], k=20, std_ratio=2.0)
    idx_after_dup = np.where(dup_mask)[0]
    keep = idx_after_dup[sor_mask]
    final_mask = np.zeros(len(noisy_xyz), dtype=bool)
    final_mask[keep] = True
    return final_mask


# =========================================================================
# Dataset 1: Vaihingen — 4 diverse crops
# =========================================================================

VAIHINGEN_PATH = str(paths.vaihingen_train_pts())
VAIHINGEN_LABELS = {
    0: "Powerline", 1: "Low_veg", 2: "Impervious", 3: "Car",
    4: "Fence", 5: "Roof", 6: "Facade", 7: "Shrub", 8: "Tree",
}

# 4 crops spanning different scene types, centered on different parts of the tile
# (cx, cy, half_size, description)
VAIHINGEN_CROPS = [
    ("V1-Center",      497037, 5419371, 15, "Roof+Facade+Shrub+Tree"),
    ("V2-Southwest",   496987, 5419321, 15, "Fence+Roof+Facade+Veg"),
    ("V3-Southeast",   497087, 5419321, 15, "Impervious+Roof+Facade"),
    ("V4-North",       497037, 5419451, 15, "Car+Fence+Roof+Facade+Veg"),
]

_vaihingen_cache = None


def _load_vaihingen():
    global _vaihingen_cache
    if _vaihingen_cache is None:
        print("  Loading Vaihingen Training...")
        data = np.loadtxt(VAIHINGEN_PATH)
        _vaihingen_cache = (data[:, :3], data[:, 6].astype(int))
        print(f"    {len(data)} points loaded")
    return _vaihingen_cache


def prepare_vaihingen_crop(crop_def, noise_std=0.05, seed=42):
    """Return (clean_c, noisy_c, labels) for a Vaihingen crop."""
    name, cx, cy, half, desc = crop_def
    xyz_full, labels_full = _load_vaihingen()

    mask = ((xyz_full[:, 0] >= cx - half) & (xyz_full[:, 0] <= cx + half) &
            (xyz_full[:, 1] >= cy - half) & (xyz_full[:, 1] <= cy + half))
    crop_xyz = xyz_full[mask]
    crop_labels = labels_full[mask]

    rng = np.random.default_rng(seed)
    noisy_xyz = crop_xyz + rng.normal(0, noise_std, crop_xyz.shape)

    final_mask = preprocess_shared(noisy_xyz)
    clean = crop_xyz[final_mask]
    noisy = noisy_xyz[final_mask]
    labs = crop_labels[final_mask]

    centroid = noisy.mean(axis=0)
    return clean - centroid, noisy - centroid, labs


def compute_vaihingen_metrics(denoised, clean, noisy, labels):
    """Compute full metric dict for labeled data."""
    m = {}
    m["Z-RMSE"] = z_rmse(denoised, clean)
    m["Z-RMSE_noisy"] = z_rmse(noisy, clean)
    m["3D-RMSE"] = xyz_rmse(denoised, clean)
    m["3D-RMSE_noisy"] = xyz_rmse(noisy, clean)
    m["Chamfer"] = chamfer_distance(denoised, clean)
    m["Chamfer_noisy"] = chamfer_distance(noisy, clean)
    m["NormCons"] = normal_consistency(denoised, clean)
    m["NormCons_noisy"] = normal_consistency(noisy, clean)

    # Per-label Z-RMSE
    for lbl in sorted(np.unique(labels)):
        lm = labels == lbl
        if lm.sum() < 10:
            continue
        name = VAIHINGEN_LABELS.get(lbl, f"L{lbl}")
        m[f"Z_{name}"] = z_rmse(denoised[lm], clean[lm])
        m[f"Z_{name}_noisy"] = z_rmse(noisy[lm], clean[lm])

    # Groups
    manmade = np.isin(labels, [2, 3, 5, 6])
    veg = np.isin(labels, [1, 4, 7, 8])
    if manmade.sum() >= 10:
        m["Z_ManMade"] = z_rmse(denoised[manmade], clean[manmade])
        m["Z_ManMade_noisy"] = z_rmse(noisy[manmade], clean[manmade])
    if veg.sum() >= 10:
        m["Z_Vegetation"] = z_rmse(denoised[veg], clean[veg])
        m["Z_Vegetation_noisy"] = z_rmse(noisy[veg], clean[veg])

    # Boundary
    bnd = find_boundary_points(clean, labels)
    if bnd.sum() >= 10:
        m["Z_Boundary"] = z_rmse(denoised[bnd], clean[bnd])
        m["Z_Boundary_noisy"] = z_rmse(noisy[bnd], clean[bnd])
    if (~bnd).sum() >= 10:
        m["Z_Interior"] = z_rmse(denoised[~bnd], clean[~bnd])
        m["Z_Interior_noisy"] = z_rmse(noisy[~bnd], clean[~bnd])

    return m


# =========================================================================
# Dataset 2: DALES — 4 crops from different tiles
# =========================================================================

DALES_ROOT = str(paths.dales_root())
DALES_LABELS = {
    1: "Ground", 2: "Vegetation", 3: "Cars", 4: "Trucks",
    5: "Powerlines", 6: "Fences", 7: "Poles", 8: "Buildings",
}
DALES_MAX_PTS = 10000

# (name, split/file, cx, cy, half, description)
DALES_CROPS = [
    ("D1-Urban",     "train/5080_54435_new.ply", 250, 250, 10, "Ground+Buildings+Cars"),
    ("D2-Suburban",  "train/5105_54405_new.ply", 200, 300, 10, "Ground+Veg+Fences+Bld"),
    ("D3-Mixed",     "train/5095_54440_new.ply", 250, 250, 10, "Ground+Veg+Trucks+Bld"),
    ("D4-Test",      "test/5080_54400_new.ply",  250, 250, 10, "Test split scene"),
]

_dales_cache = {}


def _load_dales(rel_path):
    if rel_path in _dales_cache:
        return _dales_cache[rel_path]
    from plyfile import PlyData
    full_path = os.path.join(DALES_ROOT, rel_path)
    print(f"  Loading {rel_path}...")
    ply = PlyData.read(full_path)
    v = ply.elements[0]
    xyz = np.column_stack([v['x'].astype(np.float64),
                           v['y'].astype(np.float64),
                           v['z'].astype(np.float64)])
    labels = v['sem_class'].astype(int)
    print(f"    {len(xyz)} points loaded")
    _dales_cache[rel_path] = (xyz, labels)
    return xyz, labels


def prepare_dales_crop(crop_def, noise_std=0.05, seed=42):
    name, rel_path, cx, cy, half, desc = crop_def
    xyz_full, labels_full = _load_dales(rel_path)

    mask = ((xyz_full[:, 0] >= cx - half) & (xyz_full[:, 0] <= cx + half) &
            (xyz_full[:, 1] >= cy - half) & (xyz_full[:, 1] <= cy + half))
    crop_xyz = xyz_full[mask]
    crop_labels = labels_full[mask]

    if len(crop_xyz) > DALES_MAX_PTS:
        rng_sub = np.random.default_rng(seed + 1000)
        idx = rng_sub.choice(len(crop_xyz), DALES_MAX_PTS, replace=False)
        idx.sort()
        crop_xyz = crop_xyz[idx]
        crop_labels = crop_labels[idx]

    rng = np.random.default_rng(seed)
    noisy_xyz = crop_xyz + rng.normal(0, noise_std, crop_xyz.shape)

    final_mask = preprocess_shared(noisy_xyz)
    clean = crop_xyz[final_mask]
    noisy = noisy_xyz[final_mask]
    labs = crop_labels[final_mask]

    centroid = noisy.mean(axis=0)
    return clean - centroid, noisy - centroid, labs


def compute_dales_metrics(denoised, clean, noisy, labels):
    m = {}
    m["Z-RMSE"] = z_rmse(denoised, clean)
    m["Z-RMSE_noisy"] = z_rmse(noisy, clean)
    m["3D-RMSE"] = xyz_rmse(denoised, clean)
    m["3D-RMSE_noisy"] = xyz_rmse(noisy, clean)
    m["Chamfer"] = chamfer_distance(denoised, clean)
    m["Chamfer_noisy"] = chamfer_distance(noisy, clean)
    m["NormCons"] = normal_consistency(denoised, clean)
    m["NormCons_noisy"] = normal_consistency(noisy, clean)

    for lbl in sorted(np.unique(labels)):
        lm = labels == lbl
        if lm.sum() < 10:
            continue
        name = DALES_LABELS.get(lbl, f"L{lbl}")
        m[f"Z_{name}"] = z_rmse(denoised[lm], clean[lm])
        m[f"Z_{name}_noisy"] = z_rmse(noisy[lm], clean[lm])

    manmade = np.isin(labels, [3, 4, 6, 7, 8])
    natural = np.isin(labels, [1, 2])
    if manmade.sum() >= 10:
        m["Z_ManMade"] = z_rmse(denoised[manmade], clean[manmade])
        m["Z_ManMade_noisy"] = z_rmse(noisy[manmade], clean[manmade])
    if natural.sum() >= 10:
        m["Z_Natural"] = z_rmse(denoised[natural], clean[natural])
        m["Z_Natural_noisy"] = z_rmse(noisy[natural], clean[natural])

    bnd = find_boundary_points(clean, labels)
    if bnd.sum() >= 10:
        m["Z_Boundary"] = z_rmse(denoised[bnd], clean[bnd])
        m["Z_Boundary_noisy"] = z_rmse(noisy[bnd], clean[bnd])
    if (~bnd).sum() >= 10:
        m["Z_Interior"] = z_rmse(denoised[~bnd], clean[~bnd])
        m["Z_Interior_noisy"] = z_rmse(noisy[~bnd], clean[~bnd])

    return m


# =========================================================================
# Dataset 3: Waymo — 4 frames from different scenes (no labels)
# =========================================================================

WAYMO_ROOT = str(paths.data_root() / "waymo_ply")   # unused: Waymo dropped, see README
WAYMO_MAX_PTS = 10000

# (name, scene_dir, frame_index, description)
WAYMO_CROPS = [
    ("W1-Urban",    "10017090168044687777_6380_000_6400_000", 50,  "Urban driving"),
    ("W2-Highway",  "10023947602400723454_1120_000_1140_000", 50,  "Highway scene"),
    ("W3-Suburban", "1005081002024129653_5313_150_5333_150",  50,  "Suburban road"),
    ("W4-Mixed",    "10061305430875486848_1080_000_1100_000", 50,  "Mixed traffic"),
]


def prepare_waymo_frame(crop_def, noise_std=0.05, seed=42):
    """Load Waymo frame, subsample, add noise."""
    name, scene_dir, frame_idx, desc = crop_def
    scene_path = os.path.join(WAYMO_ROOT, scene_dir)
    frames = sorted([f for f in os.listdir(scene_path) if f.endswith('.ply')])
    if frame_idx >= len(frames):
        frame_idx = len(frames) // 2

    from plyfile import PlyData
    ply = PlyData.read(os.path.join(scene_path, frames[frame_idx]))
    v = ply['vertex']
    xyz = np.column_stack([v['x'].astype(np.float64),
                           v['y'].astype(np.float64),
                           v['z'].astype(np.float64)])

    # Subsample
    if len(xyz) > WAYMO_MAX_PTS:
        rng_sub = np.random.default_rng(seed + 2000)
        idx = rng_sub.choice(len(xyz), WAYMO_MAX_PTS, replace=False)
        idx.sort()
        xyz = xyz[idx]

    rng = np.random.default_rng(seed)
    noisy_xyz = xyz + rng.normal(0, noise_std, xyz.shape)

    final_mask = preprocess_shared(noisy_xyz)
    clean = xyz[final_mask]
    noisy = noisy_xyz[final_mask]

    centroid = noisy.mean(axis=0)
    return clean - centroid, noisy - centroid


def compute_waymo_metrics(denoised, clean, noisy):
    """Metrics for unlabeled data: no per-class, no boundary."""
    m = {}
    m["Z-RMSE"] = z_rmse(denoised, clean)
    m["Z-RMSE_noisy"] = z_rmse(noisy, clean)
    m["3D-RMSE"] = xyz_rmse(denoised, clean)
    m["3D-RMSE_noisy"] = xyz_rmse(noisy, clean)
    m["Chamfer"] = chamfer_distance(denoised, clean)
    m["Chamfer_noisy"] = chamfer_distance(noisy, clean)
    m["NormCons"] = normal_consistency(denoised, clean)
    m["NormCons_noisy"] = normal_consistency(noisy, clean)
    return m


# =========================================================================
# Table printing helpers
# =========================================================================

W = 140


def print_header(title):
    print(f"\n{'=' * W}")
    print(title)
    print(f"{'=' * W}")


def run_method(name, fn, noisy_c):
    t0 = time.time()
    denoised = fn(noisy_c)
    elapsed = time.time() - t0
    return denoised, elapsed


# =========================================================================
# Experiment runner for labeled datasets (Vaihingen / DALES)
# =========================================================================

def run_labeled_dataset(dataset_name, crops, prepare_fn, metrics_fn,
                        noise_std=0.05):
    """Run all methods on all crops for a labeled dataset.

    Returns: {crop_name: {method_name: metrics_dict}}
    """
    print_header(f"DATASET: {dataset_name} (σ={noise_std*100:.0f}cm, {len(crops)} crops)")

    all_results = {}
    for crop_def in crops:
        crop_name = crop_def[0]
        desc = crop_def[-1]
        print(f"\n  [{crop_name}] {desc}")

        clean_c, noisy_c, labs = prepare_fn(crop_def, noise_std=noise_std)
        print(f"    Points: {len(clean_c)}, Labels: {np.unique(labs)}")

        all_results[crop_name] = {}
        for mname, fn in METHODS:
            denoised, elapsed = run_method(mname, fn, noisy_c)
            m = metrics_fn(denoised, clean_c, noisy_c, labs)
            m["time"] = elapsed
            all_results[crop_name][mname] = m
            print(f"    {mname:12s}: {elapsed:.1f}s  Z-RMSE {pct(m['Z-RMSE_noisy'], m['Z-RMSE']):+.1f}%")

    return all_results


# =========================================================================
# Print: per-crop comparison table
# =========================================================================

def print_crop_table(dataset_name, all_results, key_metrics):
    """Print crop × method table for selected metrics."""
    method_names = [m[0] for m in METHODS]

    for metric_key, metric_noisy_key, metric_label in key_metrics:
        print_header(f"{dataset_name} — {metric_label}")

        # Header
        hdr = f"  {'Crop':20s}"
        for mn in method_names:
            hdr += f"  {mn:>12s}"
        print(hdr)
        print("  " + "-" * (W - 2))

        all_pcts = {mn: [] for mn in method_names}

        for crop_name, crop_results in all_results.items():
            row = f"  {crop_name:20s}"
            for mn in method_names:
                m = crop_results[mn]
                if metric_key in m and metric_noisy_key in m:
                    p = pct(m[metric_noisy_key], m[metric_key])
                    all_pcts[mn].append(p)
                    row += f"  {p:+10.1f}%"
                elif metric_key in m:
                    # For metrics like NormCons where higher=better
                    row += f"  {m[metric_key]:12.4f}"
                else:
                    row += f"  {'N/A':>12s}"
            print(row)

        # Mean ± std
        row_mean = f"  {'Mean':20s}"
        row_std = f"  {'Std':20s}"
        for mn in method_names:
            vals = all_pcts[mn]
            if len(vals) >= 2:
                row_mean += f"  {np.mean(vals):+10.1f}%"
                row_std += f"  {'±' + f'{np.std(vals):.1f}%':>10s}"
            elif len(vals) == 1:
                row_mean += f"  {vals[0]:+10.1f}%"
                row_std += f"  {'-':>12s}"
            else:
                row_mean += f"  {'N/A':>12s}"
                row_std += f"  {'N/A':>12s}"
        print("  " + "-" * (W - 2))
        print(row_mean)
        print(row_std)


# =========================================================================
# Print: extended metrics table (single crop, all metrics)
# =========================================================================

def print_extended_metrics(dataset_name, crop_name, crop_results):
    """Detailed extended metrics for one representative crop."""
    print_header(f"{dataset_name} — Extended Metrics [{crop_name}]")

    method_names = [m[0] for m in METHODS]
    hdr = f"  {'Metric':25s}  {'Noisy':>10s}"
    for mn in method_names:
        hdr += f"  {mn:>12s}"
    print(hdr)
    print("  " + "-" * (W - 2))

    rows = [
        ("Z-RMSE (cm)",       "Z-RMSE",    "Z-RMSE_noisy",    100, False),
        ("3D-RMSE (cm)",      "3D-RMSE",   "3D-RMSE_noisy",   100, False),
        ("Chamfer (cm²)",     "Chamfer",   "Chamfer_noisy",    1e4, False),
        ("Normal Consist.",   "NormCons",  "NormCons_noisy",   1,   True),
    ]

    # Add per-class if available
    first_m = list(crop_results.values())[0]
    for key in sorted(first_m.keys()):
        if key.startswith("Z_") and not key.endswith("_noisy") and key + "_noisy" in first_m:
            label = key.replace("Z_", "") + " (cm)"
            rows.append((label, key, key + "_noisy", 100, False))

    for display, key, noisy_key, scale, higher_better in rows:
        if noisy_key not in first_m:
            continue
        noisy_v = first_m[noisy_key]
        row = f"  {display:25s}  {noisy_v * scale:10.4f}"
        for mn in method_names:
            m = crop_results[mn]
            if key not in m:
                row += f"  {'N/A':>12s}"
                continue
            v = m[key]
            if higher_better:
                row += f"  {v * scale:12.4f}"
            else:
                p = pct(noisy_v, v)
                row += f"  {v*scale:6.2f}({p:+5.1f}%)"
        print(row)


# =========================================================================
# Print: runtime table
# =========================================================================

def print_runtime(all_results):
    print_header("Runtime Comparison (seconds)")
    method_names = [m[0] for m in METHODS]

    hdr = f"  {'Crop':20s}  {'Pts':>6s}"
    for mn in method_names:
        hdr += f"  {mn:>10s}"
    print(hdr)
    print("  " + "-" * (W - 2))

    for crop_name, crop_results in all_results.items():
        first_m = list(crop_results.values())[0]
        # Infer point count from Z-RMSE computation... just use time
        row = f"  {crop_name:20s}  {'~8K':>6s}"
        for mn in method_names:
            m = crop_results[mn]
            row += f"  {m['time']:10.1f}"
        print(row)


# =========================================================================
# Waymo experiment (unlabeled)
# =========================================================================

def run_waymo():
    print_header("DATASET: Waymo Open (automotive LiDAR, σ=5cm, 4 frames)")

    all_results = {}
    for crop_def in WAYMO_CROPS:
        crop_name = crop_def[0]
        desc = crop_def[-1]
        print(f"\n  [{crop_name}] {desc}")

        clean_c, noisy_c = prepare_waymo_frame(crop_def, noise_std=0.05)
        print(f"    Points: {len(clean_c)}")

        all_results[crop_name] = {}
        for mname, fn in METHODS:
            denoised, elapsed = run_method(mname, fn, noisy_c)
            m = compute_waymo_metrics(denoised, clean_c, noisy_c)
            m["time"] = elapsed
            all_results[crop_name][mname] = m
            print(f"    {mname:12s}: {elapsed:.1f}s  Z-RMSE {pct(m['Z-RMSE_noisy'], m['Z-RMSE']):+.1f}%")

    # Print tables
    print_crop_table("Waymo", all_results, [
        ("Z-RMSE", "Z-RMSE_noisy", "Z-RMSE Improvement (%)"),
        ("3D-RMSE", "3D-RMSE_noisy", "3D-RMSE Improvement (%)"),
    ])

    # Extended metrics for first crop
    first_crop = list(all_results.keys())[0]
    print_extended_metrics("Waymo", first_crop, all_results[first_crop])

    return all_results


# =========================================================================
# Multi-noise experiment (Vaihingen only, key methods)
# =========================================================================

def run_multi_noise():
    print_header("Multi-Noise Robustness — Vaihingen V1-Residential")

    noise_levels = [0.02, 0.05, 0.10]
    noise_names = ["σ=2cm", "σ=5cm", "σ=10cm"]
    crop_def = VAIHINGEN_CROPS[0]
    method_names = [m[0] for m in METHODS]

    results = {}
    for ns, nn in zip(noise_levels, noise_names):
        print(f"\n  Noise level: {nn}")
        clean_c, noisy_c, labs = prepare_vaihingen_crop(crop_def, noise_std=ns)
        print(f"    Points: {len(clean_c)}")
        results[nn] = {}
        for mname, fn in METHODS:
            denoised, elapsed = run_method(mname, fn, noisy_c)
            m = compute_vaihingen_metrics(denoised, clean_c, noisy_c, labs)
            m["time"] = elapsed
            results[nn][mname] = m
            print(f"    {mname:12s}: {elapsed:.1f}s")

    # Print table
    print_header("Multi-Noise — Z-RMSE Improvement (%)")
    cats = ["Z-RMSE", "Z_ManMade", "Z_Vegetation", "Z_Roof", "Z_Facade", "Z_Boundary"]

    hdr = f"  {'Category':15s}"
    for nn in noise_names:
        for mn in method_names:
            hdr += f" {mn[:5]:>6s}"
        hdr += " |"
    print(hdr)

    sub = f"  {'':15s}"
    for nn in noise_names:
        w = 7 * len(method_names)
        sub += f" {nn:^{w}s}|"
    print(sub)
    print("  " + "-" * (W - 2))

    for cat in cats:
        nk = cat + "_noisy" if cat != "Z-RMSE" else "Z-RMSE_noisy"
        dk = cat if cat != "Z-RMSE" else "Z-RMSE"
        row = f"  {cat:15s}"
        for nn in noise_names:
            for mn in method_names:
                m = results[nn][mn]
                if dk in m and nk in m:
                    p = pct(m[nk], m[dk])
                    row += f" {p:+5.1f}%"
                else:
                    row += f" {'N/A':>5s}"
            row += " |"
        print(row)

    return results


def run_multi_noise_dales():
    print_header("Multi-Noise Robustness — DALES D1-Urban")

    noise_levels = [0.02, 0.05, 0.10]
    noise_names = ["σ=2cm", "σ=5cm", "σ=10cm"]
    crop_def = DALES_CROPS[0]
    method_names = [m[0] for m in METHODS]

    results = {}
    for ns, nn in zip(noise_levels, noise_names):
        print(f"\n  Noise level: {nn}")
        clean_c, noisy_c, labs = prepare_dales_crop(crop_def, noise_std=ns)
        print(f"    Points: {len(clean_c)}")
        results[nn] = {}
        for mname, fn in METHODS:
            denoised, elapsed = run_method(mname, fn, noisy_c)
            m = compute_dales_metrics(denoised, clean_c, noisy_c, labs)
            m["time"] = elapsed
            results[nn][mname] = m
            print(f"    {mname:12s}: {elapsed:.1f}s")

    # Print table
    print_header("Multi-Noise — Z-RMSE Improvement (%) [DALES]")
    cats = ["Z-RMSE", "Z_ManMade", "Z_Natural", "Z_Buildings", "Z_Ground",
            "Z_Vegetation", "Z_Boundary"]

    hdr = f"  {'Category':15s}"
    for nn in noise_names:
        for mn in method_names:
            hdr += f" {mn[:5]:>6s}"
        hdr += " |"
    print(hdr)

    sub = f"  {'':15s}"
    for nn in noise_names:
        w = 7 * len(method_names)
        sub += f" {nn:^{w}s}|"
    print(sub)
    print("  " + "-" * (W - 2))

    for cat in cats:
        nk = cat + "_noisy" if cat != "Z-RMSE" else "Z-RMSE_noisy"
        dk = cat if cat != "Z-RMSE" else "Z-RMSE"
        row = f"  {cat:15s}"
        for nn in noise_names:
            for mn in method_names:
                m = results[nn][mn]
                if dk in m and nk in m:
                    p = pct(m[nk], m[dk])
                    row += f" {p:+5.1f}%"
                else:
                    row += f" {'N/A':>5s}"
            row += " |"
        print(row)

    return results


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * W)
    print("COMPREHENSIVE MULTI-DATASET EVALUATION FOR TGRS")
    print(f"Methods: {', '.join(m[0] for m in METHODS)}")
    print("=" * W)

    tasks = sys.argv[1:] if len(sys.argv) > 1 else ["all"]

    # --- Vaihingen ---
    if "all" in tasks or "vaihingen" in tasks:
        vai_results = run_labeled_dataset(
            "Vaihingen (ALS, Urban)", VAIHINGEN_CROPS,
            prepare_vaihingen_crop, compute_vaihingen_metrics)

        print_crop_table("Vaihingen", vai_results, [
            ("Z-RMSE", "Z-RMSE_noisy", "Z-RMSE Improvement (%)"),
            ("Z_Facade", "Z_Facade_noisy", "Facade Z-RMSE Improvement (%)"),
            ("Z_Boundary", "Z_Boundary_noisy", "Boundary Z-RMSE Improvement (%)"),
        ])

        first_crop = list(vai_results.keys())[0]
        print_extended_metrics("Vaihingen", first_crop, vai_results[first_crop])

    # --- DALES ---
    if "all" in tasks or "dales" in tasks:
        dales_results = run_labeled_dataset(
            "DALES (ALS, Large-scale)", DALES_CROPS,
            prepare_dales_crop, compute_dales_metrics)

        print_crop_table("DALES", dales_results, [
            ("Z-RMSE", "Z-RMSE_noisy", "Z-RMSE Improvement (%)"),
            ("Z_Buildings", "Z_Buildings_noisy", "Buildings Z-RMSE Improvement (%)"),
            ("Z_Boundary", "Z_Boundary_noisy", "Boundary Z-RMSE Improvement (%)"),
        ])

        first_crop = list(dales_results.keys())[0]
        print_extended_metrics("DALES", first_crop, dales_results[first_crop])

    # --- Waymo ---
    if "all" in tasks or "waymo" in tasks:
        waymo_results = run_waymo()

    # --- Multi-noise ---
    if "all" in tasks or "noise" in tasks:
        noise_results = run_multi_noise()
        noise_dales_results = run_multi_noise_dales()

    # --- Runtime ---
    if "all" in tasks or "vaihingen" in tasks:
        print_runtime(vai_results)

    print(f"\n{'=' * W}")
    print("EVALUATION COMPLETE")
    print(f"{'=' * W}")


if __name__ == "__main__":
    main()
