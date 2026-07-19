"""FINAL TEST sample generation — deterministic, from the pre-registered
split (eval_splits.py). Shared by the classical and DL runners so every
method sees byte-identical samples. Run ONCE per method; no design change
may follow from these results.
"""
import gzip
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
import numpy as np

from eval_splits import (TEST_VAIHINGEN_SOURCE, TEST_DALES_TILES,
                         TEST_MODELNET, TEST_SEEDS, TEST_NOISE_LEVELS,
                         TEST_MN_NOISE, pick_window)
from eval_datasets import preprocess_shared, _load_dales

_vai_cache = None


def _load_vai_eval():
    global _vai_cache
    if _vai_cache is None:
        print("  Loading Vaihingen EVAL_WITH_REF...", flush=True)
        with gzip.open(TEST_VAIHINGEN_SOURCE, "rt") as f:
            data = np.loadtxt(f)
        _vai_cache = (data[:, :3], data[:, 6].astype(int))
        print(f"    {len(data)} points", flush=True)
    return _vai_cache


def _vai_windows(n=4):
    xyz, labels = _load_vai_eval()
    xmin, ymin = xyz[:, 0].min(), xyz[:, 1].min()
    xmax, ymax = xyz[:, 0].max(), xyz[:, 1].max()
    cands = []
    for cx in np.arange(xmin + 15, xmax - 15, 20.0):
        for cy in np.arange(ymin + 15, ymax - 15, 20.0):
            m = ((np.abs(xyz[:, 0] - cx) <= 15) & (np.abs(xyz[:, 1] - cy) <= 15))
            npts = int(m.sum())
            if npts < 6000 or npts > 14000:
                continue
            lab = labels[m]
            gf = float(np.isin(lab, [1, 2]).mean())
            bf = float(np.isin(lab, [5, 6]).mean())
            cands.append((min(gf, bf), float(cx), float(cy)))
    cands.sort(reverse=True)
    picked = []
    for s, cx, cy in cands:
        if all(np.hypot(cx - a, cy - b) >= 35 for _, a, b in picked):
            picked.append((s, cx, cy))
        if len(picked) == n:
            break
    return [(f"TV{i+1}", cx, cy) for i, (s, cx, cy) in enumerate(picked)]


def _crop(xyz, labels, cx, cy, half, noise_std, seed, max_pts=None):
    m = ((np.abs(xyz[:, 0] - cx) <= half) & (np.abs(xyz[:, 1] - cy) <= half))
    crop_xyz, crop_lab = xyz[m], labels[m]
    if max_pts and len(crop_xyz) > max_pts:
        sub = np.random.default_rng(seed + 1000).choice(
            len(crop_xyz), max_pts, replace=False)
        sub.sort()
        crop_xyz, crop_lab = crop_xyz[sub], crop_lab[sub]
    rng = np.random.default_rng(seed)
    noisy = crop_xyz + rng.normal(0, noise_std, crop_xyz.shape)
    keep = preprocess_shared(noisy)
    c = noisy[keep].mean(axis=0)
    return crop_xyz[keep] - c, noisy[keep] - c, crop_lab[keep]


def als_samples():
    """Yield (sample_id, sigma, seed, clean, noisy, labels, planar_lbls, edge_lbls)."""
    xyz_v, lab_v = _load_vai_eval()
    windows = _vai_windows()
    print("  Vaihingen EVAL windows:", [(n, round(cx), round(cy))
                                        for n, cx, cy in windows], flush=True)
    for sigma in TEST_NOISE_LEVELS:
        for seed in TEST_SEEDS:
            for name, cx, cy in windows:
                clean, noisy, lab = _crop(xyz_v, lab_v, cx, cy, 15, sigma, seed)
                yield (name, sigma, seed, clean, noisy, lab, [2, 5], [4, 6])
    for tname, rel in TEST_DALES_TILES:
        xyz_d, lab_d = _load_dales(rel)
        win = pick_window(xyz_d, lab_d, ground_lbls=[1], building_lbls=[8],
                          half=10.0, grid_step=25.0, min_pts=6000, max_pts=60000)
        cx, cy, _ = win
        print(f"  {tname} window ({cx:.0f},{cy:.0f})", flush=True)
        for sigma in TEST_NOISE_LEVELS:
            for seed in TEST_SEEDS:
                clean, noisy, lab = _crop(xyz_d, lab_d, cx, cy, 10, sigma, seed,
                                          max_pts=10000)
                yield (tname, sigma, seed, clean, noisy, lab, [1, 8], [6, 7])


def mn_samples():
    """Yield (sample_id, seed, clean_pts, noisy_pts)."""
    import glob
    import trimesh
    for cat in TEST_MODELNET:
        f = sorted(glob.glob(str(paths.modelnet40_root() / cat / "test" / "*.off")))[0]
        mesh = trimesh.load(f)
        for seed in TEST_SEEDS:
            rng = np.random.default_rng(seed)
            # seed also controls surface sampling for full independence
            pts, _ = trimesh.sample.sample_surface(
                mesh, 5000, seed=seed)
            pts = np.asarray(pts, dtype=np.float64)
            pts = pts / np.linalg.norm(pts.max(0) - pts.min(0))
            noisy = pts + rng.normal(0, TEST_MN_NOISE, pts.shape)
            yield (cat, seed, pts, noisy)
