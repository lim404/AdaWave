"""
Complete paper evaluation suite:
  1. Baseline comparison (uniform smooth, bilateral/PCA smooth, ours-initial, ours-optimized)
  2. Ablation study (w/o each module)
  3. Absolute RMSE table with category grouping (man-made / vegetation / overall)
"""

import paths
import numpy as np
import time
import sys
sys.path.insert(0, ".")

# ---------------------------------------------------------------------------
# Data preparation (shared across all experiments)
# ---------------------------------------------------------------------------

LABEL_NAMES = {
    0: "Powerline", 1: "Low_veg", 2: "Impervious", 3: "Car",
    4: "Fence/Hedge", 5: "Roof", 6: "Facade", 7: "Shrub", 8: "Tree",
}
MANMADE = {2, 3, 5, 6}      # Impervious, Car, Roof, Facade
VEGETATION = {1, 4, 7, 8}   # Low_veg, Fence/Hedge, Shrub, Tree


def prepare_data():
    """Load Vaihingen training set, crop, add noise, preprocess once."""
    pts_path = paths.require(paths.vaihingen_train_pts(), "Vaihingen training tile")
    data = np.loadtxt(pts_path)
    xyz = data[:, :3]
    labels = data[:, 6].astype(int)

    cx, cy = np.median(xyz[:, 0]), np.median(xyz[:, 1])
    half = 15.0
    mask = ((xyz[:, 0] >= cx - half) & (xyz[:, 0] <= cx + half) &
            (xyz[:, 1] >= cy - half) & (xyz[:, 1] <= cy + half))
    crop_xyz = xyz[mask]
    crop_labels = labels[mask]

    rng = np.random.default_rng(42)
    noisy_xyz = crop_xyz + rng.normal(0, 0.05, crop_xyz.shape)

    # Pre-run preprocessing so all methods share the same point set
    from adawave.preprocess import remove_duplicates, statistical_outlier_removal
    dup_mask = remove_duplicates(noisy_xyz)
    sor_mask = statistical_outlier_removal(noisy_xyz[dup_mask], k=20, std_ratio=2.0)
    idx_after_dup = np.where(dup_mask)[0]
    keep = idx_after_dup[sor_mask]
    final_mask = np.zeros(len(noisy_xyz), dtype=bool)
    final_mask[keep] = True

    clean = crop_xyz[final_mask]
    noisy = noisy_xyz[final_mask]
    labs  = crop_labels[final_mask]

    centroid = noisy.mean(axis=0)
    clean_c = clean - centroid
    noisy_c = noisy - centroid

    print(f"Data: {len(clean)} points after preprocessing "
          f"(from {len(crop_xyz)} crop)")
    return clean_c, noisy_c, labs, centroid


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def rmse(a, b, axis=None):
    return np.sqrt(np.mean((a - b) ** 2, axis=axis))


def find_boundary_points(xyz, labels, k=12, mixed_ratio=0.3):
    """Find points near label boundaries (where neighbors have different labels)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(xyz)
    _, indices = tree.query(xyz, k=k + 1)
    indices = indices[:, 1:]

    boundary = np.zeros(len(xyz), dtype=bool)
    for i in range(len(xyz)):
        nbr_labels = labels[indices[i]]
        same = (nbr_labels == labels[i]).mean()
        if same < (1.0 - mixed_ratio):
            boundary[i] = True
    return boundary


def compute_metrics(denoised, clean, noisy, labels, boundary_mask=None):
    """Return dict of RMSE metrics for every label + groups + boundary."""
    res = {}

    # Per-axis and overall
    res["RMSE_3D_noisy"]    = rmse(noisy, clean)
    res["RMSE_3D_denoised"] = rmse(denoised, clean)
    res["RMSE_Z_noisy"]     = rmse(noisy[:, 2], clean[:, 2])
    res["RMSE_Z_denoised"]  = rmse(denoised[:, 2], clean[:, 2])

    # Per-label Z
    for lbl in sorted(np.unique(labels)):
        m = labels == lbl
        if m.sum() < 10:
            continue
        name = LABEL_NAMES.get(lbl, f"Lbl{lbl}")
        res[f"Z_{name}_noisy"]    = rmse(noisy[m, 2], clean[m, 2])
        res[f"Z_{name}_denoised"] = rmse(denoised[m, 2], clean[m, 2])

    # Category groups
    for group_name, group_lbls in [("ManMade", MANMADE), ("Vegetation", VEGETATION)]:
        gm = np.isin(labels, list(group_lbls))
        if gm.sum() < 10:
            continue
        res[f"Z_{group_name}_noisy"]    = rmse(noisy[gm, 2], clean[gm, 2])
        res[f"Z_{group_name}_denoised"] = rmse(denoised[gm, 2], clean[gm, 2])

    # Boundary / interior split
    if boundary_mask is not None:
        interior = ~boundary_mask
        if boundary_mask.sum() >= 10:
            res["Z_Boundary_noisy"]    = rmse(noisy[boundary_mask, 2], clean[boundary_mask, 2])
            res["Z_Boundary_denoised"] = rmse(denoised[boundary_mask, 2], clean[boundary_mask, 2])
            res["RMSE_3D_Boundary_noisy"]    = rmse(noisy[boundary_mask], clean[boundary_mask])
            res["RMSE_3D_Boundary_denoised"] = rmse(denoised[boundary_mask], clean[boundary_mask])
        if interior.sum() >= 10:
            res["Z_Interior_noisy"]    = rmse(noisy[interior, 2], clean[interior, 2])
            res["Z_Interior_denoised"] = rmse(denoised[interior, 2], clean[interior, 2])
            res["RMSE_3D_Interior_noisy"]    = rmse(noisy[interior], clean[interior])
            res["RMSE_3D_Interior_denoised"] = rmse(denoised[interior], clean[interior])

    return res


# ---------------------------------------------------------------------------
# 1. Baseline methods
# ---------------------------------------------------------------------------

def baseline_uniform_smooth(noisy_c, k=12, n_iters=2, weight=0.2):
    """Baseline A: simple kNN uniform average smoothing."""
    from scipy.spatial import cKDTree
    xyz = noisy_c.copy()
    for _ in range(n_iters):
        tree = cKDTree(xyz)
        _, indices = tree.query(xyz, k=k + 1)
        indices = indices[:, 1:]  # exclude self
        for i in range(len(xyz)):
            xyz[i] = (1 - weight) * xyz[i] + weight * xyz[indices[i]].mean(axis=0)
    return xyz


def baseline_bilateral_normal(noisy_c, k=12, n_iters=2, weight=0.2):
    """Baseline B: PCA normal + bilateral smoothing (no directional analysis)."""
    from adawave.neighbors import NeighborQuery
    from adawave.geometry import estimate_local_frames

    xyz = noisy_c.copy()
    for _ in range(n_iters):
        nq = NeighborQuery(xyz)
        dists, indices = nq.query_knn(k)
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
            shift = np.dot(w, np_proj) / ws
            shift = np.clip(shift, -0.5 * median_spacing, 0.5 * median_spacing)
            xyz_new[i] += weight * shift * normals[i]
        xyz = xyz_new
    return xyz


def ours_initial(noisy_c, k=12, n_iters=2):
    """Our method, initial version: no neighbor filtering, no adaptive feature, alignment^2."""
    from adawave.neighbors import NeighborQuery
    from adawave.geometry import estimate_local_frames
    from adawave.patch import build_density_patches
    from adawave.directional import compute_directional_responses
    from adawave.descriptors import compute_descriptors

    xyz = noisy_c.copy()
    for _ in range(n_iters):
        nq = NeighborQuery(xyz)
        dists, indices = nq.query_knn(k)
        frames = estimate_local_frames(xyz, indices)
        patches = build_density_patches(
            xyz, indices, frames["normals"],
            frames["tangent1"], frames["tangent2"], grid_size=12)
        responses = compute_directional_responses(patches, n_dirs=8)
        desc = compute_descriptors(responses, n_dirs=8)

        median_spacing = np.median(dists[:, 0])
        xyz_new = xyz.copy()
        for i in range(len(xyz)):
            diff = xyz[indices[i]] - xyz[i]
            d = np.linalg.norm(diff, axis=1)
            np_proj = diff @ frames["normals"][i]
            sigma_s = np.median(d)
            if sigma_s < 1e-12:
                continue
            sigma_r = 0.3 * sigma_s
            w_s = np.exp(-0.5 * (d / sigma_s) ** 2)
            w_r = np.exp(-0.5 * (np_proj / sigma_r) ** 2)

            if not desc["is_feature"][i]:
                w = w_s * w_r
                ws = w.sum()
                if ws < 1e-12:
                    continue
                shift = np.clip(np.dot(w, np_proj) / ws,
                                -0.5 * median_spacing, 0.5 * median_spacing)
                xyz_new[i] += 0.2 * shift * frames["normals"][i]
            else:
                theta = desc["principal_direction"][i]
                t1 = frames["tangent1"][i]
                n_vec = frames["normals"][i]
                t2 = np.cross(n_vec, t1)
                t2n = np.linalg.norm(t2)
                if t2n > 1e-12:
                    t2 /= t2n
                else:
                    t2 = np.zeros(3)
                p3d = np.cos(theta) * t1 + np.sin(theta) * t2
                dn = diff / (d[:, None] + 1e-12)
                align = np.abs(dn @ p3d) ** 2  # alignment^2 (original)
                w = w_s * w_r * align
                ws = w.sum()
                if ws < 1e-12:
                    continue
                shift = np.clip(np.dot(w, np_proj) / ws,
                                -0.3 * median_spacing, 0.3 * median_spacing)
                # No adaptive scaling (original)
                xyz_new[i] += 0.12 * shift * frames["normals"][i]
        xyz = xyz_new
    return xyz


def ours_optimized(noisy_c):
    """Our full optimized pipeline."""
    from adawave.neighbors import NeighborQuery
    from adawave.denoise import iterative_denoise

    nq = NeighborQuery(noisy_c.copy())
    result = iterative_denoise(
        noisy_c.copy(), nq, k=12, n_iters=2, n_dirs=8, grid_size=12,
        smooth_weight=0.2, feature_weight=0.12,
        patch_type="density", verbose=False,
    )
    return result["xyz"]


# ---------------------------------------------------------------------------
# 2. Ablation variants
# ---------------------------------------------------------------------------

def ablation_no_neighbor_filter(noisy_c):
    """Ablation: full pipeline but neighbor_quality = all ones (no filtering)."""
    from adawave.neighbors import NeighborQuery
    from adawave.geometry import estimate_local_frames
    from adawave.patch import build_density_patches
    from adawave.directional import compute_directional_responses
    from adawave.descriptors import compute_descriptors
    from adawave.denoise import denoise_once

    xyz = noisy_c.copy()
    for _ in range(2):
        nq = NeighborQuery(xyz)
        dists, indices = nq.query_knn(12)
        frames = estimate_local_frames(xyz, indices)
        # Skip filter_neighbors_by_normal — use uniform weights
        ones = np.ones_like(dists)
        patches = build_density_patches(
            xyz, indices, frames["normals"],
            frames["tangent1"], frames["tangent2"],
            grid_size=12, neighbor_weights=ones)
        responses = compute_directional_responses(patches, n_dirs=8)
        desc = compute_descriptors(responses, n_dirs=8)
        xyz = denoise_once(
            xyz, indices, dists, frames["normals"], frames["tangent1"],
            desc["singularity_strength"], desc["principal_direction"],
            desc["is_feature"], neighbor_quality=None,
            smooth_weight=0.2, feature_weight=0.12)
    return xyz


def ablation_all_smooth(noisy_c):
    """Ablation: no feature branch — all points use bilateral smoothing."""
    from adawave.neighbors import NeighborQuery
    from adawave.geometry import estimate_local_frames, filter_neighbors_by_normal
    from adawave.patch import build_density_patches
    from adawave.directional import compute_directional_responses
    from adawave.descriptors import compute_descriptors
    from adawave.denoise import denoise_once

    xyz = noisy_c.copy()
    for _ in range(2):
        nq = NeighborQuery(xyz)
        dists, indices = nq.query_knn(12)
        frames = estimate_local_frames(xyz, indices)
        nq_w = filter_neighbors_by_normal(
            xyz, indices, dists, frames["normals"], 0.3, 1.5)
        patches = build_density_patches(
            xyz, indices, frames["normals"],
            frames["tangent1"], frames["tangent2"],
            grid_size=12, neighbor_weights=nq_w)
        responses = compute_directional_responses(patches, n_dirs=8)
        desc = compute_descriptors(responses, n_dirs=8)
        desc["is_feature"][:] = False
        xyz = denoise_once(
            xyz, indices, dists, frames["normals"], frames["tangent1"],
            desc["singularity_strength"], desc["principal_direction"],
            desc["is_feature"], neighbor_quality=nq_w,
            smooth_weight=0.2, feature_weight=0.12)
    return xyz


def ablation_no_propagation(noisy_c):
    """Ablation: full pipeline but without structural propagation."""
    from adawave.neighbors import NeighborQuery
    from adawave.geometry import estimate_local_frames, filter_neighbors_by_normal
    from adawave.patch import build_density_patches
    from adawave.directional import compute_directional_responses
    from adawave.descriptors import compute_descriptors
    from adawave.denoise import denoise_once

    xyz = noisy_c.copy()
    for _ in range(2):
        nq = NeighborQuery(xyz)
        dists, indices = nq.query_knn(12)
        frames = estimate_local_frames(xyz, indices)
        nq_w = filter_neighbors_by_normal(
            xyz, indices, dists, frames["normals"], 0.3, 1.5)
        patches = build_density_patches(
            xyz, indices, frames["normals"],
            frames["tangent1"], frames["tangent2"],
            grid_size=12, neighbor_weights=nq_w)
        responses = compute_directional_responses(patches, n_dirs=8)
        desc = compute_descriptors(responses, n_dirs=8)
        # No propagation — use raw descriptors directly
        xyz = denoise_once(
            xyz, indices, dists, frames["normals"], frames["tangent1"],
            desc["singularity_strength"], desc["principal_direction"],
            desc["is_feature"], neighbor_quality=nq_w,
            smooth_weight=0.2, feature_weight=0.12)
    return xyz


def ablation_height_patch(noisy_c):
    """Ablation: use height patch instead of density patch."""
    from adawave.neighbors import NeighborQuery
    from adawave.denoise import iterative_denoise

    nq = NeighborQuery(noisy_c.copy())
    result = iterative_denoise(
        noisy_c.copy(), nq, k=12, n_iters=2, n_dirs=8, grid_size=12,
        smooth_weight=0.2, feature_weight=0.12,
        patch_type="height", verbose=False,
    )
    return result["xyz"]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_experiment(name, fn, noisy_c, clean_c, labels, boundary_mask):
    """Run one method, measure time, return metrics."""
    t0 = time.time()
    denoised = fn(noisy_c)
    elapsed = time.time() - t0
    metrics = compute_metrics(denoised, clean_c, noisy_c, labels, boundary_mask)
    metrics["time_s"] = elapsed
    print(f"  {name:35s}  done in {elapsed:.1f}s")
    return metrics


def pct(noisy_val, denoised_val):
    return (1 - denoised_val / noisy_val) * 100


def print_table(title, rows, col_keys, col_headers, label_order):
    """Print a formatted table."""
    print(f"\n{'=' * 90}")
    print(title)
    print(f"{'=' * 90}")

    # Header
    hdr = f"{'':17s}"
    for h in col_headers:
        hdr += f"  {h:>12s}"
    print(hdr)
    print("-" * 90)

    for label_key in label_order:
        row_str = f"  {label_key:15s}"
        for method_metrics in rows:
            nk = f"Z_{label_key}_noisy"
            dk = f"Z_{label_key}_denoised"
            if nk in method_metrics and dk in method_metrics:
                nv = method_metrics[nk]
                dv = method_metrics[dk]
                p = pct(nv, dv)
                row_str += f"  {dv*100:5.2f}({p:+5.1f}%)"
            else:
                row_str += f"  {'N/A':>12s}"
        print(row_str)


def main():
    print("=" * 90)
    print("PAPER EVALUATION SUITE")
    print("=" * 90)

    clean_c, noisy_c, labels, centroid = prepare_data()

    # Pre-compute boundary mask using clean data
    boundary_mask = find_boundary_points(clean_c, labels, k=12, mixed_ratio=0.3)
    n_bnd = boundary_mask.sum()
    n_int = (~boundary_mask).sum()
    print(f"Boundary points: {n_bnd} ({100*n_bnd/len(labels):.1f}%), "
          f"Interior points: {n_int} ({100*n_int/len(labels):.1f}%)")

    # =======================================================================
    # Part 1: Baseline Comparison
    # =======================================================================
    print("\n>>> PART 1: BASELINE COMPARISON")

    from adawave.baselines import (denoise_mls, denoise_wlop, denoise_glr,
                                      denoise_jet, denoise_clop)

    experiments_baseline = [
        ("(A) Uniform kNN",          lambda x: baseline_uniform_smooth(x, k=12, n_iters=2, weight=0.2)),
        ("(B) Bilateral",            lambda x: baseline_bilateral_normal(x, k=12, n_iters=2, weight=0.2)),
        ("(C) MLS",                  lambda x: denoise_mls(x, k=12, n_iters=2)),
        ("(D) Jet Fitting",          lambda x: denoise_jet(x, k=20, n_iters=2)),
        ("(E) CLOP (edge-aware)",    lambda x: denoise_clop(x, k=12, n_iters=2)),
        ("(F) GLR",                  lambda x: denoise_glr(x, k=12, mu=0.5)),
        ("(G) WLOP",                 lambda x: denoise_wlop(x, k=20, n_iters=2)),
        ("(H) Ours (initial)",       lambda x: ours_initial(x, k=12, n_iters=2)),
        ("(I) Ours (full)",          lambda x: ours_optimized(x)),
    ]

    baseline_results = {}
    for name, fn in experiments_baseline:
        baseline_results[name] = run_experiment(name, fn, noisy_c, clean_c, labels, boundary_mask)

    # =======================================================================
    # Part 2: Ablation Study
    # =======================================================================
    print("\n>>> PART 2: ABLATION STUDY")

    experiments_ablation = [
        ("Full model",                lambda x: ours_optimized(x)),
        ("w/o neighbor filtering",    lambda x: ablation_no_neighbor_filter(x)),
        ("w/o feature branch",        lambda x: ablation_all_smooth(x)),
        ("w/o propagation",           lambda x: ablation_no_propagation(x)),
        ("w/ height patch",           lambda x: ablation_height_patch(x)),
    ]

    ablation_results = {}
    for name, fn in experiments_ablation:
        ablation_results[name] = run_experiment(name, fn, noisy_c, clean_c, labels, boundary_mask)

    # =======================================================================
    # Part 3: Full Results Tables
    # =======================================================================

    all_labels_present = sorted([
        LABEL_NAMES[l] for l in np.unique(labels) if l in LABEL_NAMES
        and (labels == l).sum() >= 10
    ])

    # -----------------------------------------------------------------------
    # TABLE 1: Baseline comparison — Z-RMSE per class
    # -----------------------------------------------------------------------
    W = 140
    print(f"\n{'=' * W}")
    print("TABLE 1: Baseline Comparison — Z-axis RMSE (cm) and improvement (%)")
    print(f"{'=' * W}")

    methods_b = list(baseline_results.keys())
    hdr = f"  {'Category':15s}  {'Noisy':>8s}"
    for m in methods_b:
        short = m.split(")")[0] + ")"
        hdr += f"  {short:>13s}"
    print(hdr)
    print("  " + "-" * (W - 2))

    label_groups = [
        ("--- Man-made ---", None),
    ]
    for lbl in sorted(MANMADE):
        n = LABEL_NAMES.get(lbl)
        if n and n in all_labels_present:
            label_groups.append((n, lbl))
    label_groups.append(("ManMade (avg)", "ManMade"))
    label_groups.append(("--- Vegetation ---", None))
    for lbl in sorted(VEGETATION):
        n = LABEL_NAMES.get(lbl)
        if n and n in all_labels_present:
            label_groups.append((n, lbl))
    label_groups.append(("Vegetation (avg)", "Vegetation"))
    label_groups.append(("--- Overall ---", None))
    label_groups.append(("Overall", None))

    for display_name, key in label_groups:
        if key is None and "---" in display_name:
            print(f"  {display_name}")
            continue

        if display_name == "Overall":
            nk, dk = "RMSE_Z_noisy", "RMSE_Z_denoised"
        elif isinstance(key, str):
            nk, dk = f"Z_{key}_noisy", f"Z_{key}_denoised"
        else:
            name = LABEL_NAMES.get(key, "")
            nk, dk = f"Z_{name}_noisy", f"Z_{name}_denoised"

        # Get noisy value from first method (all same)
        first = list(baseline_results.values())[0]
        if nk not in first:
            continue
        noisy_v = first[nk]

        row = f"  {display_name:15s}  {noisy_v*100:6.2f}cm"
        for m in methods_b:
            mr = baseline_results[m]
            if dk in mr:
                dv = mr[dk]
                p = pct(noisy_v, dv)
                row += f"  {dv*100:5.2f}({p:+5.1f}%)"
            else:
                row += f"  {'N/A':>13s}"
        print(row)

    # -----------------------------------------------------------------------
    # TABLE 2: Ablation study
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 90}")
    print("TABLE 2: Ablation Study — Z-axis RMSE (cm) and improvement (%)")
    print(f"{'=' * 90}")

    methods_a = list(ablation_results.keys())
    hdr = f"  {'Category':15s}  {'Noisy':>10s}"
    for m in methods_a:
        short = m[:18]
        hdr += f"  {short:>14s}"
    print(hdr)
    print("  " + "-" * (17 + 12 + len(methods_a) * 16))

    for display_name, key in label_groups:
        if key is None and "---" in display_name:
            print(f"  {display_name}")
            continue

        if display_name == "Overall":
            nk, dk = "RMSE_Z_noisy", "RMSE_Z_denoised"
        elif isinstance(key, str):
            nk, dk = f"Z_{key}_noisy", f"Z_{key}_denoised"
        else:
            name = LABEL_NAMES.get(key, "")
            nk, dk = f"Z_{name}_noisy", f"Z_{name}_denoised"

        first = list(ablation_results.values())[0]
        if nk not in first:
            continue
        noisy_v = first[nk]

        row = f"  {display_name:15s}  {noisy_v*100:8.2f}cm"
        for m in methods_a:
            mr = ablation_results[m]
            if dk in mr:
                dv = mr[dk]
                p = pct(noisy_v, dv)
                row += f"  {dv*100:7.2f}({p:+5.1f}%)"
            else:
                row += f"  {'N/A':>14s}"
        print(row)

    # -----------------------------------------------------------------------
    # TABLE 3: Absolute statistics summary
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 90}")
    print("TABLE 3: Summary Statistics — Absolute RMSE (cm)")
    print(f"{'=' * 90}")

    summary_methods = {**baseline_results, **{k: v for k, v in ablation_results.items() if k != "Full model"}}
    all_methods = ["Noisy"] + list(baseline_results.keys())

    hdr = f"  {'Metric':20s}"
    for m in all_methods:
        short = m.split(")")[0] + ")" if ")" in m else m[:12]
        hdr += f"  {short:>10s}"
    print(hdr)
    print("  " + "-" * (22 + len(all_methods) * 12))

    ref = list(baseline_results.values())[0]

    summary_rows = [
        ("3D RMSE",         "RMSE_3D_noisy",       "RMSE_3D_denoised"),
        ("Z RMSE",          "RMSE_Z_noisy",        "RMSE_Z_denoised"),
        ("Z ManMade",       "Z_ManMade_noisy",     "Z_ManMade_denoised"),
        ("Z Vegetation",    "Z_Vegetation_noisy",  "Z_Vegetation_denoised"),
    ]
    for lbl in sorted(np.unique(labels)):
        name = LABEL_NAMES.get(lbl)
        if name and (labels == lbl).sum() >= 10:
            summary_rows.append((f"Z {name}", f"Z_{name}_noisy", f"Z_{name}_denoised"))

    for row_name, nk, dk in summary_rows:
        if nk not in ref:
            continue
        noisy_v = ref[nk]
        row = f"  {row_name:20s}  {noisy_v*100:8.2f}cm"
        for m in list(baseline_results.keys()):
            mr = baseline_results[m]
            if dk in mr:
                row += f"  {mr[dk]*100:8.2f}cm"
            else:
                row += f"  {'N/A':>10s}"
        print(row)

    # -----------------------------------------------------------------------
    # TABLE 4: Boundary vs Interior (Feature Preservation)
    # -----------------------------------------------------------------------
    print(f"\n{'=' * W}")
    print("TABLE 4: Boundary vs Interior — 3D RMSE (cm) and Z-RMSE (cm)")
    print("  (Boundary = points near label transitions; Interior = homogeneous regions)")
    print(f"{'=' * W}")

    methods_b = list(baseline_results.keys())
    hdr = f"  {'Region':15s}  {'Noisy':>8s}"
    for m in methods_b:
        short = m.split(")")[0] + ")"
        hdr += f"  {short:>13s}"
    print(hdr)
    print("  " + "-" * (W - 2))

    for region, prefix in [("Boundary Z", "Z_Boundary"),
                           ("Interior Z", "Z_Interior"),
                           ("Boundary 3D", "RMSE_3D_Boundary"),
                           ("Interior 3D", "RMSE_3D_Interior")]:
        nk = f"{prefix}_noisy"
        dk = f"{prefix}_denoised"
        first = list(baseline_results.values())[0]
        if nk not in first:
            continue
        noisy_v = first[nk]
        row = f"  {region:15s}  {noisy_v*100:6.2f}cm"
        for m in methods_b:
            mr = baseline_results[m]
            if dk in mr:
                dv = mr[dk]
                p = pct(noisy_v, dv)
                row += f"  {dv*100:5.2f}({p:+5.1f}%)"
            else:
                row += f"  {'N/A':>13s}"
        print(row)

    # -----------------------------------------------------------------------
    # Timing
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 90}")
    print("Computation Time")
    print(f"{'=' * 90}")
    for name, mr in {**baseline_results, **ablation_results}.items():
        print(f"  {name:35s}  {mr['time_s']:.1f}s")

    print(f"\n{'=' * 90}")
    print("EVALUATION COMPLETE")
    print(f"{'=' * 90}")


def prepare_data_noise(noise_std):
    """Load Vaihingen, crop, add noise at given std (meters)."""
    pts_path = paths.require(paths.vaihingen_train_pts(), "Vaihingen training tile")
    data = np.loadtxt(pts_path)
    xyz = data[:, :3]
    labels = data[:, 6].astype(int)

    cx, cy = np.median(xyz[:, 0]), np.median(xyz[:, 1])
    half = 15.0
    mask = ((xyz[:, 0] >= cx - half) & (xyz[:, 0] <= cx + half) &
            (xyz[:, 1] >= cy - half) & (xyz[:, 1] <= cy + half))
    crop_xyz = xyz[mask]
    crop_labels = labels[mask]

    rng = np.random.default_rng(42)
    noisy_xyz = crop_xyz + rng.normal(0, noise_std, crop_xyz.shape)

    from adawave.preprocess import remove_duplicates, statistical_outlier_removal
    dup_mask = remove_duplicates(noisy_xyz)
    sor_mask = statistical_outlier_removal(noisy_xyz[dup_mask], k=20, std_ratio=2.0)
    idx_after_dup = np.where(dup_mask)[0]
    keep = idx_after_dup[sor_mask]
    final_mask = np.zeros(len(noisy_xyz), dtype=bool)
    final_mask[keep] = True

    clean = crop_xyz[final_mask]
    noisy = noisy_xyz[final_mask]
    labs = crop_labels[final_mask]

    centroid = noisy.mean(axis=0)
    return clean - centroid, noisy - centroid, labs, centroid


def multi_noise_experiment():
    """TABLE 5: Multi-noise-level robustness comparison."""
    noise_levels = [0.02, 0.05, 0.10]  # 2cm, 5cm, 10cm
    noise_names = ["2cm (low)", "5cm (medium)", "10cm (high)"]

    from adawave.baselines import (denoise_mls, denoise_wlop, denoise_glr,
                                      denoise_jet, denoise_clop)
    methods = [
        ("Bilateral",  lambda x: baseline_bilateral_normal(x, k=12, n_iters=2, weight=0.2)),
        ("MLS",        lambda x: denoise_mls(x, k=12, n_iters=2)),
        ("Jet",        lambda x: denoise_jet(x, k=20, n_iters=2)),
        ("CLOP",       lambda x: denoise_clop(x, k=12, n_iters=2)),
        ("GLR",        lambda x: denoise_glr(x, k=12, mu=0.5)),
        ("WLOP",       lambda x: denoise_wlop(x, k=20, n_iters=2)),
        ("Ours",       lambda x: ours_optimized(x)),
    ]

    # Collect results: {noise_name: {method_name: metrics}}
    all_results = {}

    for noise_std, nname in zip(noise_levels, noise_names):
        print(f"\n>>> Noise level: sigma = {nname}")
        clean_c, noisy_c, labels, centroid = prepare_data_noise(noise_std)
        boundary_mask = find_boundary_points(clean_c, labels, k=12, mixed_ratio=0.3)
        print(f"  Points: {len(clean_c)}, Boundary: {boundary_mask.sum()}")

        all_results[nname] = {}
        for mname, fn in methods:
            mr = run_experiment(mname, fn, noisy_c, clean_c, labels, boundary_mask)
            all_results[nname][mname] = mr

    # -------------------------------------------------------------------
    # TABLE 5: Multi-noise comparison
    # -------------------------------------------------------------------
    print(f"\n{'=' * 100}")
    print("TABLE 5: Robustness to Noise Level — Z-RMSE improvement (%)")
    print(f"{'=' * 100}")

    categories = ["Overall", "Roof", "Impervious", "Facade", "Tree", "ManMade", "Vegetation"]
    cat_keys = {
        "Overall": ("RMSE_Z_noisy", "RMSE_Z_denoised"),
        "Roof": ("Z_Roof_noisy", "Z_Roof_denoised"),
        "Impervious": ("Z_Impervious_noisy", "Z_Impervious_denoised"),
        "Facade": ("Z_Facade_noisy", "Z_Facade_denoised"),
        "Tree": ("Z_Tree_noisy", "Z_Tree_denoised"),
        "ManMade": ("Z_ManMade_noisy", "Z_ManMade_denoised"),
        "Vegetation": ("Z_Vegetation_noisy", "Z_Vegetation_denoised"),
    }

    # Header
    hdr = f"  {'Category':15s}"
    for nname in noise_names:
        for mname, _ in methods:
            short = mname.split(")")[0] + ")"
            hdr += f"  {short:>10s}"
        hdr += "  |"
    print(hdr)

    sub_hdr = f"  {'':15s}"
    for nname in noise_names:
        sub_hdr += f"  {'σ=' + nname.split('(')[0].strip():^{10 * len(methods) + 2 * (len(methods)-1) + 2}s}|"
    print(sub_hdr)
    print("  " + "-" * 95)

    for cat in categories:
        nk, dk = cat_keys[cat]
        row = f"  {cat:15s}"
        for nname in noise_names:
            for mname, _ in methods:
                mr = all_results[nname][mname]
                if nk in mr and dk in mr:
                    p = pct(mr[nk], mr[dk])
                    row += f"  {p:+8.1f}%"
                else:
                    row += f"  {'N/A':>9s}"
            row += "  |"
        print(row)

    # Boundary/Interior across noise levels
    print()
    for region, nk, dk in [("Boundary Z", "Z_Boundary_noisy", "Z_Boundary_denoised"),
                            ("Interior Z", "Z_Interior_noisy", "Z_Interior_denoised")]:
        row = f"  {region:15s}"
        for nname in noise_names:
            for mname, _ in methods:
                mr = all_results[nname][mname]
                if nk in mr and dk in mr:
                    p = pct(mr[nk], mr[dk])
                    row += f"  {p:+8.1f}%"
                else:
                    row += f"  {'N/A':>9s}"
            row += "  |"
        print(row)

    # Absolute RMSE table
    print(f"\n{'=' * 100}")
    print("TABLE 5b: Absolute Z-RMSE (cm) across noise levels")
    print(f"{'=' * 100}")

    hdr2 = f"  {'Category':15s}"
    for nname in noise_names:
        hdr2 += f"  {'Noisy':>8s}"
        for mname, _ in methods:
            short = mname.split(")")[0] + ")"
            hdr2 += f"  {short:>8s}"
        hdr2 += "  |"
    print(hdr2)
    print("  " + "-" * 95)

    for cat in categories:
        nk, dk = cat_keys[cat]
        row = f"  {cat:15s}"
        for nname in noise_names:
            first_method = methods[0][0]
            mr0 = all_results[nname][first_method]
            if nk in mr0:
                row += f"  {mr0[nk]*100:7.2f}cm"
            else:
                row += f"  {'N/A':>8s}"
            for mname, _ in methods:
                mr = all_results[nname][mname]
                if dk in mr:
                    row += f"  {mr[dk]*100:7.2f}cm"
                else:
                    row += f"  {'N/A':>8s}"
            row += "  |"
        print(row)

    print(f"\n{'=' * 100}")
    print("MULTI-NOISE EVALUATION COMPLETE")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    import sys
    if "--noise" in sys.argv:
        multi_noise_experiment()
    else:
        main()
