"""Pre-registered development / validation / final-test splits.

FROZEN 2026-07-18, after the method freeze (see FREEZE_v2.md).  The final
test set must be evaluated ONCE, after all method and configuration
decisions; it must never inform any design change.

Window selection for new ALS crops is deterministic and method-blind: a
fixed grid search picks the window with the best ground/building class mix
(scene composition only — no denoising result is ever consulted).
"""
import paths
import numpy as np

# ---------------------------------------------------------------- ALS: dev
# Used throughout method development (Q75, cap, buffer, aniso weights,
# anchor decision). Listed for the record; defined in eval_datasets.py.
DEV_VAIHINGEN = ["V1-Center", "V2-Southwest", "V3-Southeast", "V4-North"]
DEV_DALES = ["D1-Urban", "D2-Suburban", "D3-Mixed", "D4-Test"]
DEV_MODELNET = ["airplane", "chair", "car", "table",
                "guitar", "cone", "bottle", "piano"]
DEV_SEED = 42

# --------------------------------------------------------------- validation
# Purpose: one-shot confirmation of the frozen configuration.  If the frozen
# config fails badly here, the failure is REPORTED and any re-design must be
# declared as such (validation then becomes development and fresh validation
# data must be carved out).
VAL_VAIHINGEN_CROPS = [
    # picked by the deterministic ground/building mix search below over the
    # training tile, excluding any window within 45 m of a dev-crop centre
    # (>=15 m gap to every dev window); registered before any evaluation
    ("VV1", 496984, 5419394, 15, "validation; ground 0.51 / bldg 0.46"),
    ("VV2", 497144, 5419394, 15, "validation; ground 0.45 / bldg 0.45"),
]
VAL_DALES_TILES = [
    ("DV1", "train/5110_54460_new.ply"),
    ("DV2", "train/5145_54405_new.ply"),
]
VAL_MODELNET = ["sofa", "lamp", "toilet", "monitor"]
VAL_SEED = 7

# --------------------------------------------------------------- final test
# Run once, after freeze + validation.  Pre-registered here in full.
TEST_VAIHINGEN_SOURCE = str(paths.vaihingen_eval_pts_gz())     # official eval
# four 30x30 m windows found by the deterministic mix search below
TEST_VAIHINGEN_N_WINDOWS = 4
TEST_DALES_TILES = [
    ("DT1", "test/5100_54440_new.ply"),
    ("DT2", "test/5120_54445_new.ply"),
    ("DT3", "test/5135_54430_new.ply"),
    ("DT4", "test/5150_54325_new.ply"),
]
TEST_MODELNET = ["bed", "desk", "dresser", "vase",
                 "sink", "stairs", "laptop", "bench"]
TEST_SEEDS = [7, 19, 101]          # three seeds for mean +/- std
TEST_NOISE_LEVELS = [0.02, 0.05, 0.10]          # ALS sigma (m)
TEST_MN_NOISE = 0.015                            # fraction of bbox diagonal
# metrics: identical definitions to development scripts (z_rmse per region,
# boundary via find_boundary_points(k=12); p2p RMSE + Chamfer on CAD).


def pick_window(xyz, labels, ground_lbls, building_lbls, half=10.0,
                grid_step=25.0, min_pts=6000, max_pts=None):
    """Deterministic, method-blind crop-window search: grid-scan the tile
    and return the (cx, cy) whose window best balances ground and building
    content (maximise min(ground_frac, building_frac), tie-break on point
    count). Never touches any denoising output."""
    xmin, ymin = xyz[:, 0].min(), xyz[:, 1].min()
    xmax, ymax = xyz[:, 0].max(), xyz[:, 1].max()
    best, best_score = None, -1.0
    for cx in np.arange(xmin + half, xmax - half, grid_step):
        for cy in np.arange(ymin + half, ymax - half, grid_step):
            m = ((np.abs(xyz[:, 0] - cx) <= half)
                 & (np.abs(xyz[:, 1] - cy) <= half))
            n = int(m.sum())
            if n < min_pts or (max_pts and n > max_pts):
                continue
            lab = labels[m]
            gf = float(np.isin(lab, ground_lbls).mean())
            bf = float(np.isin(lab, building_lbls).mean())
            score = min(gf, bf) + 1e-9 * n
            if score > best_score:
                best_score, best = score, (float(cx), float(cy), n)
    return best
