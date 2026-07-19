"""Sensitivity of the frozen AdaWave globals (dev crops V1-Center + D1-Urban,
sigma=5cm): each parameter varied around its frozen value, others fixed.
Writes sensitivity_v2.csv."""
import sys, csv
sys.path.insert(0, ".")
import numpy as np
from eval_datasets import (VAIHINGEN_CROPS, DALES_CROPS, prepare_vaihingen_crop,
                           prepare_dales_crop, z_rmse, find_boundary_points)
from adawave.restoration import denoise_wavefront_prior_v2

data = []
for kind, idx, pl, el in [('vai', 0, [2,5], [4,6]), ('dal', 0, [1,8], [6,7])]:
    if kind == 'vai':
        clean, noisy, labels = prepare_vaihingen_crop(VAIHINGEN_CROPS[idx], noise_std=0.05, seed=42)
    else:
        clean, noisy, labels = prepare_dales_crop(DALES_CROPS[idx], noise_std=0.05, seed=42)
    bd = find_boundary_points(clean, labels, k=12)
    data.append((clean, noisy, bd))

SWEEPS = [("lam_n", [6.25, 12.5, 25.0, 50.0]),
          ("anchor", [300.0, 1000.0, 3000.0]),
          ("cap_sigma", [1.5, 2.0, 2.5, 3.0]),
          ("prop_decay", [0.6, 0.75, 0.9]),
          ("k", [8, 12, 16, 20])]
rows = []
for pname, vals in SWEEPS:
    for v in vals:
        Z, B = [], []
        for clean, noisy, bd in data:
            out = denoise_wavefront_prior_v2(noisy, **{pname: v})
            d = out["xyz"]
            Z.append((1 - z_rmse(d, clean) / z_rmse(noisy, clean)) * 100)
            B.append((1 - z_rmse(d[bd], clean[bd]) / z_rmse(noisy[bd], clean[bd])) * 100)
        rows.append(dict(param=pname, value=v, Z=np.mean(Z), Bdry=np.mean(B)))
        print(f"{pname}={v}: Z={np.mean(Z):+.2f} Bdry={np.mean(B):+.2f}", flush=True)
with open("sensitivity_v2.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["param", "value", "Z", "Bdry"])
    w.writeheader(); w.writerows(rows)
print("saved")
