# AdaWave — Adaptive Wavefront-Constrained Restoration of Irregular Point Clouds

Training-free, noise-and-structure adaptive point cloud restoration.
Reference implementation and full reproduction package for the TIP
submission (`paper_v2/main.tex`).

## Method entry point

```python
from adawave import restore
out = restore(xyz_noisy)                      # frozen defaults = paper config
xyz_restored = out["xyz"]                     # also: sigma_g, rel_noise, info
```

The frozen configuration (see `FREEZE_v2.md`) is the default parameter set;
**every experiment in the paper uses these defaults unchanged.**

Optional vectorised implementation (numerically equivalent; end-to-end
deviation < 1e-12 m, ~6x faster; verified by `test_fast_equivalence.py`):

```python
import adawave.fast_geometry as fg
fg.patch()          # swaps batched implementations into the pipeline
```

## Environment

Python ≥3.10, numpy, scipy, trimesh, plyfile, matplotlib; `pymeshlab`
(reconstruction task), `lz4` (ScanNet++ depth decoding), `torch`+CUDA and
`torch_geometric`, `einops`, `gdown` (learned baselines only).
Learned-baseline repos live under `external/` (ScoreDenoise, IterativePFN,
PathNet, StraightPCF) with dependency shims — see `run_*.py` headers for
checkpoint provenance.

## Data layout

| Dataset | Path | Used for |
|---|---|---|
| ISPRS Vaihingen 3D | `/mnt/a/data/Vaihingen/3DLabeling/` | dev/val (training tile), **final test (EVAL_WITH_REF)** |
| DALES | `/mnt/a/data/DALESObjects/` | dev (train tiles), final test (test tiles), full-tile run |
| ModelNet40 | `/mnt/a/data/ModelNet40/` | dev/val/test category splits |
| PU-Net test set | `/mnt/a/data/PUNet_denoise/PUNet/pointclouds/` | DL in-domain sanity |
| ScanNet++ | `/mnt/e/ScanNet++/data/` | World B safety, normals task |

Pre-registered development/validation/final-test splits: `eval_splits.py`
(binding; see `FREEZE_v2.md` §3).

## Reproducing the paper

| Paper asset | Command | Output |
|---|---|---|
| Table I + stats (final ALS) | `python final_test.py`; `python final_test_dl.py <score|pfn|pathnet|straightpcf>` | `final_test_*.csv` |
| Table II (ModelNet40) | same as above | `final_test_mn.csv` |
| World A table | `python worldA_run.py` (spec: `worldA_spec.py`) | `worldA_results.csv` |
| Safety table + Fig. adaptivity | `python worldB_safety.py` | `worldB_safety.csv` |
| Reconstruction table | `python downstream_reconstruction.py` | `downstream_reconstruction.csv` |
| Normals table | `python worldB_normals.py` | `worldB_normals.csv` |
| Ablation table | `python ablation_v2.py`; `python freeze_experiments.py` | `ablation_v2.csv`, `freeze_experiments.csv` |
| Sensitivity table | `python sensitivity_v2.py` | `sensitivity_v2.csv` |
| Large-scale run | `python large_scale_tile.py` | `large_scale_tile_summary.csv` |
| DL fairness sanity | `python sanity_punet.py <method>`; `python viz_dl_sanity.py <method>` | `sanity_punet_*.csv`, `paper_figures/dl_sanity/` |
| Cross-domain scatter / σ-gain figs | scripts embedded in session logs; data from the CSVs above | `paper_figures/*.pdf` |
| Qualitative panel | `python viz_adawave.py` | `paper_figures/qualitative_panel.pdf` |
| Failure case | (script in repo history) | `paper_figures/failure_vegetation.pdf` |
| All LaTeX tables | `python make_tip_tables.py` | `paper_v2/tables/*.tex` |
| Manuscript | `pdflatex` + `bibtex` in `paper_v2/` | `paper_v2/main.pdf` |

Numbers in the manuscript are **generated from the CSVs, never typed by
hand** (`make_tip_tables.py`).

## Integrity notes

- The method was **frozen** (`FREEZE_v2.md`, 2026-07-18) before the
  validation and final-test runs; final-test data were evaluated once.
- Learned baselines: official checkpoints; shims replace build
  dependencies only (numerically identical ops); each model reproduces its
  published PU-Net accuracy in this harness (`sanity_punet_*.csv`).
- Known limitations and failure cases are reported in the manuscript §Limitations
  and reproduced by `worldB_safety.py` (vegetation, correlated bias).

## License

Code: MIT (proposed — confirm before release). Datasets retain their
original licenses/terms (ISPRS, DALES, ModelNet40, PU-Net, ScanNet++).
