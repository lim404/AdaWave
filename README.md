# AdaWave — Adaptive Wavefront-Constrained Restoration of Irregular Point Clouds

Training-free, noise-and-structure adaptive point cloud restoration.
Reference implementation, evaluation harness, and raw results for the
TPAMI submission.

## Install

```bash
git clone https://github.com/lim404/AdaWave.git
cd AdaWave
pip install -e .            # core method only
pip install -e ".[eval]"    # + evaluation harness (trimesh, plyfile, matplotlib)
```

Python ≥3.10. Optional extras: `.[recon]` (pymeshlab, reconstruction task),
`.[scannetpp]` (lz4, ScanNet++ depth decoding).

## Method entry point

```python
from adawave import restore
out = restore(xyz_noisy)                      # frozen defaults = paper config
xyz_restored = out["xyz"]                     # also: sigma_g, rel_noise, info
```

The frozen configuration (see `FREEZE_v2.md`) is the default parameter set;
**every experiment in the paper uses these defaults unchanged.**

Optional vectorised implementation (numerically equivalent; end-to-end
deviation < 1e-12 m, ~6x faster; verified by `tests/test_fast_equivalence.py`):

```python
import adawave.fast_geometry as fg
fg.patch()          # swaps batched implementations into the pipeline
```

## Data

Datasets are **not** redistributed here. Obtain each from its original
source (terms and registration as required by the provider), then either
place them under `./data/` or point `ADAWAVE_DATA_ROOT` at the directory
holding them:

```bash
export ADAWAVE_DATA_ROOT=/path/to/datasets
```

The harness expects this layout under that root:

| Dataset | Expected path | Source | Used for |
|---|---|---|---|
| ISPRS Vaihingen 3D | `Vaihingen/3DLabeling/` | [ISPRS benchmark](https://www.isprs.org/education/benchmarks/UrbanSemLab/) (registration) | dev/val (training tile), **final test** (`EVAL_WITH_REF`) |
| DALES | `DALESObjects/` | [Univ. of Dayton](https://udayton.edu/engineering/research/centers/vision_lab/research/was_data_analysis_and_processing/dale.php) | dev (train tiles), final test (test tiles), full-tile run |
| ModelNet40 | `ModelNet40/` | [Princeton](https://modelnet.cs.princeton.edu/) | dev/val/test category splits |
| PU-Net test set | `PUNet_denoise/PUNet/pointclouds/` | as distributed with score-denoise | DL in-domain sanity |
| ScanNet++ | `ScanNet++/data/` | [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/) (registration) | World B safety, normals task |

ScanNet++ often lives on a separate volume; `ADAWAVE_SCANNETPP_ROOT`
overrides its location independently.

Missing data produces an explicit error naming the expected path and the
download source, not a bare traceback (`eval/paths.py`).

Pre-registered development/validation/final-test splits: `eval/eval_splits.py`
(binding; see `FREEZE_v2.md` §3).

## Reproducing the paper

Run from the repository root after `pip install -e ".[eval]"`.

| Paper asset | Command | Output |
|---|---|---|
| Table I + stats (final ALS) | `python eval/final_test.py` | `final_test_als.csv`, `final_test_mn.csv` |
| World A table | `python eval/worldA_run.py` (spec: `eval/worldA_spec.py`) | `worldA_results.csv` |
| Safety table + Fig. adaptivity | `python eval/worldB_safety.py` | `worldB_safety.csv` |
| Normals table | `python eval/worldB_normals.py` | `worldB_normals.csv` |
| Reconstruction table | `python eval/downstream_reconstruction.py` | `downstream_reconstruction.csv` |
| Ablation table | `python eval/ablation_v2.py`; `python eval/freeze_experiments.py` | `ablation_v2.csv`, `freeze_experiments.csv` |
| Sensitivity table | `python eval/sensitivity_v2.py` | `sensitivity_v2.csv` |
| Large-scale run | `python eval/large_scale_tile.py` | `large_scale_tile_summary.csv` |
| All LaTeX tables | `python eval/make_tip_tables.py` | `tables/*.tex` |

Numbers in the manuscript are **generated from the CSVs, never typed by
hand** (`eval/make_tip_tables.py`).

### Not included in this repository

The learned-baseline comparisons require third-party code and checkpoints
that we cannot redistribute (each carries its own license, ~680 MB total):
**ScoreDenoise**, **IterativePFN**, **PathNet**, **StraightPCF**. The
corresponding runners (`final_test_dl.py`, `sanity_punet.py`,
`run_*.py`) and the qualitative panel script are therefore omitted.

Their *outputs* are included in full under `supplementary/raw_csv/`
(`final_test_dl_*.csv`, `sanity_punet_*.csv`), so every number we report
for them is auditable even without rerunning. To reproduce them, clone
each upstream repository, fetch its official checkpoint, and follow
`protocols/baseline_settings.md`, which records the exact settings and
checkpoint provenance used.

The manuscript source (`paper_v2/`) is not part of this release.

## Integrity notes

- The method was **frozen** (`FREEZE_v2.md`, 2026-07-18) before the
  validation and final-test runs; final-test data were evaluated once.
  `COMMIT_HASH.txt` pins the freeze commit and explains how to audit that
  later commits did not alter the method.
- Learned baselines: official checkpoints; shims replace build
  dependencies only (numerically identical ops); each model reproduces its
  published PU-Net accuracy in this harness (`supplementary/raw_csv/sanity_punet_*.csv`).
- Known limitations and failure cases are reported in the manuscript
  §Limitations and reproduced by `eval/worldB_safety.py` (vegetation,
  correlated bias).

## Tests

```bash
python -m pytest tests/ -q       # 5 certificate tests, ~17 s, no data needed
```

## License

Code: MIT (`LICENSE`). Datasets retain their original licenses and terms
(ISPRS, DALES, ModelNet40, PU-Net, ScanNet++).
