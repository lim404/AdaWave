# Baseline deployment settings (fairness protocol)

## Classical (all training-free; settings fixed across every experiment)
- Bilateral: k=12, 2 iterations, weight 0.2 (normal-guided)
- MLS: k=12, 2 iterations; Jet: k=20, 2 iterations
- WLOP: k=20, 2 iterations; CLOP: k=12, 2 iterations; GLR: k=12
- WF-Fixed (non-adaptive predecessor): lam_n=1, lam_b=2, 80 GD iters,
  step 0.05, momentum 0.5, cap 0.15*spacing, anchor 100
- SOR: k=20, alpha=2; LOF: n_neighbors=20, sklearn defaults

## Learned (public checkpoints, zero-shot; shims replace build deps only)
| Model | Venue | Checkpoint | Params |
|---|---|---|---|
| ScoreDenoise | ICCV 2021 | authors' PU-Net ckpt.pt | 187k |
| IterativePFN | CVPR 2023 | authors' denoisenet-ep-99.ckpt | 3.20M |
| PathNet | TPAMI 2024 | authors' GDrive best_model.pth | 15.6M |
| StraightPCF | CVPR 2024 | in-repo ckpt_straightpcf.pt (+CVM) | 533k |

Deployment: authors' patch-based inference (patch size / seed_k /
iterations as released); unit-sphere (Score/PFN/StraightPCF) or per-patch
PCA (PathNet) normalisation as in the respective repos; outputs
denormalised and NN-aligned to the clean cloud for region metrics
(patch pipelines reorder points). Sanity: each model reproduces its
published PU-Net accuracy in this harness before any cross-domain claim
(sanity CSVs in supplementary/raw_csv/).
