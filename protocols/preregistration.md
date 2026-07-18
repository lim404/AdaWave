# Wavefront-Prior v2 — Method Freeze (2026-07-18)

This document freezes the v2 method. From this point on, **no formula,
mechanism, or constant below may change** based on validation or final-test
results. Any change reopens development and invalidates the split.

## 1. Frozen formulation

Displacement parameterisation (binormal variable REMOVED after the freeze
experiments showed it inert, <=0.03pp everywhere):

    x_i = x_noisy_i + alpha_i * n_i

Energy (convex quadratic, alpha-only):

    E(alpha) = sum_i d_i alpha_i^2
             + s2 * lam_n * sum_ij w^wf_ij ((x_i - x_j) . n_i)^2

- Noise floor `sigma_g`: per-point MAD of plane residuals -> median over the
  flattest half (sv <= median) -> shrinkage correction
  `sigma = sigma_raw / max(0.77 - 0.38 * rel_raw, 0.40)`.
- Structure excess `e_i`: Q75 of |residual - median| vs `tau=1.5` times its
  flat-half floor, `e = clip((s^2 - (tau f)^2) / (s^2 + (tau f)^2), 0, 1)`.
- Buffer `e^p_i`: 2 rounds of max-propagation, decay 0.75.
- Relative-noise balance `s2 = (sigma_g / ell_g)^2`.
- Data weight `d_i = 1 + anchor * e_i^2` (**anchor kept**: the freeze battery
  showed anchor-none/auto degrade worst-crop Edge/Bdry monotonically at all
  three noise levels; the anchor is a tail-risk control, not a mean booster).
- Wavefront-anisotropic graph weights (per side, on top of the
  normal-consistency neighbour quality):
  `f_i = (1 - e^p_i (1 - r_i <u_ij, p_i>^2))^2`, r_i = descriptor strength,
  p_i = wavefront (principal singular) direction -> 2-D diffusion on smooth
  regions, ~1-D diffusion along the singular fibre on structure.
- Displacement cap `cap_i = cap_sigma * sigma_g * (1 - e^p_i)^3`.
- Exact solve: matrix-free Jacobi-PCG on H alpha = -q, tol 1e-6.

## 2. Frozen configuration (single global, all domains)

    k=12, n_dirs=8, grid_size=12, normal_thresh=0.7,
    lam_n=12.5, anchor=1000, cap_sigma=2.0, prop_decay=0.75,
    n_outer=1, cg_tol=1e-6
    (internal constants: tau=1.5, shrinkage 0.77/0.38/0.40, buffer rounds=2,
     cap exponent 3, weight exponent 2, flat-half quantile 0.5)

Reference implementation: `wavefront/wavefront_prior_v2.py`
(`denoise_wavefront_prior_v2` defaults; `beta_channel=False`,
`use_geom_weights='aniso'`, `anchor_mode='const'`).

## 3. Data splits (pre-registered in `eval_splits.py`)

- **Development** (all design decisions were made here): Vaihingen V1-V4,
  DALES D1-D4, ModelNet40 {airplane, chair, car, table, guitar, cone,
  bottle, piano}, seed 42, sigma in {2,5,10} cm / 1.5% diag.
- **Validation** (one-shot confirmation of the frozen config; failures are
  reported, not patched silently): Vaihingen VV1/VV2 (deterministic
  method-blind window search, >=15 m gap from dev windows), DALES tiles
  train/5110_54460 + train/5145_54405, ModelNet40 {sofa, lamp, toilet,
  monitor}, seed 7.
- **Final test** (run ONCE, after validation, all methods same protocol):
  official Vaihingen EVAL_WITH_REF region (4 deterministic windows), DALES
  test tiles 5100_54440 / 5120_54445 / 5135_54430 / 5150_54325, ModelNet40
  {bed, desk, dresser, vase, sink, stairs, laptop, bench}, seeds {7,19,101},
  sigma in {2,5,10} cm, MN noise 1.5% diag. Metrics identical to dev
  definitions. World-A synthetic suite must equally be spec'd (shapes,
  noise types, sampling, seeds) before any method sees it.

## 4. DL-baseline fairness protocol (required before final tables)

For each of ScoreDenoise / IterativePFN / PathNet / StraightPCF document:
official checkpoint id + training data; input normalisation; patching,
overlap and fusion; inference iterations; normals usage; any shim (deps
only vs numerics); noise config vs authors' recommendation. Sanity checks
per method: (1) reproduce near-paper numbers on the authors' own benchmark
data; (2) verify output centroid / scale / mean displacement magnitudes;
(3) visualise one in-domain success and one ALS failure case.

## 5. Locked paper claims

1. **Noise-structure adaptation** — consensus noise floor + quantile
   structure excess decide per-point restoration strength and protection
   under unknown noise.
2. **Wavefront-directed graph restoration** — the wavefront direction is
   embedded in the graph connectivity: 2-D diffusion on smooth regions,
   directional diffusion along singularities.
3. **Exact convex inference + cross-domain evidence** — matrix-free PCG,
   provably unique minimiser; one global configuration across CAD and
   irregular airborne LiDAR.

Approved wording for the DL comparison: "the zero-shot deployment gap
persists across four representative architectures released between 2021 and
2024"; in-domain gains do not automatically transfer to unseen scanning
distributions. Do NOT claim the gap "does not shrink with model evolution",
and do NOT rank the four failures as evidence newer models are worse.
