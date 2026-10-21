# Post-hoc pre-registration: where does a physics reference stop helping?

Written 2026-10-21, before any run of this experiment. Package commit at writing: a4656fb (Round 10 v4) plus the files in this folder. This is a post-hoc mechanism study outside `prereg/round10_v4.json`; it cannot change the frozen Round 10 recipe and it never reads the test set.

## Question

Round 10 showed two groups of output references and nothing in between:

- References whose training residual is below about 1% of the target spread (inlet anchor for T_inner, the 1-D physics model for T_outer and T_avg) helped, although each of them is less accurate on its own than the direct model.
- References that leave 49% or more (initial value, inlet for T_outer, ratio outputs) were neutral or harmful.

The literal rule "a reference helps iff its own error is below the direct model's error" was refuted. This study locates the boundary between the two groups and tests whether the residual share alone predicts the gain.

## Design

One architecture and protocol, identical to registered arm C11 (`PHYS_MODE=residual`, h128x1, T_avg mask `split`, vs3 split, loss 1:6:3, 2000-epoch cap, TF32=0, SEAL_TEST=1). The only thing that changes between levels is the physics cache the residual is taken against.

| Level | Family | Reference for T_outer and T_avg | Training residual share, T_outer / T_avg (all 284 training files; T_avg on the 169 line-average ones) |
|---|---|---|---|
| (C11) | nominal | 1-D model, FE-report properties | 0.6% / 0.5% |
| s080 | diffusivity | 1-D model with k(T) x 0.80 | 8.3% / 6.3% |
| s060 | diffusivity | k(T) x 0.60 | 18.2% / 13.7% |
| s040 | diffusivity | k(T) x 0.40 | 29.8% / 22.8% |
| s025 | diffusivity | k(T) x 0.25 | 39.5% / 31.2% |
| b005 | blend | (1 - 0.05) x nominal + 0.05 x T0 | 2.4% / 3.3% |
| b010 | blend | alpha = 0.10 | 4.8% / 6.6% |
| b020 | blend | alpha = 0.20 | 9.8% / 13.3% |
| b035 | blend | alpha = 0.35 | 17.2% / 23.2% |
| b050 | blend | alpha = 0.50 | 24.6% / 33.2% |
| (C2) | initial value | T0 only (registered arm C2, alpha = 1 limit) | 49% / 66% |

T0 is the start temperature of each file, which equals the first inlet value and the first value of the nominal physics series (all files start isothermal), so the blend uses no label. The diffusivity family is the physically meaningful one: it is a physics model with wrong material properties. The blend family connects C11 (alpha = 0) with C2 (alpha = 1) along one path. The T_inner inlet anchor is unchanged in every level. The shares in the table were computed on all 284 training files by `build_threshold_caches.py` (written to `level_stats.csv`) on 2026-10-21, before any training of this study.

Seeds: the registered S20 set (1, 2, 3, 5, 7, 11, 13, 21, 42, 123, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53) for every level, run S10 first and the remaining ten second. 9 levels x 20 = 180 runs. Baselines reused from Round 10 v4 without retraining: B2_mask (direct, C1), C11 (nominal), C2 (initial value), all S20.

## Readouts (validation only, 20 files; T_avg on the 17 line-average files)

1. Per level and channel: seed-paired difference level minus C1 (negative = the reference helps), 95% CI from 10,000 seed bootstraps; also ToTa = (T_outer + T_avg) / 2. Bands for wording: T_inner 0.02, T_outer 0.05, T_avg 0.08 C.
2. Gain curve: difference against the training residual share (log axis), both families on one plot, C11 and C2 as end points.
3. Boundary rho*: per family and channel, the residual share at which the mean difference crosses 0 (linear interpolation on log share between adjacent levels); also the first level whose CI upper bound is at or above 0.
4. Collapse test: at matched residual share, the two families' differences differ by less than the band (T_outer 0.05, T_avg 0.08 C). Interpolated on log share inside the common range.
5. Literal-rule check: each level's own validation MAE against the direct model's (0.340 / 0.330).

## Predictions written before running

- P1: the gain shrinks monotonically with the residual share in both families.
- P2: rho* lies between 5% and 30% for T_outer and T_avg [guess, not derived].
- P3: the two families collapse within the band; if not, the error type matters and the paper may not state the boundary in residual share alone.
- P4: every degraded reference is less accurate on its own than the direct model, so any level that still helps is further evidence against the literal rule.

## What will and will not be claimed

Whatever the outcome, every level is reported. No result here changes the Round 10 recipe, which stays C12. Statements are about this geometry, data and GRU at h128; the diffusivity family speaks to "a physics reference with wrong properties", not to a new material's FE data.
