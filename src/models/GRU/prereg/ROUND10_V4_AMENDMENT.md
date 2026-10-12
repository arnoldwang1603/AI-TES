# Round 10 v4 amendment — 2026-10-12

## Reason and timing

The owner requested one complete execution of the agreed scientific comparisons, without a ten-seed screening round or a later human decision to top up selected arms. This amendment also corrects the F1 leave-one-family-out fitting and early-stopping scope. It does not assert that a residual output parameterisation is novel or that any candidate has won.

The owner reported stopping v3 shortly after S1/B1_mask began; the posted progress showed two completed seeds and no B6/F11 stage. That is a user-reported status, not a substitute for the server-side audit. The `--restart-v4` audit must refuse an existing real test-access reservation or record, including an interrupted access. No old result is automatically reused.

`round10_v3.json` remains the frozen historical protocol. The active protocol is `round10_v4.json`; this is a disclosed amendment after v3 training began, not a claim that v3 never launched. The old data, run directories, reports and access records remain intact.

## Execution and seed allocation

| Block | v4 allocation | Training runs |
|---|---|---:|
| P1/S1 | Eight scientific arms × 20; RETRAIN machine bridge × 10 | 170 |
| P2/S2 | All eight arms × 20, including C14 and C15 | 160 |
| P3/S3 | R3, R1, R2, R2C, C3, C2 × 20 | 120 |
| P4/S4 | All nine arms × 20; G7c/G7d run unconditionally | 180 |
| P5/S5 | C11, C12, C13 × 20 | 60 |
| C17/S6 | Up to six eligible confirmations × 20 at h160 | 0–120 |
| F4/S7 | Two new splits; direct and, if selected, challenger; 10 per split/arm | 20–40 |
| F1/S7b | All ten families; direct and, if different, frozen final recipe; 10 per fold/arm, h128 | 100–200 |
| F9/F10/S8 | Frozen final recipe and, if different, direct; same 20 fresh seeds | 20–40 |
| **Total** | **34 base scientific arms × 20 plus the bridge and follow-ups** | **830–1090** |

The seed labels are specified in `round10_v4.json`; every comparison uses the shared registered seed labels. F9/F10 use the registered fresh set. The upper bound is one executable branch, not the union of every possible named candidate.

F4 reuses the full S20 main-experiment comparison for split seed 42. New split seeds 43 and 44 each use S10. Within each split, direct and challenger use the same training seeds and receive a separate paired verdict; results from the three splits are not pooled into one statistical sample. Reuse of split 42 adds no training runs to the table above.

Every base scientific comparison is completed even if it does not qualify for adoption. Model selection remains automatic under registered validation-side rules. C17 is a capacity confirmation of eligible candidates, not a replacement for the full base comparison table. Negative or inconclusive h128 comparisons remain reportable as h128 findings; they do not establish a result across every capacity.

## F1 scope correction and ordering

S7 runs F4. S7b applies the F4 decision and freezes the final recipe before F1 begins; S8 reads that frozen recipe. F1 compares direct with that recipe at the same h128 capacity, with all ten families and ten paired seeds per family. If the recipe is direct, duplicate training is omitted. F1 is diagnostic and cannot reopen recipe selection.

Each `loto_v4_foldN.json` separates fold training, internal validation and the held-out family. Within each F1 fold, held-out-family labels are excluded from network fitting, fitted statistics, early stopping and scheduler decisions. The recipe is frozen by the preceding main-validation/F4 protocol before F1 begins; F1 results cannot change it. Earlier main-validation/F4 data include these families, so this is post-selection family-holdout evaluation of a fixed recipe, not nested family-unseen selection of the complete research pipeline. `eval_checkpoint.py --set holdout` scores it from the training-source collection after training; this operation does not read `Test_70_cases`.

F1 fits its scaler and data-dependent model constants using only the training partition of that fold. Full-284 constants are no longer injected. This includes applicable residual/ratio normalisation and anchor constants, with the training T_avg mask respected. C11/C12 use the fixed FE-report nominal physical reference (r_o=0.127 m, diffusivity scale s=1.0), driven only by inlet history; material properties and geometry are not recalibrated within F1. Residual normalization (mu, sigma, z0) is fitted separately on each fold train. The historical full-284 E1 grid search selected the same nominal values and remains disclosed; the globally calibrated branch is not used as a fold-fitted reference. A fold with no scored T_avg retains missing T_avg metrics; the absence is never converted to zero.

Both changes matter: the v3 full-train constants exposed held-out-family statistics, and its held-out family also served as the early-stopping validation set. v4 measures the held-out family after fitting and stopping have used only the allowed partitions.

## Preserved scientific scope and reporting

The four agreed items remain: validity and controls, startup behaviour, ratio/physical/output references, and encoder size. Registered diagnostics, linear/physical baselines and generalisation analyses remain part of the package.

The name `direct` denotes the existing baseline, which retains the T_inner inlet anchor. C14 is the all-absolute output control. C3 adds the inlet anchor to T_outer while T_avg remains absolute; its label must not imply that all three channels use an inlet residual. The full table retains per-channel outputs, uncertainty, failures, startup/jump metrics and the material-refit column.

The temperature-input dataset, seven exclusions, masks and base split remain unchanged. Model selection remains on validation data. B6 and F11 remain the two registered test-report access windows; test results cannot change the recipe. Statistical thresholds and decision definitions remain those explicitly carried into v4, with the updated comparison seed populations. Twenty seeds estimate training variability; they do not create twenty new independent physical systems or establish material transfer.

## Isolation, migration and verification

- Formal run prefix: `r10v4_`; smoke prefix: `r10v4smoke_`.
- Results and journal: `runs/_round10_v4_results/`; smoke records: `runs/_round10_v4_smoke/`.
- `--restart-v4` writes the idempotent audit `runs/_round10_v4_results/migration_v3_audit.json`, refuses existing real test access, and does not train, score, delete, rename or import old runs.
- The old `r10_` directories and `runs/_round10_results/` remain preserved. v4 resume accepts only its own matching package, dataset, protocol and run identities.
- The owner performs Git commits/pushes; the teammate updates the complete package and follows `ROUND10_RUNBOOK.md`.

**Validation status:** local complete preflight (30 check scripts plus the independent P0 regression) and `--all --dry-run` passed. The executable branch is 830–1090 runs. Per the owner's later instruction, full-arm smoke is deferred to the lab server. The automatic local smoke was interrupted after its first two-epoch training run; no full-smoke PASS is claimed. See `ROUND10_V4_VALIDATION_2026-10-12.md`. The v3 69/69 smoke does not certify v4.
