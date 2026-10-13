# Round 10 v4: A7 scaler compatibility and P4 sweep validation (2026-10-12)

After the case-ID recovery and server preflight succeeded, the pipeline skipped completed S0/S1 and stopped in S1b's `a7_bridge.py`: `AttributeError: 'MinMaxScaler' object has no attribute 'feature_names_in_'`. This failure occurs after B6 and G1 in the fixed stage order. It is a validation-analysis compatibility error, not a training failure.

## Correction

A7 formerly required sklearn's optional feature-name metadata to select temperature scales. The own-split scaler fitting, sidecar restoration and prediction inverse scaling already use the frozen order `(Time, T_outer, T_inner, T_avg, Input)`. A7 now selects output scales in `(T_inner, T_outer, T_avg)` order with indices `(2, 1, 3)`, whether or not optional names exist. Present names must match the frozen order; invalid dimensions, nonfinite/nonpositive scales or conflicting feature counts are rejected.

The correction does not add metadata to a fitted scaler, refit or save new constants, alter predictions, change weights/loss aggregation, relax the existing A7 reproduction tolerance, or modify training/evaluation code. The available traceback does not establish which sklearn version or fitting detail caused the optional field to be absent.

## Downstream P4 correction

The follow-up audit reproduced a silent false pass: the startup gate used Pandas `max()`, which skips NaN. A synthetic sweep containing an invalid observation and otherwise small offsets passed; a missing temperature/channel observation also passed. Such evidence cannot establish the registered requirement that every channel stays within 0.5 degrees C across the complete constant-inlet scan.

The gate now validates each seed's unchanged `50:600:10` grid (56 temperatures, three channels), unique rows, matching seed and finite nonnegative absolute offsets before taking the maximum. This validation applies to freshly generated and cached scans. Invalid or incomplete scans cannot pass; the decision records per-seed reasons and a null maximum instead of presenting a partial maximum as a complete measurement. The original 0.5-degree bound, first-five-step reduction, whole-run condition and legitimate masked validation T_avg values are unchanged. The pipeline still supplies the registered S20 run panel.

## Recovery after completed B6

The previous package seal is archived as `prereg/round10_package_hashes_v4_before_a7_scaler_fix.json`, predecessor SHA256 `0bdad6d0b77239bf2ac22105ee92315a81bb6ac3217ea572ebf12f8ec81acddf`. Relative to that seal, `a7_bridge.py` and the downstream P4 sweep gate in `round10_decisions.py` change; the recovery utility, its checks, the scaler/P4 checks and predecessor archive are added. The preregistration, data, training code, test-report code and completed runs remain unchanged.

The user confirmed that the first A7 patch had not been applied on the server. Recovery version `round10.v4.a7-scaler/2` therefore combines both corrections in one migration from the same stopped 760-input predecessor package. It does not silently migrate a different or progressed package state.

`round10_a7_scaler_hotfix.py` defaults to read-only inspection; `--apply` performs the specifically audited recovery. It requires the stopped S0/S1-complete state, the A7 attribute-error log, unchanged registered S1 run membership, and an already completed B6 with the same 40 checkpoint/config/sidecar fingerprints and saved report hash. It verifies the B6 manifest and access evidence, hashes B6 output files without inspecting test scores, and preserves those files and journals. It backs up the original pipeline state and failed A7 log before changing only the state `package_sha256`. Unexpected package changes, later stages/access, missing or changed artifacts and live processes are refused.

The existing report entrypoint returns from the completed-B6 path before obtaining test-access capability or scoring. Resuming `--all --parallel 3` therefore skips S0/S1, regenerates validation-only S1b readouts, reuses B6 and continues G1/A7. Existing P1 interpretation entries remain in the decision journal; a stage retry can append the same validation interpretation again. The earlier case-ID hotfix covers a pre-B6 failure and is not the recovery command for this incident.

Follow the A7 section of `ROUND10_RUNBOOK.md` after the maintainer commits/pushes the complete correction. Do not regenerate seals or delete stage state/ledgers on the server.

## Validation scope

- The original missing-attribute failure was reproduced with an unnamed scaler.
- Scaler regression: 13/13 PASS using actual sklearn scalers fitted from arrays/DataFrames and synthetic predictions with unequal channel ranges; known weighted MAE, unchanged scaler arrays and invalid-schema rejection are checked.
- Recovery regression: 56/56 PASS, including production B6 manifest serialization and the real completed-report entrypoint with all test-access/scoring functions forbidden. Recovery changes only the package identity and preserves B6 artifacts and access records; repeated application and interrupted recovery are covered.
- P4 scan regression: 36/36 PASS, covering finite and threshold-boundary values, NaN/Inf, missing or duplicate observations, empty grids, wrong seeds and malformed columns on both fresh and cached paths.
- Follow-up interface audit: 330 further checks pass across the pipeline/decisions, actual F4/F1 split and producer contracts, S5b analysis interfaces and synthetic final-report branches. This includes direct, R3, C11 and ENC_linear reports, fixed seed/ensemble counts, material-refit columns and report-cache reuse. Predictions/solver/plot calls are replaced in some interface checks; these are not a complete server execution.
- Real archived validation replay: all 60 packaged Round9 checkpoints pass the existing absolute scalar tolerance `0.001`, with optional feature-name metadata deliberately removed. Maximum absolute difference from each archived validation scalar is `8.5403389e-06`. Only training/validation data and archived checkpoints are used, with sealed-test CSV reads blocked; no model is trained.
- The entire A7 table/export path also passes with these 60 archived checkpoints and ten explicitly reused archived RETRAIN proxies, including all four CSV/Markdown tables. These proxies test interfaces and do not represent the server's new RETRAIN experiments.
- The server's ten new RETRAIN checkpoints are unavailable locally and will be checked by A7 on the server. The 60-checkpoint replay is not a claim that the server's complete A7 stage has already passed.

Recovery regression and installation/dry-run results are recorded with the release evidence. No full-arm training smoke or real Test70 scoring is part of this repair.
