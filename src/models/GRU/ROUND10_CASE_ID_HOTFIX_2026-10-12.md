# Round 10 v4: validation case identity correction (2026-10-12)

The S1 training stage completed on the server, but S1b failed with `Index contains duplicate entries, cannot reshape`. The teammate traced the error to basename-based validation identifiers and reported that all 120 runs in the six S1b comparison arms retained the same 20 unique full validation paths. That server inventory is teammate-provided evidence; the migration audits the server's actual saved state and S1 artifacts before changing it.

## Cause and bounded correction

`tes_gru/evaluate.py` already exports the full relative dataset path in `Case` and a display-only basename in `CaseFile`. The frozen split has 20 unique full paths, including three repeated basenames (`Case 3.csv`, `Case 15.csv`, `Case 17.csv`) in different thermal-profile folders. Three validation readers incorrectly keyed statistical rows by `CaseFile`:

- `round10_pipeline.comparison`: S1b and main family comparisons.
- `round10_decisions.long_for`: later adoption/confirmation and F4 recipe decisions.
- `paper_stats.long_table_from_runs`: standalone statistical replay.

All three now use `Case`. Actual duplicate `(arm, seed, file, channel)` rows remain errors with an explicit diagnostic; no rows are averaged or dropped to suppress the collision. The run-directory loader requires `Case` rather than falling back to an ambiguous basename. Synthetic regression fixtures use real split-relative paths and duplicate basenames.

Training, evaluation exports, checkpoints, splits, seeds, preregistered statistical policy and test-access code are unchanged. This corrects an implementation error in case identification, not a new experimental protocol or a result-driven analysis choice. Existing completed S1 training remains reusable. No real Test70 inputs or formal test scores are needed to implement or verify the correction.

## Frozen-package provenance and recovery

The previous package manifest is preserved byte-for-byte as `prereg/round10_package_hashes_v4_before_case_id_fix.json`. The current `prereg/round10_package_hashes.json` seals the corrected implementation and new checks. The original `round10_v4.json`, execution protocol and training-source manifest are unchanged.

`round10_case_id_hotfix.py --apply` is an explicit, narrowly scoped migration. It audits the approved package delta and the original S0/S1-complete, S1b-duplicate-index-failure state before B6; verifies saved S1 runs; preserves an original state backup and recovery receipt; and changes only the package identity in the state. Ordinary pipeline hash checks remain strict. The migration is safe to repeat and does not train or score Test70. Unexpected package, stage, run or ledger evidence causes refusal.

The first recovery-script release incorrectly compared different path bases: split `val` keys are relative to the training directory, while `load_all_data._case_id` and the exported `Case` are relative to `DATA_ROOT`. The corrected audit constructs the expected identity as the exact frozen `train_subdir + '/' + val_key` (for example, `Latest Database (Use this for training)/Case 1 (linear charging)/Case 16.csv`). Only slash direction is normalized; basenames, arbitrary suffix matches and unknown leading directories are not accepted. A mismatch now reports the run and examples of missing/unexpected paths. This correction changes neither stored summaries nor their statistical keys. The earlier refusal occurs before migration writes, so the same recovery command can be retried after updating the complete patch.

After the maintainer commits and pushes the complete correction, follow the dedicated recovery commands in `ROUND10_RUNBOOK.md`. Do not reseal files on the server, delete state, use `--restart-v4`, or discard completed training to work around this error.

## Validation

The original implementation reproduces the reported exception with the new 120-run synthetic fixture (2/8 checks pass before correction). After correction:

- Case-identity regression: 8/8 PASS, including all three readers, 20/20/17 effective files, S20, known effects and duplicate/missing-identity refusal.
- Existing decision regression with repeated basenames: 13/13 PASS.
- A5 statistics/calibration: 27/27 PASS.
- Pipeline controls: 48/48 PASS; paper-report controls: 51/51 PASS; synthetic F11: 24/24 PASS.
- Hotfix migration with the corrected production-format fixture: 37/37 PASS, including complete 170-run inventory, 120 comparison summaries, exact frozen case paths and directory prefix, refusal boundaries, original-state preservation, interrupted-write recovery and idempotence. Production stage-loop regression skips completed S0/S1 and enters S1b with training explicitly forbidden.

The initial migration's 31 passing checks used unprefixed split keys as exported `Case` values and missed the producer's path-base difference. The revised fixture reproduces the reported refusal against the original migrator, then passes with the correction. It also rejects missing/wrong prefixes, arbitrary leading directories, absolute paths and basename-only identifiers; Windows slash conversion remains supported. The 8/8 statistical-reader checks now also use prefixed production-format `Case` values and the corresponding mask identities.

No full model smoke was run for this analysis-only hotfix; training and model code are unchanged. These are local regression checks, not a claim that the server has resumed. The package hash review and full `--all --dry-run` are recorded in the installation evidence; the registered workload remains 830–1090 runs without dry-run training or test access.
