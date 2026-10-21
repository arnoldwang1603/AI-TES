# Round 10 post-hoc studies (2026-10-21)

Two studies outside the frozen Round 10 v4 protocol. Both reuse the registered arm definitions from `round10_pipeline.py`, change exactly one setting, score the validation set only (SEAL_TEST=1) and never read `Test_70_cases`. Neither can change the frozen recipe (C12).

| Study | What changes | Arms x seeds | Runs | Pre-registration |
|---|---|---|---|---|
| Unmasked supplement | `TAVG_MASK=""` for C12, R3, C11 | C12 x S20, R3 x S10, C11 x S10 | 40 | this README |
| Reference threshold | the physics cache behind C11 is degraded in 9 controlled levels | 9 x S20 | 180 | `PREREG_reference_threshold.md` |

Baselines are reused from the Round 10 v4 runs (`r10v4_` run directories): B2_mask (direct, C1), C11, C2. The analysis needs those directories reachable (`--baseline-root`).

## Server commands (run from `src/models/GRU`)

Python < 3.12 is fine; all files here parse under the Python 3.8 grammar.

```bash
# 0. sanity: same package as Round 10 v4 (expect a4656fb or a descendant that only adds round10_posthoc/)
git log -1 --format=%h

# 1. reference threshold: build the nine degraded caches (CPU, about 2-3 minutes; writes runs/_threshold_caches/)
python round10_posthoc/build_threshold_caches.py
cat runs/_threshold_caches/level_stats.csv

# 2. dry run: prints the nine run names, the single env difference from C11, and the planned run count (180)
python round10_posthoc/run_threshold.py --dry-run

# 3. train (pass 1 = S10 seeds of every level, pass 2 = the other ten; resume-safe, rerun the same command after an interruption)
nohup python round10_posthoc/run_threshold.py --workers 3 > threshold_run.log 2>&1 &

# 4. analysis (point --baseline-root at the directory that holds the r10v4_ runs of B2_mask, C11 and C2)
python round10_posthoc/analyze_threshold.py --runs-root runs --baseline-root runs
```

Outputs of step 4 go to `runs/_threshold_results/`: `threshold_table.md/.csv` (per level and channel: seed-paired difference to direct with 95% CI, training residual share, the degraded reference's own validation MAE), `threshold_boundary.csv` (where the gain reaches 0 per family and channel), `threshold_collapse.csv` (do the two families agree at the same residual share), `threshold_curve.png`.

Unmasked supplement (already running on the owner's PC; the same script works on the server):

```bash
python round10_posthoc/run_nomask_supplement.py --dry-run
python round10_posthoc/run_nomask_supplement.py --workers 3
```

## What to hand back

`runs/_threshold_caches/level_stats.csv`, `runs/_threshold_results/`, and the run directories `runs/r10th_*` (and `runs/r10v4_*_vs3_seed*` without `_mk` for the unmasked supplement), plus `threshold_run.log`.
