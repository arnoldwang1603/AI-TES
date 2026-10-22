# Round 10 post-hoc studies (2026-10-21)

Two studies outside the frozen Round 10 v4 protocol. Both reuse the registered arm definitions from `round10_pipeline.py` and change exactly one setting. Training runs sealed (SEAL_TEST=1, validation plots only); the 70 test files are then scored afterwards with `plot_test_posthoc.py`, and since 2026-10-22 the test set is the primary readout for every conclusion (owner decision, Addendum 3 of the pre-registration); validation is secondary. Neither study can change the frozen recipe (C12).

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

# 4. analysis (point --baseline-root at the directory that holds the r10v4_ runs of B2_mask, C11 and C2).
#    Default is the test set, so run the test readout section below first; --set val gives the validation tables.
python round10_posthoc/analyze_threshold.py --runs-root runs --baseline-root runs
python round10_posthoc/analyze_threshold.py --set val --runs-root runs --baseline-root runs
```

Outputs of step 4 go to `runs/_threshold_results/`: `threshold_table.md/.csv` (per level and channel: seed-paired difference to direct with 95% CI, training residual share, the degraded reference's own validation MAE), `threshold_boundary.csv` (where the gain reaches 0 per family and channel), `threshold_collapse.csv` (do the two families agree at the same residual share), `threshold_curve.png`.

Unmasked supplement (already running on the owner's PC; the same script works on the server):

```bash
python round10_posthoc/run_nomask_supplement.py --dry-run
python round10_posthoc/run_nomask_supplement.py --workers 3
```

## What to hand back

`runs/_threshold_caches/level_stats.csv`, `runs/_threshold_results/`, and the run directories `runs/r10th_*` (and `runs/r10v4_*_vs3_seed*` without `_mk` for the unmasked supplement), plus `threshold_run.log`.

## Test-set readout of the threshold study (primary readout, Addendums 2 and 3 in the pre-registration)

Run after the threshold training is complete. The analysis of step 4 above then defaults to the test set; run it with `--set val` for the secondary validation tables. It scores every threshold checkpoint against its own degraded reference and the three baselines on all 70 test files, writes 70 test plots per run into each run's `variants/forward_direct/plots/`, plus `summary_errors_test_posthoc.csv`, `predictions_test_posthoc.npz`, `test_posthoc_aggregate.json` and `POSTHOC_TEST_NOTE.md`; every read is logged in `round10_posthoc/posthoc_test_access.log`. Files are only added, nothing is overwritten. Resume-safe: rerun the same command after an interruption.

```bash
nohup python round10_posthoc/plot_test_posthoc.py --reason "threshold study test readout (Addendum 2)" --runs-root runs --glob "r10th_*_seed*" --workers 3 > test_threshold.log 2>&1 &
```

```bash
python round10_posthoc/plot_test_posthoc.py --reason "threshold baselines test readout (Addendum 2)" --runs-root runs --glob "r10v4_forward_direct_Tin-anchor_l1_w1-6-3_oth-abs_h128x1dp0.0_ep2000_vs3_mk_seed*" --workers 3
python round10_posthoc/plot_test_posthoc.py --reason "threshold baselines test readout (Addendum 2)" --runs-root runs --glob "r10v4_forward_direct_Tin-anchor_l1_w1-6-3_oth-abs_h128x1dp0.0_phr_ep2000_vs3_mk_seed*" --workers 3
python round10_posthoc/plot_test_posthoc.py --reason "threshold baselines test readout (Addendum 2)" --runs-root runs --glob "r10v4_forward_direct_Tin-anchor_l1_w1-6-3_oth-anchor_initial_h128x1dp0.0_init_ep2000_vs3_mk_seed*" --workers 3
```

```bash
python round10_posthoc/analyze_threshold.py --set test --runs-root runs --baseline-root runs
```

The test outputs carry a `_test` suffix next to the validation ones in `runs/_threshold_results/`.

## CPU threads per run (2026-10-22)

Every training or scoring process used to open PyTorch thread pools as wide as the machine; on the 48-thread server three training runs kept about 45 cores busy while the GPU sat at 37%. The pool threads mostly spin: an epoch is paced by the GRU's sequential GPU steps. `round10_pipeline.py`, `run_threshold.py`, `run_nomask_supplement.py` and `plot_test_posthoc.py` now take `--threads` (default 1, `0` = old behaviour) and pass it to every child process.

Local benchmark (RTX 4090, 32 threads, C11, 40 epochs; `bench_threads.py`): 3 workers unlimited 8.6 epochs/s in total, 3 x 1 thread 10.4, 6 x 1 12.0, 9 x 1 12.4, 12 x 1 12.6, 3 x 4 10.2. The loss history of the shared seed was identical in every setting. Confirm the worker count on the server when nothing else is training:

```bash
python round10_posthoc/bench_threads.py --grid 3x0,3x1,6x1,8x1,10x1,6x2 --epochs 60
```

It prints epochs per second for each setting and writes `runs/_bench_threads/bench_threads.csv`; then pass the best worker count as `--workers` (or `--parallel` for `round10_pipeline.py`).
