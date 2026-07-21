# TES GRU Surrogate — Experiment Log

Lab notebook for the GRU input-feature ablation pipeline
(`GRU_input_ablation.py` → `tes_gru/` package). Newest entries on top.

**Conventions**
- One dated entry per change or run. Record *what* changed, *why*, the config
  delta, the expected effect, and (later) the observed result.
- Status tags: `[CODE]` landed in source, not yet run · `[RUNNING]` ·
  `[DONE]` results in hand · `[SUPERSEDED]`.
- Each sweep writes to `runs/<RUN_NAME_BASE>_seed<N>/`. Keep the `RUN_NAME_BASE`
  here in sync with `config.py` so the log maps 1:1 to output folders.

| `RUN_NAME_BASE` | pad mode | variants × seeds | status |
|---|---|---|---|
| `2026-05-25_8var_1200ep_P0_seqsliding` | `zero` | 8 × 4 = 32 runs | `[DONE]` ~116 GPU-h (rejected baseline) |
| `2026-06-28_abs_sliding_1200ep_P0_init` | `init` | `abs_sliding` × 4 = 4 | `[DONE]` see 2026-07-16 results |
| `2026-07-16_abs_sliding_800ep_ES150_P0_variable` | `variable` | `abs_sliding` × 4 = 4 | `[DONE]` **t=0 winner** — see results |
| `2026-07-20_abs_sliding_W{5,10,20,50}_1000ep_ES150_P0_variable` | `variable` | 4 W × **2 seeds {7,42}** = 8 (screening) | `[CODE]` pending — `run_window_sweep_2gpu.py` (2×4090) / `run_window_sweep.py` (1 GPU) |

The open question (report Future Work) is which `t=0` fix wins on the **forward
sliding variant** (`abs_sliding`): **`init`** — pad the pre-history with the `t=0`
steady-state value (Yiming's suggestion at the final review) — vs **`variable`** —
a variable-length window feeding only the real steps (the presenter's proposal;
the report's "cleanest fix"). The old **`zero`** padding was rejected at that
review (it caused the ~9 °C `t=0` undershoot); its data already exists
(`2026-05-25`), so it is the fixed baseline, **not re-run**. Both new sweeps use
the full seed set `{7, 21, 42, 123}`; every non-sliding variant is untouched, so
its `2026-05-25` numbers still stand.

Shared protocol: stacked GRU `hidden=128, layers=5, dropout=0.3`, `lr=0.0025`,
`batch=16`, best-val checkpoint; P0 stability provisions (grad-clip `1.0`,
orthogonal recurrent init, `60`-epoch LR warmup); learned `InitStateEncoder`
(h0); teacher forcing `0.5 → 0` over `100` epochs; `WINDOW_SIZE=10`; seeds
`{7, 21, 42, 123}`; fixed 70-case test set (`MANUAL_SPLIT_ENABLED=False`).
Epoch budget: runs through 2026-06-28 used `1200` epochs / no early stop;
**since 2026-07-16 (Change C): `800` epochs + early stop (patience `150`)** —
best-val checkpointing is unchanged, so results remain comparable.

---

## 2026-07-20

### Change D — Window-size sweep (W ∈ {5, 10, 20, 40}) + epoch cap 800 → 1000 `[CODE]`

**What.** The window-size sweep Yiming called the primary lever at the final
review, run under the production `variable` padding.
- `WINDOW_SIZE` is now env-overridable (`WINDOW_SIZE=<W>`), **one W per
  process** — W is baked into default args and star-import snapshots at import
  time, so it must never be mutated inside a running process (same trap class
  as `RUN_NAME`).
- New driver **`run_window_sweep.py`**: one command runs all four W values
  sequentially in fresh subprocesses (`--dry-run` prints the plan). Fully
  resume-friendly — re-running skips finished (W, seed) pairs and resumes
  interrupted training, so Ctrl+C is safe.
- `max_epochs: 800 → 1000` — the variable run showed slower convergence
  (seed7 best at 748 hit the 800 wall); 1000 restores headroom, still ~2×
  cheaper than the old 1200/no-ES protocol.
- `RUN_NAME_BASE` now folds W in:
  `2026-07-20_abs_sliding_W<W>_1000ep_ES150_P0_variable`.

**W=10 note.** The W=10 leg re-runs the variable configuration under the
1000-ep cap — it doubles as the clean W=10 reference AND retires the seed7
truncation caveat from 2026-07-16.

**Cost estimate.** Rollout cost scales ~linearly with W (the GRU advances W
steps per rollout iteration). Per-seed at the 1000-ep cap: W5 ≈ 3 h, W10 ≈ 6 h,
W20 ≈ 11 h, W50 ≈ 28 h. The 2-seed screening pass (Change D.1) is 8 runs ≈
96 GPU-hours; on 2×4090 with (W,seed)-level load balancing that is ~48 h
wall-clock (~2 days). A full 4-seed sweep would be ~190 GPU-hours.

**Verification.** All modules + driver compile; per-W config resolution
checked for {5,10,20,50} (correct run names, ep=1000, pad=variable);
driver `--dry-run` prints the right plan; functional rollout test at W=5 and
W=20 confirms variable windows grow `1,…,W` then plateau at `W` with correct
output shapes (no residual hard-coded 10 anywhere on the path).

### Change D.1 — 2-seed screening design + dual-GPU driver `[CODE]`

Sweep restructured as **screening → confirmation** to cut cost ~40%:
- **Screening (this round):** all 4 W × seeds **{7, 42}** — the SAME pair for
  every W (paired design: per-seed differences cancel seed effects). `SEEDS`
  is now env-overridable (comma-separated) in `config.py`; both drivers
  default it to `7,42`.
- **Confirmation (later):** top-2 W get the remaining seeds {21, 123} by
  re-running a driver with `SEEDS="21,123"` — same run dirs base, new
  `_seed21/_seed123` folders land alongside.
- **New `run_window_sweep_2gpu.py`** for the 2×4090 machine: one process per
  GPU via `CUDA_VISIBLE_DEVICES`, shared work queue over **(W, seed) jobs**
  (not whole-W), scheduled **longest-W-first** (per-seed cost ≈ 3/6/11/28 h
  for W=5/10/20/50). (W,seed) granularity matters: a whole-W W=50 job is an
  indivisible ~56 h process that would pin one card while the other idles;
  splitting by seed balances the two lanes to ~48 h each. Per-(W,seed)
  console output → `sweep_logs/W<W>_seed<seed>.log`. Resume-friendly; Ctrl+C
  terminates children and is safe to relaunch.
- Screening wall-clock on 2×4090: **~48 h (~2 days)**. Single-GPU
  (`run_window_sweep.py`, also 2-seed now): ~96 h.

### Next steps
1. Run the screening sweep on the 2×4090 box:
   `python run_window_sweep_2gpu.py` (dry-run flag available). `[ ]`
2. Analyze W-sensitivity on the paired 2 seeds: overall/early/late MAE, t=0
   error, best-epoch drift, wall-clock vs W; shortlist top-2 W. `[ ]`
3. Confirmation pass: `SEEDS="21,123" python run_window_sweep_2gpu.py` with
   `WINDOW_SIZES` trimmed to the shortlist. `[ ]`
4. After W is settled: joint hyperparameter search (lr × dropout × layers ×
   W neighborhood) with Optuna + pruning — lr=0.0025 and dropout=0.3 were
   tuned under the pre-P0 recipe and deserve a re-check under the final one;
   at minimum do the cheap local lr re-check {0.001, 0.0025, 0.004}. `[ ]`

---

## 2026-07-16

### Results — `init` sweep `[DONE]` (4/4 seeds complete, fixed 70-case test set)

| seed | overall MAE (zero → init) | EarlyMAE T_inner (zero → init) |
|---|---|---|
| 7 | 2.36 → 2.00 | 2.32 → 1.64 |
| 21 | 1.75 → 1.68 | 1.58 → 2.04 |
| 42 | 1.80 → 1.79 | 2.01 → **5.06** |
| 123 | 1.82 → 2.18 | 2.25 → 2.16 |
| **mean** | **1.93 ± 0.29 → 1.91 ± 0.22** | **2.04 → 2.73** |

Split verified identical to the baseline (`manual_split=false`, `70_cases`).
**Verdict: inconclusive.** Overall MAE is a wash; the early-window error did
NOT reliably improve — it is seed-noisy, with a seed-42 early-window blow-up
(5.06 °C; needs a per-case look) masking a clear seed-7 improvement. The
`init` pad alone does not settle the `t=0` question; the `variable` sweep is
now the deciding run. Best epochs: 327 / 555 / 631 / 386.

### Results — `variable` sweep `[DONE]` → **`variable` wins the t=0 question**

4/4 seeds complete, fixed 70-case set, seed field now recorded correctly.
Three-way comparison (4-seed mean ± sd, °C):

| metric | `zero` | `init` (Yiming) | `variable` (ours) |
|---|---|---|---|
| **t=0 first-step error** (3 predicted ch) | 7.33 ± 3.83 | 7.00 ± 4.88 | **2.66 ± 0.70** |
| t=0 worst-case (mean over seeds) | 43.8 | 20.9 | **12.7** |
| EarlyMAE T_inner (first 10%) | 2.04 ± 0.34 | 2.73 ± 1.57 | **1.88 ± 0.16** |
| EarlyMAE T_avg | 2.95 ± 0.87 | 2.09 ± 0.44 | **1.94 ± 0.46** |
| Overall MAE | **1.93 ± 0.28** | **1.91 ± 0.22** | 2.09 ± 0.22 |
| R² overall | 0.995 | 0.996 | 0.995 |

**Verdict.** `variable` is the only mode that reliably removes the visible
`t=0` jump: first-step error drops ~2.8× vs `zero` with by far the tightest
seed spread (±0.70 vs ±3.8/±4.9), and it does so on **every** seed. `init`
does NOT fix it — on 2/4 seeds its t=0 error is as bad as or worse than
`zero` (seed123: 13.1 °C). Mechanistically this matches the report's
prediction: repeating the t=0 row still pushes 9 fabricated steps through the
recurrence and drifts the InitStateEncoder's h0, whereas the variable-length
window feeds h0 straight into the first real step. The cost is a small,
seed-noise-level overall-MAE increase (2.09 vs 1.93, overlapping ±sd; driven
by seed21 and T_outer). **Recommendation: adopt `variable` as the production
pad mode** (already the default).

**Early-stop behavior.** ES fired on 3/4 seeds (stopped at 601/602/717 ep);
seed7 hit the 800 cap with best at **748** — the variable mode converges
slower than the 1200-ep evidence base predicted (max observed best was 631
before this run). Wall-clock ~4.5–5.9 h/seed, roughly **half** of the 1200-ep
runs. Caveat: seed7's best-val may be slightly truncated by the cap; if it is
ever re-run, use a 1000-ep cap.

### Change C — Epoch budget 1200 → 800 + early stopping (patience 150); split pinned `[CODE]`

**Evidence.** Across all 36 completed full-length runs (zero baseline 8×4 +
init 4), the best-val epoch never exceeded **631** (abs_sliding median ~460;
most other variants < 500). The second half of every 1200-epoch run has never
produced a best checkpoint — pure waste (~40–50% of each run's GPU time).
At the final review Yiming also noted the late-phase val curve is flat and
safe to cut. Best-val checkpointing is unchanged, so results stay comparable
with all earlier runs.

**Config delta (all in `tes_gru/config.py` — the only file that changed):**
- `max_epochs: 1200 → 800` (headroom ≥ 27% over the worst observed best epoch)
- `early_stop_patience: 10⁹ (off) → 150` (= 3× the LR-scheduler patience, so
  training survives two LR halvings before stopping)
- `TRAIN_SUBDIR/TEST_SUBDIR/MANUAL_SPLIT_ENABLED` pinned to the baseline
  protocol (`Latest Database` / `70_cases` / `False`) — previously the repo
  copy still carried `test_in_10s`/`True` and would have silently reverted
  the runner's local fix.
- `RUN_NAME_BASE` now folds epochs + ES into the name:
  `2026-07-16_abs_sliding_800ep_ES150_P0_{pad_mode}`.
- **Default pad mode flipped `init` → `variable`** — the variable sweep is this
  round's deciding run; a plain `python GRU_input_ablation.py` now runs it
  (last round the variable pass was skipped, likely because the env var was
  never set). `init`/`zero` remain reachable via the env var.
- **Provenance fix:** `run_config.json`'s `seed` field used to record the
  frozen legacy constant `SEED = SEEDS[0]` (=7) in *every* seed's folder
  (pre-existing monolith bug — the 2026-05-21 baseline snapshots show it too).
  The main loop now rebinds `config.SEED = current_seed` and the snapshot
  reads the live value, so each folder records its true seed. Training was
  never affected (`set_seed(current_seed)` was always correct); the directory
  suffix `_seed<N>` was and remains the authoritative label for old runs.

### Next steps *(superseded by the 2026-07-20 entry)*
1. ~~Run the `variable` sweep~~ `[x]` · ~~three-way comparison~~ `[x]` — see
   Results above. `variable` adopted as production default.
2. ~~Window-size sweep~~ → set up as Change D (2026-07-20), 1000-ep cap. `[x]`
3. ~~Re-run seed7 with 1000-ep cap~~ → covered by the sweep's W=10 leg. `[x]`
4. Optional post-mortem: init seed-42 / seed-123 early-window blow-ups
   (moot for mode selection — init lost — but instructive per-case reading). `[ ]`

---

## 2026-06-28

### Change B — Refactor: monolith → `tes_gru/` package `[CODE]`

`GRU_input_ablation.py` (2957 lines) split into a package; the file is now a
thin launcher. **Run command is unchanged: `python GRU_input_ablation.py`.**

| module | role |
|---|---|
| `config.py` | all run constants + setup primitives (`set_seed`, `DEVICE`, schedules, `BASE_DIR`) |
| `utils.py` | `format_duration`, `_to_jsonable` |
| `models.py` | encoders, `ThermalGRU`, loss |
| `data.py` | `ThermalDataset`, loader / manual split |
| `rollout.py` | variant rollouts (incl. the padding logic changed in Change A) |
| `train.py` | `train_model` |
| `evaluate.py` | `test_model`, metrics, per-case plots |
| `runio.py` | run dir layout, checkpoint / resume / meta / history I/O |
| `main.py` | `main()` — sweep seeds × variants, comparison figures |

Two refactor-specific fixes (would otherwise break silently):
- **`BASE_DIR`**: path anchoring moved from `os.path.dirname(__file__)` to a
  `config.BASE_DIR` that points at `src/`, so `runs/` and the relative
  data-root candidates resolve the same as before despite living one dir deeper.
- **`RUN_NAME` live ref**: the only runtime-mutated global. `runio`/`main` use
  `config.RUN_NAME` (live) instead of an `import *` snapshot.

**Verification**
- All modules compile; `import tes_gru.main` resolves all relative imports.
- AST audit: 31/31 module-level globals preserved with byte-identical values
  (30 in `config.py`, `_DEFAULT_WEIGHTS` in `models.py`); 0 removed, 0 `global`
  statements; only `BASE_DIR` added; only `RUN_NAME` is runtime-mutable.
- Numerical equivalence vs the pre-split monolith (identical weights, same
  random inputs): forward `_rollout_sliding`, inverse `_rollout_inverse`,
  `ThermalGRU` forward, `_r2`, `_mape` all **maxdiff = 0.0**.
- Adversarial 4-dimension review (line coverage · name resolution · deliberate
  edits · extended equivalence): all PASS. Extended pass covered all 8 variants'
  feature engineering, `run_rollout_train` (incl. seeded teacher-forcing + dropout
  RNG order), gradients, and the schedules — 143 comparisons, all maxdiff 0.
  Three names resolving only via transitive `import *` (`time`/`np`/`random`) were
  hardened to explicit imports.

No behavioral change intended; this entry exists for reproducibility provenance.

### Change A — Sliding-window `t=0` padding fix: `zero` → {`init`, `variable`} `[CODE]`

**What.** `SLIDING_PAD_MODE` now selects among three ways to handle the missing
`W-1-t` window positions at rollout step `t < W-1`:
- **`variable`** — no padding: feed only the `t+1` real steps (a variable-length
  window). At `t=0` it degenerates to a single real step, identical to `abs`, so
  the `InitStateEncoder`'s clean `h0` is used uncorrupted; the window grows to `W`
  as `t` advances. *(The presenter's proposal; the report's "cleanest fix".)*
- **`init`** *(default)* — repeat the `t=0` steady-state row. *(Yiming's fix.)*
- **`zero`** — legacy zero-pad. *(Rejected baseline.)*

Applied at all four touchpoints (`_rollout_sliding`, `_rollout_inverse`, and the
two single-case mirrors in `test_model`). The mode is **overridable via the
`SLIDING_PAD_MODE` env var**, and `RUN_NAME_BASE` folds it in, so one launch
script can sweep both fixes without editing `config.py`.

**Why.** Inputs are MinMax-scaled to `[0,1]`, so a zero pad equals the *coldest*
temperature (~100 °C). Zero-padding injected a spurious "cold-system" signal at
the start of every rollout, dragging the clean `h0` cold and producing the ~9 °C
`t=0` undershoot seen across most cases in `2026-05-25`. At the final project
review **Yiming** rejected zero-padding and proposed padding with the initial
steady-state value (`init`); the presenter had proposed a variable-length window
(`variable`). Both are in the report Future Work; this round measures **both**
against the existing `zero` baseline. (Arnold was absent from that review.)

**Config delta.**
- `SLIDING_PAD_MODE`: three modes; env-overridable; default `init`.
- `RUN_NAME_BASE = f"2026-06-28_abs_sliding_1200ep_P0_{SLIDING_PAD_MODE}"` — folds
  the mode into the run dir (`..._init` / `..._variable`); no stale-cache reuse.
- `VARIANTS = ['abs_sliding']`; seeds `[7, 21, 42, 123]`.
- mode recorded in `run_config.json` / aggregate meta / startup banner.

**How to run both** (one process per mode; PowerShell):
```
$env:SLIDING_PAD_MODE="init";     python GRU_input_ablation.py   # -> ..._init_seedN
$env:SLIDING_PAD_MODE="variable"; python GRU_input_ablation.py   # -> ..._variable_seedN
```
Each invocation reads the env var once at import. `zero` reproduces `2026-05-25`.

**Expected effect (to confirm).** Both `init` and `variable` should largely remove
the `t=0` undershoot; `variable` is the cleaner fix (h0 untouched at the start).
Marginal overall-MAE change; variant ranking unchanged. Baseline —
`zero`-pad `abs_sliding`: overall MAE `1.93 ± 0.25 °C`, `R² = 0.995`, 4/4 seeds.

**Verification (code-level).**
- `init` mode is byte-identical to the pre-split monolith (maxdiff 0) — the new
  `variable` branch is inert there.
- `variable`: window lengths grow `1,2,…,W` then plateau at `W` (`t=0` = single
  step); train-time (batch) and test-time (single-case) rollouts agree exactly
  (maxdiff 0). Bad `SLIDING_PAD_MODE` fails fast (assert).

### Next steps *(superseded by the 2026-07-16 entry)*
1. Run both sweeps — `init` and `variable`, `abs_sliding` × 4 seeds each. `[x init / see above for variable]`
2. Compare `init` vs `variable` (and both vs the existing `zero` baseline): `t=0`
   undershoot, early/late-window MAE, overall MAE, per-seed spread. Pick the
   winner; update this entry to `[DONE]` with numbers. `[ ]`
3. Window-size sweep `W ∈ {5, 10, 20, 40}` under the hardened protocol — window
   size is the expected primary lever for further gains (raised by Yiming at the
   review). `[ ]`
