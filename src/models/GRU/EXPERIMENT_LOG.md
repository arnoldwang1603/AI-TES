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
| `2026-07-20_abs_sliding_W{5,10,20,50}_1000ep_ES150_P0_variable` | `variable` | 4 W × 2 seeds = 8 (screening) | **`[POSTPONED]`** — T_inner ablation takes priority (Arnold 2026-07); lab box is single-GPU, use `run_window_sweep.py` when resumed |
| `2026-07-21_abs_sliding_W10_1000ep_ES150_P0_variable_Tin-{anchor,output_only}` | `variable` | 2 modes × 2 seeds {7,42} = 4 | `[DONE]` **anchor wins** — overall 1.39 °C, early T_inner 0.37 °C |

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

## 2026-07-23

### Results — T_inner ablation `[DONE]` → **anchor wins decisively; adopt it**

4/4 runs complete (2 modes × seeds {7,42}), W=10, variable pad, fixed 70-case
set. Control = the 2026-07-16 arfed run on the same seeds.

| metric (mean of seeds 7,42) | arfed (ctrl) | **anchor** | output_only |
|---|---|---|---|
| **Early-window MAE T_inner** | 1.78 | **0.37** | 1.84 |
| MAE T_inner (whole rollout) | 1.08 | **0.38** | 1.58 |
| MaxErr T_inner | 10.73 | **2.22** | 9.01 |
| t=0 first-step err (T_inner) | 4.31 | **1.73** | 3.52 |
| **Overall MAE** | 1.93 | **1.39** | 2.17 |
| MAE T_outer | 2.35 | **1.19** | 1.45 |
| MAE T_avg | **2.35** | 2.62 | 3.48 |
| R² overall | 0.9963 | **0.9965** | 0.9954 |

**Headline.** Early-window T_inner error drops **1.78 → 0.37 °C (4.8×)**,
essentially reaching the 0.34 °C zero-parameter copy floor — i.e. the model
now exploits the T_inner~T_input relationship as well as it possibly can.
The early instability Arnold flagged is resolved. Overall MAE **1.39 °C**
also beats the previous project best (v22, 1.53 °C single-seed), on both
seeds (1.29 / 1.50) and on a larger test set.

**Unexpected bonus.** T_outer improved almost as much (2.35 → 1.19) even
though nothing about it changed — its rollout consumed the polluted T_inner
feedback, so cleaning that channel improved the whole state vector. T_avg is
marginally worse (2.35 → 2.62), the only regression.

**output_only is not competitive** (2.17 overall, early T_inner 1.84 ≈
unchanged). Removing the crutch does not by itself make the model use
T_input; it has to be forced structurally. This is a useful negative result:
v22's smooth starts were not caused by the 4-input formulation alone.

**Predicted failure mode confirmed and quantified.** As anticipated, the
anchor's advantage is conditional on the tracking assumption:
`|T_inner − T_input|` < 1 °C → anchor 0.28 vs arfed 1.18 (4× better);
gap ≥ 2 °C → anchor 3.03 vs arfed 2.88 (slightly *worse*). Those moments are
~1.8 % of all timesteps, so the aggregate is dominated by the win. Per case:
anchor beats the control on **67/70** cases; the 3 losses are all Type-9
(+0.17…+0.41 °C). Biggest wins are exactly the cases that used to have the
worst starts (Case 40: 4.26 → 0.42; Case 36: 2.95 → 0.29; Case 24:
2.52 → 0.26). Median per-case T_inner MAE 0.83 → **0.20**.

**Convection caveat stands.** The gap≥2 °C degradation is the same mechanism
that will bite on convection data (δ ≈ −100 °C at t=0). Do not port these
anchor constants; the normalization must be redesigned there (Arnold's
"temperature gradient drives heat transfer" framing + the gate).

### Plot review — two real problems the aggregate metrics hid `[OPEN]`

Visual inspection of the per-case plots (not the summary numbers) surfaced
two issues. Both are confirmed quantitatively.

**(a) REGRESSION: T_avg acquires a permanent upward offset.** Signed bias by
rollout decile (70 cases, both seeds, °C):

| | 0-10% | 20-30% | 40-50% | 60-70% | 80-90% | 90-100% |
|---|---|---|---|---|---|---|
| arfed T_avg | +0.98 | +1.00 | +0.37 | −0.08 | −0.25 | −0.49 |
| **anchor T_avg** | +0.97 | **+2.20** | **+2.53** | **+2.39** | **+2.60** | **+2.70** |
| arfed T_outer | −0.98 | −1.40 | −2.40 | −2.43 | −2.34 | −2.38 |
| **anchor T_outer** | −1.14 | −0.61 | −0.53 | −0.28 | +0.02 | **+0.11** |

arfed's T_avg bias self-corrects toward zero; anchor's climbs to ~+2.5 within
the first 30 % and **locks in for the rest of the rollout** — this is the
"late drift" visible on cases 1-4, 8-10, 19, 23, 27, 58-70. Note the
trade: anchor drove T_outer's bias from −2.4 to ~0 while T_avg went from ~0
to +2.5. Working hypothesis (unproven): the model carries a systematic
stored-energy bias, and whichever channel is least constrained absorbs it —
pinning T_inner (and thereby improving T_outer) pushed it into T_avg. Net
MAE still improves, but a locked-in offset is worse for a surrogate than
noise of the same magnitude, and T_avg is the primary reported quantity.

**(b) DESIGN FLAW: the anchor lags one step.** It uses `Input_T(t)` to
predict `T_inner(t+1)`, a choice made purely to make the causality argument
airtight. But Input_T is exogenous — the whole curve is a given boundary
condition — so `Input_T(t+1)` is equally legitimate and strictly better.
Oracle comparison of the residual the head must learn:

| residual definition | mean \|δ\| | std | max | \|δ\| at t=0 (mean / max) |
|---|---|---|---|---|
| `T_inner(t+1) − Input_T(t)` (current) | 0.756 | 1.00 | **37.5** | 1.81 / **37.5** |
| `T_inner(t+1) − Input_T(t+1)` (fix) | 0.733 | **0.49** | **8.5** | **0.35** / 8.5 |

The tail shrinks 4.4× and the std halves. This explains the Case-40 start
the plots show: at t=0 all four channels sit at 217.2 °C, then Input_T jumps
to 263.2 and T_inner follows to 254.7 in one step — anchored on the stale
217.2 the model predicts 217.3, a −37 °C first-step error (Case 54 is the
same, −34 °C). Under the fix the anchor would sit at 263.2 and the residual
becomes the physically real 8.5 °C lag. This also shrinks the large-gap
weakness (gap ≥ 2 °C is largely this one-step lag) and makes the whole
scheme more robust for convection data.

**(c) Not a problem: the "odd" initial values on cases 41-45, 48-51.** Those
cases start isothermal — T_inner = T_outer = T_avg = Input_T at t=0 (e.g.
355.0 across the board) — which is how the simulations are initialized, so
all four curves start from one point in the plots. Model first-step errors
there are ±1 °C. No action.

**Verdict revision:** anchor is still the right direction (T_inner early
error 1.78 → 0.37, 67/70 cases improved), but it should NOT go into the
report or become production until (b) is implemented and (a) is understood.

### Change F — three fixes for the T_outer / T_avg weakness `[CODE]`

Diagnosis first. T_avg's error is **not** driven by case complexity
(correlation with number of inlet direction-changes: **+0.01**); it is a
broad systematic level offset — median per-case T_avg MAE 2.61, >3 °C on
15/70 cases, and 70 % of the error is pure bias. Two of the three fixes are
taken from the LSTM line's configuration, which reports much better T_outer /
T_avg with the same data.

1. **`ANCHOR_LEAD=1`** — anchor on `Input_T(t+1)` instead of `Input_T(t)`.
   Input_T is exogenous (whole curve known up front), so this is equally
   causal and strictly better: residual max 37.5 → 8.5 °C, std 1.00 → 0.49.
   Fixes the Case-40 / Case-54 first-step blowups (−37 / −34 °C).
2. **`LOSS_WEIGHTS=1,6,3`** — per-channel loss weights (T_inner, T_outer,
   T_avg), matching the LSTM line's `Ti×1 + To×6 + Ta×3`. Our loss was
   uniform `[1,1,1]`; with T_inner anchored to ~0.38 °C, most of the
   remaining gradient budget was being spent on the channel that is already
   solved.
3. **`PHYSICS_BOUND_WEIGHT=1`** — hinge penalty when predicted T_avg leaves
   `[min(T_inner,T_outer), max(T_inner,T_outer)]`, applied in RAW temperature
   space (per-channel MinMax scaling does not preserve the ordering).
   Verified on our data: the bound holds at **98.76 %** of all timesteps and
   T_avg sits ~0.41 of the way from T_outer to T_inner. The bounds are
   **detached** in the penalty so it can only pull T_avg back inside — never
   widen the bracket by distorting the (now accurate) T_inner.

All three are env-overridable and folded into the run name
(`..._lead{0,1}_w1-6-3_pb1`). New driver `run_fix_ablation.py` runs them
cumulatively (A = lead only, B = +weights, C = +physics bound) so each
effect is attributable; `--only C` runs just the full stack.

**Verification.** Anchor-lead reconstruction matches the `Input_T(t+1)` form
exactly and differs from the old one; per-channel weights produce exactly the
expected loss values; the hinge is zero inside the bracket, linear outside,
disabled when `PHYSICS_BOUND_WEIGHT=0`, its gradient pushes T_avg back inside,
and — after the detach fix — leaves T_inner/T_outer gradients at exactly 0.
Env overrides and run naming confirmed.

### Meeting 2026-07-23 (Arnold) — asks, and what each maps to

1. **"Do you have a theory why some cases still start badly?"** — **SOLVED, with
   a clean mechanism.** The first-step error is entirely the anchor lag: it
   equals the one-step jump in Input_T, because the head was anchored on
   `Input_T(t)` while the truth follows `Input_T(t+1)`.
   **Correlation between first-step error and the Input_T jump: +0.999**, and
   **all 11** cases with >2 °C first-step error have a matching Input_T jump
   (Case 40: err −37.3 vs jump −46.0; Case 54: −34.0 vs −42.0). This also
   answers his puzzle about "nearly identical cases behaving differently" —
   it depends only on whether that case's inlet ramps instantly at t=0, not on
   the case's overall shape. Already fixed by `ANCHOR_LEAD=1` (Change F).
2. **"All your plots are missing t=0"** — confirmed: the plot slice was
   `[1:n+1]`, so the shared starting point was cut and a one-step-lagged
   prediction looked like it began somewhere else. **Fixed**: the t=0 row (the
   given initial condition) is now prepended for plotting only; `inv_actual` /
   `inv_pred` and every metric are untouched.
3. **"Just do the same thing (delta) for outer and average"** — implemented as
   `OTHER_CH_MODE=persistence` and added as **arm D**. It is deliberately
   flagged as a different mechanism from the T_inner anchor (see the config
   comment): T_inner anchors on GT exogenous Input_T so errors reset each step,
   whereas T_outer / T_avg can only anchor on their own previous prediction,
   making the rollout an integrator with no reset. Prediction to be tested: a
   per-step bias of 0.001 °C compounds to 1.4 °C over 1440 steps, and a level
   offset (the current failure mode) has no absolute reference to pull it back.
   Measurement will settle it.
4. **Email summarizing the input features** (old delta variants vs the current
   anchor) — Arnold explicitly asked for this in writing; he is not yet clear
   on what changed. `[ ]`

### Next steps
1. Run `python run_fix_ablation.py` (arms A→B→C→D, ~24 h; `--only C` for the
   recommended stack in ~6 h, `--only D` for Arnold's variant). `[ ]`
2. Judge: does T_avg's +2.5 °C locked-in offset collapse, do the Case-40 / 54
   starts come good (expect the first-step error to fall from ~37 °C to ≤8.5),
   and does D beat or drift versus C? `[ ]`
3. Send Arnold the input-feature summary email. `[ ]`
4. Then resume the window-size sweep under the winning configuration. `[ ]`

---

## 2026-07-21

### Change E — T_inner ablation: `TINNER_MODE` = arfed / **anchor** / output_only `[CODE]`

**Trigger (Arnold, email).** "T_inner essentially follows Input_T (gap ≤ 1 °C)
— there is no way the model cannot predict this accurately. Focus on fixing
the T_inner prediction."

**Diagnosis confirmed on data.** |T_inner − Input_T| gap: mean 0.6 °C in the
early window, 0.75 °C late, identical at t=0. A zero-parameter
"copy Input_T" baseline scores **0.34 °C** early-window MAE — our trained
model scores **1.9–2.7 °C** there (5–8× worse than free). The model leans on
its AR-fed T_inner input (which is GT during teacher forcing, hence learned)
instead of the clean exogenous Input_T; at inference that slot carries its
own h₀-transient error and copies it forward. v22 (4-input) had no such slot
→ smooth starts. The early spike is T_inner-specific in every configuration
(early/late ≈ 2.2–2.6 vs 0.5–0.8 for the other channels), and the pad fix
did not move it — all consistent.

**Two fixes, one flag (`TINNER_MODE`, env-overridable, abs_sliding only):**
- **`anchor`** *(priority arm)* — inputs unchanged; the T_inner head predicts
  a z-scored residual `δ(t+1) = T_inner(t+1) − Input_T(t)` and the rollout
  reconstructs `T_inner = KA·InputT_s + KB + KC·z` (affine constants from the
  train-set δ stats + scaler, computed in `train_model`, attached to the
  model). Hard-wires the tracking relation; the fed-back T_inner is
  GT-anchor-dominated so feedback error resets each step. Same principle as
  the ODE anchor in Sid's LSTM. **Distinct from the failed `delta` INPUT
  features**: that was a time-difference input replacing the absolute anchor
  (integrates errors); this is a cross-channel offset at the OUTPUT anchored
  on GT (resets errors). z-scoring keeps the head O(1) — a raw 0.6 °C δ is
  ~0.0015 MinMax units, untrainable without it (δ std floored at 0.05 °C).
- **`output_only`** — v22-style A/B: T_inner removed from the input (4-d
  `[Time, T_outer, T_avg, Input_T]`), still predicted, never fed back.

Control arm = the 2026-07-16 variable run (Tin-arfed, 800-ep-cap caveat).

**Touchpoints.** `config.py` (flag, dynamic `INPUT_DIMS`, run-name tag),
`data.py` (4-col features under output_only), `train.py` (anchor constants),
`rollout.py` + `evaluate.py` (mode-aware windows, reconstruction, no inner
history under output_only), provenance (`run_config.json` / aggregate /
banner), new driver **`run_tinner_ablation.py`** (single GPU — lab box
confirmed 1×4090; anchor first, then output_only; 2-seed screening {7,42};
resumable; logs → `sweep_logs/Tin-<mode>.log`).

**Verification (all exact unless noted).** `arfed` is bitwise-identical to
the pre-change rollout from git HEAD at tf=0 AND seeded tf=0.5 (RNG
consumption order preserved — existing results unaffected). `anchor`
reconstruction matches the closed form (maxdiff 0) and the raw-space
round-trip `T_inner = Input_T + μ + σz` checks to 1e-13 °C through the
scaler affine. `output_only` windows are 4-d everywhere with correct output
shapes. Train rollout == hand-rolled test mirror for both new modes
(0 / 3e-8). Bad mode value fails fast at import.

**Decision (Arnold, email, 2026-07-22): GO on anchor.** "I think right now
it's safe to just implement the anchor to eliminate the early errors and go
from there." He also endorsed the gate concept as part of the long-term
framework (branch models selected by input-condition scenario: temperature
vs fluid, solid-only vs solid+fluid).

**Physics clarification from Arnold (corrects our mental model).** The
current temperature-input cases are **solid-only conduction — no fluid is
simulated at all.** T_input is the inner surface temperature of the embedded
heat-exchanger pipe; T_inner is the inner surface of the storage media; the
small gap is conduction across the pipe wall (< 0.5 in). Consequence: the
T_inner~T_input tracking in THIS dataset is structural (thin-wall
conduction), not flow-dependent — the pump-stop/standby decoupling scenarios
we worried about cannot occur inside this data family, so the anchor's
assumption is even safer here than we estimated.

**Known upcoming issue — convection data (flagged by Arnold).** In the
convection datasets (training "soon"), some cases start with
T_inner/T_outer/T_avg at ~100 °C while T_input starts at ~200 °C to emulate
hot fluid arriving → **δ ≈ −100 °C at t=0**, decaying over the transient.
That breaks the current narrow-band z-scoring (train-set σ would jump from
~0.8 °C to tens of °C). Arnold still wants a delta formulation there ("the
temperature gradient is always the one that drives heat transfer" — i.e., δ
becomes a physically meaningful driving-force signal, not noise), details to
be discussed when convection training starts. Do NOT reuse the current
anchor constants on convection data without revisiting the normalization.

**W sweep postponed** until this ablation lands (it decides the formulation
the sweep should run under).

### Next steps
1. `python run_tinner_ablation.py` on the lab 4090 (~24 h total: 2 modes ×
   2 seeds × ~6 h). `[ ]`
2. Judge on early-window T_inner MAE + t=0 first-step error vs the arfed
   control and the 0.34 °C copy-baseline floor; report back to Arnold. `[ ]`
3. Resume the W sweep under the winning `TINNER_MODE`. `[ ]`

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
