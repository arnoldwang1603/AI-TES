# TES GRU Surrogate — Experiment Log

Newest entries on top. **New to the project? Read "Start here" below first.**
It explains what the model does and what the words in the entries mean. The
entries themselves are a lab record, written for whoever runs the next round.

---

## Start here

### What we are building

A thermal energy storage tank is charged with hot fluid and later discharged.
Simulating one run properly (CFD / PyAnsys) takes hours. This model is a
**surrogate**: give it the same boundary condition and it reproduces the
tank's temperature curves in about a second. A design study that would cost a
week of simulation then costs minutes.

### The data

One **case** is one complete run of the tank, stored as a CSV time series
(~1440 rows, 10 s apart). Each row holds the time, the **inlet fluid
temperature** (`Input Temperature`), and the three temperatures we predict:

| | what it is | how hard |
|---|---|---|
| `T_inner` | inner wall of the tank | easy — it tracks the inlet temperature closely |
| `T_outer` | outer wall | medium |
| `T_avg` | average temperature of the storage material | hardest, and the one the application cares about |

311 runs are used for training, and a
**fixed set of 70 runs is the test set** — never trained on, unchanged since
the project started, so a number from any round is comparable with any other.
Data lives in `AI-TES/data/` (`Latest Database (Use this for training)` and
`Test_70_cases`).

A few of the 311 training runs are held out as the **validation set**: the
model never trains on them, and they decide when training stops and which
epoch's weights are kept. Rounds 4-8 used the first 15 files in path order,
which (found 2026-09-16) are all linear-charging runs. From round 9 the
validation set is 2 runs from each of the 10 case-type folders (20 runs,
list frozen in `splits/stratified_v2_s42.json`, tag `vs2` in the run name).
Numbers from the two protocols are comparable only within a protocol.

### What the model sees and produces

Input: the whole `[time, inlet temperature]` sequence. The inlet temperature
is a **boundary condition** — at deployment the user specifies it — so the
model may see all of it up front, including the future. Output: the three
temperature curves, one value per timestep.

Nothing the model predicts is fed back into it. This is the **direct**
variant; one seed trains in about three minutes. An older **autoregressive**
variant feeds its own predictions back step by step; it costs 4-6 hours per
seed, scores worse, and survives only as a reference.

### The three ideas that got the error down

Everything in this log is variations on these three.

1. **`T_inner`: predict the offset from the inlet temperature.** The model
   outputs how far `T_inner` sits above the inlet — a few degrees — and the
   inlet carries the large swings for free. `T_inner`'s error has been 0.3 °C
   ever since (2026-07-23).
2. **`T_avg`: predict its position between the two walls.** `T_avg` almost
   always lies between them, so the model outputs
   `pos = (T_avg − T_outer) / (T_inner − T_outer)`, a number near 0.28, and
   `T_avg = T_outer + pos × (T_inner − T_outer)` reconstructs it. `T_avg`
   went 3.0 → 1.9 °C (2026-08-12).
3. **Weight the loss towards `T_outer`.** The three errors are added with
   weights. `T_avg` is computed *from* `T_outer`, so spending the training
   budget on `T_avg` starves what it depends on. Dropping `T_avg`'s weight
   from 3 to 1 took the overall error 0.97 → 0.77 °C (2026-08-26) — the
   biggest win of round 6, out of a weight rather than a new mechanism.

### The one known blind spot

In 20 of the 70 test cases the run both charges *and* discharges. For a few
minutes in those runs the true `T_avg` climbs **above both walls** (the walls
cool first while the interior still holds heat), and the walls are only a few
degrees apart. Idea 2 cannot express that: `pos` would have to reach 10 or 50
instead of 0.28. Those cases — 32-40 and 51-53 are the worst 12 — carry a
`T_avg` error of 2.9 °C against 1.7 °C elsewhere. Rounds 6 and 7 are mostly
about what `T_avg` should do when the position idea stops applying; the five
answers are in "Methods, plainly" below.

### How a result is judged

- **MAE in °C** per temperature; **overall MAE** is the plain mean of the
  three. Lower is better. Current best: 0.72 °C overall for a single model (round 7, `cgate-ah`), 0.68 °C as a 20-seed ensemble.
- **Never one run.** Training is stochastic, so every configuration is trained
  10-20 times from different random starts (**seeds**) and we report the mean
  and the spread. A difference smaller than the spread is not a result.
- **Stability counts as much as the mean.** A seed ending above 1.5 °C is a
  **blow-up**; a good mean with two blow-ups loses to a slightly worse mean
  with none.
- **The deployment recipe is the ensemble**: average the predictions of all
  seeds of a configuration, then score that. It is free accuracy (0.774 →
  0.742 in round 6). `ensemble_eval.py` computes it.

### Vocabulary used in the entries

| word | meaning |
|---|---|
| case | one complete tank run = one CSV = one test-set item |
| seed | one training run from one random start; "20 seeds" = the same configuration trained 20 times |
| arm / config | one setting being compared in a round |
| excursion case | a case where `T_avg` leaves the interval between the walls (the blind spot above); 20 of the 70 |
| blow-up | a seed whose final overall MAE exceeds 1.5 °C |
| channel | one of the model's outputs (3 normally; the gates add a 4th) |
| gate | a rule that decides, per case or per timestep, to stop using the position formula and take `T_avg` from somewhere else |
| direct / AR | one forward pass over the whole sequence / feeding predictions back step by step |
| anchor | predicting an offset from a known reference rather than an absolute temperature (idea 1) |

### Where things are, and how to run

    tes_gru/                the package: config, data, models, rollout, train, evaluate
    GRU_input_ablation.py   thin launcher -- runs ONE configuration (env vars pick it)
    run_roundN.py           a round: many configurations x many seeds, resumable
    export_results.py       package a finished round for the shared drive
    ensemble_eval.py        seed-ensemble scores (the deployment metric)
    sanity_compare.py       two configurations on their common cases, paired by seed
    eval_checkpoint.py      re-score a finished run's best or final checkpoint, no retraining
    splits/                 the frozen validation lists (round 9 on)
    runs/<name>_seed<N>/    one directory per run: metrics, plots, predictions
    ../../../../calit2-thermal-wiki/HANDOVER.md   state of the work for whoever picks it up next (house rules,
                            open questions, plan) -- lives in the wiki with the results and notes

Every knob is an environment variable read in `config.py`, and `run_roundN.py`
pins all of them for every arm, so nothing left over in your shell can leak
into a run. To reproduce a round: `python run_round9.py` (`--dry-run` shows
the plan, `--stage S1` runs one stage). Runs are resumable — rerunning skips
anything with a `done.flag`.

Current defaults: GRU, hidden 128, 2 layers, dropout 0.3, lr 0.0025, batch 16,
up to 1000 epochs with early stopping (patience 150; under the old validation
set most runs stopped between 450 and 800; round 9 raises the cap to 2000
because the new validation set keeps improving past 1000), best-validation
checkpoint.

### Methods, plainly

The entries and the round-7 runner refer to "method 2/3/4". They are five
answers to one question: what should `T_avg` do where the position idea does
not apply?

1. **Position head** (round 5, the current base). `T_avg` is reconstructed
   from its position between the walls. Idea 2 above.
2. **Case gate → plain extra output** (`cgate-ah`). A fixed rule reads the
   inlet-temperature curve, decides "this run both charges and discharges",
   and for those runs takes `T_avg` from a fourth output that predicts the
   temperature directly. Best on the 12 hard cases. Under the round-6 weights
   it blew up in 2 of 20 seeds; under the round-7 weights (idea 3) it is the
   most accurate and most stable configuration we have.
3. **Case gate → anchored extra output** (`cgate-afb`). Same rule, but the
   fourth output gives only the distance from the midpoint of the two walls,
   a few degrees. Never blew up in 20 seeds.
4. **Timestep gate** (`tgate10-soft40`). No case rule at all: at each
   timestep, wide gap between the walls → use the position, gap down to a few
   degrees → fade over to the fourth output. Best overall score.
5. **Regime flag as an input** (round 7, `fic` / `fip`). Methods 2-4 switch
   the model's output from outside and the model itself never learns which
   regime it is in. Here the same rule's verdict goes in as an extra input
   column, so the model can adapt on its own.

One failure worth knowing so nobody repeats it: giving the *same* output
channel two meanings — position in normal runs, temperature in gated ones —
destroys the runs it never touches, because the model cannot tell which
meaning is wanted. That is round 6's `cgate` arm, and it is the plot that
caused the 2026-09-05 misunderstanding.

Two more things round 9 changes, which are about *how* the model is trained
rather than what it predicts:

- **Which runs decide when training stops.** Training stops when the error
  on the validation runs stops improving, and the weights from the best
  validation epoch are kept. Until round 8 those runs were all
  linear-charging cases, so a model could be stopped while it was still
  learning the discharge runs. Round 9 uses 2 validation runs of every case
  type (tag `vs2`).
- **One network per temperature** (tag `ind`). Rounds 1-8 used one network
  with three outputs; the three temperatures share every internal weight and
  one stopping decision. Round 9 also trains three separate networks, one
  after another — inner wall, outer wall, then the average against the
  finished walls — so no temperature can stop or distort another's training.
  A two-network form (tag `inda`) keeps the two walls in one network and
  separates only the average.
- **Epoch cap 2000.** With the new validation set training keeps improving
  past 1000 epochs, so the cap was raised for round 9; a run that reaches
  the cap is reported as such.
- **A time curve for the position** (tag `tp`). Early in a run the interior
  is still cold, so the true position sits near the outer wall and climbs
  over the first ~200 steps; the model's position started at its long-run
  value instead, which is why the first minutes of `T_avg` were too high in
  every charging case (worst where the walls part fast). The climb is the
  same in every run, so it is now a fixed curve (the per-step median over
  the training runs) and the network only predicts the deviation from it.

### Conventions for the entries below

- One dated entry per change or run: *what* changed, *why*, the config delta,
  the expected effect, and (later) the observed result.
- Status tags: `[CODE]` landed in source, not yet run · `[RUNNING]` ·
  `[DONE]` results in hand · `[SUPERSEDED]`.
- Each sweep writes to `runs/<RUN_NAME_BASE>_seed<N>/`; the run name encodes
  the configuration, so the log maps 1:1 onto output folders.
- Decision rules are written down **before** a round launches.

### What each round asked

| date | round | question | answer |
|---|---|---|---|
| 2026-05-25 | — | 8 input layouts, zero-padded history | rejected: 9 °C error at t=0 |
| 2026-06-28 | — | refactor the 2957-line script into a package | done; run command unchanged |
| 2026-07-16 | — | how to handle the start of a run | a variable-length window wins; epoch budget cut to 800 + early stop |
| 2026-07-21/23 | — | can `T_inner` be made easy? (Arnold) | yes — anchor it on the inlet temperature: 1.93 → 1.39 °C |
| 2026-07-25 | — | five candidate fixes | loss weights + no feedback: 1.27 °C |
| 2026-08-06 | — | are the "weird jumps" caused by dropping feedback? | no, by the fixed anchors |
| 2026-08-09 | 4 | do fixed anchors on `T_outer` / `T_avg` help? | no, all three hypotheses refuted |
| 2026-08-12 | 5 | predict `T_avg` as a position between the walls | best so far: 0.890 °C, no blow-ups |
| 2026-08-26 | 6 | where should the position stop being trusted? | the lever was the loss weight: 0.774 °C; methods 3 and 4 are the safe gates |
| 2026-09-06 | 7 | combine the weight with the gates; tell the model its regime | `[CODE]` — see 2026-09-12 |
| 2026-09-12 | 8 | combine the two survivors; give the outer wall a reference (the inlet, and a slow running average of it -- the physical one tracks the wall to 5 °C with no model at all); AR at 2 layers | `[CODE]` |
| 2026-09-12 | 7 | (results) | the case gate with a plain 4th channel wins under the new weights: 0.72 °C, no blow-ups, the 12 hard cases halved; the regime flag helps the other 49 cases instead; removing both-phase runs from training hurts by 0.55 °C |
| 2026-09-16 | 8 | (results) | the survivors do not stack (0.707 vs 0.718, noise); the outer-wall reference fits T_outer and breaks T_avg, worst on the gated cases; every anchored arm early-stops ~300 epochs sooner on a lower validation loss — and the validation set turns out to be 15 linear-charging runs |
| 2026-09-19 | 9 | re-test rounds 4–8 — every configuration ever trained (146) and every loss-weight vector — with a validation set that contains every case type; one network per temperature (and a two-network form: walls shared, T_avg separate); a time curve for the position head (the first-50-step T_avg problem) | `[CODE]` — runner ready: everything by default, 218 GPU-h ≈ 108 h wall with 3 seeds in parallel, S1 decision arms first; 3-seed pilots: the validation set caused the round-8 collapse (1.53 → 0.89), the reference still costs T_avg on the switched cases; the time curve cuts the first-50-step error 1.24 → 0.48 with nothing worse |

---

## 2026-09-19 — Round 9 `[CODE]` — a validation set that sees every case type; one network per temperature

**In one line:** round 8 ended with a suspect rather than a verdict: every
arm with the outer-wall reference stopped training ~300 epochs early on a
lower validation loss and came out worse on the test set, and the validation
set turned out to be 15 linear-charging runs. Round 9 changes the protocol
(validation = 2 runs of each of the 10 case types) and re-runs the
configuration each round hinged on, so every earlier conclusion is re-tested
under the new protocol against its old-protocol twin (branch V). Then it asks
the round-8 question the way it should have been asked: three separate
networks, one per temperature, trained one after another so nothing one
temperature does can stop another's training (branch I). Direct prediction
only; AR stays parked. Decision 2026-09-17 (user), code 2026-09-17/19.

**What was checked before coding.** (1) The "final" checkpoint every run
saves (`gru_variant_<v>.pth`) is byte-identical to the best one — `train_model`
restores the best weights before returning and main.py writes the file
afterwards — so the cheap test "score the final-epoch weights" (HANDOVER,
2026-09-16) is impossible with existing runs; verified on all 20 seeds of the
round-8 EMA arm (max |Δ| = 0, re-scores identical to meta.json). The only test
is to retrain. (2) The EMA reference's seed conversion bug (round-8 audit) is
fixed and unit-tested; on the base data it changed nothing (both scalers span
[1, 610]). (3) `fit_model_constants` was extracted from `train_model` so a
saved run can be rebuilt and re-scored without retraining
(`eval_checkpoint.py`; reproduces recorded metrics to 3 decimals, shared and
independent models alike).

**Code (`tes_gru`)**
- `VAL_SPLIT=stratified`, `VAL_PER_TYPE` (config, data, runio): 2 runs per
  case-type folder, drawn once with `SPLIT_SEED=42` and frozen in
  `splits/stratified_v2_s42.json` (committed with the code; the lab server
  reads it and draws nothing). 291 train / 20 val / 70 test; 6 of the 20
  validation runs are both-phase runs (none were before). Tag `vs2`;
  provenance `data.val_split`; export family prefix `vs2_`. Legacy mode is
  untouched (296/15/70, same 15 files as before).
- `MODEL_SPLIT=independent` (config, models, train, runio, export): a
  wrapper of three complete `ThermalGRU`s (init encoder + GRU + head) behind
  the shared model's interface — same `(B, T, OUTPUT_SIZE)` output, hidden
  state a tuple — so rollout, anchors, position formula, gates and
  evaluate.py are unchanged. The T_avg network owns the extra channels (4th
  absolute head, learned gate). `train_model` runs three stages in channel
  order, each with its own optimiser, plateau schedule, warm-up, early
  stopping (patience 150) and best epoch; the finished networks are frozen
  and kept in eval mode (dropout off) while the next trains, so the T_avg
  network fits against the walls it will see at test time. Stage loss = the
  configured weight on that stage's channel only (the 1-6-1 tag is inherited
  for pairing; with one channel per network the relative weights do
  nothing). One best checkpoint file holds every network's best state.
  `train_history.json` gains `stages` (per-network best/total epoch, best
  val); the top-level best_epoch indexes the concatenated curve, total_epoch
  is the sum. Tag `ind`; provenance `model_split`; export family
  `pos_head_independent[...]`, knob doc `ind`. In-run resume is off for
  independent seeds (done.flag resume across seeds unchanged). Direct
  prediction only. Parameter counts: shared h128×2 = 283k (GRU 150k); three
  h128×2 = 848k (GRU 449k); three h64×2 = 414k (GRU 114k) — the init-encoder
  MLPs are fixed-width and dominate the small nets, which is why the capacity
  control matches the GRU, not the total.
- Shared mode is bit-for-bit what it was: the loop was refactored into
  `_train_stage` and a 2-epoch smoke run reproduces the pre-refactor numbers
  exactly (46.814 / 81.197 / 58.557 on the fip arm).
- `MODEL_SPLIT=avg_only` (tag `inda`): the two-network form of the same
  wrapper — the walls keep one shared trunk (stage 1, loss on T_inner +
  T_outer), T_avg has its own (stage 2). Added after the pilot showed the
  lone T_outer net worse than the shared trunk's T_outer. The wrapper is
  generic over channel groups (`MODEL_GROUPS` in config).
- Validation loss = the exact mean over validation files (was the mean of
  per-batch means: with 20 files and batch 16 the last four files weighed
  1/8 each; the legacy 15-file set is one batch and is unchanged).
  `train_history.json` also records `val_hist_ch`, the unweighted per-channel
  validation MAE per epoch, so which channel drove an arm's stop can be read
  off the run; meta.json carries flat `Best_Epoch_<net>` / `Total_Epoch_<net>`
  for split models.
- `MAX_EPOCHS=2000` for every round-9 arm (tag `ep2000`; the stale-dir guard
  now refuses only ep < 151), see "Epoch cap" below.
- `POS_TIME_PRIOR=median` (tag `tp`; config, train, rollout, evaluate,
  runio, export, eval_checkpoint) — **the early-window fix.** Diagnosis on
  the 20-seed cgate-ah ensemble (2026-09-19, prompted by the user: "the core
  problem since round 5 is still there — T_avg in the first ~50 steps of
  cases 15–24 / 58–65"): the head's position output is flat at ~0.24 from
  t = 0, while the true position of a charging run starts near 0.05 and
  climbs to ~0.25 over the first ~200 steps (the heat has not penetrated; the
  bulk average sits at the cold outer wall). Error = Δpos × wall gap, so it
  is worst where the gap opens fast: case 24 (gap 157 °C at step 50) carries
  10.6 °C over the first 50 steps, cases 29 / 54–57 3–8 °C; cases 58–65 are
  fine (< 0.8 °C, they are gated). The window is 3.5 % of the steps and 2.9 %
  of the T_avg error (t < 200: 11.5 %) — a curve-shape problem more than a
  score problem. On the 311 training runs the climb is the same in every
  run: the per-step median position is 0.04 / 0.16 / 0.25 / 0.28 at steps
  1 / 50 / 200 / 500, spread 0.037 around it in the first 200 steps against
  0.08 for the flat mean (a √t least-squares fit over the whole run is too
  flat — it is dominated by the plateau — so the prior is the nonparametric
  per-step median). The head now predicts the residual from that curve;
  mu/sd are the residual's; the step index comes from the scaled Time input
  (column 0), direct and AR paths, train and test; gated cases are
  unaffected. Offline, with the TRUE walls and the curve alone (network
  output 0), the first-200-step T_avg error on the 49 non-gated test cases
  is 0.51 °C against 2.64 for the flat mean and 1.39 for the trained
  ensemble; the full-run error is not worse (1.55 vs 1.59); case 24's first
  50 steps go from 13.7 (flat) / 10.6 (ensemble) to 0.2.
- `TOUTER_EMA_FORM = euler | exact` (config, rollout, train, runio, export;
  tag `To-em<τ>x` for exact). The user derived the reference from the outer
  wall's lumped energy balance (2026-09-20, notebook pages b1–b3): ρcV
  dT_out/dt = hA (T_in − T_out) → τ = ρcV/hA → integrating factor →
  zero-order hold over one step gives the exact discrete form
  T_out(t) = T_out(t−Δt)·e^(−Δt/τ) + T_in(t)·(1 − e^(−Δt/τ)); the Taylor step
  e^(−Δt/τ) ≈ 1 − Δt/τ gives the forward-Euler EMA rounds 8–9 use. Both are
  e_t = r·e_(t−1) + (1−r)·x_t with r = 1 − 1/τ (euler) or e^(−1/τ) (exact);
  one function `config.touter_ema_r()` feeds the rollout, the training-time
  twin and the preflight. At τ = 1200 the weights differ by 0.04 %; the
  exact form is there for a medium with a short τ, where the Taylor step is
  no longer small. Default euler, so no existing run name changes; the
  derivation is in config.py next to the knob.
- Runner: `--parallel N` (default 3) deals each arm's seeds round-robin to N
  concurrent launcher processes on the one GPU, each with its own run dirs
  and log; measured on the dev 4090: 3 → 2.03×, 4 → 2.27× the throughput of
  one (the models are small; each process is launch-bound). GPU memory per
  process 1.1–1.4 GB (shared models), 3.8 GB (three-network arms); the
  preflight reads the free memory and lowers N if needed. An arm's verdict
  is now its done.flags, not the exit code. Default stages: everything —
  S1, S2, S1b, S1c — 218 GPU-h, ~108 h wall at 3 workers (user 2026-09-20:
  "it is 7 days anyway, run it all").
- `rollout.touter_reference`: the 1440-step Python loop that computed the
  EMA reference per batch is replaced by its closed form (float64; equal to
  the recurrence to 5×10⁻⁷ in scaled units, i.e. < 10⁻³ °C). The EMA arms
  trained ~3× slower than the others in round 8 because of that loop (4.5 vs
  1.5 s/epoch on the dev box); now they cost the same. The runner's preflight
  re-checks closed form against recurrence every launch.
- `eval_checkpoint.py`, `fit_model_constants`, `build_model` /
  `model_parts` (models.py): the one place a model is built, used by
  train.py and the re-scorer.

**Runner `run_round9.py`** (S1 + S1b + S2 by default, 218 GPU-h at the 2000-epoch cap, ≈ 108 h wall with `--parallel 3` — in order S1 ~44 GPU-h (the decision arms: configs, the weight scans, the early-window pair), S2 ~24 (the split models), S1b ~105 (the 90 legacy configurations the old validation set measurably cut short), S1c ~44 (the 24 it did not); all on by default; the stages run in that order so the decision arms finish first, and `--stage S1` / `--stage S1,S2` run subsets; every launch imports `tes_gru.config` under each arm's knobs and refuses duplicate run names; `--stage S1` first if time is short; 20 seeds where the original had 20, 12 where it had 12 — the seed list every round has used, so each arm pairs per seed with its twin)

| stage | arms | pairs with | question |
|---|---|---|---|
| S1 (branch V, `vs2`; 19 arms) | r4-abs-h128x5, r4-anchor_avg-h128x5; r5-abs-h128x2, r5-pos-h128x5, r5-pos-h128x2; r6-tg10s40, r6-cgate-afb, r6-cgate-ah, r6-ah; r7-w161, r7-w161-ah, r7-w161-cgate-ah, r7-w161-fip, r7-w161-fiz, r7-w161-tg30s30; r8-w161-fip-cgate-ah, r8-w161-fiz-cgate-ah, r8-w161-To-em1200, r8-w161-To-em1200-cgate-ah | its legacy-split twin (rounds 6–8 in `runs/`, rounds 4–5 in the wiki's result folders; r5-pos-h128x5 on its 10 seeds); the `ah` / `fiz` arms are the 4-channel and input-column nulls rounds 6–8 paired against | what the protocol change moved, per arm; and whether each round's conclusion (position head > direct, at 5 and at 2 layers; the fixed T_avg anchor loses; 1-6-1 > 1-6-3; each gate vs its 4-channel null; fip vs fiz; fip-cgate-ah vs fiz-cgate-ah; the two reference arms) still holds when early stopping can see a discharge. r5-abs-h128x2 is Arnold's 2026-09-17 "use the absolute value for T_avg as well" at the current capacity |
| S1, part 3 (the early-window fix; 2 arms, new) | tp-w161, tp-w161-cgate-ah (20 seeds) | r7-w161, r7-w161-cgate-ah | does a fixed elapsed-time curve for the position (`POS_TIME_PRIOR=median`, tag `tp`) remove the first-200-step T_avg overestimate without costing anything later — the user's "core problem since round 5" (cases 15–24 / 58–65, first 50 steps); primary = first-200-step T_avg error on the 49 non-gated cases, paired; passes only with overall not worse and no veto |
| S1b (the legacy configurations the old validation set demonstrably cut short; 90 arms, tier "keep" in `splits/round9_legacy_configs.json`) | the entire abs-model capacity, lookahead, weight and anchor-scale scans of rounds 4–5, the fixed anchors, the learned gates, the round-6 soft timestep gates and floors at 1-6-3, To-ai, os2, To-em900, the full stack — each with its legacy seed list | its legacy twin, per seed | tiered by a **measurement**, after the user pointed out (2026-09-20) that the first tiering did not follow its own criterion: for every legacy run, how much the training loss still fell in the 150 epochs after the checkpoint the old set picked (`legacy_post_checkpoint_train_drop`, median over seeds, stored in the manifest). Calibration on S1's twins: cgate-ah 12.8 % and w161 11.4 % (vs2 moved cgate-ah by +0.03 — nothing), To-em1200-cgate-ah 25.5 % (vs2: 1.53 → 0.89), abs-h128x5 31.4 %. Everything above the calibrated no-change point (12.8 %) is S1b: the abs scans sit at 32–56 % with best epochs 290–410, the learned gates at 33–48 %, the round-6 soft gates and floors at 13–30 %. The first, argued tiering had put exactly these in the archive and the least-cut-short arms (median 10 %) in "keep" — backwards |
| S1c (the 24 legacy configurations the old set did not cut short; off by default) | drop ≤ 12.8 %: the 1-6-1 gate/flag family (cgate-afb, fip/fiz-cgate-afb, tg10s40, tg20/30, fic), the h192/h256 arms at 1-6-1 and 1-6-3 (best epochs 680–830, many at the 1000 cap), fl20/fl40/fl40s, the shared-channel cgate, To-em1500 | its legacy twin, per seed | these trained at least as long as the winner did, and vs2 did not move the winner; `--stage S1c` (~44 h) completes the record. The manifest is built from the runs' own `run_config.json`, so nothing is hand-picked; the 5 exclude-both-phase sanity arms (a different test set) and the 4 AR arms (user decision) are the only omissions |
| S2 (branch I, `ind` / `inda`, `vs2`; 7 arms) | ind-w161, ind-w161-cgate-ah, ind-w161-To-em1200, ind-w161-To-em1200-cgate-ah; ind-h64-w161-cgate-ah; inda-w161-cgate-ah, inda-w161-To-em1200-cgate-ah | the S1 arm with the same knobs; the h64 control against ind-w161-cgate-ah; inda against both its shared and its ind twin | does one network per temperature beat the shared one, alone and on the winner; does the outer-wall reference help once it cannot stop T_avg's training; is any gain a GRU-capacity gain; does the wall pair want its shared trunk (`inda`: walls shared, T_avg separate — the pilot's suggestion) |
| S1, part 2 (the weight scans; 14 arms) | r6-w1-6-0.5, r6-w1-6-6, r6-w1-6-10, r6-w1-3-3, r6-w1-4-3, r6-w1-8-3, r6-w2-6-3, r7-w1-8-1, r7-w1-8-2, r7-w1-10-1, r7-w1-12-1 (12 seeds, plain position head); r8-w163-To-em1200, r8-w131-To-em1200 (20 seeds); w166-cgate-ah (12 seeds, new: Arnold's "explore increasing the weight average" on the winner) | their round-6/7/8 twins (same base, same seeds); each vs r7-w161 on the common seeds, the EMA pair vs r8-w161-To-em1200, w166-cgate-ah vs r7-w161-cgate-ah and r6-cgate-ah | is 1-6-1 still the weight vector — the T_avg side (0.5 / 1 / 3 / 6 / 10) and the T_outer side (3 / 4 / 6 / 8 / 10 / 12) — now that early stopping sees every case type. The user's rule: everything decided under the old validation set is re-run, weights included; nothing carries over untested |

**Decision rule, fixed before the run** (the round-7 statistic throughout:
paired difference per seed, sign-flip p, 95 % bootstrap CI, blow-up seeds on
either side counted and the comparison repeated without them, jump veto,
ensemble deployment metric, simplicity order — which now reads "shared < inda
< ind" and "no T_outer reference < reference"). The 3-seed pilot below was in
hand when the numeric thresholds were set; the sweep's 20 seeds include those
three, and every comparison is also reported on the 17 non-pilot seeds.
- *Protocol effect* (S1 vs legacy twins): reported for every arm, sign not
  assumed. What changes between the twins is the protocol as a whole — the
  validation set, the training set (296 → 291; 13 of the 15 old validation
  files move into training, so the scaler and the fitted constants move a
  little) and the epoch cap (1000 → 2000, below). The rest-49 shift is the
  sensitivity indicator for the training-set part.
- *The specific hypothesis*, numeric, on the paired (To-em1200-cgate-ah −
  cgate-ah) switched-21 T_avg under vs2 (legacy +3.40 [+1.76, +5.57]):
  (a) "the validation set caused the round-8 collapse" if the vs2 CI's upper
  bound is below +1.0 °C; (b) "the reference itself still costs T_avg" if the
  vs2 CI excludes 0. Both can be true, and the pilot says they will be
  (+0.69, seeds +1.04 / +0.64 / +0.38); then branch I asks whether the cost is
  the coupling. Descriptive: the difference-in-differences per seed, and both
  arms' best epochs with the count of seeds at the cap (censored).
- *Branch I:* each split arm vs its S1 twin; ind-To-em1200-cgate-ah vs
  ind-cgate-ah and inda-To-em1200-cgate-ah vs inda-cgate-ah (the round-8
  question re-asked where the reference cannot stop T_avg's training); inda
  vs ind on the same knobs; ind-h64-cgate-ah within noise of ind-cgate-ah
  means an ind gain is not GRU capacity. "Independent" is tested as the whole
  bundle (separate weights, stopping, optimiser, sequential training on
  frozen walls), not attributed to one ingredient. The recipe changes only if
  a vs2 arm beats the legacy cgate-ah ensemble (0.676) AND its own null, with
  no veto.
- *The early-window fix* (S1 part 3): tp-w161-cgate-ah vs r7-w161-cgate-ah
  (and tp-w161 vs r7-w161), paired per seed: primary = T_avg |error| over
  the first 200 steps on the 49 non-gated cases; the arm passes if that CI
  is below zero AND the overall paired difference is not above zero AND no
  veto. Secondary: the first 50 steps; cases 24 / 29 / 54–57 individually;
  the switched 21 unchanged. Expected: the first-200-step error at least
  halved; overall within noise or slightly better.
- *Expected from the pilot:* the anchored arms recover most of their T_avg
  loss under vs2 but stay behind cgate-ah on the switched cases; cgate-ah
  itself moves little; ind beats shared on the 49 normal cases and loses on
  the 21 switched and on T_outer; inda keeps the shared T_outer and the ind
  rest-49 gain and is the arm to beat. If every split arm matches its shared
  twin, one network was never the problem and the split is dropped.

**Epoch cap 2000 (ep2000 tag).** Under vs2 the 1000 cap binds: in the pilot
all three cgate-ah seeds and one of three EMA seeds were still improving on
validation at epoch 1000 (best epochs 950–993; the plateau scheduler had the
LR at 1e-5 and the running best still moved 1–3 % per 150 epochs), whereas
the legacy set went flat by 600–800 (4 % of the round 6–8 runs hit 1000). A
binding cap turns "stop when validation stops improving" into "train to the
cap" and right-censors every best-epoch comparison, so every round-9 arm runs
with MAX_EPOCHS=2000 (the July precedent: 800 → 1000 when a seed hit the
wall). Cost ~1.5× per seed. Seeds at the cap are counted per arm and treated
as censored.

**Preflights** (every launch): package compiles under the server's Python;
the split file is the committed one (10 types × 2, train/val disjoint) and
`data.py` refuses a train_dir that differs from the frozen train list; stale-
dir guard (ep < 151 smoke dirs, Total_Epoch < 151, `vs`/`ind` tags without
matching provenance); no two arms of the plan share a run name; child process
1 — the split resolves to 291/20/70 from the file and the numpy/torch case
detectors agree on all 381 runs; child process 2 — EMA closed form vs
recurrence for τ = 100…1500 (NaN fails), and the wrapper: three nets, no
shared parameter, output layout, the stage loss vectors train.py uses summing
to the weight vector. A check passes on its `CHECK_OK` line, not on its exit
code (the Windows dev box exits 0xC0000409 at teardown after any CUDA model
run — also the launcher; complete outputs, deterministic, not seen on Linux).

**Adversarial review (2026-09-19, 4 lenses × 2 refuters; 33 findings, 11
survived, 22 had one or both refuters cut off by the session limit and were
judged by hand).** Acted on: the binding epoch cap (above); the vs2
validation loss was a mean of two unequal batches (16 + 4 files: the last four
carried 1/8 each) → now the exact mean over files (legacy 15-file set
unchanged; the pilot ran with the old weighting); S3's w1-6-3 arm was
byte-identical to S1's r6-cgate-ah → dropped, plus a unique-run-name assert;
the 4-channel and input-column nulls (r6-ah, r7-w161-ah, r8-w161-fiz-cgate-ah)
added so each round's own pairing is repeated; r5-abs-h128x2's legacy figure
corrected to 1.37 (20 seeds; 1.11 was the n=6 false positive); hypothesis
made numeric and censor-aware; the training-set change named in the protocol
clause; `eval_checkpoint.py` overwrote the original run's `predictions.npz`
(or left stray `runs/<name>/` dirs) through a module-global side write → off,
and knobs absent from an older run_config are now reset to defaults instead
of leaking from the shell; a stage with no validation improvement now raises
instead of silently restoring the previous stage's file; `TOUTER_TAU ≥ 3`
asserted (the closed form overflows below 2.6); the preflight's stage-mask
check now exercises train.py's own helper; per-channel validation curves
(`val_hist_ch`) and flat per-stage epochs in meta.json; the "weights do
nothing in ind mode" claim corrected (they act only through the gradient
clip, measured not to engage on the anchored arms); progress tail counts only
the current launch; `Round*_upload/` ignored by git. Not acted on: a code
stamp in run_config.json (pilot dirs are distinguishable by the missing
ep2000 tag); caching the frozen nets' outputs per stage (the pilot's 7–9
min/seed is budgeted as 3×).

**Smoke tests (dev box, 2026-09-19)**: 2-epoch shared runs of
r4-anchor_avg-h128x5, r7-w161-fip, r8-w161-To-em1200-cgate-ah under vs2 (the
fip run reproduces the pre-refactor numbers exactly, before and after the
review fixes); 3-epoch split runs of ind-w161, ind-w161-cgate-ah (4
channels), ind-w161-To-em1200, ind-w161-To-em1200-cgate-ah at h64,
inda-w161-To-em1200-cgate-ah (2 nets); provenance (`val_split`,
`model_split`, `output_size`), `stages` / `val_hist_ch` in the history, flat
stage keys in meta.json, one checkpoint with the parts, eval_checkpoint
re-score identical to meta.json; all dirs deleted afterwards.

**Dev-box pilot (3 seeds, 2026-09-19; 1000-epoch cap, batch-weighted
validation — the two things the sweep changes).** r8-w161-To-em1200-cgate-ah,
r7-w161-cgate-ah and ind-w161-To-em1200-cgate-ah under vs2, seeds 7/21/42;
legacy twins from `calit2-thermal-wiki/round8`:

| arm | overall | T_outer | T_avg | 12 hard | 21 switched | 49 rest | best/total epoch |
|---|---|---|---|---|---|---|---|
| To-em1200-cgate-ah, legacy | 1.533 (1.89 / 1.84 / 0.86) | 0.387 | 3.95 | 8.13 | 7.66 | 2.35 | 224 / 374 |
| **To-em1200-cgate-ah, vs2** | **0.886** (0.90 / 0.90 / 0.86) | 0.327 | 2.05 | 2.10 | 1.44 | 2.32 | 674 / 776 (1 at cap) |
| cgate-ah, legacy | 0.747 (0.67 / 0.85 / 0.72) | 0.483 | 1.48 | 1.31 | 1.15 | 1.62 | 618 / 768 |
| cgate-ah, vs2 | 0.780 (0.73 / 0.73 / 0.87) | 0.384 | 1.69 | 1.07 | 0.76 | 2.08 | 955 / 1000 (3 at cap) |
| ind-To-em1200-cgate-ah, vs2 | 0.919 (0.84 / 1.04 / 0.88) | 0.423 | 2.03 | 2.72 | 2.23 | 1.94 | stages: inner ~170 / outer 515–697 / avg 213–267 |

Paired per seed: To-em1200-cgate-ah vs2 − legacy = −0.65 overall (−0.99 /
−0.94 / −0.01), switched-21 −6.2 (−8.6 / −9.2 / −0.9) → the validation set
caused the collapse. cgate-ah vs2 − legacy = +0.03 (noise), T_outer −0.10,
switched −0.40, rest-49 +0.46 (all three seeds) — the protocol moves the
winner's error from the switched to the normal cases. To-em1200-cgate-ah vs
cgate-ah under vs2 = +0.11: T_outer −0.06 (3/3 better), switched +0.69 (3/3
worse), rest-49 +0.23 — the reference still costs T_avg. ind vs shared
(both vs2, EMA arm) = +0.03: rest-49 −0.38 (3/3), switched +0.79 (3/3),
T_outer +0.10 (3/3, the lone T_outer net is worse than the shared trunk's) —
hence the `inda` arms. n = 3; the sweep decides.

**Dev-box pilot of the early-window fix (3 seeds, 2026-09-20; vs2, cap 2000,
exact validation mean — the sweep's settings).** tp-w161-cgate-ah, seeds
7/21/42, paired against the cgate-ah vs2 seeds above (which ran at cap 1000):

| | overall | T_outer | T_avg | non-gated 49: t < 50 (signed) | t < 200 | all t | switched 21 | best / total epoch |
|---|---|---|---|---|---|---|---|---|
| cgate-ah vs2 | 0.780 | 0.384 | 1.685 | 1.24 (+0.92) | 1.28 | 2.08 | 0.76 | 955 / 1000 (at cap) |
| **tp-w161-cgate-ah vs2** | **0.746** | 0.379 | 1.591 | **0.48** (+0.19) | **0.79** | 1.94 | 0.77 | 831–1210 / 981–1360 (stopped) |

Paired per seed: non-gated t < 50 −0.76 (−0.83 / −0.85 / −0.59), t < 200
−0.49 (3/3), non-gated all-t −0.14 (−0.13 / 0.00 / −0.28), switched 21
+0.01 (noise, as expected — the gate bypasses the position), T_outer 0.00,
overall −0.03. Per case, first 50 steps: 24: 9.3 → 2.6; 29: 8.6 → 1.4; 55:
3.3 → 0.6; 15 / 20 / 22: 1.3 / 1.0 / 0.7 → 0.3 / 0.2 / 0.1. The remaining
early error sits on 54 / 56 (1.9 / 2.6): the sign of the mean is now
+0.19, so the systematic overestimate is gone and what is left is
case-specific. Nothing got worse anywhere; the three seeds early-stopped at
981 / 1102 / 1360 under the 2000 cap (none at the cap). n = 3; the sweep
decides.

## 2026-09-16 — Round 8 results `[DONE]` — the survivors do not stack; the outer-wall reference fits T_outer and breaks T_avg; the validation set is the suspect

**In one line:** combining the phase flag with the case gate gives 0.707 vs
0.718 — inside the seed noise, so `cgate-ah` alone stays the recipe. The
outer-wall reference Arnold asked for fails in both forms: the instantaneous
inlet (`To-ai`) blows up 11 of 20 seeds, and the smoothed inlet (`To-em1200`)
leaves T_outer where it was and makes T_avg worse on every case subset; put on
the winner it wrecks the 21 gated cases (0.86 → 4.26) with 4 blow-ups. The
anchored arms all stop training ~300 epochs earlier than their un-anchored
twins on a *lower* validation loss — and a code audit found the validation
set has been 15 linear-charging runs since round 4. That is the lead round 9
follows. Deployment stays the 20-seed `cgate-ah` ensemble, 0.676.

Results in `calit2-thermal-wiki/round8` (rounds 6–8 in one folder, 20 seeds
per arm); every number below recomputed from `meta.json` /
`summary_errors.csv` / `train_history.json` (2026-09-16, re-checked
2026-09-19).

### The table

| arm | overall | T_outer | T_avg | 12 hard | 21 switched | 49 rest | >1.5 °C | best epoch | val loss ×10³ |
|---|---|---|---|---|---|---|---|---|---|
| w161 (base) | 0.787 | 0.448 | 1.631 | 2.47 | 1.69 | 1.61 | 0/20 | 628 | 2.06 |
| **w161-cgate-ah** (recipe) | **0.718** | 0.447 | 1.430 | 1.09 | 0.86 | 1.67 | 0/20 | 619 | 1.99 |
| w161-fip-cgate-ah | 0.707 | 0.433 | 1.412 | 1.02 | 0.80 | 1.67 | 0/20 | 649 | 2.01 |
| w161-To-ai | 1.668 | 2.218 | 2.560 | 3.27 | 2.62 | 2.53 | 11/20 | 294 | 2.30 |
| w161-To-em900 | 0.931 | 0.443 | 2.067 | 2.58 | 1.80 | 2.18 | 0/20 | 354 | 0.88 |
| w161-To-em1200 | 0.935 | 0.437 | 2.085 | 2.90 | 1.97 | 2.14 | 0/20 | 422 | 0.94 |
| w161-To-em1500 | 0.912 | 0.352 | 2.138 | 3.12 | 2.06 | 2.17 | 0/20 | 452 | 1.23 |
| w161-To-em1200-cgate-ah | 1.147 | 0.365 | 2.825 | 4.67 | 4.26 | 2.21 | 4/20 | 324 | 1.54 |
| w161-fip-To-em1200-cgate-ah | 1.012 | 0.356 | 2.434 | 3.07 | 2.89 | 2.24 | 1/20 | 298 | 1.49 |
| w161-To-em1200-os2 | 0.993 | 0.591 | 2.083 | 2.77 | 1.94 | 2.14 | 0/20 | 332 | 1.03 |
| w163-To-em1200 | 0.989 | 0.515 | 2.162 | 2.65 | 1.83 | 2.30 | 0/20 | 349 | 1.64 |
| w131-To-em1200 | 1.015 | 0.566 | 2.186 | 2.79 | 1.93 | 2.30 | 0/20 | 326 | 0.73 |

Paired per seed (mean difference, 95 % bootstrap CI, sign-flip p):

| comparison | overall | T_outer | 21 switched T_avg |
|---|---|---|---|
| fip-cgate-ah vs cgate-ah | −0.011 [−0.045, +0.018] p=0.50 | −0.014 ns | −0.06 ns |
| fip-cgate-ah vs its null fiz-cgate-ah | −0.031 [−0.069, +0.006] | | |
| To-ai vs w161 | +0.881 [+0.71, +1.09] | +1.770 | +0.93 |
| To-em1200 vs w161 | +0.148 [+0.118, +0.175] | −0.011 [−0.061, +0.040] p=0.69 | +0.28 [+0.20, +0.36] |
| To-em1500 vs w161 | +0.125 [+0.074, +0.194] | −0.096 [−0.128, −0.067] | +0.37 |
| To-em1200-cgate-ah vs cgate-ah | +0.429 [+0.24, +0.67] | −0.083 [−0.109, −0.055] | +3.40 [+1.76, +5.57] |
| os2 / w163 / w131 vs To-em1200 | +0.058 / +0.054 / +0.080, all CI above zero | +0.15 / +0.08 / +0.13 | ns |

### Decision under the pre-registered rule

1. **fip-cgate-ah: within noise, not adopted.** The expected −0.16 on the 49
   untouched cases did not appear on top of the gate (r49 1.67 → 1.67); the
   flag's round-7 effect came through the same channel the gate already
   fixes. Simplicity clause → `cgate-ah` stays.
2. **To-ai: refuted, as expected.** Its formula floor is 73 °C; 11 seeds above
   1.5 °C. Arnold's literal reference is out.
3. **To-em1200 alone: refuted.** T_outer does not move (−0.011, ns) — the
   network already predicted T_outer to 0.45 without a reference, and the
   reference's own floor is 5.3 — while T_avg gets worse on all three
   subsets (+0.5 on the 49 untouched cases, not just the gated ones). τ = 900
   / 1500 tell the same story; 1500 gains 0.10 on T_outer and loses more on
   T_avg. Companion tests (os2, 1-6-3, 1-3-1) are all worse than 1-6-1 under
   the anchor — the weight is not the lever here.
4. **To-em1200 on the winner: refuted with a veto.** 4/20 blow-ups; the 21
   switched cases go from 0.86 to 4.26. Same on the full stack (1/20, 2.89).
5. **Deployment unchanged:** `cgate-ah`, 20-seed ensemble 0.676
   (`ensemble_eval.py`, common seeds).

### Why the reference failed — what the data say, and what they do not

The T_avg damage is not where a bad T_outer reference would put it. The
reference feeds T_avg only through the position formula, and T_outer itself
is as good or better under every EMA arm; yet T_avg is worse everywhere, and
worst exactly where the 4th channel does the work (the 21 switched cases
under cgate-ah). Two facts point at training, not at the formula:

- **Every anchored arm stops ~300 epochs earlier** (best epoch 294–452 vs
  619–649 for the un-anchored twins) **on a lower validation loss** (0.9 vs
  2.0 ×10⁻³ for To-em1200 vs w161). Lower validation, worse test.
- **The validation set is 15 linear-charging runs.** `data.py` sorts the 311
  training files by path and takes the first 5 % — all from
  `Case 1 (linear charging)/`. Early stopping and best-checkpoint selection
  have never seen a charge-then-discharge run, in any round since round 4.
  The 4th channel is trained only on the ~107 gated runs, none of which are
  in validation; when the reference makes T_outer easy the validation curve
  bottoms out early, and the checkpoint is taken before that channel has
  converged. (All rounds share this, so the internal comparisons of rounds
  4–8 stand; the absolute numbers are "under this protocol".)

Not proven: this is a hypothesis about early stopping, and the position
formula could still be part of it. Round 9 tests it directly (a validation
set with 2 runs of each case type, then the same arms again) — see the
round-9 `[CODE]` entry.

A second audit finding, fixed 2026-09-17 before any new run: the EMA
reference at test time was seeded with T_outer(0) in T_outer's scaled units
but run in Input_T's scaled units. On this dataset the two scalers coincide
(all temperature columns span the same range) so round 8 is unaffected
(mean |Δref| ≈ 5×10⁻⁵ °C); on data where they differ the seed would be off
by tens of degrees. The conversion now goes through `model.touter_ref_affine`
and is unit-tested.

### AR
Not run (user decision 2026-09-17). The S3 arms stay in `run_round8.py` for a
single later run of the base arm and the direct winner, if the paper needs it.

## 2026-09-12 — Round 8 `[CODE]` — close the base-data line: combine the survivors, a physical outer-surface reference, a fair AR comparison

**In one line:** round 7 left two things that each work on different cases
(the case gate on the 12 hard runs, the phase flag on the other 49) and were
never combined; this round combines them. It also runs the outer-surface
anchor Arnold asked for on 2026-09-10 in two forms — his literal one (the
inlet temperature at that moment) and the physical one found while preparing
it (a slow running average of the inlet, which tracks the outer wall to
within a few degrees) — on its own, on the winner, and on the full stack,
with the same protocol the position head went through. The autoregressive
branch gets its first run at the direct model's depth. Whatever wins becomes
the deployment recipe. The convection phase is deferred until this is settled
(2026-09-13).

**The finding behind the anchor arm.** Scanning references for T_outer on the
training set (residual std, T_outer's own std is 115.8 °C): raw Input_T(t)
97.2; Input_T lagged 720 steps 39.8; exponential moving average of Input_T
with τ = 480 / 720 / 1000 / 1200 / 1500 steps → 38.3 / 21.3 / 9.4 / **7.3** /
12.0. The outer wall behaves as a first-order low-pass of the inlet (fitted
slope 1.001, offset +2.8 °C); EMA(1200) removes 94% of its variance. For
comparison the T_inner anchor removes 99% and won; the round-4 initial-value
anchor removed 50% and lost. Arnold's intuition ("the input is the
fundamental reference, it just has to be time-dependent") is right once the
reference is the smoothed inlet rather than the instantaneous one.

The formula alone, scored on the 70 test cases (reference + train-set mean
offset, no model), is the floor each anchor arm has to beat on T_outer:

| reference | T_outer MAE | worst case |
|---|---|---|
| Input_T(t) (To-ai) | 73.2 | 185.1 |
| EMA τ = 720 | 14.3 | 39.5 |
| EMA τ = 900 | 8.4 | 23.2 |
| EMA τ = 1000 | 6.2 | 17.9 |
| **EMA τ = 1200** | **5.3** | **12.7** |
| EMA τ = 1500 | 8.6 | 27.1 |
| the model today (w161, direct T_outer) | 0.45 | — |

So the reference on its own is a 5 °C model and the network already does
0.45 without it; the question S2 answers is whether "reference + learned
correction" beats "learned absolute value" — the T_inner precedent says yes,
the round-4 T_avg precedent (a 96%-variance anchor that lost to a head that
already learned the dynamics) says not necessarily.

**Code (`tes_gru`)**
- `TOUTER_MODE` = `abs` | `anchor_input` | `anchor_ema`, `TOUTER_TAU`
  (config/train/rollout/evaluate): T_outer(t+1) = ref(t+ANCHOR_LEAD) + delta,
  delta z-scored on the train set (ANCHOR_SCALE included) — the T_inner recipe
  applied verbatim to head 1 in the direct and the AR path, at train and test
  time; the position formula consumes the anchored value. ref = Input_T or
  its EMA (`rollout.touter_reference`, causal, affine-equivariant; numpy twin
  in train.py). Tags `To-ai` / `To-em<tau>`; provenance `touter_mode`,
  `touter_tau`; export family `pos_head_touter_anchor`. Asserted exclusive
  with OTHER_CH_MODE's own T_outer re-parametrisations (persistence, anchor).
- `CASE_FLAG_INPUT` extended to `abs_sliding`: data.py appends the column
  after the variant branches (both variants share it), `INPUT_DIMS` grows by
  one, `_rollout_sliding` and evaluate.py's mirror concatenate the column
  onto every window row (GT, never fed back). The runner's preflight builds
  one AR dataset and checks width, the flag column against the numpy flag,
  and the Input_T column against the dataset's own scaled inlet series.
- The case gate's verdict is computed once per rollout / per case and passed
  into `apply_other_anchor(case_gate=...)` instead of 1440× per AR pass;
  identical boolean (smoke run reproduced the previous number to 4 decimals),
  AR evaluation 25% faster.
- `MAX_EPOCHS` env override in `LATEST_PARAMS` for smoke tests only. A
  non-1000 cap adds an `ep<N>` tag to RUN_NAME, so a smoke run never shares a
  directory with a real seed; `save_run_config_snapshot` no longer rewrites a
  finished run's provenance; and the runner refuses any finished dir with
  Total_Epoch < 151 (unreachable for a real run under patience 150), which
  survives any run_config rewrite.

**Runner `run_round8.py`** (~12 h: S1 + S2 by default; the AR stage runs afterwards on request, with the base arm and the direct winner only, ~40 h, instead of all six AR arms at ~120 h)

| stage | arms | seeds | question |
|---|---|---|---|
| S1 | w161-cgate-ah (resumes), w161-fip-cgate-ah, w161-fiz-cgate-ah | 20 | do the two survivors stack; fiz-cgate-ah is the null |
| S2 | w161-To-ai; w161-To-em900 / -em1200 / -em1500; w161-To-em1200-cgate-ah; w161-fip-To-em1200-cgate-ah; w161-To-em1200-os2, w163-To-em1200, w131-To-em1200 | 20 | Arnold's literal reference vs w161; the physical one as a τ bracket vs w161, then on the winner (vs cgate-ah) and on the full stack (vs fip-cgate-ah); plus the companions the position head got in rounds 5-6: the head's output range (the residual's tail is 6 σ) and the loss balance under the anchor. The anchor adds no channel or input, so its null is the same arm without it |
| S3 | AR-w161-L2, AR-w161-cgate-ah-L2, AR-w161-fip-cgate-ah-L2; AR-w161-To-em1200-L2, AR-w161-To-em1200-cgate-ah-L2, AR-w161-fip-To-em1200-cgate-ah-L2 | 4 | off by default, run after the direct results with `--stage S3 --only ...` for the base arm and the direct winner; the AR branch at 2 layers with the round-7 recipe, then the S2 main line mirrored (EMA anchor alone, on the winner, on the full stack); descriptive (the fiz null was cut: at n=4 it cannot resolve a 0.04 °C effect against a 0.3 °C seed spread) |

**Expected progression** (the round-7 rule checks each step): cgate-ah 0.718 (measured) → fip-cgate-ah ≈ 0.70 (fip's −0.16 on the 49 untouched cases on top of the gate's −0.84 on the 21 switched) → To-em1200-cgate-ah, T_outer below its 0.45 with T_avg following through the position formula (the mechanism that made 1-6-1 win), ≈ 0.68 → fip-To-em1200-cgate-ah ≈ 0.65. Each step is checked against its parent (the same arm without the added piece): paired 95% CI below zero, no veto, and fip-cgate-ah also against fiz-cgate-ah (null shift reported; comparison repeated without any blow-up seed of the null). A step that does not materialise is dropped and the next one is tested on the previous survivor; the last survivor ships as its 20-seed ensemble. Also expected: To-ai within noise of w161 (its floor is 73 °C); the τ bracket a curve with 1200 best; os2 mattering only on the tail cases; the weight arms confirming 1-6-1, with 1-3-1 the candidate if the anchor has made T_outer easy enough that its weight was starving T_avg. Simplicity order adds “no T_outer anchor < anchor”; deployment metric on the common 20 seeds.

**Review (2026-09-12, 4 lenses × 2 refuters, 20 findings kept)**: the two
high-severity ones — the AR fip/fiz pair at n=4 (cut the null, kept the arm
the PI asked for as descriptive) and the smoke-test guard reading a file every
launch rewrites (fixed three ways, above). The lead-shifted-reference gap
became the EMA scan and the To-em arms.

**Smoke tests (dev box, 2026-09-12/13)**: 3-epoch direct runs of To-ai +
cgate-ah, fip + cgate-ah, To-em1200 + cgate-ah and fip + To-em1200 + cgate-ah;
2-epoch AR runs of fip + cgate-ah and of fip + To-em1200 + cgate-ah at L2
(6-wide window rows and the per-step EMA anchor through training and the
70-case evaluation); all artefacts present; the anchor fit printed the
scan's numbers (EMA 1200: mean +2.76, std 7.34); all dirs deleted afterwards.

**Expected**: fip-cgate-ah ≈ 0.70 single / ≈ 0.66 ensemble if the two effects
add; To-ai within noise of w161; To-em1200 the arm to watch — the tightest
reference T_outer has ever had, and a T_outer gain propagates into T_avg
through the position formula; AR-L2 between 0.9 and 1.1, the last AR run
unless the paper needs it.

## 2026-09-12 — Round 7 results `[DONE]` — the case gate with a plain 4th channel wins under 1-6-1; the regime flag helps the other cases

**In one line:** under the new loss weights the ranking of the gates turned
over — the case gate with a raw absolute 4th channel (method 2, `cgate-ah`),
which blew up twice in round 6, is now the best configuration the project has
produced (0.718 ± 0.058, 0 blow-ups, ensemble 0.676) and cuts every one of the
12 hard cases roughly in half; round 6's best gate (`tgate10-soft40`) blew up
instead. Feeding the "discharge has begun" flag to the network (method 5,
`fip`) does nothing for the 12 hard cases but improves the other 49; the two
mechanisms are complementary and were not combined this round. Arnold's
sanity check: with the both-phase runs removed, the gated arms reproduce their
ungated twins bit for bit, and removing those runs from training makes the
remaining 49 cases 0.55 °C worse.

448/448 round-7 runs complete (plus the 916 round-6 runs in the same folder as
references). Every number below was independently recomputed from the raw
`meta.json` / `summary_errors.csv` files and agrees to two decimals; the
decision-rule application was audited against the docstring and every call
stands.

### The table

Overall MAE mean ± sd over seeds; f12 / sw21 / r49 = T_avg MAE on the 12
flagged cases / the 21 the case gate switches / the 49 it never touches;
paired = per-seed difference vs the arm's pre-registered null (negative = arm
better) with the 95% bootstrap CI; ens = seed-ensemble MAE on the common 12
seeds (`ensemble_eval.py --seeds S12`).

| arm | n | overall | worst | blow | f12 | sw21 | r49 | null | paired overall (CI) | f12 paired | jumps paired | ens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w161 (new base) | 20 | 0.787 ± 0.046 | 0.92 | 0 | 2.47 | 1.69 | 1.61 | r6 base 1-6-3 | −0.187 [−0.40, −0.07] | −0.45 | −36 | 0.742 |
| w161-ah (4-ch null) | 20 | 0.818 ± 0.090 | 1.09 | 0 | 2.48 | 1.70 | 1.68 | w161 | +0.031 [−0.01, +0.08] ns | ns | ns | 0.778 |
| **w161-cgate-ah** (method 2) | 20 | **0.718 ± 0.058** | **0.85** | **0** | **1.09** | **0.86** | 1.67 | w161-ah | **−0.099 [−0.153, −0.048]** | **−1.39** | −14 ns | **0.681** |
| w161-tg30s30 (method 4, soft) | 20 | 0.752 ± 0.076 | 0.96 | 0 | 1.88 | 1.41 | 1.56 | w161-ah | −0.066 [−0.119, −0.015] | −0.60 | +14 ns | 0.693 |
| w161-cgate-afb (method 3) | 20 | 0.768 ± 0.051 | 0.89 | 0 | 1.93 | 1.41 | 1.73 | w161-ah | −0.050 [−0.096, −0.004] | −0.55 | +12 ns | 0.722 |
| w161-tg20 (hard) | 20 | 0.770 ± 0.074 | 0.99 | 0 | 2.18 | 1.54 | 1.60 | w161-ah | −0.048 [−0.096, −0.004] | −0.30 | **+56, p=0.001 → veto** | 0.712 |
| w161-tg30 (hard) | 20 | 0.784 ± 0.053 | 0.89 | 0 | 2.15 | 1.54 | 1.62 | w161-ah | −0.034 ns | −0.32 | **+73 → veto** | 0.727 |
| w161-tg10s40 (round-6 best) | 20 | 1.004 ± 0.683 | 3.07 | **2** | 2.54 | 1.88 | 1.91 | w161-ah | +0.186 ns | ns | +79 | 0.798 |
| w161-tg10s40-cgate-afb | 20 | 0.770 ± 0.079 | 1.01 | 0 | 1.87 | 1.38 | 1.76 | w161-ah | −0.047 [−0.086, −0.002] | −0.60 | **+47 → veto**; r49 +0.07 [+0.01, +0.15] | 0.729 |
| w161-fiz (input-col null) | 20 | 0.775 ± 0.070 | 1.02 | 0 | 2.42 | 1.66 | 1.59 | w161 | −0.011 ns | ns | ns | 0.746 |
| **w161-fip** (method 5, phase flag) | 20 | **0.733 ± 0.044** | 0.89 | 0 | 2.39 | 1.64 | **1.43** | w161-fiz | −0.042 [−0.079, −0.008] | −0.02 ns | **−34, p=0.001** | 0.705 |
| w161-fic (case flag) | 20 | 0.786 ± 0.069 | 0.99 | 0 | 2.60 | 1.80 | 1.51 | w161-fiz | +0.011 ns | **+0.18 [+0.12, +0.24] worse** | −17 | 0.742 |
| w161-fip-cgate-afb | 20 | 0.732 ± 0.038 | 0.84 | 0 | 1.70 | 1.30 | 1.66 | w161-fiz-cgate-afb | −0.077 [−0.201, −0.007] (null has 1 blow-up) | −0.16 | ns | 0.691 |
| w161-fiz-cgate-afb (its null) | 20 | 0.809 ± 0.270 | 1.94 | 1 | 1.86 | 1.40 | 1.76 | — | — | — | — | 0.722 |
| w1-8-1 / 1-10-1 / 1-12-1 / 1-8-2 | 12 | 0.810 / 0.846 / 0.820 / 0.794 | — | 0 | — | — | — | w161 | +0.035 [+0.00, +0.08] / +0.072 / +0.045 / +0.019 ns | — | — | 0.771 / 0.780 / 0.782 / 0.765 |
| w161-h256x3 | 12 | 0.823 ± 0.041 | 0.91 | 0 | 2.80 | 1.87 | 1.67 | w161 | +0.048 [+0.02, +0.08] | +0.36 | +62 | — |
| w161-h192x2 | 12 | 0.798 ± 0.063 | 0.96 | 0 | 2.52 | 1.71 | 1.67 | w161 | +0.024 ns | ns | ns | 0.765 |
| w161-h256x3-{ah, tg10s40, cgate-afb} | 12 | 0.821 / 0.885 (1 blow-up) / 0.847 | | | | | | h256x3-ah | −0.002 / +0.064 ns / +0.026 ns; cgate-afb r49 +0.16 [+0.05, +0.30] | | | |
| AR-w161 / AR-w161-cgate-afb | 4 | 1.121 ± 0.354 (1 blow-up) / 1.055 ± 0.319 | | | | | | r6 AR-pos_head 1.207 | −0.086 ns / −0.066 ns | | | descriptive |

Round-7 seeds match round 6's lists, so every cross-round pair is per seed.
Ensemble on all 20 seeds where available: cgate-ah **0.676** (f12 1.00, sw21
0.70), fip-cgate-afb 0.699, fip 0.701, tg30s30 0.703, w161 0.753.

### Decision under the pre-registered rule

1. **Winner: `w161-cgate-ah`.** Beats its 4-channel null by 0.099 with the CI
   clear of zero; f12 −1.39 (every one of the 20 seeds negative), sw21 −0.84;
   r49 unchanged (−0.01, CI [−0.11, +0.08]); jumps −14 (no increase); 0 seeds
   above 1.5 °C, worst 0.85. Simplicity clause closed directly: against the
   simplest arm, w161 itself, the paired difference is −0.068, CI [−0.103,
   −0.032] — no simpler arm is within noise. Against the other two survivors
   in its class it is also ahead (vs cgate-afb −0.049 [−0.087, −0.011]; vs
   tg30s30 −0.033 [−0.068, +0.006], within noise on the mean but −0.79 on the
   hard cases). Per case, cgate-ah roughly halves all 12 (e.g. case 32: 2.13 →
   0.77; case 53: 4.10 → 2.00); nothing gets worse.
2. **Pass, ranked below:** tg30s30 (−0.066, soft ramp adds no jumps),
   cgate-afb (−0.050), fip (−0.042 vs its input-column null), fip-cgate-afb
   (−0.077, but its null carries a blow-up, so the comparison is one-seed
   sensitive; vs the plain cgate-afb it is −0.036 [−0.063, −0.010]).
3. **Vetoed by jumps:** tg20 (+56), tg30 (+73), tg10s40-cgate-afb (+47, and a
   paired degradation on the 49). Hard per-timestep switches create the
   discontinuities Arnold flagged; the soft ramp (tg30s30) does not.
4. **Vetoed by blow-ups:** tg10s40 (2/20, both T_outer: 4.06 and 3.48),
   h256x3-tg10s40 (1/12).
5. **Weights closed.** T_outer weight above 6 is worse by the CI criterion
   (1-8-1 +0.035 [+0.001, +0.077]; 1-10-1 and 1-12-1 likewise); 1-8-2 within
   noise. 1-6-1 stays.
6. **Capacity closed.** h256x3 is worse than h128x2 under 1-6-1 (+0.048, CI
   clear of zero, and +62 jumps); h192x2 within noise and loses on
   simplicity; h256x3 with either gate does not beat its own null and
   cgate-afb degrades the 49. Round 6's "h256x3 is the best capacity point"
   was a 1-6-3 artefact.
7. **fic** (constant case flag) is within noise overall and makes the flagged
   cases worse (+0.18); the constant tells the network *that* a discharge
   comes but not *when*, and it hedges.
8. **S5** descriptive: AR-w161 1.12 vs the round-6 AR pair 1.21, n=4, within
   noise; the AR branch stays ~50% behind the direct model.

### What changed between the rounds, and why it matters

- The gate ranking **inverted** with the weight change. Under 1-6-3 the raw
  absolute 4th channel (cgate-ah) blew up in 2/20 seeds and the soft timestep
  gate (tg10s40) was the safe choice; under 1-6-1 cgate-ah has its worst seed
  at 0.85 and tg10s40 blows up in 2/20. Both failure modes were T_outer
  blow-ups (round-6 cgate-ah worst seeds: T_outer 3.10, 2.64; round-7
  tg10s40: 4.06, 3.48), so the weight lever, which stabilised T_outer, moved
  the stability boundary rather than the mechanism. Consequence: gate
  conclusions do not transfer across loss weights; every gate has to be
  re-measured with its null under the weights it will ship with.
- The two survivors work on **different cases**. cgate-ah vs fip: overall
  −0.015 (within noise), but cgate-ah wins the 21 switched cases by 0.78 and
  fip wins the 49 untouched ones by 0.25 (CI [+0.18, +0.32]). fip also cuts
  spurious jumps by 34 per seed and T_outer 0.448 → 0.420 across the board.
  The flag stacks with a gate (fip-cgate-afb beats cgate-afb by 0.036 with the
  CI clear of zero), so **fip × cgate-ah** is the obvious next arm.
- The phase flag's onset definition (first step after the peak plateau) was
  fixed before launch on a review finding; the earlier argmax version had
  fired 24–37% of the run early on plateau cases.

### S0 — Arnold's sanity check (2026-09-05 meeting)

- **Identity.** With every both-phase run removed from train, val and test
  (296/15/70 → 189/15/49) the gates cannot fire, and the gated arms reproduce
  their ungated twins **exactly**: xb-cgate = xb-base and xb-cgate-afb = xb-ah
  on all 10 seeds, all three channels, all 490 per-case rows (max |diff| =
  0.0000). The shared code is clean; the only thing that ever differed was
  what the gate did when it fired.
- **Training-set question.** Removing those runs from training makes the
  remaining 49 cases *worse*, like for like on the 49 (`sanity_compare.py`):
  xb-w161 vs w161 +0.563 ± 0.077 overall (worse on every one of 12 seeds,
  T_outer +0.86, T_avg +0.73, p = 0.001); xb-base vs the round-6 base +0.546 ±
  0.371 (p = 0.002), with 2/10 blow-ups against 1/20. The both-phase runs are
  36% of the training set and the model needs them; they are not
  contaminating the normal regime.

### Next (round 8, `[PLAN]`)

- **fip × cgate-ah** (the two survivors combined), with its nulls fiz ×
  cgate-ah and cgate-ah itself; 20 seeds. Expected to take the 49 from fip and
  the 21 from cgate-ah at once.
- **Arnold's request (2026-09-10 meeting):** anchor T_outer on the live
  Input_T(t) the way T_inner is. Measured on the training set the residual
  is 97 °C against 116 absolute (−16%) and 58 for the round-4 initial-value
  anchor, and T_outer(0) = Input_T(0) exactly in every run, so the round-4
  anchor already was "Input_T at t = 0"; the expectation is a null or a loss,
  but it is cheap and it settles the question by measurement. One arm plus a
  lead-shifted variant, 20 seeds.
- Confirm cgate-ah's ensemble as the deployment recipe (0.676 at 20 seeds)
  and hand the per-case plots to the shared drive.
- Drop: hard timestep gates, tg10s40 under 1-6-1, further weights, larger
  capacity, the constant case flag.

## 2026-09-06 — Round 7 `[CODE]` — combine the weight lever with the robust gates; tell the GRU which regime it is in

**In one line:** round 6's two separate wins — the loss weight and the two
safe gates (methods 3 and 4) — have never been used together, so this round
stacks them; it also tries feeding the model the regime flag as an input
(method 5) and runs the check Arnold asked for on 2026-09-05.

Where round 6 left us: the loss weight was the lever (1-6-1: 0.774 ± 0.02 over
12 seeds vs 0.973 for 1-6-3, monotone in T_avg's weight, T_outer 0.70 → 0.44
doing the work); the gates that repair the 12 excursion cases without damaging
the other 58 are tgate10-soft40 (0.822, 0/20 blow-ups), cgate-afb (0.841, 0/20)
and cgate-ah (0.948, 2/20), all measured under 1-6-3; floors and learned gates
are dead; h256/L3 is the best capacity point (0.845 under 1-6-3). The
2026-09-05 meeting added two things: Arnold's sanity protocol (drop every
both-phase run from train and test, rerun, see whether the rest recovers), and
the diagnosis that the shared-channel cgate failed because the GRU is never
told which regime a step is in (the 49 never-switched cases went 1.90 → 7.90).

**Code (`tes_gru`)**
- `CASE_FLAG_INPUT` = `case` | `phase` | `zero` (config/data): one extra
  forward_direct input column derived from the input-temperature profile
  alone, appended last so Input_T stays at idx 1. `case` = the per-run
  detector flag; `phase` = 1 from the first step after the input-temperature
  peak plateau (the steps from the maximum onward within 1% of the span), i.e.
  once the discharge has begun — argmax alone flipped 24–37% of the run early
  on profiles that hold their top temperature (review finding); `zero` = a
  constant column, the null for the init-RNG shift an extra input causes.
  Tag `fic`/`fip`/`fiz`. Asserted forward_direct-only.
- `EXCLUDE_BOTH_PHASE` (config/data): removes every run the detector fires
  on from train, val and test before the scaler is fitted (296/15/70 →
  189/15/49). Tag `xb`. `export_results` files these under `sanity_xb_*`
  families, keeps them out of the ranking / plots / "(cur best)", writes them
  to `summary/sanity_xb_reduced_testset.csv`, and adds a `test_cases` column.
- `data.input_case_flag_np` / `input_phase_flag_np`: numpy twins of
  `rollout._input_case_gate`. `run_round7.py` runs the twin check in a child
  process (BASE env pinned, CUDA hidden) over every train / val / test run,
  comparing the raw-column numpy verdict with the torch verdict on the scaled,
  last-row-dropped X the gates actually see; a second child under
  EXCLUDE_BOTH_PHASE=1 asserts no retained run fires. Refuses to launch on any
  disagreement (dev box: 381/381 agree; test 21 fire; retained 189/15/49).
- Provenance keys `case_flag_input`, `exclude_both_phase` (runio); export
  naming and README legend updated.
- `ensemble_eval.py`: per configuration, average the per-seed predictions
  case by case and score the average (the deployment recipe; round 6 gave
  0.774 → 0.742 for w1-6-1). Per-channel layout self-check against meta.json;
  `--seeds` for a common n across stages; `--subset-size/--subset-reps` for a
  subset sd; T_avg on the 12 flagged / 21 switched / 49 untouched cases.
- `sanity_compare.py`: two configurations on their common test cases, paired
  by seed, with a sign-flip p — the reading of S0 (see below).

**Runner `run_round7.py`** (~27 h direct + ~40 h AR, all stages default)

| stage | arms | seeds | question |
|---|---|---|---|
| S0 | xb-base, xb-ah, xb-cgate, xb-cgate-afb (1-6-3); xb-w161 | 10; 12 | code check: xb-cgate == xb-base and xb-cgate-afb == xb-ah seed by seed (the gate is inert; a 4th channel shifts the init RNG, so afb pairs with ah, not base). Training-set question via sanity_compare: xb-base vs round-6 base, xb-w161 vs w161, on the 49 retained cases |
| S1 | w161, w161-ah, w161-tg10s40, -tg20, -tg30, -tg30s30, -cgate-afb, -cgate-ah, -tg10s40-cgate-afb | 20 | how much gate gain survives the weight change, and which gate; ah = 4-channel null |
| S2 | w1-8-1, w1-10-1, w1-12-1, w1-8-2 | 12 | T_avg pinned at 1, is more T_outer weight still better |
| S3 | w161-fic, -fip, -fiz, -fip-cgate-afb, -fiz-cgate-afb | 20 | does telling the GRU the regime beat switching it from outside; fiz = input-column null, fiz-cgate-afb = the combo's null |
| S4 | w161-h256x3, -h192x2, -h256x3-ah, -h256x3-tg10s40, -h256x3-cgate-afb | 12 | capacity under the new weights, with the two robust gates and their null |
| S5 | AR-w161, AR-w161-cgate-afb | 4 | does the lever transfer to the autoregressive branch (descriptive) |

**Decision rule (pre-registered, full text in the runner docstring)**: every
arm is paired per seed against its own null (same input and output channel
count): S2 vs w161, 4-channel gates vs w161-ah, fic/fip vs w161-fiz,
fip-cgate-afb vs fiz-cgate-afb, h256x3 gates vs h256x3-ah. Primary = paired
overall-MAE difference with an exact sign-flip p and 95% CI, arms ranked;
"within noise" = CI includes 0. Secondary = T_avg on the 12 flagged cases, plus
the 21 switched and the 49 untouched, with no paired degradation on the 49.
Veto = any seed > 1.5 °C on the common seeds, or a paired increase in
JumpSteps_gt2C with p < 0.05. Deployment metric = seed-ensemble MAE at a common
n (--seeds S12) with a subset sd. Simplicity order: 3-ch < 4-ch, no flag <
flag, h128x2 < larger, one gate < two.

**Dev-box smoke runs (2026-09-06)**: one seed (7) each of `w161-fip` and
`xb-cgate-afb` through the runner end to end — done.flag, npz, provenance all
correct; fip 0.708 overall / T_avg 1.43 (single seed, before the phase-onset
fix; that run dir was deleted so w161-fip starts clean), xb-cgate-afb 1.084 on
49 cases (kept; resumes as seed 7). Exit code 0xC0000409 after DONE is the
known Windows CUDA teardown crash.

**Expected**: S0 pairs bit-identical (anything else is a shared-code bug);
S1 w161 ≈ 0.77 at n=20, gates adding at most a few hundredths on the mean but
halving the 12-case T_avg; S3 `fip` is the arm to watch — if the phase flag
lets the GRU handle the excursion itself, it should beat every output-side
gate on the 12 cases without the `ah` penalty; S4 h256x3 × w161 ≈ 0.70.

## 2026-08-26 — Round 6 results `[DONE]` — the lever was the loss weight, not the gate

**In one line:** the round set out to fix the 12 hard cases and found a
bigger win elsewhere — counting `T_avg`'s error for less during training
improved everything (0.97 → 0.77 °C). Of the four ways to handle the hard
cases, two are safe (methods 3 and 4), one is unstable (method 2) and one
wrecks the other cases (the shared-channel arm).

916/916 runs, 55 configs, nothing lost. Base (pos_head h128x2) re-run fresh
on the lab server reproduced round 5 seed-for-seed to 2 decimals — the direct
path is deterministic across machines.

### Headline: T_avg loss weight 3 → 1 is the biggest single gain of the project

| weights (Ti/To/Ta) | n | overall | sd | T_outer | T_avg | blow-ups |
|---|---|---|---|---|---|---|
| **1-6-1** | 12 | **0.7745** | **0.022** | **0.436** | **1.605** | 0/12 |
| 1-6-0.5 | 12 | 0.799 | 0.114 | 0.469 | 1.650 | 0/12 |
| 1-8-3 | 12 | 0.916 | 0.250 | 0.608 | 1.842 | 1/12 |
| 1-6-3 (champion) | 20 | 0.973 | 0.453 | 0.701 | 1.933 | 1/20 |
| 1-6-6 | 12 | 1.617 | 0.678 | 1.592 | 2.912 | 4/12 |
| 1-6-10 | 12 | 1.685 | 0.436 | 1.715 | 3.008 | 9/12 |

Monotone dose-response over six settings, and the tightest spread of any
configuration ever measured here (sd 0.022). Mechanism: under pos_head,
T_avg = T_outer + pos·gap, so T_avg accuracy is DOWNSTREAM of T_outer. The
two heads compete for the shared trunk; a heavy T_avg weight starves T_outer
(0.70) and thereby T_avg itself. Dropping it to 1 improves BOTH (T_outer
0.70 → 0.44, T_avg 1.93 → 1.61). Round 5's "1/6/3 is optimal" was true under
the old parametrization only — exactly the concern that put S4 in this round.
Needs 20-seed confirmation (n=12), but the dose-response makes a fluke
unlikely. 12-seed prediction ensemble: **0.742**.

### Gating (S1, 34 arms × 20 seeds) under the pre-registered rule

**Primary (overall MAE, paired vs base 0.973 ± 0.453):**

| arm | overall | sd | worst | excursion-case T_avg | jumps |
|---|---|---|---|---|---|
| tgate10-soft40 | **0.822** | 0.067 | 0.98 | 1.65 | 1409 |
| tgate30 (hard) | 0.831 | 0.105 | 1.18 | 1.65 | 1450 |
| cgate-afb | 0.841 | 0.127 | 1.20 | 1.42 | 1419 |
| tgate20-soft20 | 0.851 | 0.130 | 1.25 | 1.73 | 1401 |
| cgate-ah | 0.948 | 0.528 | 2.87 | **1.30** | 1410 |
| base | 0.973 | 0.453 | 2.87 | 2.09 | 1407 |

tgate10-soft40 is the only arm significant vs base (paired sign-flip
p = 0.024) — with 31 selection arms that p is not to be taken at face value;
its real credential is **zero blow-ups and sd 0.067 against base's 1/20
blow-up (seed 17 → 2.87) and sd 0.45**. Nineteen arms had zero blow-ups;
the gates buy robustness more than they buy mean.

**Secondary (20 excursion cases):** the case-gate family wins outright —
floor40-cgate-ah 1.05, cgate-ah 1.30, cgate-afb 1.42 vs ~1.65–1.75 for the
timestep gates and 2.09 base. On the 12 originally flagged cases cgate-ah
cuts T_avg from 2.93 to **1.45** (−50 %) but pays on the other 58 (1.87 vs
1.73); tgate10-soft40 improves both (2.25 / 1.55). By median cgate-ah is
actually the best arm (0.761, 15/20 seeds beat base) — but it carries one
catastrophic seed.

**Tertiary (jumps):** no gate materially increased jumps except cgate
(2150); hard timestep gates sit at 1440–1460 vs base 1407. The pre-registered
fear that hard gates manufacture discontinuities did **not** materialize.

**Verdicts on the pre-registered questions:**
- *Timestep gate beats case gate?* (my prediction) — **not confirmed.** Different
  profiles: case gate has the better median and far better excursion cases,
  timestep gates are more robust. A draw, not a win.
- *Anchored fallback (afb)?* — **mixed.** Helps the case gate (cgate-afb 0.841,
  zero blow-ups, vs cgate-ah 0.948 with one), hurts the timestep gate
  (tgate30s30-afb 0.961 vs 0.899). The 0.778 smoke number was one lucky seed.
- *Learned gates?* — **dead.** Every variant (per-step, pooled, three biases)
  is worse than its own inert 5-channel null (1.08): 1.19–1.26.
- *Dual-use channel (Arnold's proposal as literally stated)?* — **catastrophic
  and now measured:** cgate 2.18 ± 0.05, T_avg 5.86, 20/20 blow-ups; with a
  dedicated fallback channel (cgate-ah) 0.948. The review finding was right.
- *Floors?* — **dead.** floor10/20/80 ≈ base, floor40 worse (1.06), soft
  floors worse still. Reconditioning without a handover does nothing.
- *Clean fit?* — no effect (0.944 vs 0.973).
- *Channel-count controls:* 4 channels −0.03 vs base (noise); 5 channels +0.11.

### S3 / S5

Under pos_head, **depth 5 hurts**: h128/L5 1.10, h256/L5 1.68 ± 2.41 (blow-ups);
2–3 layers fine, h256/L3 0.845 ± 0.056. Round 5's "architecture doesn't
matter" held only for the old parametrization.

AR pair (n=4): AR-pos_head 1.207 ± 0.638 vs AR-base 1.434 ± 0.208. The
position idea transfers (T_avg 3.08 → 2.04) but is unstable in the AR loop and
the direct model remains ~40 % better. AR stays the scenario branch, not the
accuracy contender.

### Still true

Case 40/54 first-step MaxErr unchanged (8.4–8.6) across every arm — a
single-timestep artifact no gate touches, as predicted.

### Round 7 (`[PLAN]`)

Everything points one way: **combine w1-6-1 with the robust gates**. Confirm
w1-6-1 at 20 seeds; w1-6-1 × {tgate10-soft40, tgate30, cgate-afb, cgate-ah};
a weight sweep around it (1-6-0.5 / 1-6-2 / 1-8-1 / 1-10-1); h256/L3 × w1-6-1.
Drop floors and learned gates. ~12 arms × 20 seeds ≈ 10 h. Deployment recipe
= seed ensemble (0.742 already, likely ~0.70 with the combination).

## 2026-08-22 — Round 6 `[CODE]` + adversarial design review

Round 6 asks one question: WHERE should the position parametrization stop
being trusted? Four mechanism families in `run_round6.py` S1 (34 arms x 20
seeds): denominator floor (hard/soft), per-timestep temperature gate
(hard/ramped), per-case gate (Arnold's both-phases rule), and a learned gate
(per-step and per-case pooled). Plus S3 arch / S4 weights re-scans under
pos_head and S5, the AR transfer pair. 55 configs / 916 runs / ~73 h.
Decision rule pre-registered in the runner docstring (primary overall MAE
paired vs base; secondary excursion-case T_avg; tertiary a spurious-jump
veto), because a best-of-31 selection at n=20 otherwise overstates its edge.

Before launch the whole setup went through an adversarial review (3 lenses +
independent checks). What it caught, all fixed:

- **AR path never applied pos_head** — `apply_other_anchor` was wired only
  into forward_direct; `_rollout_sliding` and evaluate's AR branch fed the
  raw position z to the loss AND back into the AR window. S5 would have
  burned ~40 h comparing two effectively identical arms. Wired in per step
  (reconstructed T_avg is what feeds back), unit-verified exact.
- **Stale done.flag dirs** — a pre-round-6 smoke run (`_tg30s30_seed7`) would
  have been silently absorbed as seed 7 of an arm. Deleted, and the runner
  now refuses to launch if any round-6-tagged dir carries pre-round-6
  provenance.
- **Exporter merged the S5 pair** — `family()` bucketed every abs_sliding run
  as "AR" regardless of mode, so AR-pos_head and AR-base averaged together
  and overwrote each other's per-seed files. Fixed (`AR_pos_head` family).
- **Env leakage** — the runner passed the parent shell's environment through,
  so a leftover `set POS_CASE_GATE=1` would silently gate every arm. All 15
  knobs now pinned in BASE.
- **Confounded controls** — 5-channel learned arms had no channel-count null
  (added `learned-b10`: bias pins the gate inert); `floor40-cgate` was the
  only cgate combo without the dedicated fallback channel (now `-ah`).
- **Decision metrics did not exist** — the jump veto and excursion-case
  aggregate were prose only, and predictions.npz is not retained for most
  arms after export. The jump metric (`MaxJump (C)`, `JumpSteps_gt2C`) now
  lands in every summary_errors.csv at evaluation time.
- **Two new mechanism arms from the review**: `*-afb` — gates hand over to a
  surface-anchored fallback (midpoint + z*sd, fitted) instead of the raw
  absolute head this project has refuted three times (midpoint rather than
  the proposed max+softplus, because gated regions also contain
  between-surface steps a >=max fallback cannot express); `base-cleanfit` —
  excursion steps excluded from the mu/sd fit (they inflate sd ~34%).
  Cut in exchange: floor120, tgate80, floor80-tgate20s20 (operate far
  outside the failure regime, median excursion gap is 7 C).
- **S5 rebooked from measurement**: round 4 ran the same AR config at
  ~4.25 h/seed, not the folklore 18 h — so 4 seeds/arm now fit in ~40 h and
  n=2 (below the project's own evidence bar) is avoided.
- **Case-gate detector validated on TRAINING data** (was test-only): fires on
  107/311, catches 105/105 excursion cases, 2 false fires. 34% of training
  cases carry excursions, so the fallback path gets real training signal.
- **Free win measured**: averaging the 10 round-5 champion seeds' predictions
  gives overall MAE 0.829 vs 0.890 single-seed mean (T_outer 0.56 -> 0.43),
  zero training cost. The deployment recipe should be an ensemble.

Known, accepted: S3/S4 still run under the UNGATED head (they are
null-checks; any n=12 winner needs confirmation), and S5 uses the plain
position head — if S1 names a gate, the AR pair is worth re-running with it.

## 2026-08-12 — Round 5 results `[DONE]` — **the position head wins**

384/384 runs complete, 36 configs, nothing lost this time (the `anchor_avg_grad`
control ran, confirming the config fix).

### Headline: `pos_head` is the best configuration the project has produced

| config | n | overall MAE | worst run | runs >1.5 °C |
|---|---|---|---|---|
| **pos_head h128×2dp0.3** | 10 | **0.890 ± 0.077** | 1.047 | **0/10** |
| pos_head h128×5dp0.3 | 10 | 1.072 ± 0.368 | 1.816 | 2/10 |
| baseline h128×5 (previous best) | 20 | 1.284 ± 0.236 | 1.670 | 5/20 |
| AR baseline (arm B, 6 seeds total) | 6 | ~1.52 | — | — |

It is not only the lowest mean but by far the **tightest** distribution ever
measured here (sd 0.077 vs 0.24–0.54 for everything else) — and the first
configuration with **zero** runs above 1.5 °C. Against the autoregressive
baseline that started this line of work, that is a **~41 % improvement**.

Per channel (h128×2): T_avg **2.996 → 1.829** (permutation p = 0.0000),
T_outer **0.812 → 0.561**, T_inner unchanged (0.291 → 0.280, by design — the
position head does not touch it). Per case it wins on **56/70** for T_avg and
**66/70** for T_outer.

**On the cases flagged in review** — T_avg MAE:

| group | default h128×5 | pos_head | change |
|---|---|---|---|
| **late-drift group** (58-65, 1-4, 8-10, 19, 23, 27) | 2.86 | **0.80** | **−72 %** |
| start-jump (40, 54) | 3.39 | 2.65 | −22 % |
| "odd initial value" (41-45, 48-51) | 2.34 | 1.84 | −22 % |

The late-drift group — the one Arnold and I kept returning to and that three
previous attempts failed on — is essentially solved. The diagnosis that
preceded it (error = position error × surface gap, with 36 % headroom in the
per-case position) predicted this, and the measured gain lands where predicted
(T_avg 2.42 → 1.83 against an oracle of 1.54).

A minority of cases got worse (67: 2.89 → 3.74, 33: 2.69 → 3.62, 35: 2.88 →
3.39); case 70 remains the worst overall (6.31). Worth a look before the
formulation is frozen.

### Two more refutations

**The round-4 architecture win did not replicate.** h128×2 measured
1.114 ± 0.181 at n=6; at n=20 it is **1.366 ± 0.307**, and against the
5-layer default p = 0.35 — indistinguishable. h256×5dp0.1 likewise
(1.343, p = 0.76). So that "capacity is the only lever that worked"
conclusion was an n=6 false positive, and the honest read is that within this
grid **architecture barely matters**; the capacity grid (T2) spans only
1.135–1.483 with overlapping error bars. This is the clearest argument yet
for not trusting anything below ~15 seeds in this project.

**`ANCHOR_SCALE` did nothing for Case 40** — MaxErr_T_inner 8.42 (as=1) →
8.57 / 8.60 / 8.60 (as=3/6/12), with or without lookahead. So the residual is
**neither an information limit nor a range limit**: at as=12 the required
−8.5 °C correction is only ~1.1 σ, comfortably reachable, and the model still
does not make it. The remaining explanation is that it is a *single timestep
out of 1440* with no meaningful weight in the loss — Case 40's case-level
T_inner MAE is 0.23 °C, i.e. excellent. It is a visible artifact on a plot
rather than a modelling failure, and closing it would require explicitly
weighting the first steps.

### Anchor family, final word

`anchor_avg_grad` (the control the config bug killed in round 4) came in at
1.655 ± 1.318 with a worst run of 4.97 — matching the detached variant's
instability. The whole fixed-reference anchor family for T_outer/T_avg is
closed: it loses on the mean and is wildly unstable. `pos_head` succeeds
precisely because it does the opposite — it predicts the blend weight rather
than fixing it.

---

## 2026-08-09 — Round 4 results `[DONE]` + two fixes + Round 5 `[CODE]`

300 runs launched, **288 usable** (see the config bug below). All on the
fixed 70-case set, forward_direct unless noted.

### Three hypotheses tested; three refuted

**1. Anchors on T_outer / T_avg — settled, they lose.** At n=20 the
round-3 "0.87 best run" is exposed as a lucky tail:

| | n | overall MAE | range | runs >1.5 °C |
|---|---|---|---|---|
| **E (no anchor)** | 20 | **1.284 ± 0.236** | 0.89–1.67 | 5/20 |
| F (both anchors) | 12 | 1.930 ± 0.810 | 0.85–3.26 | 8/12 |
| H (T_avg anchor, detached) | 20 | 1.867 ± 1.200 | 0.81–**4.31** | 9/20 |

Permutation test H vs E: **p = 0.037**. Per channel the trade is plainly
bad — the anchors wreck T_outer (0.825 → 2.57/2.77) while T_avg barely
moves (2.761 → 2.708/2.731). Anchoring is dropped for T_outer/T_avg; the
T_inner anchor (a genuinely tight reference) stays.

**2. Inlet lookahead does NOT fix Case 40 — my hypothesis was wrong.**

| | Case 40 MaxErr | Case 54 | all-case EarlyMAE T_inner | overall |
|---|---|---|---|---|
| la=0 | **8.42** | 7.83 | 0.31 | **1.284** |
| la=1 | 8.38 | 7.71 | 0.32 | 1.641 |
| la=3 | 8.46 | 7.86 | 0.31 | 1.495 |
| la=5 | 8.49 | 7.84 | 0.33 | 1.755 |

The target metric is untouched while overall MAE degrades significantly
(p = 0.003). So the residual is **not** an information limit. It is a
**range limit of the anchor head**: with the fitted delta std ≈ 0.65 °C, the
8.5 °C correction Case 40 needs is ~13 σ — unreachable no matter what the
model knows. (This is the same "magazine size" argument used earlier to
explain why the anchor is safe in-distribution; it applies here too and I
should have seen it before spending the runs.) Round 5's `ANCHOR_SCALE`
tests this directly.

**3. Loss weights — Arnold's suspicion also refuted; 1/6/3 is already best.**

| weights | overall | T_avg |
|---|---|---|
| 1-3-3 | 1.655 | 3.49 |
| 1-4-4 | 1.485 | 2.94 |
| **1-6-3 (current)** | **1.284** | **2.76** |
| 1-6-5 | 1.453 | 2.95 |
| 1-6-6 | 1.445 | 2.95 |
| 1-6-8 | 1.551 | 3.18 |

Counter-intuitively, raising T_avg's weight makes **T_avg itself worse**.
Nothing to change here; the answer for Arnold is "tested, current ratio wins".

### What did work: capacity

| | L=2 | L=3 | L=5 |
|---|---|---|---|
| h64 dp0.3 | 1.427 | 1.394 | 1.339 |
| h64 dp0.1 | 1.316 | 1.431 | 1.252 |
| **h128 dp0.3** | **1.114 ± 0.181** | 1.454 | 1.284 *(default)* |
| h128 dp0.1 | 1.269 | 1.446 | 1.444 |
| h256 dp0.3 | 1.363 | 1.578 | 1.369 |
| h256 dp0.1 | 1.256 | 1.305 | **1.152 ± 0.110** |

**h128×2 is the leader** (p = 0.123 vs default — suggestive at n=6, not yet
proven), and capacity is also the **only** lever that improved T_avg
(2.76 → 2.42 at h128×2, → 2.32 at h256×5dp0.1). The 5-layer recipe was
tuned for the AR formulation; the direct model wants fewer layers.

AR baseline (arm B) on seeds 21/123: 1.597 → combined 4-seed ≈ 1.43, worse
than every direct configuration.

### Two fixes

- **`config.py` assert list was missing `anchor_avg_grad`** — it killed the
  round-4 G arm at import time on the runner (12 runs lost, the detach
  control). The list is now a named `_OTHER_OK` tuple with all five modes.
- **`ANCHOR_SCALE`** (new): multiplier on the T_inner anchor's z-scaling
  std, so the head can express corrections beyond a few σ. Folded into the
  run name (`_as<k>`) and `run_config.json`.

### Diagnosis — what T_avg's error actually IS (zero-cost, no GPU)

Before designing another T_avg fix, the round-4 data was mined for what makes
the persistently-bad cases bad. No single physical feature explains much
(strongest correlation: surface gap r = +0.36, swing r = +0.34), but
reparametrizing exposes the mechanism. Define the **position** of T_avg
between the two surfaces:

```
pos = (T_avg − T_outer) / (T_inner − T_outer)
```

- **Within a case, pos is essentially constant** — median std 0.037 over 187
  cases.
- **Between cases it genuinely varies** — 0.238 … 0.332, sd 0.027.
- The surface gap averages 78 °C (max 195), so a pos error of just **0.02
  costs 1.6 °C** of T_avg error (3.9 °C on the widest-gap cases).

So T_avg's error is a **position error amplified by the gap** — which is why
the worst cases (70, 42, 15, 5, 13, 6) are exactly the large-gap,
large-swing ones, and why extra loss weight never helped: the problem is
resolving one number, not spending more effort.

**Oracle ladder** (all using the TRUE surfaces, so these bound any
reparametrization), T_avg MAE on the 70-case test set:

| | MAE | worst case |
|---|---|---|
| fixed global blend w = 0.289 *(the round-4 anchor)* | 2.41 | 5.42 |
| **model as it stands (predicts T_avg directly)** | **2.42** | — |
| **per-case optimal constant pos** | **1.54** | 4.67 |

This is the retrospective explanation for the round-4 anchor failure: its
oracle ceiling (2.41) was already the model's own level, so a fixed w could
not buy anything no matter how it was trained. The headroom lives in the
**per-case** position — 2.42 → 1.54, about 36 %.

### Change K — T_avg position head (`OTHER_CH_MODE=pos_head`) `[CODE]`

Acting on the diagnosis: instead of predicting T_avg, the head predicts the
position and T_avg is reconstructed,

```
T_avg = T_outer + pos · (T_inner − T_outer),    pos = μ + σ · z_head
```

with μ = 0.272 and σ widened to 0.055 (2× the observed spread) so ±3σ spans
0.11 … 0.44, comfortably covering the 0.238 … 0.332 range the fixed w could
not follow. Reference channels are **detached** — the same rule the anchor
work established, so T_avg's objective cannot reshape the T_inner / T_outer
heads. Verified: reconstruction exact (maxdiff 0), pos stays in a sane band,
and T_avg's loss produces exactly zero gradient into the other two channels.

Distinct from the failed round-4 anchor in the one way that matters: there
the blend weight was a **fixed constant** and the head could only add a
scalar correction on top; here the head predicts the weight itself, which is
precisely the per-case degree of freedom the oracle says is worth 36 %.

### Change J — Round 5 (`run_round5.py`) `[CODE]`

34 configs / 364 runs / ~23 h, everything aimed at the open questions:

- **T1 confirm** — the two leaders + the default at **20 seeds** each, to
  turn p = 0.12 into a verdict.
- **T2 refine** — capacity grid around the winner: h {96,128,160,192} ×
  L {1,2,3} × dropout {0.2,0.3}, 10 seeds. Adds **L=1** (never tried) and
  dropout 0.2 (skipped in round 4).
- **T3 range** — `ANCHOR_SCALE` {3,6,12} × {la0, la1}, 10 seeds: the direct
  test of the Case-40 range hypothesis. Lookahead is paired back in because
  it may only pay once the head can express the correction it enables.
- **T4 gaps** — the G control that the config bug killed, plus 2 more AR
  seeds for a 6-seed baseline.
- **T6 position head** — `pos_head` on two architectures, 10 seeds each: the
  targeted attack on T_avg from the diagnosis above.
- **T5 champion** — `--champion H=..,L=..,DP=..,AS=..` at 20 seeds, run
  after reading T1–T3/T6 (deliberately not planned blind).

36 configs / 384 runs / ~24 h.

### Status of every case flagged by Arnold and in review

| flagged | targeted change | outcome |
|---|---|---|
| Case 40 / 54 bad start (−37 °C) | `ANCHOR_LEAD` (anchor on Input_T(t+1)) | **fixed** −37.3 → +8.5 |
| the remaining +8.4 °C | `INPUT_LOOKAHEAD` k = 1/3/5 | **failed** — 8.42 → 8.38/8.46/8.49, untouched; hypothesis was wrong (range limit, not information limit). Second attempt = `ANCHOR_SCALE`, untested |
| late drift on 58-65, 1-4, 8-10, 19, 23, 27 (T_avg) | physics bound / loss weights / fixed-w anchor | **all three failed**; only capacity helped (2.82 → 2.15 on that group), which is a global gain, not a targeted fix. Third attempt = `pos_head`, untested |
| "odd initial values" on 41-45, 48-51 | plot the t = 0 point | **resolved** — it was a plotting artifact (isothermal start), never a model error; that group is now the best of the three (1.96) |
| "weird jumps" (Arnold, from the plots) | traced to the anchor, not to dropping AR | **resolved by removing the anchor** — E showed 7 cases with spurious jumps vs G's 43 |

---

## 2026-08-06 — meeting follow-ups (Arnold)

### Verified: the "weird jumps" Arnold saw are caused by the ANCHOR, not by dropping AR

He compared plots of arm B (AR) against arm G (anchored, no AR), saw spurious
step discontinuities in G that B didn't have, and concluded B looked better
case-by-case. The observation is real, but the attribution needs correcting —
the right comparison is B against **E** (plain forward_direct, no anchor).
Counting cases with a single-step jump in the prediction larger than the
truth's own step by >2 °C, seed 42:

| arm | cases with spurious jump >2 °C | >5 °C | worst |
|---|---|---|---|
| B (AR) | 8 | 3 | 9.3 °C |
| **E (direct, no anchor)** | **7** | **0** | **3.9 °C** |
| G (direct + T_avg anchor) | **43** | 3 | 10.1 °C |

So removing autoregression does **not** introduce jumps — E is the cleanest of
the three. The jumps come from the T_avg blend anchor, and they land exactly
where the mechanism predicts: in G they appear in **T_avg (39 cases) and
T_outer (40 cases)** while T_inner is unaffected (4 cases, same as E). T_avg
is reconstructed from T_outer, so any step artifact in T_outer is copied
straight into T_avg. This is more evidence that the anchored arms are not
ready, independent of the seed instability.

### Confirmed: the T_avg physics constraint is physically WRONG, not just weak

Arnold: "you shouldn't apply that, because for some cases, for some period,
it's actually above both surfaces." Checked on the 70-case test set: T_avg
exceeds **both** surface temperatures at **1.24 %** of timesteps, across
**21/70 cases**, by up to **13.2 °C** (Case 36). It never falls below both.
So the earlier finding that the hinge "fires on only 3.3 % of steps and hurt
performance" was understating the problem — a fraction of those firings were
penalising physically correct predictions. The constraint stays off, and
should not be revisited for the convection data either without re-checking
this first.

### Change I — Round-4 sweep set up (`run_round4.py`) `[CODE]`

One driver covers every open ask, exploiting forward_direct's ~4 min/seed
(planned ~18 h of the ~3-day budget; SEEDS env widens any stage):

- **S1 stability** — E/F/G/H × 12 seeds: replaces the 2-seed coin flips on
  the anchor bimodality with real distributions.
- **S2 lookahead** — new `INPUT_LOOKAHEAD=k`: forward_direct gains k extra
  input columns carrying Input_T(t+1..t+k) (legal — the curve is a given
  boundary condition; tail-held). The targeted Case-40 fix: the model can
  finally SEE the inlet jump coming and predict the physical lag. k ∈ {1,3}
  × 12 seeds.
- **S3 weights** — Arnold's ask: T_avg emphasis grid w ∈ {1-6-5, 1-6-6,
  1-4-4, 1-3-3} + one interaction config (la1 + 1-6-6), × 6 seeds.
- **S4 arch** — `HIDDEN_SIZE`/`NUM_LAYERS`/`DROPOUT` now env-overridable;
  grid h ∈ {64,128,256} × L ∈ {2,3,5} × 4 seeds (control h128×5 = S1's E).
  The 5×128 recipe was tuned for AR; the direct formulation may want less.
- **S5 AR baseline** — arm B × seeds {21,123}, completing its 4-seed number.

All stages run with post-meeting defaults: physics bound OFF everywhere
(Arnold: the constraint is physically wrong), TINNER anchor + lead 1.
Run-name scheme rebuilt (short tags, varying knobs only) — the old scheme
plus new tags was approaching the Windows 260-char path limit (now worst
case 229). Verified: lookahead columns match shifted Input_T exactly, arch
overrides fold into names, end-to-end forward passes with la=1 + h64×2.

### Open asks from the meeting

1. **Raise the T_avg loss weight.** Arnold raised it twice: the current
   Ti/To/Ta = 1/6/3 came from the LSTM line, and he suspects the ratio is not
   optimal for us — T_avg is our weakest channel and is weighted below T_outer.
   Worth a small sweep (e.g. Ta = 3 → 5/6) once the seed question is settled. `[ ]`
2. **Case 40's residual error is consistent across seeds** — he expects it to
   persist with more seeds, so it needs a targeted fix rather than more
   averaging. (Current state: first-step error +8.4 °C, which is the physical
   T_inner lag ceiling; going below it would require feeding the inlet's
   near-future, which is available but not currently used.) `[ ]`
3. **More seeds for F/G/H** before drawing any conclusion (our own plan; he
   agreed but pushed to look case-by-case rather than at averages). `[ ]`
4. **The 0.29/0.71 blend coefficient is geometry-specific.** He flagged that
   all 311 training cases share one system geometry and size, so the fitted
   relationship will not transfer to a different geometry — a new one has to be
   refitted per scenario. Noted as a hard constraint on the anchor approach; the
   planned gate/branch structure is where this would be handled. `[ ]`

---

## 2026-07-25

### Results — 5-arm fix ablation `[DONE]` → **arm B wins; new project best 1.27 °C**

10/10 runs complete (5 arms × seeds {7,42}), all configs verified, fixed
70-case set. Means over the two seeds:

| metric | arfed (07-16) | anchor (07-21) | **A** lead | **B** +wts | **C** +bound | **D** persist | **E** direct |
|---|---|---|---|---|---|---|---|
| **Overall MAE** | 1.93 | 1.39 | 1.70 | **1.27** | 1.44 | 2.51 | 2.10 |
| MAE T_inner | 1.08 | 0.38 | 0.23 | 0.25 | 0.27 | 0.20 | 2.33\* |
| MAE T_outer | 2.35 | 1.19 | 1.35 | **0.77** | 0.83 | 1.94 | 1.11 |
| MAE T_avg | 2.35 | 2.62 | 3.53 | 2.79 | 3.21 | 5.40 | 2.87 |
| MaxErr T_inner | 10.73 | 2.22 | 0.92 | 0.95 | 0.98 | 0.80 | 10.07\* |
| R² overall | 0.9963 | 0.9965 | 0.9954 | **0.9970** | 0.9958 | 0.9916 | 0.9931 |
| infer ms/case | ~560 | ~560 | 583 | 582 | 578 | 590 | **4.0** |
| train h/seed | ~5 | ~3 | 3.7 | 4.2 | 4.2 | 2.2 | **0.1** |

\* E lacked the anchor — see the confound note below.

**1. The `ANCHOR_LEAD` fix worked exactly as predicted.** Case 40's first-step
error went **−37.3 → +8.5 °C**, Case 54 **−34.0 → +7.7 °C** — landing right on
the predicted residual (the physical T_inner-lag ceiling is 8.5 °C, so the
anchor cannot do better without also seeing the inlet's future). MaxErr
T_inner across all cases: 2.22 → 0.92. This closes Arnold's "why do some cases
start badly" question end-to-end: cause identified (r = 0.999 with the inlet
jump), fix implemented, magnitude predicted, result matches.

**2. Loss re-weighting is the single biggest win** (A → B: 1.70 → 1.27).
T_outer improved 3× versus the pre-fix baseline (2.35 → 0.77). Confirms the
diagnosis that a uniform loss was spending a third of its gradient budget on
the already-solved T_inner channel. Adopting the LSTM line's 1/6/3 was right.

**3. The physics bound is HARMFUL — default flipped to OFF.** Arm C scored
1.44 vs B's 1.27, and T_avg (the channel it targeted) got *worse* (2.79 →
3.21; late bias +3.01 → +3.31). Mechanism: the bracket |T_inner − T_outer|
averages 79 °C while the bias is ~2.5 °C, so the hinge fires on only 3.3 % of
steps — a rare spiky gradient that perturbs optimization without ever touching
the in-bracket level offset. `PHYSICS_BOUND_WEIGHT` now defaults to 0 (kept
available for convection data, where near-isothermal starts make the bracket
genuinely tight).

**4. Arnold's persistence suggestion is empirically refuted (arm D).** Overall
2.51 (worst arm), T_avg 5.40, and the T_avg bias balloons to +5.17 mid-rollout.
Training plateaued almost immediately (best epoch 85–94 vs ~330–430 elsewhere).
This matches the predicted mechanism exactly: anchoring on the channel's *own*
previous prediction makes the rollout an integrator with no reset, so per-step
errors compound and a level offset can never be pulled back.

**5. T_avg's systematic offset is STILL UNSOLVED.** Every arm retains a late
bias of roughly +2.8 to +3.3 °C (B: +1.47 → +3.01 across the rollout). Neither
loss weighting nor the physics bound removed it. This is now the top open
problem — and notably, arm E (no AR feedback at all) has the *flattest* profile
and beats B on T_avg in **38/70** cases, which points at the AR feedback loop
as the offset's origin rather than at loss shaping.

**6. Arm E was confounded — my design error, now fixed.** The anchor was
implemented as `abs_sliding`-only, so `forward_direct` ran without it: its
T_inner (2.33) and Case-40 start (−35.8, i.e. unfixed) measured *the missing
anchor*, not the formulation. What the arm still shows, cleanly, is that on the
channels it could compete on, the seq2seq formulation holds up — T_outer 1.11,
T_avg 2.87 (better than C, and the best T_avg bias profile of any arm) — at
**4.0 ms/case inference (145× faster) and 0.1 h/seed training (40× faster)**.
The anchor now applies to `forward_direct` too (verified: reconstruction exact,
still a single forward pass, gated by `TINNER_MODE`), so a re-run gives the
fair formulation test. Seed variance is high (1.53 vs 2.68) and needs the extra
seeds before any conclusion.

### Results — arm E re-run WITH the anchor `[DONE]` → **seq2seq wins outright**

The fair formulation test (anchor now applies to `forward_direct`, same loss
shaping as B/C, same 70-case set, same seeds):

| | **B** (best AR) | **E** (seq2seq + anchor) |
|---|---|---|
| **Overall MAE** | 1.27 | **1.14** |
| T_inner | 0.25 | 0.26 |
| T_outer | 0.77 | **0.75** |
| **T_avg** | 2.79 | **2.40** |
| R² | 0.9970 | 0.9967 |
| per-seed spread | 1.35 / 1.19 (**0.16**) | 1.12 / 1.15 (**0.03**) |
| inference | 582 ms/case | **13.3 ms** (44×) |
| training | ~4.2 h/seed | **~4 min/seed** (60×) |

**The formulation, not the cell, was the gap.** With the anchor restored, the
exogenous-only single-pass model beats our best autoregressive configuration
on overall MAE while being 44× faster at inference and 60× faster to train,
and it is **5× more stable across seeds** (0.03 vs 0.16 spread). This is the
clean answer to "is the LSTM line ahead because it's an LSTM?" — no: it is
ahead because it does not feed its own predictions back.

**It also cracked the T_avg offset** — the one problem no arm had solved. The
late bias drops from +3.01 to +2.55 and the whole profile flattens (E is lower
at every decile), and E beats B on T_avg in **60/70 cases**. This confirms the
hypothesis from the 5-arm round: the offset originates in the AR feedback
path, which is why more loss shaping could never remove it. It is reduced, not
eliminated — T_avg remains the weakest channel and the next target.

**What AR still owns:** nothing measurable on this dataset. Both formulations
give the same Case 40 / 54 first-step residual (+8.4 / +7.6 — the physical
lag ceiling) and the same t=0 error to within 0.3 °C. AR's remaining
justification is scenario-based, not accuracy-based: closed-loop / streaming
settings where the full Input_T curve is not known in advance (Arnold's
solid+fluid coupling branch).

**Note on the exit code.** The runner reported `FAILED (3221226505)` =
`0xC0000409`, a Windows shutdown-time crash in CUDA teardown *after* both
seeds finished and wrote `done.flag`. Results are complete and valid; only the
process exit status is garbage. Worth a `--only` re-check of `done.flag` files
rather than trusting the runner's summary line on Windows.

### Next steps
1. ~~Re-run arm E with the anchor~~ `[x]` — seq2seq wins; see above.
2. Extend E to seeds {21,123} for a 4-seed number (~10 min) before reporting. `[ ]`
3. Report to Arnold: recommend `forward_direct` as the production formulation
   for the offline surrogate; keep AR+anchor as the causal branch. `[ ]`
4. Re-run the window-size sweep? Note W is irrelevant for `forward_direct`
   (no sliding window) — the sweep only matters if AR is retained. `[ ]`
5. Remaining T_avg offset (+2.55 late) is now the top accuracy target. `[ ]`
2. Adopt **B** as the working configuration (lead on, weights 1/6/3, bound
   off) — now the defaults. `[ ]`
3. Attack the remaining T_avg offset; arm E's flat profile suggests probing the
   AR feedback path rather than more loss shaping. `[ ]`
4. Extend the winner to seeds {21,123} for a 4-seed number before it goes in
   the report / to Arnold. `[ ]`

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

### Change G — arm E: `forward_direct` (formulation test) `[CODE]`

Comparing our numbers against the LSTM line conflates three things: the cell
(LSTM vs GRU — near-irrelevant), the **problem formulation** (their seq2seq
over exogenous inputs vs our 1450-step causal AR rollout — the big one), and
extras (bidirectional/attention/ODE features). Their formulation is
structurally immune to error accumulation and is legitimate for the offline
surrogate use case (the full Input_T curve IS the given boundary condition —
even MPC evaluates full candidate curves). Our 8-variant ablation never
tested a non-AR forward variant — a genuine blind spot.

New variant **`forward_direct`**: inputs = exogenous `[Time, Input_T]` only
(2-d), all three state channels predicted in ONE forward pass, zero AR
feedback, h0 still from the InitStateEncoder. `VARIANTS` is now
env-overridable; run name folds the variant tag. Added as **arm E** with
loss shaping identical to C, so formulation is the only variable vs C.
Single-pass training (no rollout loop) makes it the cheapest arm by far;
inference should be ~4 ms (like the non-sliding inverse variants) vs
~500 ms.

Interpretation guide: if E ≈/> C, the LSTM line's edge is the formulation,
not the cell — adopt seq2seq for the offline surrogate (with our multi-seed
error bars), keep AR+anchor as the causal branch for future closed-loop /
streaming scenarios in Arnold's framework. Verified: dataset (N,2)/(N,3)
shapes, one forward call per rollout (spy), TF no-op, default config
unchanged, runner dry-run correct.

### Next steps
1. Run `python run_fix_ablation.py` (arms A→B→C→D→E, ~24-26 h; `--only C`
   ~6 h, `--only D` for Arnold's variant, `--only E` ~1-2 h for the
   formulation test). `[ ]`
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
