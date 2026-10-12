"""Configuration, hyperparameters, and training-setup primitives.

Split out of the original GRU_input_ablation.py monolith. Holds every run
constant plus the small bootstrap helpers (set_seed, DEVICE, warmup_lr,
teacher_forcing_prob, apply_orthogonal_init_gru, is_inverse). Other modules
do `from .config import *`; runio/main also `from . import config` so they
can read/write the mutable RUN_NAME live.
"""
import os
import json
import random

import numpy as np
import torch
import torch.nn as nn

# Directory that CONTAINS this package (i.e. src/). Anchors the runs/ output
# dir and the relative data-root candidates regardless of which module file
# __file__ points at. config.py lives at src/tes_gru/config.py, so two
# dirnames up = src/.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# 1. Config
# ============================================================
# Seeds: the script now loops over ALL of these in one invocation, so a
# single `python GRU_input_ablation.py` produces 4 seeds x 8 variants = 32
# runs. Each (seed, variant) gets its own resume / done flag so a crash
# mid-experiment can pick up exactly where it left off.
# Env-overridable (comma-separated) so sweep drivers can run a 2-seed
# screening pass (e.g. SEEDS="7,42") without editing this file. Default is
# the full 4-seed protocol. Paired design: use the SAME seed list for every
# configuration being compared so per-seed differences cancel seed effects.
SEEDS = [int(s) for s in os.environ.get("SEEDS", "7,21,42,123").split(",") if s.strip()]
assert SEEDS, "SEEDS parsed to an empty list"
SEED = SEEDS[0]                 # legacy: kept for any code referencing it as a single-seed default
# Sliding-window length W (steps the GRU sees per rollout iteration in the
# sliding variants; also abs_window's WindowInitStateEncoder length).
# 2026-07-20: env-overridable for the window-size sweep (W in {5,10,20,50},
# per Yiming at the final review). One W per PROCESS -- W is baked into
# default args and star-import snapshots at import time, so it must never be
# mutated at runtime. Use run_window_sweep.py to sweep all values.
WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", "10"))
assert 1 <= WINDOW_SIZE <= 200, f"unreasonable WINDOW_SIZE={WINDOW_SIZE}"

# 2026-08-06: architecture env-overridable for the round-4 capacity sweep.
# The 5-layer/128-hidden recipe was tuned for the AR formulation; the direct
# (one-pass) formulation may want a different capacity, and Arnold flagged
# layers + hidden size as the first knobs to analyze. lr is intentionally NOT
# exposed (coupled with batch size; out of scope this round).
HIDDEN_SIZE = int(os.environ.get("HIDDEN_SIZE", "128"))
NUM_LAYERS = int(os.environ.get("NUM_LAYERS", "5"))
DROPOUT = float(os.environ.get("DROPOUT", "0.3"))
assert 8 <= HIDDEN_SIZE <= 1024 and 1 <= NUM_LAYERS <= 8 and 0.0 <= DROPOUT < 0.9
# Round 10 / H9 (2026-10-11): nn.GRU applies dropout BETWEEN layers only, so at
# NUM_LAYERS == 1 any DROPOUT is a no-op (models.py already passes 0.0). The
# run name still carried the requested value, so one-layer dp0.2 / dp0.3 runs
# were distinct directories holding the same model. Normalise here: every
# one-layer run is tagged dp0.0, and preflight_arm_configs catches the
# collision instead of launching a duplicate.
if NUM_LAYERS == 1 and DROPOUT != 0.0:
    print("NOTE: NUM_LAYERS=1 -> GRU dropout is a no-op; DROPOUT %g normalised to 0.0 (run name dp0.0)" % DROPOUT)
    DROPOUT = 0.0
_DP_TAG = "0.0" if NUM_LAYERS == 1 else "%g" % DROPOUT

LATEST_PARAMS = dict(
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    lr=0.0025,
    # 2026-07-16: epoch budget cut 1200 -> 800 and early stopping enabled,
    # based on ALL full-length (1200-ep) runs to date: across 36 completed
    # runs (8 variants x 4 seeds zero-pad baseline + abs_sliding x 4 seeds
    # init-pad), the best-val epoch NEVER exceeded 631 (abs_sliding
    # median ~460; most variants < 500). The final ~600 epochs of a
    # 1200-ep run have never produced a best checkpoint. Best-val
    # checkpointing is unchanged (still restored before testing), so
    # results stay comparable with the earlier 1200-ep runs.
    # (Arnold's earlier 2026-05-20 instruction was full 1200 ep / no early
    # stop -- that was before this best-epoch evidence existed.)
    # 2026-07-20: cap raised 800 -> 1000. The variable-pad run showed slower
    # convergence than the 1200-ep evidence base predicted (seed7 best at
    # epoch 748 hit the 800 wall); 1000 restores headroom while still saving
    # ~2x vs the old 1200-ep/no-ES protocol.
    # MAX_EPOCHS env override: smoke tests (a 2-epoch run to exercise a new
    # code path end to end), and since round 9 the cap itself. Any cap other
    # than 1000 is folded into the run name ("ep<N>"), so a smoke run can
    # never be resumed as a real seed and a run under a raised cap never
    # shares a directory with a 1000-cap one; the runners refuse to reuse a
    # finished dir whose cap or Total_Epoch is below 151 (unreachable for a
    # real run under patience 150).
    # 2026-09-19: under the stratified validation set (round 9, VAL_SPLIT)
    # the 1000 cap BINDS: in the dev-box pilot all three cgate-ah seeds and
    # one of three EMA-anchor seeds were still improving on validation at
    # epoch 1000 (best epoch 950-993; the plateau scheduler had the LR at
    # 1e-5 and the running best still moved 1-3% per 150 epochs). The old
    # validation set (15 linear-charging runs) went flat by ~600-800 epochs,
    # which is what the 1000 figure was sized to. run_round9.py therefore
    # pins MAX_EPOCHS=2000 for every arm, so "stop when validation stops
    # improving" keeps its meaning; the cost is ~1.5x per seed.
    max_epochs=int(os.environ.get("MAX_EPOCHS", "1000")),
    batch_size=16,
    # Stop after this many consecutive epochs without a val improvement.
    # 150 > 2 full ReduceLROnPlateau cycles (patience=50), so training
    # survives two LR halvings before giving up. Largest observed gap
    # from stagnation to a new best was well under this.
    early_stop_patience=150,
    sched_patience=50,
    sched_factor=0.5,
)

# ------------------------------------------------------------
# Teacher forcing schedule (per Arnold, 2026-05-20)
# ------------------------------------------------------------
# Probability of using GT (teacher-forced) instead of the model's own
# previous prediction for the per-step AR feedback. Linear decay from
# TF_INIT_PROB at epoch 0 down to 0 at epoch TF_DECAY_EPOCHS; held at 0
# thereafter. Apples to TRAINING only -- test/inference is always full
# autoregressive (tf_prob = 0).
TF_INIT_PROB = 0.5
TF_DECAY_EPOCHS = 100


# ------------------------------------------------------------
# Modern RNN training defaults (Pascanu 2013 / Chen 2019 / Liu 2019)
# ------------------------------------------------------------
# Added 2026-05-21 after literature review (Greff 2017 "LSTM Search Space
# Odyssey", Pascanu 2013 "On the Difficulty of Training RNNs", Chen 2019
# "Dynamical Isometry...GRU", Liu 2019 RAdam). These are the documented
# best-practice defaults for stacked GRU training -- our V1-V5 pipelines
# never had them, which likely explains the seed-dependent cliff at lr=0.0025.

# (1) Gradient clipping: cap global grad norm. 1.0 is the standard RNN
#     default. Set to None or 0 to disable.
GRADIENT_CLIP_NORM = 1.0

# (2) Orthogonal initialization on recurrent weight matrices (weight_hh).
#     Per Chen 2019, this is the unique stable init regime for stacked
#     GRU at depth >= 3. Input-to-hidden weights (weight_ih) keep
#     PyTorch's default Kaiming-uniform init.
ORTHOGONAL_INIT_RECURRENT = True

# (3) Linear LR warmup: ramp lr from 0 to LATEST_PARAMS['lr'] linearly
#     over the first LR_WARMUP_EPOCHS epochs, bypassing Adam's early-
#     step LR variance issue (Liu 2019). ReduceLROnPlateau only starts
#     adjusting LR AFTER warmup is complete. Set to 0 to disable.
LR_WARMUP_EPOCHS = 60


# ------------------------------------------------------------
# Sliding-window padding mode (abs_sliding / inverse_abs_sliding)
# ------------------------------------------------------------
# At rollout step t the GRU sees a window ending at t. For t < W-1 the oldest
# (W-1-t) positions have no real step. SLIDING_PAD_MODE decides what happens.
# The two live candidates (report Future Work) are "variable" and "init";
# "zero" is the rejected baseline whose data already exists (2026-05-25).
#   "variable" : NO padding -- feed only the t+1 real steps (a variable-length
#                window). At t=0 it degenerates to a single real step, so the
#                rollout is identical to `abs` there and the InitStateEncoder's
#                clean h0 is used uncorrupted; the window grows to W as t
#                advances. This is the "cleanest fix" proposed in the report
#                (our / the presenter's proposal at the final review).
#   "init"     : repeat the t=0 steady-state row for the missing positions. The
#                FE system sits at its initial temperature indefinitely before
#                the simulation starts, so the pre-history IS that steady state.
#                (Yiming's suggestion at the final review: pad with the initial
#                value, NOT zeros.)
#   "zero"     : legacy zero-padding. In MinMax-scaled space a zero is the
#                COLDEST temperature (~100 C), so it injects a spurious
#                cold-system signal that drifts h0 and produces the ~9 C t=0
#                undershoot. Rejected at the review (Yiming); kept only to
#                reproduce the 2026-05-25 seqsliding run exactly.
# Overridable via the SLIDING_PAD_MODE env var; RUN_NAME_BASE below folds the
# mode into the run-dir name.
# 2026-07-16: default flipped "init" -> "variable". The init sweep is DONE
# (inconclusive; see EXPERIMENT_LOG). The variable sweep is this round's
# deciding run, so a plain `python GRU_input_ablation.py` with NO env var now
# runs it -- last round the variable pass was skipped, likely because the env
# var was never set.
SLIDING_PAD_MODE = os.environ.get("SLIDING_PAD_MODE", "variable")   # "variable" | "init" | "zero"
assert SLIDING_PAD_MODE in ("variable", "init", "zero"), \
    f"SLIDING_PAD_MODE must be variable/init/zero, got {SLIDING_PAD_MODE!r}"


# ------------------------------------------------------------
# T_inner handling in the forward sliding variant (Arnold 2026-07)
# ------------------------------------------------------------
# Context: T_inner tracks Input_T within ~1 C in the data (gap mean 0.6 C,
# identical at t=0), yet the model's early-window T_inner error (1.9-2.7 C)
# is 5-8x WORSE than a zero-parameter "copy Input_T" baseline (0.34 C) --
# the model leans on its own AR-fed T_inner input instead of the clean
# exogenous Input_T, so small h0-transient errors copy themselves forward.
# Applies to the FORWARD `abs_sliding` variant only. One mode per PROCESS
# (baked into input dims / dataset columns at import time).
#   "arfed"       current 5-input recipe: T_inner is an input and its own
#                 predictions feed back (status quo / control arm).
#   "anchor"      inputs unchanged (5-d), but the T_inner head predicts a
#                 z-scored residual delta = T_inner(t+1) - Input_T(t); the
#                 rollout reconstructs T_inner = Input_T + delta. Hard-wires
#                 the tracking relationship; feedback error resets at every
#                 step from the GT anchor. (Same principle as the ODE anchor
#                 in Sid's LSTM.) NOTE: this is a cross-channel physical
#                 offset predicted at the OUTPUT -- not the (failed) delta
#                 time-difference INPUT features of the 8-variant ablation.
#   "output_only" v22-style: T_inner removed from the input (4-d:
#                 [Time, T_outer, T_avg, Input_T]); still predicted, never
#                 fed back. The clean A/B that removes the crutch.
# 2026-07-22: default flipped "arfed" -> "anchor" after Arnold's GO ("safe to
# just implement the anchor and go from there"). A bare launcher run now lands
# on the approved arm; the arfed control already exists (2026-07-16 run).
TINNER_MODE = os.environ.get("TINNER_MODE", "anchor")   # "arfed" | "anchor" | "output_only"
assert TINNER_MODE in ("arfed", "anchor", "output_only"), \
    f"TINNER_MODE must be arfed/anchor/output_only, got {TINNER_MODE!r}"

# ------------------------------------------------------------
# T_outer reference (2026-09-12, Round 8) -- Arnold, 2026-09-10 meeting
# ------------------------------------------------------------
# "Use T_input again as T_outer's reference as well, exactly what you do for
# the inner surface": every change is triggered by the input, so the input
# should be the fundamental reference, and unlike the round-4 initial-value
# anchor it is time-dependent.
#   "abs"           (default) head 1 is T_outer directly
#   "anchor_input"  T_outer(t+1) = Input_T(t+ANCHOR_LEAD) + delta_o, with
#                   delta_o z-scored on the train set -- the T_inner anchor
#                   applied verbatim to T_outer
# Measured on the training set before launch: the residual T_outer - Input_T
# has std 97 C against 116 C for T_outer itself (-16%), while the round-4
# initial-value anchor removed 50% and still lost; and T_outer(0) == Input_T(0)
# in every run, so that round-4 anchor already WAS "Input_T at t=0". The
# expectation is a null or a loss. It runs anyway so the answer is a
# measurement, not an argument.
#   "anchor_ema"    same reconstruction, but the reference is an exponential
#                   moving average of the input temperature with time
#                   constant TOUTER_TAU steps (e[0] = T_outer(0), the given
#                   initial condition, converted into Input_T's scale; causal; affine-
#                   equivariant, so the raw-degree fit and the scaled-space
#                   use agree exactly). Scanned on the training set before
#                   launch (2026-09-12): residual std vs T_outer's own 115.8
#                       raw Input_T(t)  97.2   lag 720 steps  39.8
#                       EMA tau=480     38.3   EMA tau=720    21.3
#                       EMA tau=1000     9.4   EMA tau=1200    7.3  <- best
#                       EMA tau=1500    12.0   EMA tau=2000   20.5
#                   i.e. the outer wall is a first-order low-pass of the
#                   inlet (fitted slope 1.001), and the reference removes
#                   94% of T_outer's variance -- the T_inner anchor removes
#                   99%, the round-4 initial-value anchor removed 50%.
TOUTER_MODE = os.environ.get("TOUTER_MODE", "abs")
assert TOUTER_MODE in ("abs", "anchor_input", "anchor_ema"), TOUTER_MODE
TOUTER_TAU = int(os.environ.get("TOUTER_TAU", "1200"))      # steps of 10 s
# >= 3: the closed-form EMA (rollout.touter_reference) weights the inputs by
# (1 - 1/tau)^-k over k <= 1439, which overflows float64 below tau = 2.6.
assert 3 <= TOUTER_TAU <= 100000, TOUTER_TAU
# TOUTER_SCALE widens the outer head's usable range the way ANCHOR_SCALE does
# for the inner one: the train-set residual has std 7.3 C but a maximum of
# 46 C (6.3 sigma), which a unit head cannot reach. Kept separate from
# ANCHOR_SCALE so the two anchors are scaled independently.
TOUTER_SCALE = float(os.environ.get("TOUTER_SCALE", "1.0"))
assert 0.1 <= TOUTER_SCALE <= 20.0
# ------------------------------------------------------------
# Where the EMA reference comes from (user's derivation, 2026-09-20, pages
# b1-b3 of the notebook), kept here so the formula can be swapped when the
# medium changes:
#   energy balance of the outer wall, lumped:  rho c V dT_out/dt = h A (T_in - T_out)
#   units: rho c V [J/K], h A [J/s/K]  ->  tau = rho c V / (h A)  [s]
#   =>  tau dT_out/dt + T_out = T_in                      (first-order standard form)
#   integrating factor e^(t/tau):  d/dt [T_out e^(t/tau)] = (1/tau) e^(t/tau) T_in
#   integrate over one step (t - dt, t):
#     T_out(t) = T_out(t-dt) e^(-dt/tau) + e^(-t/tau) (1/tau) INT_{t-dt}^{t} e^(s/tau) T_in(s) ds
#   zero-order hold, T_in(s) = T_in(t) over the step (dt = 10 s):
#     T_out(t) = T_out(t-dt) e^(-dt/tau) + T_in(t) (1 - e^(-dt/tau))     [EXACT, "the red line"]
#   and because dt/tau = 1/1200 is small, e^(-dt/tau) ~= 1 - dt/tau:
#     R(t) = R(t-1) + (1/tau) (T_in(t) - R(t-1))                        [EULER, rounds 8-9]
# Both are the recurrence e_t = r e_{t-1} + (1 - r) x_t with
#   r = 1 - 1/TAU              (euler)     TAU in steps
#   r = exp(-1/TAU)            (exact)
# At tau = 1200 steps the two differ by 0.04% in the weight (0.000833 vs
# 0.000833); the exact form matters for a medium with a small tau (tau of a
# few steps), where the Taylor step is no longer small. Default euler so
# every round-8/9 EMA arm keeps its formula; the exact form gets its own
# tag ("To-em<tau>x") and is never mixed into an existing run name.
TOUTER_EMA_FORM = os.environ.get("TOUTER_EMA_FORM", "euler")
assert TOUTER_EMA_FORM in ("euler", "exact"), TOUTER_EMA_FORM


def touter_ema_r():
    """The recurrence weight r of the EMA reference (e_t = r e_{t-1} + (1-r) x_t),
    shared by rollout.touter_reference, train.py's raw-degree twin and the
    runner's preflight, so the three can never disagree on the form."""
    import math
    return math.exp(-1.0 / TOUTER_TAU) if TOUTER_EMA_FORM == "exact" else 1.0 - 1.0 / TOUTER_TAU
# The EMA is seeded at T_outer(0), the GIVEN initial condition, not at
# Input_T(0): identical on the base data (they are equal in every run) and
# right for data where the tank does not start at the inlet temperature.
# (exclusivity against OTHER_CH_MODE's own T_outer re-parametrisations --
#  "persistence" and the round-4 "anchor" -- is asserted where OTHER_CH_MODE
#  is defined, further down)

# ------------------------------------------------------------
# Anchor lead (2026-07-23)
# ------------------------------------------------------------
# The anchor predicts T_inner(t+1). ANCHOR_LEAD=1 anchors it on Input_T(t+1),
# =0 on Input_T(t). Input_T is EXOGENOUS -- the whole curve is a given boundary
# condition known before the rollout starts -- so t+1 is equally causal and
# strictly better: the residual the head must learn drops from max 37.5 C /
# std 1.00 to max 8.5 C / std 0.49, because Input_T(t) is stale whenever the
# the input temperature jumps between steps (Case 40: input temperature 217->263 in one step, giving a
# -37 C first-step error under lead=0).
ANCHOR_LEAD = int(os.environ.get("ANCHOR_LEAD", "1"))
assert ANCHOR_LEAD in (0, 1)

# ------------------------------------------------------------
# Loss shaping (2026-07-23) -- borrowed from the LSTM team's recipe
# ------------------------------------------------------------
# Per-channel loss weights in OUTPUT order (T_inner, T_outer, T_avg).
# Uniform [1,1,1] under-serves the hard channels: with T_inner anchored its
# error is ~0.38 C while T_avg sits at 2.62 C, of which 2.52 C is a pure
# level offset -- the model has little gradient incentive left to fix it.
# The LSTM line uses Ti x1 + To x6 + Ta x3 and reports far better T_outer /
# T_avg, so we adopt the same emphasis. Set "1,1,1" to restore the old
# behaviour.
LOSS_WEIGHTS = [float(x) for x in
                os.environ.get("LOSS_WEIGHTS", "1,6,3").split(",")]
assert len(LOSS_WEIGHTS) == 3, "LOSS_WEIGHTS needs 3 comma-separated values"
LOSS_W0_MODE = os.environ.get("LOSS_W0_MODE", "").strip()
assert LOSS_W0_MODE in ("", "inv_kc"), LOSS_W0_MODE
if LOSS_W0_MODE:
    assert TINNER_MODE == "anchor", "LOSS_W0_MODE=inv_kc needs the T_inner anchor"
ENC_WIDTHS = os.environ.get("ENC_WIDTHS", "256,256").strip()
ENC_WIDTHS_LIST = [int(v) for v in ENC_WIDTHS.split(",") if v.strip()]
assert all(v > 0 for v in ENC_WIDTHS_LIST), ENC_WIDTHS
H0_MODE = os.environ.get("H0_MODE", "linear").strip()
assert H0_MODE in ("linear", "tanh", "zero"), H0_MODE
INIT_ENCODER = os.environ.get("INIT_ENCODER", "point").strip()
assert INIT_ENCODER in ("point", "window"), INIT_ENCODER
EARLY_STEP_WEIGHT = float(os.environ.get("EARLY_STEP_WEIGHT", "1"))
assert 0 < EARLY_STEP_WEIGHT <= 100, EARLY_STEP_WEIGHT
INIT_BURNIN_STEPS = int(os.environ.get('INIT_BURNIN_STEPS', '0'))
SYNTHETIC_REST_FILES = int(os.environ.get('SYNTHETIC_REST_FILES', '0'))
assert 0 <= INIT_BURNIN_STEPS <= 200 and 0 <= SYNTHETIC_REST_FILES <= 100, 'invalid startup controls'
CONSTANTS_FROM = os.environ.get('CONSTANTS_FROM', os.environ.get('FROZEN_CONSTANTS', '')).strip()
if CONSTANTS_FROM:
    assert os.path.isfile(CONSTANTS_FROM), 'missing frozen constants sidecar: %s' % CONSTANTS_FROM

# Physics bound penalty (also from the LSTM recipe): T_avg must lie between
# T_inner and T_outer. Verified on our data: holds at 98.76% of all timesteps
# (T_avg sits ~0.41 of the way from T_outer to T_inner). The penalty is a
# hinge on violations, applied in RAW temperature space (the MinMax scaling
# is per-channel, so the ordering is not preserved in scaled space).
# 0.0 disables it.
# 2026-07-25: default flipped 1.0 -> 0.0. MEASURED HARMFUL: arm C (bound on)
# scored 1.44 overall vs arm B (bound off) 1.27, and T_avg -- the channel it
# was meant to help -- got WORSE (2.79 -> 3.21; late bias +3.01 -> +3.31).
# Likely cause: the bracket |T_inner - T_outer| averages 79 C while the bias
# is only ~2.5 C, so the hinge fires on just 3.3% of steps -- a rare, spiky
# gradient that perturbs optimization without ever addressing the in-bracket
# level offset. Kept available for the convection data, where near-isothermal
# starts make the bracket genuinely tight.
PHYSICS_BOUND_WEIGHT = float(os.environ.get("PHYSICS_BOUND_WEIGHT", "0.0"))

# ------------------------------------------------------------
# T_outer / T_avg output parameterization (Arnold's request, 2026-07-23 mtg)
# ------------------------------------------------------------
# Arnold: "the error only happens for average and outer... I'm guessing it is
# because you're using delta for inner -- then yes, just do the same thing for
# outer and average."
#   "abs"         predict T_outer(t+1) / T_avg(t+1) directly (status quo).
#   "persistence" predict the per-step CHANGE and add it to the previous
#                 value: T(t+1) = T_AR(t) + delta, with delta z-scored on the
#                 train set.
# NOTE this is NOT the same mechanism as the T_inner anchor. T_inner is
# anchored on GROUND-TRUTH exogenous Input_T, so its error resets every step.
# T_outer / T_avg have no such external reference (they sit 78 C / 58 C away
# from Input_T with large variance), so the only available delta is against
# their OWN previous prediction -- which makes the rollout an integrator with
# no reset: a per-step bias of just 0.001 C compounds to 1.4 C over the 1440
# steps, and the true per-step change is only 0.071 / 0.091 C. It also cannot
# by itself remove a level offset (the current failure mode) because there is
# no absolute reference to pull the level back. Kept as an experimental arm
# so the question is settled by measurement rather than argument.
#   "anchor"      (2026-08-06) anchor BOTH remaining channels on fixed
#                 references instead of their own previous prediction, so
#                 nothing integrates:
#                   T_outer(t) = T_outer(0) + delta_o
#                       T_outer(0) is the GIVEN initial condition (GT), so the
#                       reference is constant -- residual std 56.6 C vs 115.4 C
#                       absolute (-51%).
#                   T_avg(t)   = w*T_inner(t) + (1-w)*T_outer(t) + delta_a
#                       T_avg is physically the volume average between the two
#                       surfaces. Fitting w on the train set gives w = 0.289
#                       with residual std 3.81 C vs 107.0 C absolute (-96%) --
#                       an anchor as strong as the T_inner one. w is stable
#                       across cases (per-case mean 0.288, std 0.034; only
#                       2/311 cases have within-case residual std > 10 C).
#                       This is the physics-bound idea done right: the bound
#                       only said "T_avg is somewhere between", which was too
#                       weak to help; this pins where.
#                 Both deltas are z-scored like the T_inner anchor. Safe in
#                 forward_direct because there is no AR loop for the errors to
#                 compound through.
#   "anchor_avg"  (2026-08-06, MEASURED BEST) anchor ONLY T_avg on the blend;
#                 leave T_outer absolute. The 5-arm F result split the two:
#                 the T_avg blend anchor worked (seed42 T_avg 2.45 -> 1.76,
#                 overall 0.912 -- best ever) while the T_outer initial-value
#                 anchor was unstable (seed7 T_outer 0.74 -> 2.14, wrecking
#                 that seed). Consistent with the fitted residuals: the T_avg
#                 anchor removes 96% of the variance, the T_outer one only
#                 51% -- too weak to pay for the extra constraint.
#   "anchor_avg_grad"  same as anchor_avg but WITHOUT detaching the blend's
#                 reference channels, so T_avg's loss backprops into the
#                 T_inner/T_outer heads. Kept as the ablation control for the
#                 detach decision. (This value was missing from the assert
#                 list below, which silently killed the round-4 G arm at
#                 import time -- 12 runs lost.)
#   "pos_head"    (2026-08-09) T_avg is reparametrized as a POSITION between
#                 the two surfaces, predicted per step:
#                     T_avg = T_outer + pos * (T_inner - T_outer)
#                     pos   = pos_mean + pos_sd * z_head
#                 Diagnosis behind it: T_avg's error is a position error
#                 amplified by the surface gap. pos is nearly constant within
#                 a case (std 0.037) but genuinely varies BETWEEN cases
#                 (0.238-0.332). Oracle ladder on the 70-case test set:
#                     fixed global blend w=0.289 -> T_avg MAE 2.41
#                     model predicting T_avg     -> 2.42   (i.e. we sit exactly
#                                                   at the fixed-blend ceiling)
#                     per-case optimal pos       -> 1.54   <- 36% headroom
#                 That ceiling is why the round-4 fixed-w anchor could not
#                 help: its oracle was already the model's own level. Letting
#                 the head predict pos targets the 1.54 instead. References
#                 are detached, as in anchor_avg.
OTHER_CH_MODE = os.environ.get("OTHER_CH_MODE", "abs")
_OTHER_OK = ("abs", "persistence", "anchor", "anchor_avg", "anchor_avg_grad",
             "pos_head", "ratio_input", "anchor_initial")
assert OTHER_CH_MODE in _OTHER_OK, \
    f"OTHER_CH_MODE must be one of {_OTHER_OK}, got {OTHER_CH_MODE!r}"
RATIO_CHANNELS = os.environ.get("RATIO_CHANNELS", "R3").strip().upper()
RATIO_REF = os.environ.get("RATIO_REF", "kelvin").strip().lower()
assert RATIO_CHANNELS in ("R1", "R2", "R3"), RATIO_CHANNELS
assert RATIO_REF in ("kelvin", "celsius20"), RATIO_REF
RATIO_INDICES = {"R1": (1,), "R2": (1, 2), "R3": (0, 1, 2)}[RATIO_CHANNELS]
PHYS_MODE = os.environ.get("PHYS_MODE", "").strip()
PHYSICS_CACHE_DIR = os.environ.get("PHYSICS_CACHE_DIR", "").strip()
assert PHYS_MODE in ("", "residual", "input", "input_residual"), PHYS_MODE
if PHYS_MODE:
    assert PHYSICS_CACHE_DIR and os.path.isdir(PHYSICS_CACHE_DIR), "PHYS_MODE needs an existing PHYSICS_CACHE_DIR"
    assert OTHER_CH_MODE == "abs" and TOUTER_MODE == "abs", "physics and other reference heads are exclusive"
    assert PHYSICS_BOUND_WEIGHT == 0, "physics residual heads require zero bound weight"


def warmup_lr(epoch, base_lr, warmup_epochs=None):
    """Return the LR to use at this epoch during the warmup phase.

    Linear ramp: lr(0) = base_lr * (1 / warmup_epochs); lr(warmup-1) = base_lr.
    Returns None once warmup is complete (caller leaves optimizer LR alone
    so the regular scheduler controls it).
    """
    if warmup_epochs is None:
        warmup_epochs = LR_WARMUP_EPOCHS
    if warmup_epochs <= 0 or epoch >= warmup_epochs:
        return None
    return base_lr * (epoch + 1) / warmup_epochs


def apply_orthogonal_init_gru(gru):
    """Apply orthogonal init to each per-gate sub-matrix of weight_hh_l*.

    PyTorch's nn.GRU packs the 3 gate weight matrices into one tensor of
    shape (3*hidden_size, hidden_size). Per Saxe 2013 / Chen 2019, we
    apply orthogonal_ to each (H, H) gate sub-matrix INDEPENDENTLY rather
    than to the full (3H, H) block (the per-gate slices are the recurrent
    transitions whose spectral radius we want at 1.0).
    """
    H = gru.hidden_size
    with torch.no_grad():
        for name, param in gru.named_parameters():
            if 'weight_hh' in name:
                for gate in range(3):
                    nn.init.orthogonal_(param[gate * H:(gate + 1) * H])


def teacher_forcing_prob(epoch):
    """Linear TF prob decay: 0.5 at ep 0 -> 0.0 at ep 100, then 0."""
    if epoch >= TF_DECAY_EPOCHS:
        return 0.0
    return TF_INIT_PROB * (1.0 - epoch / TF_DECAY_EPOCHS)

# Default variants run in a fresh sweep. abs_window has been retired from
# the default set because it requires reading the first W ground-truth
# observations at inference time (not available in the surrogate-vs-CFD
# deployment). abs_sliding is the production-valid replacement that uses
# only past predictions + exogenous Input_T.
#
# delta / abs+delta are also strictly leaky (they feed GT deltas at every
# step) but are kept as diagnostic upper bounds. abs_window is left in the
# code paths below so it can be opted back in for further analysis, but
# is not part of the default 4-variant ablation.
# 2026-05-21: extended to 8 variants. The first 4 are the original "forward"
# ablation (predict T_inner/T_outer/T_avg from [Time, T_outer, T_inner, T_avg,
# Input_T]). The next 4 are Arnold's "inverse" ablation -- only T_avg in the
# input, predict [Input_T, T_inner, T_outer]. The idea is to use the observed
# T_avg as an indicator to recover the unknown driving Input_T.
# 2026-06-28: this run validates the init-pad fix (SLIDING_PAD_MODE='init') on
# the FORWARD sliding-window variant ONLY (abs_sliding -- the production-valid
# headline config). Seeds are kept at the full [7,21,42,123]. Restore the
# commented full list below to re-run the complete 8-variant ablation.
# 2026-07-24: env-overridable so drivers can select the variant per arm
# (run_fix_ablation.py arm E runs 'forward_direct'). Default unchanged.
VARIANTS = [v.strip() for v in
            os.environ.get("VARIANTS", "abs_sliding").split(",") if v.strip()]
# Full 8-variant ablation (uncomment to run everything):
# VARIANTS = [
#     'delta', 'abs+delta', 'abs', 'abs_sliding',
#     'inverse_delta', 'inverse_abs+delta', 'inverse_abs', 'inverse_abs_sliding',
# ]


# Per-variant model OUTPUT channel labels (output_size stays at 3 for both
# forward and inverse; only the semantics of the 3 channels differ).
# Used for metric labelling, plot titles, and CSV column naming.
VARIANT_OUTPUT_CHANNELS = {
    # Forward variants -- output is (T_inner, T_outer, T_avg) at t+1.
    'delta':              ['T_inner', 'T_outer', 'T_avg'],
    'abs+delta':          ['T_inner', 'T_outer', 'T_avg'],
    'abs':                ['T_inner', 'T_outer', 'T_avg'],
    'abs_window':         ['T_inner', 'T_outer', 'T_avg'],
    'abs_sliding':        ['T_inner', 'T_outer', 'T_avg'],
    # forward_direct (2026-07-24): seq2seq formulation test. Inputs are ONLY
    # the exogenous [Time, Input_T]; ALL state channels predicted in ONE
    # forward pass, no AR feedback anywhere. Mirrors the LSTM line's problem
    # setup (minus bidirectionality) so the formulation-vs-cell question is
    # answered inside our own multi-seed harness. Legitimate for the offline
    # surrogate use case: the full Input_T curve is the given boundary
    # condition. Structurally immune to error accumulation; single-pass
    # inference (~4 ms, like the non-sliding inverse variants).
    'forward_direct':     ['T_inner', 'T_outer', 'T_avg'],
    # Inverse variants -- output is (Input_T, T_inner, T_outer) at t+1.
    'inverse_delta':       ['Input_T', 'T_inner', 'T_outer'],
    'inverse_abs+delta':   ['Input_T', 'T_inner', 'T_outer'],
    'inverse_abs':         ['Input_T', 'T_inner', 'T_outer'],
    'inverse_abs_sliding': ['Input_T', 'T_inner', 'T_outer'],
}


def is_inverse(variant):
    return variant.startswith('inverse_')

# Per-variant GRU input dimensionality.
#   abs_sliding is 4 (current step's Time / T_outer / Input_T / T_avg) +
#   2*W (past W T_outer + past W T_avg, STRICTLY past steps, not including
#   current) = 4 + 20 = 24 for W=10.
#
# Per Arnold-aligned spec (2026-05-17): at step t the past lookback covers
# steps t-W..t-1. The current step (t) is in idx 1, 3 only. When fewer than
# W past values are available (t < W), the oldest lookback slots are
# zero-padded.
INPUT_DIMS = {
    # ---- Forward variants ----
    # 5-input refactor per Arnold (2026-05-20): all 5 data columns
    # (Time, T_outer, T_inner, T_avg, Input_T) are now used as inputs.
    # Previously T_inner was target-only.
    'delta':       5,    # Time + 4 deltas (dT_outer, dT_inner, dT_avg, dInput_T)
    'abs+delta':   9,    # 5 abs + 4 deltas
    'abs':         5,    # Time + T_outer + T_inner + T_avg + Input_T
    'abs_window':  5,    # same as abs
    # abs_sliding (Arnold 2026-05-25 sequence-input refactor):
    # 5-d per step; the W-step sliding window is in the TIME dimension (the
    # GRU input becomes (B, W, 5) instead of (B, 1, 5+3*W)). See
    # _rollout_sliding for the new semantics.
    # Under TINNER_MODE == "output_only" (v22-style A/B) T_inner is removed
    # from the input -> 4-d [Time, T_outer, T_avg, Input_T].
    'abs_sliding': 4 if TINNER_MODE == "output_only" else 5,
    # ---- Inverse variants (Arnold 2026-05-21) ----
    # Only Time + T_avg (and dT_avg / window of T_avg for delta / sliding flavors).
    # T_avg is GT exogenous (always observable at deployment), so there is no
    # AR feedback in the input -- a single forward pass suffices for the
    # non-sliding flavors. inverse_abs_sliding processes a W-step (B, W, 2)
    # sequence per rollout step (Arnold 2026-05-25 sequence-input refactor).
    'inverse_delta':       2,                   # Time + dT_avg
    'inverse_abs+delta':   3,                   # Time + T_avg + dT_avg
    'inverse_abs':         2,                   # Time + T_avg
    'inverse_abs_sliding': 2,                   # Time + T_avg per step, W-step sequence input
    # forward_direct: exogenous-only [Time, Input_T]; single pass, no AR.
    # +INPUT_LOOKAHEAD extra columns of near-future Input_T (see below).
    'forward_direct': 2,   # patched to 2 + INPUT_LOOKAHEAD right below
}

# ------------------------------------------------------------
# Input-temperature lookahead (2026-08-06) -- the targeted Case-40 fix
# ------------------------------------------------------------
# The remaining +8.5 C first-step error on input-jump cases is the physical
# T_inner lag: the model at step t cannot see that the input temperature jumps at t+1,
# so it cannot predict the lag. But the FULL Input_T curve is a given
# boundary condition, so showing the model the next k values is legal.
# INPUT_LOOKAHEAD=k appends k extra features to forward_direct's input:
# Input_T(t+1) .. Input_T(t+k), tail-held at the last value. 0 = off.
# (forward_direct only; the AR variants are untouched.)
INPUT_LOOKAHEAD = int(os.environ.get("INPUT_LOOKAHEAD", "0"))
assert 0 <= INPUT_LOOKAHEAD <= 30
FEAT_COLS = os.environ.get("FEAT_COLS", "Time (s),Input Temperature (C)").strip()
NO_TIME = FEAT_COLS.lower() in ("input temperature (c)", "input_t", "input", "no_time")
assert NO_TIME or FEAT_COLS in ("Time (s),Input Temperature (C)", "time,input"), FEAT_COLS
FORWARD_INPUT_INDEX = 0 if NO_TIME else 1
if NO_TIME:
    assert all(v == "forward_direct" for v in VARIANTS), "NO_TIME supports forward_direct only"

# ------------------------------------------------------------
# Case-flag input (2026-09-06, Round 7)
# ------------------------------------------------------------
# Round 6 established that the case gate has to live OUTSIDE the GRU: the
# network never sees the gate decision, and the shared-channel arm failed
# precisely because one channel had to serve two regimes the GRU could not
# tell apart (2026-09-05 meeting). The direct fix is to TELL it. One extra
# input column, derived from the input-temperature profile alone (a given
# boundary condition, so legal at inference and identical in training),
# appended LAST so every existing column index still holds:
#   "case"   the per-run detector flag (both charging and discharging),
#            constant over the run
#   "phase"  per timestep: 1 once the discharge has begun in a both-phase
#            run (the excursion starts right there), 0 everywhere else.
#            "Begun" = the first step after the peak plateau (the steps
#            from the maximum onward that stay within 1% of the span), so
#            a held top temperature still counts as charging.
#   "zero"   a constant-zero column. The control: one more input column
#            shifts the RNG stream at init exactly like the extra output
#            channel did in round 6, so "the flag helps" needs this null.
# forward_direct and abs_sliding (round 8 extended it to the AR path: the
# sliding window carries the column as its last feature); asserted below,
# where VARIANTS is in scope.
CASE_FLAG_INPUT = os.environ.get("CASE_FLAG_INPUT", "").strip().lower()
assert CASE_FLAG_INPUT in ("", "case", "phase", "zero"), CASE_FLAG_INPUT
INPUT_DIMS['forward_direct'] = (1 if NO_TIME else 2) + INPUT_LOOKAHEAD + (2 if PHYS_MODE in ('input', 'input_residual') else 0) + (1 if CASE_FLAG_INPUT else 0)
INPUT_DIMS['abs_sliding'] += 1 if CASE_FLAG_INPUT else 0

# EXCLUDE_BOTH_PHASE (Arnold, 2026-09-05): drop every run the case detector
# fires on -- from train, val AND test -- and rerun. Under this flag no gate
# can ever fire, so a gated arm has to reproduce the ungated one on the
# remaining cases; if it does not, the shared code is broken. Runs made this
# way are scored on a 49-case test set and must never be compared with
# full-set numbers (export_results keeps them in their own family).
EXCLUDE_BOTH_PHASE = os.environ.get("EXCLUDE_BOTH_PHASE", "0") not in ("0", "", "false", "False")

# ------------------------------------------------------------
# Anchor output range (2026-08-09)
# ------------------------------------------------------------
# Multiplier on the T_inner anchor's z-scaling std. Round 4 established that
# the stubborn +8.4 C residual on the input-jump cases is a RANGE limit, not
# an information limit: giving the model the input temperature's future (lookahead 1/3/5)
# left Case 40 at 8.38 / 8.46 / 8.49 C, i.e. untouched, while making overall
# MAE significantly worse (p=0.003). With the fitted delta std ~0.65 C, an
# 8.5 C correction sits ~13 sigma out -- unreachable for the head. Raising
# this widens what the head can express, at the cost of resolution in the
# near-zero regime where almost all steps live. 1.0 = round-4 behaviour.
ANCHOR_SCALE = float(os.environ.get("ANCHOR_SCALE", "1.0"))
assert 0.1 <= ANCHOR_SCALE <= 50.0

# ------------------------------------------------------------
# pos_head excursion handling (2026-08-20)
# ------------------------------------------------------------
# Round 5 left cases 32-40 / 51-53 weak. Mechanism: for 3-6% of their
# timesteps T_avg rises ABOVE both surfaces (up to 13 C above), so the
# "position between the surfaces" assumption breaks. Ranking all 70 test
# cases by how far the position falls outside the head's reachable band puts
# exactly those 12 on top -- no misses, no extras.
#
# Two independent remedies, composable so the round-6 sweep can separate them:
#
# POS_GAP_FLOOR (timestep level, ours). The excursions coincide with the two
#   surfaces being close (median 7 C apart), where pos = excess/gap is
#   ill-conditioned -- measured max 66 against a typical 0.27. Flooring the
#   denominator bounds it. Reconstruction stays exact, since pos is defined
#   through the same floored denominator. 0 disables. A simulated +-3 sigma
#   head shows the forced T_avg error on the 12 cases falling 0.14 -> 0.02 ->
#   0.00 for floors of 0 / 20 / 30 C, with the other 58 cases untouched.
#   (Same trick already used for the T_inner anchor's std floor.)
#
# POS_CASE_GATE (case level, Arnold's suggestion, 2026-08-20 meeting). Only
#   runs with both a charging and a discharging phase show the excursion --
#   confirmed on all 70 cases: 19 of the 20 excursion cases qualify, one false
#   positive (case 58, 0.3% excursion) and one miss (case 50, discharging for
#   only 3% of the run). Gated cases fall back to predicting T_avg directly.
#   The gate is read off the INPUT TEMPERATURE profile alone (rises then falls back by
#   >25% of its span, peak in the interior): 20/20 hits, 1 false alarm, 69/70
#   correct. The input temperature is a given boundary condition, so this is legal
#   at inference -- note that the more obvious "input temperature crosses the initial
#   temperature" test is NOT usable, it catches only 3/20, because the phase
#   is defined against the medium's evolving temperature rather than its
#   initial one.
# The knobs below are composable; each answers a different question about
# WHERE the position parametrization should stop being trusted.
#
#   POS_GAP_FLOOR / POS_FLOOR_SOFT   never switch away, just recondition:
#       bound the denominator so pos stays finite. Hard = max(|gap|, floor);
#       soft = sqrt(gap^2 + floor^2), same asymptote without the kink at the
#       threshold. Reconstruction stays exact either way, because train.py
#       fits mu/sd through the same denominator.
#   POS_TEMP_GATE / POS_TEMP_SOFT    timestep level: hand T_avg over to a
#       dedicated absolute head while the surfaces are within POS_TEMP_GATE
#       degrees of each other, which is exactly when the excursions happen
#       (median separation 7 C). POS_TEMP_SOFT > 0 ramps the handover over
#       that many degrees instead of switching abruptly.
#   POS_CASE_GATE                    case level (Arnold, 2026-08-20 meeting):
#       hand the whole run over when it contains both a charging and a
#       discharging phase. Blunter -- it gives up the parametrization for
#       100% of a case to repair 3-6% of it -- but it needs no in-sequence
#       decision. Detected from the input temperature profile alone (see _input_case_gate).
#
# POS_ABS_HEAD adds a 4th output channel holding that absolute T_avg estimate.
# It exists because channel 2 cannot serve both roles: it carries a z-scored
# position (~N(0,1)) whereas an absolute T_avg is a MinMax-scaled temperature
# (~0.5), so reusing it would force the head to jump between two unrelated
# output ranges mid-sequence. The 4th channel is consumed inside the rollout,
# which still returns 3 channels, so the loss is untouched. Auto-enabled by
# POS_TEMP_GATE. Caveat to watch: this head only receives gradient where the
# blend actually uses it, so in runs whose surfaces never converge it stays
# untrained -- harmless while unused, but it is why the soft band is worth
# preferring over a hard switch.
POS_GAP_FLOOR = float(os.environ.get("POS_GAP_FLOOR", "0"))
assert 0.0 <= POS_GAP_FLOOR <= 300.0
POS_FLOOR_SOFT = os.environ.get("POS_FLOOR_SOFT", "0") not in ("0", "", "false", "False")
POS_TEMP_GATE = float(os.environ.get("POS_TEMP_GATE", "0"))
assert 0.0 <= POS_TEMP_GATE <= 300.0
POS_TEMP_SOFT = float(os.environ.get("POS_TEMP_SOFT", "0"))
assert 0.0 <= POS_TEMP_SOFT <= 300.0
POS_CASE_GATE = os.environ.get("POS_CASE_GATE", "0") not in ("0", "", "false", "False")
#   POS_LEARNED_GATE                 the model decides. A 5th channel is fed
#       through a sigmoid to produce the handover weight directly, instead of
#       us picking a temperature threshold for it. Worth testing against the
#       hand-designed gates: our threshold is derived from where the
#       excursions happen to sit in THIS dataset (surfaces a median 7 C apart),
#       which may not be the boundary that actually matters. POS_GATE_BIAS
#       shifts the sigmoid's starting point so training begins trusting the
#       position formula (+2 -> w = 0.88) rather than sitting at an
#       uncommitted 0.5. If a hand-designed gate is on at the same time, the
#       two weights multiply: both have to want the position formula.
POS_LEARNED_GATE = os.environ.get("POS_LEARNED_GATE", "0") not in ("0", "", "false", "False")
POS_GATE_BIAS = float(os.environ.get("POS_GATE_BIAS", "2.0"))
assert -10.0 <= POS_GATE_BIAS <= 10.0
# POS_LEARNED_POOL: pool the learned-gate logit over the whole sequence
# before the sigmoid, making it a per-CASE router instead of a per-step
# one -- jump-free by construction (2026-08-22 review: the per-step gate
# can flap between two T_avg estimates that differ by degrees, creating
# exactly the discontinuities the decision rule vetoes).
POS_LEARNED_POOL = os.environ.get("POS_LEARNED_POOL", "0") not in ("0", "", "false", "False")
# POS_ANCHORED_FALLBACK: replace the raw absolute fallback with a
# surface-anchored one, T_avg_fb = midpoint(T_inner,T_outer) + mu + sd*z.
# Review argument: every gate falls back to a raw absolute head -- the
# mechanism family this project refuted three times -- while the model's
# own surfaces are its two most accurate outputs (0.28 / 0.56 C).
# Anchoring on their midpoint inherits that accuracy and stays exact as
# the gap closes (midpoint -> T_avg when the surfaces converge). The
# midpoint is used rather than max(T_i,T_o)+softplus (as first proposed)
# because gated regions also contain BETWEEN-surface steps, which a
# >=max fallback cannot express.
POS_ANCHORED_FALLBACK = os.environ.get("POS_ANCHORED_FALLBACK", "0") not in ("0", "", "false", "False")
# POS_FIT_CLEAN: drop excursion steps (T_avg outside the surface bracket)
# from the no-floor mu/sd fit. Review measurement: those steps inflate
# the fitted sd ~34% (0.164 vs 0.122), coarsening the head's resolution
# in the normal regime for every ungated arm.
POS_FIT_CLEAN = os.environ.get("POS_FIT_CLEAN", "0") not in ("0", "", "false", "False")
POS_ABS_HEAD = (os.environ.get("POS_ABS_HEAD", "0") not in ("0", "", "false", "False")
                or POS_TEMP_GATE > 0 or POS_LEARNED_GATE
                or POS_ANCHORED_FALLBACK)
# Round 9 (2026-09-19, user: "the core problem since round 5 is still
# there -- T_avg in the first ~50 steps"): the position head's early-time
# error. Measured on the 20-seed cgate-ah ensemble: the head's pos output is
# ~flat at 0.24 from t=0, while the true position of a charging run starts
# near 0.05 and climbs to ~0.25 over the first ~200 steps (the heat has not
# penetrated; the bulk average sits at the cold outer wall). Error =
# delta-pos x wall gap, so it is worst where the gap opens fast (case 24:
# gap 157 C at step 50 -> 10.6 C). On the 311 training runs the climb is a
# near-universal function of elapsed time (heat-penetration depth ~
# sqrt(alpha t)): the per-step median position over the runs is 0.04 / 0.16
# / 0.25 / 0.28 at steps 1 / 50 / 200 / 500, and the spread around it in
# the first 200 steps is 0.037 against 0.08 for the flat mean. With the
# TRUE walls and that curve alone (no network) the first-200-step T_avg
# error on the 49 non-gated test cases is 0.5 C against 2.6 for the flat
# mean and 1.4 for the trained ensemble; the full-run error does not get
# worse (1.55 vs 1.59).
# POS_TIME_PRIOR = median  the head predicts the RESIDUAL from that curve:
#                          pos = c[t] + mu + sd*z, c the per-step median of
#                          the training runs' position (usable steps only,
#                          |gap| > 5 C, gaps interpolated), mu/sd the
#                          residual's -- the anchor idea (T_inner, T_outer)
#                          applied to the position's time axis. Step index
#                          from the scaled Time input (column 0). Direct and
#                          AR paths, train and test; gated cases unaffected
#                          (the gate takes T_avg from the absolute channel).
#                = 0       the flat prior of rounds 5-8 (pos = mu + sd*z).
POS_TIME_PRIOR = os.environ.get("POS_TIME_PRIOR", "0")
assert POS_TIME_PRIOR in ("0", "median"), POS_TIME_PRIOR
if POS_TIME_PRIOR != "0":
    assert OTHER_CH_MODE == "pos_head", "POS_TIME_PRIOR needs OTHER_CH_MODE=pos_head"
    assert not POS_GAP_FLOOR, "POS_TIME_PRIOR is fitted on the plain (unfloored) position"

# Output channels the head emits (the rollout always returns 3): the 4th is
# the absolute T_avg fallback, the 5th is the learned handover weight.
if OTHER_CH_MODE == "pos_head" and POS_LEARNED_GATE:
    OUTPUT_SIZE = 5
elif OTHER_CH_MODE == "pos_head" and POS_ABS_HEAD:
    OUTPUT_SIZE = 4
else:
    OUTPUT_SIZE = 3

# ------------------------------------------------------------
# Round 9 (2026-09-17 decision): one network per temperature.
# ------------------------------------------------------------
# MODEL_SPLIT = shared       one GRU trunk + one linear head emits every
#                            channel (rounds 1-8).
#             = independent  three complete models (init encoder + GRU +
#                            head), one for T_inner, one for T_outer, one
#                            for T_avg (the T_avg model also owns the extra
#                            channels: the 4th absolute head, the learned
#                            gate). Trained in three stages: T_inner, then
#                            T_outer, then T_avg -- each with its own
#                            optimiser, LR schedule, early stopping and best
#                            checkpoint, the finished models frozen (eval
#                            mode, no dropout) while the next one trains. So
#                            the T_avg model is fitted against the walls it
#                            will actually see at test time, and no channel's
#                            gradient or validation curve can stop another's
#                            training -- the coupling suspected behind the
#                            round-8 T_outer-anchor failure.
#             = avg_only     two models: the two walls share one trunk as in
#                            rounds 1-8 (stage 1, loss on T_inner + T_outer),
#                            T_avg has its own (stage 2, trained against the
#                            frozen walls). The dev-box pilot (2026-09-19, 3
#                            seeds) showed the lone T_outer net WORSE than the
#                            shared trunk's T_outer (0.42 vs 0.33) while the
#                            lone T_avg net was better on the 49 normal cases;
#                            this keeps the wall pair's shared trunk and
#                            decouples only T_avg's training and stopping.
#   The per-stage loss is the configured LOSS_WEIGHTS entry of that stage's
#   channel(s); with one channel per stage the relative weights act only
#   through the gradient clip (GRADIENT_CLIP_NORM, measured not to engage on
#   the anchored arms), so the weight tag is kept mainly so an independent
#   arm pairs by name with its shared twin.
#   The rollout, the anchors and the position formula are unchanged: the
#   wrapper returns the same (B, T, OUTPUT_SIZE) layout. HIDDEN_SIZE /
#   NUM_LAYERS / DROPOUT apply to each of the three (three h128x2 nets carry
#   ~3x the parameters of one; HIDDEN_SIZE=64 is the parameter-matched
#   control, see run_round9.py). Direct prediction only (the AR path feeds
#   predictions back through one input row and has not been exercised with
#   the wrapper). In-run resume is off for this mode (a killed seed restarts
#   from scratch; done.flag resume across seeds is unaffected).
MODEL_SPLIT = os.environ.get("MODEL_SPLIT", "shared")
assert MODEL_SPLIT in ("shared", "independent", "avg_only"), MODEL_SPLIT
if OTHER_CH_MODE in ("ratio_input", "anchor_initial"):
    assert MODEL_SPLIT == "shared" and TOUTER_MODE == "abs", "new heads require shared model and absolute T_outer"
    assert PHYSICS_BOUND_WEIGHT == 0, "z heads require zero physics-bound weight"
    assert all(v == "forward_direct" for v in VARIANTS), "new heads support forward_direct only"
if PHYS_MODE:
    assert MODEL_SPLIT == "shared" and all(v == "forward_direct" for v in VARIANTS), "physics modes require shared forward_direct"
if OTHER_CH_MODE == "ratio_input" and RATIO_CHANNELS == "R3":
    assert TINNER_MODE != "anchor", "R3 replaces the T_inner anchor; set TINNER_MODE=arfed"
# Output-channel groups, one model (and one training stage) per group.
MODEL_GROUPS = {"shared": None,
                "independent": ((0,), (1,), tuple(range(2, OUTPUT_SIZE))),
                "avg_only": ((0, 1), tuple(range(2, OUTPUT_SIZE)))}[MODEL_SPLIT]
if MODEL_SPLIT != "shared":
    assert all(v == "forward_direct" for v in VARIANTS), \
        "MODEL_SPLIT=%s supports forward_direct only (got %r)" % (MODEL_SPLIT, VARIANTS)

# ------------------------------------------------------------
# Output layout
# ------------------------------------------------------------
# Every run writes into  src/runs/<RUN_NAME>/  with the structure:
#
#   runs/<RUN_NAME>/
#     run_config.json                    timestamp + config snapshot
#     checkpoints/
#       best_gru_variant_<v>.pth         best-val checkpoint
#       gru_variant_<v>.pth              final-state checkpoint
#     variants/
#       <variant>/
#         plots/                         70 per-case plots
#         summary_errors.csv             per-case metrics
#         train_history.json             train/val/lr histories
#         predictions.npz                raw inv_pred / inv_actual
#         meta.json                      scalar metrics dict
#         resume_state.pt                in-progress (deleted on done)
#         done.flag                      sentinel
#     comparison/
#       variant_comparison.csv
#       variant_loss_curves.png
#       variant_test_mae.png
#       variant_capacity_speed.png
#       variant_results.json
#
# RUN_NAME_BASE is the parent directory name. The script automatically
# expands it per seed: `runs/<RUN_NAME_BASE>_seed<N>/...` for each
# seed in SEEDS. To resume / continue an interrupted run, KEEP the same
# RUN_NAME_BASE between invocations -- each (seed, variant) tracks its
# own resume/done state. To start a fresh experiment, change RUN_NAME_BASE.
# Convention: prefix with the date (YYYY-MM-DD) so chronology is obvious.
# ------------------------------------------------------------
# Validation split (2026-09-17, Round 9)
# ------------------------------------------------------------
# "legacy"      the protocol every round through 8 used: sort the training
#               files by path, the first 5% (15 of 311) are validation. Sorted,
#               those 15 are ALL in "Case 1 (linear charging)/", so early
#               stopping and checkpoint selection never saw a charge-then-
#               discharge run (audit 2026-09-16; the leading hypothesis for
#               the round-8 T_outer-anchor failure).
# "stratified"  VAL_PER_TYPE files from EACH case-type folder under the
#               training dir, drawn once with SPLIT_SEED, written to
#               splits/<name>.json on first use and READ from there after
#               that, so every seed of every arm sees the same lists. Test
#               set untouched.
# The tag "vs<N>" enters RUN_NAME so old- and new-protocol runs never share a
# directory; export_results files them under separate families.
VAL_SPLIT = os.environ.get("VAL_SPLIT", "legacy")
assert VAL_SPLIT in ("legacy", "stratified"), VAL_SPLIT
VAL_PER_TYPE = int(os.environ.get("VAL_PER_TYPE", "2"))
assert 1 <= VAL_PER_TYPE <= 5
# Round 9 add-on (2026-09-23): leave-one-type-out (LOTO) generalisation test.
# TRAIN_EXCLUDE_TYPE = "" | the number of a case-type folder ("4" -> the
#   folder whose name starts with "Case 4 ("). Every run of that type is
#   removed from training AND validation -- the frozen stratified split is
#   FILTERED, not redrawn, so the other nine types' validation runs are the
#   same in every fold -- and appended to the test set, so a fold is scored
#   on the 70 standard cases plus the held-out type's own runs (one row per
#   case in summary_errors.csv; the analysis separates them by name, and
#   export_results files these runs under a "loto_" family so their mixed
#   Test_MAE_Overall never sits next to a full-set number). Answers "what
#   does the model do on an operating regime it has never seen", with the
#   data already in hand. Tag `xt<N>`.
TRAIN_EXCLUDE_TYPE = os.environ.get("TRAIN_EXCLUDE_TYPE", "").strip()
if TRAIN_EXCLUDE_TYPE:
    assert TRAIN_EXCLUDE_TYPE.isdigit(), "TRAIN_EXCLUDE_TYPE is the case-type number, got %r" % TRAIN_EXCLUDE_TYPE
    assert VAL_SPLIT == "stratified", "TRAIN_EXCLUDE_TYPE needs VAL_SPLIT=stratified (the frozen split is filtered)"

# ------------------------------------------------------------
# Split file (2026-10-11, Round 10 / H1)
# ------------------------------------------------------------
# SPLIT_NAME = "" (default: the legacy formula stratified_v<VAL_PER_TYPE>_s<SPLIT_SEED>.json)
#            | the basename of a json in splits/. Besides the legacy keys a
#            split json may carry `drop` (files removed from the training
#            pool before anything else -- never test files), `tavg_mask`
#            (H2), `tag` (the run-name token, e.g. vs3; default vs<N>) and
#            `split_name`. The json is read again, and validated, in
#            data.load_all_data; here only the tag is needed for RUN_NAME.
SPLIT_NAME = os.environ.get("SPLIT_NAME", "").strip()
SPLIT_TAG = "vs%d" % VAL_PER_TYPE
if SPLIT_NAME:
    assert VAL_SPLIT == "stratified", "SPLIT_NAME needs VAL_SPLIT=stratified"
    assert os.path.basename(SPLIT_NAME) == SPLIT_NAME and SPLIT_NAME.endswith(".json"), \
        "SPLIT_NAME is the basename of a json in splits/, got %r" % SPLIT_NAME
    _split_p = os.path.join(BASE_DIR, "splits", SPLIT_NAME)
    assert os.path.isfile(_split_p), "SPLIT_NAME: %s does not exist" % _split_p
    _split_tag = json.load(open(_split_p, encoding="utf-8")).get("tag")
    if _split_tag:
        assert str(_split_tag).isalnum(), "split tag must be alphanumeric, got %r" % (_split_tag,)
        SPLIT_TAG = str(_split_tag)


# ------------------------------------------------------------
# Per-file T_avg loss mask (2026-10-11, Round 10 / H2)
# ------------------------------------------------------------
# TAVG_MASK = ""            legacy: every file's T_avg label is trained on
#           | "split"       the json's `tavg_mask` list: those training files
#                           lose their T_avg term in the loss, those validation
#                           files are excluded from the T_avg part of the
#                           validation scalar (tag mk)
#           | "random:<s>"  B7 control: the SAME NUMBER of training files as
#                           the json masks among training files, drawn
#                           uniformly from the training list with seed s;
#                           validation mask = the json's (tag mkr<s>)
#           | "valonly"     B8 control: training labels untouched, validation
#                           mask = the json's (tag mkv)
# The dataset carries a per-file loss_mask (3,) with the T_avg entry 0 when
# masked; models.weighted_loss and train.validate_epoch consume it. The
# scaler fit is untouched. data.get_tavg_mask_sets is the single routing
# point for every later T_avg statistic (A4, H5, H6).
TAVG_MASK = os.environ.get("TAVG_MASK", "").strip()
TAVG_MASK_SEED = None
TAVG_MASK_TAG = ""
if TAVG_MASK:
    assert VAL_SPLIT == "stratified", "TAVG_MASK needs VAL_SPLIT=stratified (the mask lives in the split json)"
    assert PHYSICS_BOUND_WEIGHT == 0.0, "TAVG_MASK and PHYSICS_BOUND_WEIGHT > 0 are exclusive (masked files have no T_avg term)"
    if TAVG_MASK.startswith("random:"):
        TAVG_MASK_SEED = int(TAVG_MASK.split(":", 1)[1])
        TAVG_MASK_TAG = "mkr%d" % TAVG_MASK_SEED
    elif TAVG_MASK == "split":
        TAVG_MASK_TAG = "mk"
    elif TAVG_MASK == "valonly":
        TAVG_MASK_TAG = "mkv"
    else:
        raise AssertionError("TAVG_MASK must be '', split, valonly or random:<seed>, got %r" % TAVG_MASK)


# ------------------------------------------------------------
# Algebraic T_avg relabel (2026-10-11, Round 10 / A4)
# ------------------------------------------------------------
# RELABEL_CSV = ""  (default) labels as given
#             | the label table tavg_convention_by_file.csv (a basename in
#               splits/, or a path). For every TRAINING / VALIDATION file the
#               table marks inner-weighted or borderline, with inner share
#               s = inner_share_pct / 100, T_avg is replaced by the line
#               average  T_line = (T_avg - s * T_inner) / (1 - s)  in
#               data.load_all_data -- before the scaler fit and before any
#               mask logic. Test files are never relabelled. Tag `rl`; path
#               + sha256 into run_config.json; eval_checkpoint re-applies it.
RELABEL_CSV = os.environ.get("RELABEL_CSV", "").strip()
RELABEL_CSV_PATH = None
RELABEL_TAG = ""
if RELABEL_CSV:
    _rl = RELABEL_CSV
    if not (os.path.isabs(_rl) or "/" in _rl or os.sep in _rl):
        _rl = os.path.join(BASE_DIR, "splits", _rl)
    assert os.path.isfile(_rl), "RELABEL_CSV: %s does not exist" % _rl
    RELABEL_CSV_PATH = os.path.normpath(_rl)
    RELABEL_TAG = "rl"


def split_file_name():
    """Basename of the split json in use (legacy formula unless SPLIT_NAME)."""
    return SPLIT_NAME or "stratified_v{}_s{}.json".format(VAL_PER_TYPE, SPLIT_SEED)


def split_file_path():
    return os.path.join(BASE_DIR, "splits", split_file_name())


# ------------------------------------------------------------
# TF32 (2026-10-11, Round 10 / H9)
# ------------------------------------------------------------
# Nothing pinned the Ampere+ TF32 switches before Round 10, so the numbers
# depended on the framework default and on the GPU generation.
#   "0"      (default) matmul.allow_tf32 = cudnn.allow_tf32 = False: exact
#            fp32 on every GPU. New Round 10 runs.
#   "1"      both True.
#   "legacy" the torch default every run through Round 9 actually used:
#            matmul False, cuDNN True. Verified 2026-10-11 on the 4090: it
#            reproduces the champion's archived predictions.npz BIT-FOR-BIT
#            (3 test cases, every step but the H9-fixed last one), whereas
#            "0" and "1" differ from it by 1e-2 .. 6e-2 C per step. Use it
#            for every re-score of a pre-Round-10 checkpoint
#            (eval_checkpoint does so by default when run_config.json has
#            no tf32 key).
# Applied in set_seed (every seed, and eval_checkpoint's re-import); recorded
# in run_config.json with the torch / CUDA / cuDNN versions and the GPU name.
# Tags: "1" -> tf32, "legacy" -> tf32c, "0" -> none (legacy names unchanged).
TF32 = os.environ.get("TF32", "0").strip()
assert TF32 in ("0", "1", "legacy"), "TF32 must be 0, 1 or legacy, got %r" % TF32


def tf32_switches(mode):
    """(matmul_allow_tf32, cudnn_allow_tf32) for a TF32 knob value."""
    return (mode == "1", mode in ("1", "legacy"))


TF32_MATMUL, TF32_CUDNN = tf32_switches(TF32)
TF32_ENABLED = (TF32 != "0")           # any TF32 switch on

# ------------------------------------------------------------
# SEAL_TEST (2026-10-11, Round 10 / H3)
# ------------------------------------------------------------
# "1": a training run never scores the test set. main.py scores the
#      VALIDATION files through the same test_model path instead (masked
#      files unscored on T_avg), writes Val_<metric> keys to meta.json,
#      summary_errors_val.csv / predictions_val.npz, and "sealed": true to
#      run_config.json and meta; no Test_* key is ever written. The test set
#      is read once, later, by eval_checkpoint --set test (F11). A sealed and
#      a test-scored run share a run name (no tag); the done.flag resume
#      refuses to reuse one as the other (main.check_cached_meta_seal).
# "0": (default) the legacy flow -- train, then score the 70 test cases.
SEAL_TEST = os.environ.get("SEAL_TEST", "0").strip() not in ("0", "", "false", "False")

_LW = "-".join(f"{w:g}" for w in LOSS_WEIGHTS)
_VTAG = "-".join(VARIANTS)
# Run-name scheme (2026-08-06 rebuild): only the knobs that VARY between runs
# are folded in; constants (epoch cap, ES patience, pad mode when default) live
# in run_config.json. Kept short deliberately -- the old scheme plus new tags
# was pushing full plot paths toward the Windows 260-char limit.
# RUN_PREFIX (2026-10-12): the leading token used to be the fixed string
# "2026-08-06" (the date of the naming rebuild), which read like a run date.
# Round 10 runs are prefixed "r10"; set RUN_PREFIX=2026-08-06 to rebuild a
# Round 9 name. run_round9.RUN_PREFIX reads the same env var.
RUN_PREFIX = os.environ.get("RUN_PREFIX", "r10").strip() or "r10"
_parts = [f"{RUN_PREFIX}_{_VTAG}"]
if any(v != "forward_direct" for v in VARIANTS):
    _parts.append(f"W{WINDOW_SIZE}")             # window matters for sliding variants only
_parts += [f"Tin-{TINNER_MODE}", f"l{ANCHOR_LEAD}", f"w{_LW}",
           f"oth-{OTHER_CH_MODE}",
           f"h{HIDDEN_SIZE}x{NUM_LAYERS}dp{_DP_TAG}"]
if LOSS_W0_MODE:
    _parts.append("w0ikc")
if ENC_WIDTHS != "256,256":
    _parts.append("enc" + ("-".join(str(v) for v in ENC_WIDTHS_LIST) or "l"))
if H0_MODE != "linear":
    _parts.append("h0t" if H0_MODE == "tanh" else "h0z")
if INIT_ENCODER != "point":
    _parts.append("ew%d" % WINDOW_SIZE)
if NO_TIME:
    assert POS_TIME_PRIOR == "0", "NO_TIME cannot use a position time prior"
    _parts.append("nt")
if EARLY_STEP_WEIGHT != 1:
    _parts.append("esw%g" % EARLY_STEP_WEIGHT)
if OTHER_CH_MODE == "ratio_input":
    _parts.append("rt%s%s" % (RATIO_CHANNELS, "k" if RATIO_REF == "kelvin" else "c"))
if OTHER_CH_MODE == "anchor_initial":
    _parts.append("init")
if PHYS_MODE:
    _parts.append("ph" + {"residual": "r", "input": "i", "input_residual": "ir"}[PHYS_MODE])
if INIT_BURNIN_STEPS:
    _parts.append('bi%d' % INIT_BURNIN_STEPS)
if SYNTHETIC_REST_FILES:
    _parts.append('rest%d' % SYNTHETIC_REST_FILES)
if CONSTANTS_FROM:
    import hashlib
    with open(CONSTANTS_FROM, 'rb') as _frozen_file:
        _parts.append('cf' + hashlib.sha256(_frozen_file.read()).hexdigest()[:12])
if INPUT_LOOKAHEAD:
    _parts.append(f"la{INPUT_LOOKAHEAD}")
if CASE_FLAG_INPUT:
    assert all(v in ("forward_direct", "abs_sliding") for v in VARIANTS), \
        "CASE_FLAG_INPUT supports forward_direct / abs_sliding only (got %r)" % (VARIANTS,)
    _parts.append("fi" + CASE_FLAG_INPUT[0])          # fic / fip / fiz
if EXCLUDE_BOTH_PHASE:
    _parts.append("xb")
if TOUTER_MODE != "abs":
    assert OTHER_CH_MODE in ("abs", "pos_head", "anchor_avg", "anchor_avg_grad"), \
        "TOUTER_MODE=anchor_input re-parametrises head 1; OTHER_CH_MODE=%r does too" % (OTHER_CH_MODE,)
    _parts.append("To-ai" if TOUTER_MODE == "anchor_input"   # T_outer anchored on Input_T
                  else "To-em%d%s" % (TOUTER_TAU, "x" if TOUTER_EMA_FORM == "exact" else ""))  # ... on its EMA(tau); x = exact discretisation
    if TOUTER_SCALE != 1.0:
        _parts.append("os%g" % TOUTER_SCALE)
if LATEST_PARAMS['max_epochs'] != 1000:
    # A MAX_EPOCHS smoke run gets its own directory, so it can never share a
    # RUN_NAME with (and be resumed as) a real seed (2026-09-12 review).
    _parts.append("ep%d" % LATEST_PARAMS['max_epochs'])
if PHYSICS_BOUND_WEIGHT:
    _parts.append(f"pb{PHYSICS_BOUND_WEIGHT:g}")
if ANCHOR_SCALE != 1.0:
    _parts.append(f"as{ANCHOR_SCALE:g}")
if POS_GAP_FLOOR:
    _parts.append(f"fl{POS_GAP_FLOOR:g}" + ("s" if POS_FLOOR_SOFT else ""))
if POS_TEMP_GATE:
    _parts.append(f"tg{POS_TEMP_GATE:g}" + (f"s{POS_TEMP_SOFT:g}" if POS_TEMP_SOFT else ""))
if POS_CASE_GATE:
    _parts.append("cgate")
if POS_LEARNED_GATE:
    _parts.append("lg" + ("p" if POS_LEARNED_POOL else "")
                  + (f"b{POS_GATE_BIAS:g}" if POS_GATE_BIAS != 2.0 else ""))
if POS_ANCHORED_FALLBACK:
    _parts.append("afb")
if POS_FIT_CLEAN:
    _parts.append("cf")
if POS_TIME_PRIOR != "0":
    _parts.append("tp")                      # time prior on the position head
if (POS_ABS_HEAD and not POS_TEMP_GATE and not POS_LEARNED_GATE
        and not POS_ANCHORED_FALLBACK):
    _parts.append("ah")
if MODEL_SPLIT == "independent":
    _parts.append("ind")
elif MODEL_SPLIT == "avg_only":
    _parts.append("inda")
if VAL_SPLIT == "stratified":
    _parts.append(SPLIT_TAG)                 # vs<N>, or the json's tag (vs3) under SPLIT_NAME (H1)
    if TAVG_MASK_TAG:
        _parts.append(TAVG_MASK_TAG)         # mk / mkr<seed> / mkv (H2)
if RELABEL_TAG:
    _parts.append(RELABEL_TAG)               # rl: algebraically relabelled T_avg (A4)
if TRAIN_EXCLUDE_TYPE:
    _parts.append("xt%s" % TRAIN_EXCLUDE_TYPE)      # leave-one-type-out fold
if TF32 == "1":
    _parts.append("tf32")                    # H9: both TF32 switches on
elif TF32 == "legacy":
    _parts.append("tf32c")                   # H9: cuDNN-only TF32 (pre-Round-10 numerics)
if SLIDING_PAD_MODE != "variable":
    _parts.append(SLIDING_PAD_MODE)
RUN_NAME_BASE = "_".join(_parts)
RUN_NAME = RUN_NAME_BASE                       # mutated per-seed at runtime in main loop
RUNS_ROOT_DIR = "runs"                         # parent folder under src/

# ------------------------------------------------------------
# Persistence / resume
# ------------------------------------------------------------
# RESUME_ENABLED = True:
#   * If `runs/<RUN_NAME>/variants/<v>/done.flag` exists, skip that variant
#     entirely and reuse its saved meta + history + summary CSV.
#   * If `runs/<RUN_NAME>/variants/<v>/resume_state.pt` exists (training was
#     interrupted), restore optimizer / scheduler / model / RNG state
#     and continue from the next epoch.
# RESUME_ENABLED = False:
#   * Always retrain from scratch (overwrites previous artifacts).
RESUME_ENABLED = True
SAVE_PREDICTIONS = True            # raw inv_pred / inv_actual per case (for replotting)
CHECKPOINT_EVERY_N_EPOCHS = 1      # save resume_state.pt every N epochs

# ------------------------------------------------------------
# Data source -- 2026-04-26 newly uploaded ML-Group dataset
# ------------------------------------------------------------
# The 2026-04-26 upload lives at:
#   raw/R199 299 AI-TES Project_Arnold Team/2026 Spring/ML Group/data/
#   |- training_data/{10s, 10s_with_burn_in, 150s, 150s_with_burn_in, original}/
#   |- tests/{test_in_10s, test_in_10s_with_burn_in, test_in_150s,
#   |         test_in_150s_with_burn_in, test_with_inputs}/
#   |- Convection/    (6-col schema -- NOT used by this ablation)
#
# Set TRAIN_SUBDIR / TEST_SUBDIR to switch between burn-in / no-burn-in
# variants without touching the loader.
# 2026-07-16: aligned with the protocol every comparable run actually used
# (see run_config.json of the 2026-05-21 zero baseline AND the init sweep):
# train on the Latest Database, test on the fixed 70-case set.
# 2026-08-06: data now lives in the repo at AI-TES/data/ with a FLAT layout
# (no training_data/ or tests/ level). Verified byte-identical to the data all
# previous runs used: train 311/311 and test 70/70 files match by MD5, so
# results stay directly comparable.
TRAIN_SUBDIR = "Latest Database (Use this for training)"
TEST_SUBDIR = "Test_70_cases"  # fixed 70-case test set (baseline protocol)

# Explicit data-root candidates, tried in order. The first existing path wins.
NEW_DATA_ROOT_CANDIDATES = [
    # 2026-08-06: repo-local data (AI-TES/data/), flat layout. BASE_DIR is
    # src/models/GRU, so three levels up is the repo root. Tried first so a
    # fresh clone works with no machine-specific path editing.
    os.path.join("..", "..", "..", "data"),
    # When the script is run from this repo's src/ dir, the new data lives
    # outside ML-model-for-thermal-predictions/ -- under the Arnold-team folder.
    os.path.join(
        "..", "..", "..",
        "R199 299 AI-TES Project_Arnold Team",
        "2026 Spring", "ML Group", "data",
    ),
    # Fallback: alternative project layouts.
    os.path.join(
        "..", "..", "..", "..",
        "R199 299 AI-TES Project_Arnold Team",
        "2026 Spring", "ML Group", "data",
    ),
    os.path.join(
        "..", "..",
        "R199 299 AI-TES Project_Arnold Team",
        "2026 Spring", "ML Group", "data",
    ),
]
# =============================================================
# >>> EDIT THIS IF THE RELATIVE PATHS ABOVE DON'T RESOLVE ON YOUR MACHINE <<<
#
# Point it at the directory that DIRECTLY CONTAINS the three subfolders
#     `Convection/`, `tests/`, `training_data/`.
#
# Leave the empty string ("") to disable.
#
# Example (this is what currently works on my machine, replace with YOUR own path):
#     NEW_DATA_ROOT_ABSOLUTE = (
#         r"C:\Users\FEvMo\Documents\GitHub\calit2-thermal-wiki\raw"
#         r"\R199 299 AI-TES Project_Arnold Team\2026 Spring\ML Group\data"
#     )
#
# Other plausible shapes:
#     NEW_DATA_ROOT_ABSOLUTE = r"D:\projects\TES\ML Group\data"            # Windows
#     NEW_DATA_ROOT_ABSOLUTE = "/home/<user>/calit2-thermal-wiki/raw/R199 299 AI-TES Project_Arnold Team/2026 Spring/ML Group/data"  # Linux
#     NEW_DATA_ROOT_ABSOLUTE = "/Users/<user>/.../ML Group/data"           # macOS
# =============================================================
DATA_ROOT = os.environ.get("DATA_ROOT", "").strip()
NEW_DATA_ROOT_ABSOLUTE = DATA_ROOT

# Legacy fallback (old 177-file dataset under
# raw/ML-model-for-thermal-predictions/data/) -- only used if every
# new-data candidate above fails to resolve.
LEGACY_TRAIN_REL = os.path.join("data", "data_in_10s")
LEGACY_TEST_REL = os.path.join("data", "test_in_10s")

# ------------------------------------------------------------
# Manual data split (per Arnold: redo the split ourselves, don't
# rely on the FE-group-provided train_dir / test_dir partition)
# ------------------------------------------------------------
# When MANUAL_SPLIT_ENABLED = True, all CSVs from training_data/<TRAIN_SUBDIR>/
# AND tests/<TEST_SUBDIR>/ are pooled, deterministically shuffled, then split
# OUTER (test) and INNER (val) at the configured fractions.
#
#   Outer: train+val / test     = (1 - OUTER_TEST_FRAC) / OUTER_TEST_FRAC
#   Inner: train     / val      = (1 - INNER_VAL_FRAC ) / INNER_VAL_FRAC
#
# All 4 variants receive the IDENTICAL split (same train/val/test file lists)
# because shuffling uses SPLIT_SEED, applied once at load time.
# 2026-07-16: False = folder-as-given (fixed 70-case test set, no supplement).
# This is what the zero baseline AND the init sweep actually ran with
# (their run_config.json all say manual_split_enabled: false). True would
# re-pool train+test and top the test set up to 76 cases -- NOT comparable.
MANUAL_SPLIT_ENABLED = False
OUTER_TEST_FRAC = 0.20   # 80/20 train+val / test  (Arnold confirmed; used only when manual split is on)
INNER_VAL_FRAC = 0.05    # 95/5  train     / val   (existing convention since GRU V1)
SPLIT_SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # H9: pin the TF32 switches (process-global) on every seed.
    torch.backends.cuda.matmul.allow_tf32 = TF32_MATMUL
    torch.backends.cudnn.allow_tf32 = TF32_CUDNN


def runtime_env_info():
    """torch / CUDA / cuDNN versions, GPU name and the TF32 switches as they
    are set right now -- written to run_config.json (H9) so a result can be
    tied to the numerics it was produced with."""
    info = {
        'tf32': TF32,
        'torch': str(torch.__version__),
        'cuda': str(torch.version.cuda) if torch.version.cuda else None,
        'cudnn': torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'tf32_matmul': bool(torch.backends.cuda.matmul.allow_tf32),
        'tf32_cudnn': bool(torch.backends.cudnn.allow_tf32),
    }
    return info


set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


