"""Configuration, hyperparameters, and training-setup primitives.

Split out of the original GRU_input_ablation.py monolith. Holds every run
constant plus the small bootstrap helpers (set_seed, DEVICE, warmup_lr,
teacher_forcing_prob, apply_orthogonal_init_gru, is_inverse). Other modules
do `from .config import *`; runio/main also `from . import config` so they
can read/write the mutable RUN_NAME live.
"""
import os
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

LATEST_PARAMS = dict(
    hidden_size=128,
    num_layers=5,
    dropout=0.3,
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
    max_epochs=1000,
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
# Anchor lead (2026-07-23)
# ------------------------------------------------------------
# The anchor predicts T_inner(t+1). ANCHOR_LEAD=1 anchors it on Input_T(t+1),
# =0 on Input_T(t). Input_T is EXOGENOUS -- the whole curve is a given boundary
# condition known before the rollout starts -- so t+1 is equally causal and
# strictly better: the residual the head must learn drops from max 37.5 C /
# std 1.00 to max 8.5 C / std 0.49, because Input_T(t) is stale whenever the
# inlet jumps between steps (Case 40: inlet 217->263 in one step, giving a
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

# Physics bound penalty (also from the LSTM recipe): T_avg must lie between
# T_inner and T_outer. Verified on our data: holds at 98.76% of all timesteps
# (T_avg sits ~0.41 of the way from T_outer to T_inner). The penalty is a
# hinge on violations, applied in RAW temperature space (the MinMax scaling
# is per-channel, so the ordering is not preserved in scaled space).
# 0.0 disables it.
PHYSICS_BOUND_WEIGHT = float(os.environ.get("PHYSICS_BOUND_WEIGHT", "1.0"))

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
OTHER_CH_MODE = os.environ.get("OTHER_CH_MODE", "abs")   # "abs" | "persistence"
assert OTHER_CH_MODE in ("abs", "persistence"), \
    f"OTHER_CH_MODE must be abs/persistence, got {OTHER_CH_MODE!r}"


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
    'forward_direct': 2,
}

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
_LW = "-".join(f"{w:g}" for w in LOSS_WEIGHTS)
_VTAG = "-".join(VARIANTS)
RUN_NAME_BASE = (f"2026-07-23_{_VTAG}_W{WINDOW_SIZE}"
                 f"_{LATEST_PARAMS['max_epochs']}ep"
                 f"_ES{LATEST_PARAMS['early_stop_patience']}"
                 f"_P0_{SLIDING_PAD_MODE}_Tin-{TINNER_MODE}"
                 f"_lead{ANCHOR_LEAD}_w{_LW}_pb{PHYSICS_BOUND_WEIGHT:g}"
                 f"_oth-{OTHER_CH_MODE}")
# ^ window size, epoch budget, early-stop patience, pad mode, and T_inner
#   mode all folded into the run dir name; set via the WINDOW_SIZE /
#   SLIDING_PAD_MODE / TINNER_MODE env vars (see run_tinner_ablation.py and
#   run_window_sweep.py).
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
TRAIN_SUBDIR = "Latest Database (Use this for training)"
TEST_SUBDIR = "70_cases"       # fixed 70-case test set (baseline protocol)

# Explicit data-root candidates, tried in order. The first existing path wins.
NEW_DATA_ROOT_CANDIDATES = [
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
NEW_DATA_ROOT_ABSOLUTE = ""  # <-- TODO: set to your absolute data path, or leave "" if relative paths work

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


set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


