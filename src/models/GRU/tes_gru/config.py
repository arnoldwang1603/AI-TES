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
SEEDS = [7, 21, 42, 123]
SEED = SEEDS[0]                 # legacy: kept for any code referencing it as a single-seed default
WINDOW_SIZE = 10                # steps used by abs_window's WindowInitStateEncoder

LATEST_PARAMS = dict(
    hidden_size=128,
    num_layers=5,
    dropout=0.3,
    lr=0.0025,
    # Per Arnold (2026-05-20): train for the full 1200 epochs without
    # early stopping. Best-val checkpoint is still saved during training
    # and restored before testing, so we report the best-epoch model
    # rather than the final-epoch one.
    max_epochs=1200,
    batch_size=16,
    early_stop_patience=10**9,   # effectively disabled
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
# Overridable via the SLIDING_PAD_MODE env var so one launch can sweep both
# "init" and "variable" without editing this file; RUN_NAME_BASE below folds
# the mode into the run-dir name.
SLIDING_PAD_MODE = os.environ.get("SLIDING_PAD_MODE", "init")   # "variable" | "init" | "zero"
assert SLIDING_PAD_MODE in ("variable", "init", "zero"), \
    f"SLIDING_PAD_MODE must be variable/init/zero, got {SLIDING_PAD_MODE!r}"


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
VARIANTS = [
    'abs_sliding',            # forward sliding (production-valid; the headline config)
]
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
    'abs_sliding': 5,
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
RUN_NAME_BASE = f"2026-06-28_abs_sliding_1200ep_P0_{SLIDING_PAD_MODE}"   # <-- pad mode folded in; set via SLIDING_PAD_MODE env var
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
TRAIN_SUBDIR = "10s"           # or "10s_with_burn_in" / "150s" / etc.
TEST_SUBDIR = "test_in_10s"    # or "test_in_10s_with_burn_in" / etc.

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
MANUAL_SPLIT_ENABLED = True
OUTER_TEST_FRAC = 0.20   # 80/20 train+val / test  (Arnold confirmed)
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


