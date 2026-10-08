"""Configuration, hyperparameters, and training-setup primitives.

Holds every run constant plus bootstrap helpers (set_seed, DEVICE, warmup_lr,
teacher_forcing_prob, apply_orthogonal_init_gru, is_inverse).
"""
import os
import random

import numpy as np
import torch
import torch.nn as nn

# Directory that CONTAINS this package (i.e. src/). Anchors runs/ output
# dir and relative data-root candidates.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# 1. Config
# ============================================================
# Seeds protocol: paired design across all configurations.
SEEDS = [int(s) for s in os.environ.get("SEEDS", "7,21,42,123").split(",") if s.strip()]
assert SEEDS, "SEEDS parsed to an empty list"
SEED = SEEDS[0]

# Sliding-window length W (default/fallback when DYNAMIC_WINDOW_MODE == 'fixed')
WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", "10"))
assert 1 <= WINDOW_SIZE <= 200, f"unreasonable WINDOW_SIZE={WINDOW_SIZE}"

LATEST_PARAMS = dict(
    hidden_size=128,
    num_layers=5,
    dropout=0.3,
    lr=0.0025,
    max_epochs=1000,
    batch_size=16,
    early_stop_patience=150,
    sched_patience=50,
    sched_factor=0.5,
)

# ------------------------------------------------------------
# Teacher forcing schedule
# ------------------------------------------------------------
TF_INIT_PROB = 0.5
TF_DECAY_EPOCHS = 100

# ------------------------------------------------------------
# Modern RNN training defaults
# ------------------------------------------------------------
GRADIENT_CLIP_NORM = 1.0
ORTHOGONAL_INIT_RECURRENT = True
LR_WARMUP_EPOCHS = 60

# ------------------------------------------------------------
# Sliding-window padding mode (abs_sliding / inverse_abs_sliding)
# ------------------------------------------------------------
# "variable" | "init" | "zero"
SLIDING_PAD_MODE = os.environ.get("SLIDING_PAD_MODE", "variable")
assert SLIDING_PAD_MODE in ("variable", "init", "zero"), \
    f"SLIDING_PAD_MODE must be variable/init/zero, got {SLIDING_PAD_MODE!r}"

# ------------------------------------------------------------
# T_inner handling in the forward sliding variant
# ------------------------------------------------------------
# "arfed" | "anchor" | "output_only"
TINNER_MODE = os.environ.get("TINNER_MODE", "anchor")
assert TINNER_MODE in ("arfed", "anchor", "output_only"), \
    f"TINNER_MODE must be arfed/anchor/output_only, got {TINNER_MODE!r}"

# ------------------------------------------------------------
# Anchor lead (2026-07-23)
# ------------------------------------------------------------
ANCHOR_LEAD = int(os.environ.get("ANCHOR_LEAD", "1"))
assert ANCHOR_LEAD in (0, 1)

# ------------------------------------------------------------
# Loss shaping & physics bounds
# ------------------------------------------------------------
# Per-channel loss weights in OUTPUT order (T_inner, T_outer, T_avg).
LOSS_WEIGHTS = [float(x) for x in
                os.environ.get("LOSS_WEIGHTS", "1,6,3").split(",")]
assert len(LOSS_WEIGHTS) == 3, "LOSS_WEIGHTS needs 3 comma-separated values"

PHYSICS_BOUND_WEIGHT = 0.0

# ------------------------------------------------------------
# PINN Temporal Smoothness Weights (Arm F / G)
# ------------------------------------------------------------
SMOOTH_TO_WEIGHT = float(os.environ.get("SMOOTH_TO_WEIGHT", "0.0"))
SMOOTH_TA_WEIGHT = float(os.environ.get("SMOOTH_TA_WEIGHT", "0.0"))

# Dynamic Weight Annealing (Arm G)
USE_DWA = (os.environ.get("USE_DWA", "0") == "1")

# Soft-Start Rollout Blending (Arm G / Arm J)
SOFT_START_TAU_VAL = os.environ.get("SOFT_START_TAU", "0.0")
SOFT_START_TAU = float(SOFT_START_TAU_VAL) if SOFT_START_TAU_VAL else 0.0

# ------------------------------------------------------------
# Early Loss Weighting (Arm I & Baseline for subsequent arms)
# ------------------------------------------------------------
EARLY_LOSS_WEIGHT = float(os.environ.get("EARLY_LOSS_WEIGHT", "1.0"))
EARLY_LOSS_TAU = float(os.environ.get("EARLY_LOSS_TAU", "6.0"))

# ------------------------------------------------------------
# Exploration Configurations (Arms L, M, N, O, P)
# ------------------------------------------------------------
# Arm L: Channel-Decoupled Early Loss Multipliers & Taus (Ti, To, Ta)
EARLY_CH_WEIGHTS_STR = os.environ.get("EARLY_CH_WEIGHTS", "")
EARLY_CH_WEIGHTS = [float(x) for x in EARLY_CH_WEIGHTS_STR.split(",")] if EARLY_CH_WEIGHTS_STR else None
EARLY_CH_TAUS_STR = os.environ.get("EARLY_CH_TAUS", "")
EARLY_CH_TAUS = [float(x) for x in EARLY_CH_TAUS_STR.split(",")] if EARLY_CH_TAUS_STR else None

# Arm M: Early Transient Derivative / Velocity Matching Loss
EARLY_DERIV_WEIGHT = float(os.environ.get("EARLY_DERIV_WEIGHT", "0.0"))
EARLY_DERIV_TAU = float(os.environ.get("EARLY_DERIV_TAU", "6.0"))

# Arm N: Convex Spatial Bounding Head for T_avg
USE_SPATIAL_ANCHOR = (os.environ.get("USE_SPATIAL_ANCHOR", "0") == "1")

# Arm O & Arms V–Y: Transient-Localized PINN Smoothness (2nd-Order Curvature Damping)
EARLY_SMOOTH_TO_WEIGHT = float(os.environ.get("EARLY_SMOOTH_TO_WEIGHT", "0.0"))
EARLY_SMOOTH_TA_WEIGHT = float(os.environ.get("EARLY_SMOOTH_TA_WEIGHT", "0.0"))
EARLY_SMOOTH_TAU = float(os.environ.get("EARLY_SMOOTH_TAU", "6.0"))

# ------------------------------------------------------------
# PINN magnitude controls (Arms Z1-Z4, 2026-09-22)
# ------------------------------------------------------------
# PINN_NORM: "mean" (legacy) averages the exp(-t/tau)-decayed penalty over the
#   WHOLE ~1440-step horizon, diluting it ~220x. "weighted" divides by the sum
#   of the decay weights instead, so the penalty is a true weighted mean over
#   the early window it targets.
PINN_NORM = os.environ.get("PINN_NORM", "mean")
assert PINN_NORM in ("mean", "weighted"), f"PINN_NORM must be mean/weighted, got {PINN_NORM!r}"

# PINN_BALANCE_RATIO > 0 enables loss-ratio balancing: a global scale k on
# every PINN term is re-set each epoch (EMA in log space, max 2x change per
# epoch) so that  L_PINN ~= ratio * L_data.  0 disables (fixed weights).
PINN_BALANCE_RATIO = float(os.environ.get("PINN_BALANCE_RATIO", "0"))
PINN_BALANCE_START = int(os.environ.get("PINN_BALANCE_START", "60"))  # after LR warmup
PINN_SCALE_MIN, PINN_SCALE_MAX = 1e-3, 1e6

# Lumped energy-balance residual on T_avg (full horizon, raw degC):
#   dTa = a*(Ti - Ta) + b*(To - Ta) + c ,  (a, b, c) least-squares fit on the
#   training set at startup. Penalty = mean(residual^2) / sigma_fit^2.
PINN_ENERGY_WEIGHT = float(os.environ.get("PINN_ENERGY_WEIGHT", "0"))

# Validation / checkpoint metric: "total" (legacy: full training objective
# incl. PINN) or "data" (data term only -- required when PINN weights move).
VAL_METRIC = os.environ.get("VAL_METRIC", "total")
assert VAL_METRIC in ("total", "data"), f"VAL_METRIC must be total/data, got {VAL_METRIC!r}"

# Arm P: Progressive Rollout Horizon Curriculum (Short-to-Long BPTT)
CURRICULUM_WARMUP_EPOCHS = int(os.environ.get("CURRICULUM_WARMUP_EPOCHS", "0"))
CURRICULUM_MIN_HORIZON = int(os.environ.get("CURRICULUM_MIN_HORIZON", "15"))

# Recurrent Steady-State Hidden Warmup Steps (Arm K)
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "0"))

# ------------------------------------------------------------
# Dynamic Sliding Window Configurations (Arms Q, R, S, T, U, V, W, X, Y)
# ------------------------------------------------------------
# "fixed" | "step" | "ramp" | "flux_adaptive" | "dilated" | "asymmetric"
DYNAMIC_WINDOW_MODE = os.environ.get("DYNAMIC_WINDOW_MODE", "fixed")
WINDOW_EARLY = int(os.environ.get("WINDOW_EARLY", "3"))
WINDOW_LATE = int(os.environ.get("WINDOW_LATE", "20"))
WINDOW_SPLIT_STEP = int(os.environ.get("WINDOW_SPLIT_STEP", "10"))
WINDOW_MIN = int(os.environ.get("WINDOW_MIN", "2"))
WINDOW_MAX = int(os.environ.get("WINDOW_MAX", "25"))
WINDOW_RAMP_STEPS = int(os.environ.get("WINDOW_RAMP_STEPS", "40"))
FLUX_SCALE = float(os.environ.get("FLUX_SCALE", "0.05"))
WINDOW_TOKENS = int(os.environ.get("WINDOW_TOKENS", "8"))
WINDOW_FAST = int(os.environ.get("WINDOW_FAST", "3"))
WINDOW_SLOW = int(os.environ.get("WINDOW_SLOW", "18"))

# ------------------------------------------------------------
# T_outer / T_avg output parameterization
# ------------------------------------------------------------
OTHER_CH_MODE = os.environ.get("OTHER_CH_MODE", "abs")   # "abs" | "persistence"
assert OTHER_CH_MODE in ("abs", "persistence"), \
    f"OTHER_CH_MODE must be abs/persistence, got {OTHER_CH_MODE!r}"


def warmup_lr(epoch, base_lr, warmup_epochs=None):
    """Return LR to use at this epoch during the warmup phase."""
    if warmup_epochs is None:
        warmup_epochs = LR_WARMUP_EPOCHS
    if warmup_epochs <= 0 or epoch >= warmup_epochs:
        return None
    return base_lr * (epoch + 1) / warmup_epochs


def apply_orthogonal_init_gru(gru):
    """Apply orthogonal init to each per-gate sub-matrix of weight_hh_l*."""
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


# Default variants run in a fresh sweep
VARIANTS = [v.strip() for v in
            os.environ.get("VARIANTS", "abs_sliding").split(",") if v.strip()]


# Per-variant model OUTPUT channel labels
VARIANT_OUTPUT_CHANNELS = {
    # Forward variants -- output is (T_inner, T_outer, T_avg) at t+1
    'delta':              ['T_inner', 'T_outer', 'T_avg'],
    'abs+delta':          ['T_inner', 'T_outer', 'T_avg'],
    'abs':                ['T_inner', 'T_outer', 'T_avg'],
    'abs_window':         ['T_inner', 'T_outer', 'T_avg'],
    'abs_sliding':        ['T_inner', 'T_outer', 'T_avg'],
    'forward_direct':     ['T_inner', 'T_outer', 'T_avg'],
    # Inverse variants -- output is (Input_T, T_inner, T_outer) at t+1
    'inverse_delta':       ['Input_T', 'T_inner', 'T_outer'],
    'inverse_abs+delta':   ['Input_T', 'T_inner', 'T_outer'],
    'inverse_abs':         ['Input_T', 'T_inner', 'T_outer'],
    'inverse_abs_sliding': ['Input_T', 'T_inner', 'T_outer'],
}


def is_inverse(variant):
    return variant.startswith('inverse_')


# Per-variant GRU input dimensionality
INPUT_DIMS = {
    'delta':          5,    # Time + 4 deltas
    'abs+delta':      9,    # 5 abs + 4 deltas
    'abs':            5,    # Time + T_outer + T_inner + T_avg + Input_T
    'abs_window':     5,
    'abs_sliding':    4 if TINNER_MODE == "output_only" else 5,
    'inverse_delta':       2,
    'inverse_abs+delta':   3,
    'inverse_abs':         2,
    'inverse_abs_sliding': 2,
    'forward_direct':      2,
}

# ------------------------------------------------------------
# Output layout & Directory Naming Tags
# ------------------------------------------------------------
_LW = "-".join(f"{w:g}" for w in LOSS_WEIGHTS)
_VTAG = "-".join(VARIANTS)
RUN_NAME_BASE = (f"2026-07-23_{_VTAG}_W{WINDOW_SIZE}"
                 f"_{LATEST_PARAMS['max_epochs']}ep"
                 f"_ES{LATEST_PARAMS['early_stop_patience']}"
                 f"_P0_{SLIDING_PAD_MODE}_Tin-{TINNER_MODE}"
                 f"_lead{ANCHOR_LEAD}_w{_LW}_pb{PHYSICS_BOUND_WEIGHT:g}"
                 f"_oth-{OTHER_CH_MODE}")

# Dynamically append unique ablation tags
if SMOOTH_TO_WEIGHT > 0.0 or SMOOTH_TA_WEIGHT > 0.0:
    RUN_NAME_BASE += f"_smTo{SMOOTH_TO_WEIGHT:g}_smTa{SMOOTH_TA_WEIGHT:g}"
if USE_DWA:
    RUN_NAME_BASE += "_DWA"
if SOFT_START_TAU > 0.0:
    RUN_NAME_BASE += f"_softStart{SOFT_START_TAU:g}"
if EARLY_LOSS_WEIGHT > 1.0:
    RUN_NAME_BASE += f"_earlyL{EARLY_LOSS_WEIGHT:g}_tau{EARLY_LOSS_TAU:g}"
if EARLY_CH_WEIGHTS is not None:
    RUN_NAME_BASE += "_decoupledL"
if EARLY_DERIV_WEIGHT > 0.0:
    RUN_NAME_BASE += f"_derivW{EARLY_DERIV_WEIGHT:g}_tau{EARLY_DERIV_TAU:g}"
if USE_SPATIAL_ANCHOR:
    RUN_NAME_BASE += "_spatialAnc"

# Dynamic Sliding Window tags
if DYNAMIC_WINDOW_MODE == "step":
    RUN_NAME_BASE += f"_dynStep_e{WINDOW_EARLY}_l{WINDOW_LATE}_s{WINDOW_SPLIT_STEP}"
elif DYNAMIC_WINDOW_MODE == "ramp":
    RUN_NAME_BASE += f"_dynRamp_min{WINDOW_MIN}_max{WINDOW_MAX}_steps{WINDOW_RAMP_STEPS}"
elif DYNAMIC_WINDOW_MODE == "flux_adaptive":
    RUN_NAME_BASE += f"_dynFlux_min{WINDOW_MIN}_max{WINDOW_MAX}_scale{FLUX_SCALE:g}"
elif DYNAMIC_WINDOW_MODE == "dilated":
    RUN_NAME_BASE += f"_dynDilated_tok{WINDOW_TOKENS}"
elif DYNAMIC_WINDOW_MODE == "asymmetric":
    RUN_NAME_BASE += f"_dynAsym_fast{WINDOW_FAST}_slow{WINDOW_SLOW}"

# Transient Localized PINN tag (Arms O, V, W, X, Y)
if EARLY_SMOOTH_TO_WEIGHT > 0.0 or EARLY_SMOOTH_TA_WEIGHT > 0.0:
    RUN_NAME_BASE += f"_earlyPinnTo{EARLY_SMOOTH_TO_WEIGHT:g}_ta{EARLY_SMOOTH_TA_WEIGHT:g}"

if PINN_NORM == "weighted":
    RUN_NAME_BASE += "_pnormW"
if PINN_ENERGY_WEIGHT > 0.0:
    RUN_NAME_BASE += f"_energy{PINN_ENERGY_WEIGHT:g}"
if PINN_BALANCE_RATIO > 0.0:
    RUN_NAME_BASE += f"_pbal{PINN_BALANCE_RATIO:g}"
if VAL_METRIC == "data":
    RUN_NAME_BASE += "_valData"

if CURRICULUM_WARMUP_EPOCHS > 0:
    RUN_NAME_BASE += f"_currEp{CURRICULUM_WARMUP_EPOCHS}_minH{CURRICULUM_MIN_HORIZON}"
if WARMUP_STEPS > 0:
    RUN_NAME_BASE += f"_warmup{WARMUP_STEPS}"

RUN_NAME = RUN_NAME_BASE
RUNS_ROOT_DIR = "runs"

# ------------------------------------------------------------
# Persistence / resume
# ------------------------------------------------------------
RESUME_ENABLED = True
SAVE_PREDICTIONS = True
CHECKPOINT_EVERY_N_EPOCHS = 1

# ------------------------------------------------------------
# Data source
# ------------------------------------------------------------
TRAIN_SUBDIR = "Latest Database (Use this for training)"
TEST_SUBDIR = "70_cases"

NEW_DATA_ROOT_CANDIDATES = [
    os.path.join(
        "..", "..", "..",
        "R199 299 AI-TES Project_Arnold Team",
        "2026 Spring", "ML Group", "data",
    ),
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

NEW_DATA_ROOT_ABSOLUTE = "/home/cymaker/gru_project/data_temp"

LEGACY_TRAIN_REL = os.path.join("data", "data_in_10s")
LEGACY_TEST_REL = os.path.join("data", "test_in_10s")

MANUAL_SPLIT_ENABLED = False
OUTER_TEST_FRAC = 0.20
INNER_VAL_FRAC = 0.05
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
