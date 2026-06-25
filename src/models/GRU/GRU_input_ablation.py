"""
================================================================================
GRU INPUT-FEATURE ABLATION  --  horizontal comparison on the 2026-04-26 data
================================================================================

WHAT THIS SCRIPT DOES
---------------------
Trains four GRU variants on the 10s thermal dataset with the "latest"
hyperparameters held fixed across all variants, then produces a single
side-by-side comparison report. The four variants only differ in (a) which
input features the GRU sees at each step and (b) how the initial hidden
state h0 is built.

VARIANTS (all share lr=0.0025, hidden=128, layers=5, dropout=0.3,
          batch=16, max_epochs=300, full autoregressive rollout, seed=42)

  1. delta        Inputs at step t: [Time(t), dT_outer(t), dInput_T(t),
                                     dT_avg(t)]                              (4-d)
                  Per Arnold's clarification (2026-05): dT is "current
                  minus previous step", NOT "next minus current". The
                  deltas are recomputed AT EACH STEP from the AR chain:
                      dT_outer(t) = T_outer_AR(t) - T_outer_AR(t-1)
                  where T_outer_AR(t) is the model's own prediction
                  from the previous iteration (or the GT initial value
                  at t=0). dInput_T(t) is exogenous (Input_T trajectory
                  is user-specified at deployment) so it stays at the
                  precomputed GT value. h0 from InitStateEncoder.
                  PRODUCTION-VALID -- no future GT in the input.

  2. abs+delta     Inputs at step t: [Time, T_outer_AR, Input_T,
                                     T_avg_AR, dT_outer_AR, dInput_T,
                                     dT_avg_AR]                              (7-d)
                  V1 layout. T_outer / T_avg are AR-fed; dT_outer /
                  dT_avg are AR-derived (current - previous from the
                  AR chain, same rule as the 'delta' variant). dInput_T
                  remains GT exogenous. h0 from InitStateEncoder.
                  PRODUCTION-VALID.

  3. abs          Inputs: [Time, T_outer_AR, Input_T, T_avg_AR]              (4-d)
                  V3/V4 baseline. T_outer and T_avg are AR-fed,
                  Input_T is GT exogenous. h0 from InitStateEncoder.
                  PRODUCTION-VALID -- nothing in the input vector
                  references future GT.

  4. abs_sliding  SEQUENCE-INPUT sliding (Arnold 2026-05-25 spec):
                  At each rollout step t the GRU input is a W-step
                  sequence of shape (B, W, 5):
                      window[k] = [Time(t-W+1+k),
                                   T_outer_AR(t-W+1+k),
                                   T_inner_AR(t-W+1+k),
                                   T_avg_AR(t-W+1+k),
                                   Input_T(t-W+1+k)]
                      for k = 0 .. W-1.
                  Steps with k = t-W+1+offset < 0 are ZERO-padded.
                  The GRU's last-step output is the prediction for t+1.
                  Hidden state CARRIES OVER between rollout steps
                  (advanced W steps per iteration). Stride = 1
                  (one prediction per input step; windows overlap by W-1).
                  AR values in the window come from the model's own
                  predictions (or GT-via-TF at the moment they were
                  produced), NOT from ground truth. Time and Input_T
                  in the window are GT exogenous (Input_T is exogenous
                  in the forward problem).
                  h0 from InitStateEncoder.
                  PRODUCTION-VALID -- no future GT is used; all AR slots
                  in the window are either predictions or known exogenous
                  inputs.
                  NOTE: replaces an earlier (2026-05-20) implementation
                  that flattened the W-step lookback into a (5+3W)-d
                  FEATURE vector at each step (seq_len=1 for the GRU).
                  The flat-window version had wrong semantics per Arnold
                  -- the GRU should see the past as a TIME sequence so
                  the recurrent gates can actually model it.

  --- INVERSE variants (Arnold 2026-05-21) ---
  All 4 inverse variants share output = [Input_T, T_inner, T_outer] (the
  model predicts the unknown driving Input_T plus the two outer/inner
  temperatures, GIVEN the observed T_avg). T_avg is exogenous GT in
  inverse mode (treated like a sensor reading at the centre of the media
  that the user can measure even in deployment). NO AR feedback in
  the input -- a single forward pass is sufficient for inverse_delta /
  inverse_abs+delta / inverse_abs; only inverse_abs_sliding uses a
  per-step T_avg lookback window.

  5. inverse_delta       Inputs: [Time, dT_avg]                              (2-d)
  6. inverse_abs+delta   Inputs: [Time, T_avg, dT_avg]                       (3-d)
  7. inverse_abs         Inputs: [Time, T_avg]                               (2-d)
  8. inverse_abs_sliding SEQUENCE-INPUT sliding (Arnold 2026-05-25 spec):
                         per rollout step t the GRU input is a W-step
                         sequence of shape (B, W, 2):
                             window[k] = [Time(t-W+1+k), T_avg(t-W+1+k)]
                         All values are GT (no AR in inverse). Hidden
                         state carries over; GRU's last output is the
                         prediction for step t+1. Stride = 1. Replaces
                         the earlier (2026-05-21) flat-window version
                         that put the W-step lookback into the FEATURE
                         dimension (2+W d per step).

  All inverse variants: h0 from InitStateEncoder (4-d t=0 obs); model
  output_size = 3 with channels [Input_T, T_inner, T_outer]. Teacher
  forcing is a no-op for inverse variants because there is no AR
  feedback in the input vector.

  --- diagnostic-only (kept in code, removed from default VARIANTS) ---

  X. abs_window   (NOT production-valid) Same 4-d inputs as 'abs', but
                  h0 is built from the FIRST W ground-truth observations
                  via WindowInitStateEncoder. Requires reading t=0..W-1
                  of the true sequence at deployment, which we do not
                  have in the surrogate-vs-CFD setting (we only know
                  the initial state at t=0 and the exogenous Input_T
                  trajectory). Kept as an upper-bound diagnostic --
                  shows what could be achieved if a 10-step ground-truth
                  warm-up window were available.


DATA LAYOUT EXPECTED
--------------------
The script auto-resolves the data root and looks for:

  <repo>/raw/R199 299 AI-TES Project_Arnold Team/2026 Spring/ML Group/data/
    +-- training_data/<TRAIN_SUBDIR>/   (default: '10s', 312 csvs)
    +-- tests/<TEST_SUBDIR>/            (default: 'test_in_10s', 40 csvs)

Each CSV is the 5-column schema:
  Time (s), T_outer (C), T_inner (C), T_avg (C), Input Temperature (C)

If the relative path resolution fails on your machine, edit the
NEW_DATA_ROOT_ABSOLUTE constant near the top of the file. Falls back to the
legacy data/data_in_10s and data/test_in_10s layout if neither resolves.

DATA SPLIT (Arnold convention -- manual re-split)
-------------------------------------------------
Outer 80/20 train+val / test, inner 95/5 train / val. Implemented as:
  (a) all 40 files in tests/<TEST_SUBDIR>/ are kept as the CORE test set,
  (b) supplemented with 30 randomly drawn (seed=42) cases from
      training_data/<TRAIN_SUBDIR>/ to reach 70 test files total,
  (c) the remaining 282 training_data files form the train+val pool,
  (d) inner 95/5 (= 268 train + 14 val) is then a deterministic shuffle
      of that pool.

All four variants receive the IDENTICAL train/val/test split (seed-locked),
so cross-variant comparisons are apples-to-apples.

If you want the legacy "folder-as-given" split (i.e. use exactly the 312
training_data files for train+val and the 40 tests files for test, no
mixing), set MANUAL_SPLIT_ENABLED = False.

HOW TO RUN
----------
1. Make sure PyTorch + pandas + scikit-learn + matplotlib are installed.
   Tested on PyTorch >= 2.0 (CPU or CUDA).
2. Place this file under the repo
3. From that directory, simply run:
     python GRU_input_ablation.py
4. Expected runtime:
     ~30-60 minutes per variant on a single GPU,
     ~4-6 hours total on CPU. Each variant prints progress every 10 epochs.

OUTPUT ARTIFACTS
----------------
Everything for one run lives under  src/runs/<RUN_NAME>/  with this layout:

  runs/<RUN_NAME>/
    run_config.json              timestamp + full config snapshot for
                                 provenance (so an inspector of the
                                 folder doesn't have to read the script).

    checkpoints/
      best_gru_variant_<v>.pth   best-val checkpoint per variant
      gru_variant_<v>.pth        final-state checkpoint per variant

    variants/
      <variant>/                 one folder per variant (delta, abs+delta,
                                 abs, abs_window). Inside each:
        plots/
          plot_*.png             ~70 per-case prediction figures
                                 (T_inner / T_outer / T_avg actual vs
                                 predicted with error annotation)
        summary_errors.csv       71 rows (70 cases + AVERAGE) x 28
                                 columns of per-case metrics: MAE / RMSE
                                 / MaxErr per channel, Early/Late MAE,
                                 MAPE, R^2, inference time...
        train_history.json       train + val + lr histories, best_epoch,
                                 total_epoch, parameter breakdown
        predictions.npz          raw inv_actual / inv_pred per case --
                                 lets you regenerate plots later
                                 without rerunning inference
                                 (~2-5 MB compressed)
        meta.json                scalar dict matching one row of
                                 variant_comparison.csv
        resume_state.pt          in-progress optimizer / model / RNG
                                 state, atomic-written every
                                 CHECKPOINT_EVERY_N_EPOCHS epochs.
                                 Auto-deleted once done.flag is written.
                                 (~15 MB)
        done.flag                sentinel. Created only after train +
                                 test both succeed. Re-running the
                                 script skips any variant whose
                                 done.flag exists.

    comparison/
      variant_comparison.csv     4 rows x ~40 columns, ranked by
                                 overall MAE. Includes Total_Params,
                                 Best_Epoch / Total_Epoch, MAPE, R^2,
                                 EarlyMAE, LateMAE, ms/case, etc.
      variant_results.json       aggregate JSON snapshot of all_results
                                 + config -- consume this for replotting.
      variant_loss_curves.png    train / val / lr curves overlaid (1 x 3).
      variant_test_mae.png       bar chart of overall + per-channel MAE
                                 per variant, annotated with param
                                 count + time.
      variant_capacity_speed.png 2 x 2 reviewer-facing summary:
                                   A. trainable params per variant
                                   B. accuracy vs capacity scatter
                                      (proves the winner isn't bigger)
                                   C. inference speed (ms/case)
                                   D. T_inner Early vs Late MAE
                                      (autoregressive drift)

CRASH RECOVERY / RESUMING
-------------------------
RESUME_ENABLED = True (default) gives you three behaviours:

  * If `results_variant_<name>/done.flag` exists, the variant is SKIPPED
    on a re-run. Its meta + history are loaded from disk and folded into
    the cross-variant comparison automatically. Delete the done.flag to
    force a retrain.

  * If `results_variant_<name>/resume_state.pt` exists but no done.flag
    (training was interrupted), training RESUMES from the saved epoch.
    Optimizer / scheduler / RNG state are all restored, so the rollout
    is bit-for-bit identical to an uninterrupted run.

  * If neither exists, the variant trains fresh from epoch 0.

resume_state.pt is written atomically (.tmp then os.replace) every epoch,
so a power loss mid-write cannot leave a corrupted file.

Set RESUME_ENABLED = False at the top of this file to force every variant
to retrain from scratch (overwrites previous artifacts).

KEY METRICS REPORTED
--------------------
For each variant we record:

  Capacity              Total_Params, GRU_Params, FC_Params, Encoder_Params
                        (lets reviewers verify the winner isn't bigger)
  Training efficiency   Best_Epoch / Total_Epoch / Best_Epoch_Frac
  Per-channel error     MAE, RMSE, MaxErr (T_inner, T_outer, T_avg)
  AR drift              EarlyMAE (first 10% steps), LateMAE (last 10%),
                        Late/Early ratio on T_inner
  Relative error        MAPE per channel + Overall (%)
  Correlation           R^2 per channel + Overall
  Inference speed       ms/case, us/step, cases/sec
                        (surrogate-vs-CFD justification)

ADJUSTING THE EXPERIMENT
------------------------
All knobs live in the "Config" section near the top of this file:

  TRAIN_SUBDIR / TEST_SUBDIR       switch to '10s_with_burn_in', etc.
                                   DO NOT use 'original' (that is 1 s
                                   timestep, 14400 rows -- our model is
                                   trained for 10 s timestep).
                                   DO NOT use 150s subdirs.
  WINDOW_SIZE                      window length used by abs_window's
                                   WindowInitStateEncoder.
  LATEST_PARAMS                    hidden_size, num_layers, dropout, lr,
                                   max_epochs, batch_size, early_stop,
                                   ReduceLROnPlateau patience and factor.
  VARIANTS                         the four ablation variants. Drop one
                                   here to skip it.
  MANUAL_SPLIT_ENABLED             True -> 80/20 + 95/5 manual split,
                                   False -> folder-as-given (legacy).
  OUTER_TEST_FRAC / INNER_VAL_FRAC fractions of the manual split.
  SPLIT_SEED                       deterministic shuffle seed for the
                                   manual split. SAME seed is used for
                                   all variants -- DO NOT change between
                                   runs unless you intend to re-shuffle.
  RESUME_ENABLED                   skip-if-done + resume-from-checkpoint.
  CHECKPOINT_EVERY_N_EPOCHS        how often resume_state.pt is rewritten.
  SAVE_PREDICTIONS                 toggle predictions.npz dumps.

TROUBLESHOOTING
---------------
* "Data directory not found"
    The relative path candidates didn't match your machine. Set
    NEW_DATA_ROOT_ABSOLUTE to the absolute path of `.../ML Group/data`.

* CUDA out of memory
    Lower LATEST_PARAMS['batch_size'] from 16 to 8 (or 4). All four
    variants train independently, so OOM in one variant doesn't affect
    the others -- after fixing, just re-run; completed variants will
    skip themselves.

* "Duplicate case_ids after split: ..."
    Should not happen with the standard layout, but if it does, the
    relative paths from the data root collided. Move/rename the
    offending files. Each test/val case must have a unique
    relative-path identifier.

* Want to re-make the figures only (no retraining)
    The data needed to re-plot is in variant_results.json (cross-variant
    figures) and results_variant_<name>/predictions.npz (per-case plots).
    Either delete the .png files and re-run (resume will skip training),
    or write a short replot script that loads those files.

* Different machines, slightly different MAE
    Even with seed=42, GPU non-determinism (cudnn) can cause < 1% drift.
    The script enables cudnn deterministic mode, which mitigates but
    doesn't fully eliminate this. Use the same machine for paper-quality
    numbers.

================================================================================
"""

import os
import glob
import json
import shutil
import time
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import MinMaxScaler


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
VARIANTS = [
    # Forward / normal: predict (T_inner, T_outer, T_avg) at t+1.
    'delta', 'abs+delta', 'abs', 'abs_sliding',
    # Inverse / reverse: predict (Input_T, T_inner, T_outer) at t+1 from
    # T_avg observation only. T_avg is GT exogenous (always available);
    # no AR feedback needed in the input.
    'inverse_delta', 'inverse_abs+delta', 'inverse_abs', 'inverse_abs_sliding',
]


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
RUN_NAME_BASE = "2026-05-25_8var_1200ep_P0_seqsliding"   # <-- edit this for a new experiment
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


# ============================================================
# 2. Dataset (variant-aware)
# ============================================================
class ThermalDataset(Dataset):
    """
    Per variant returns the appropriate input feature layout.

    Stored arrays (per case):
      X            (N, F)   input feature rows for steps t = 0..N-1
      Y            (N, 3)   targets [T_inner, T_outer, T_avg] for steps t = 1..N
      time_values  (N,)     time at steps 1..N
      init_cond    (4,)     [T_outer, T_inner, T_avg, Input_T] at t=0
      window_obs   (W, 4)   first W rows of [T_outer, T_inner, T_avg, Input_T]
                            (only used by abs_window; otherwise ignored)
      full_*                full series for plotting / inverse_transform
    """

    BASE_COLS = ['Time (s)', 'T_outer (C)', 'T_inner (C)', 'T_avg (C)',
                 'Input Temperature (C)']

    def __init__(self, csv_or_df, variant, scaler=None, file_name=None,
                 window_size=WINDOW_SIZE):
        if isinstance(csv_or_df, str):
            self.file_name = os.path.basename(csv_or_df)
            df = pd.read_csv(csv_or_df)
        else:
            self.file_name = file_name or "unknown"
            df = csv_or_df.copy()

        self.variant = variant
        self.window_size = window_size

        df.rename(columns={"T_ave (C)": "T_avg (C)"}, inplace=True)
        if "Input Temperature (C)" not in df.columns:
            df["Input Temperature (C)"] = df["T_inner (C)"]
        df["FileName"] = self.file_name

        if scaler is None:
            self.scaler = MinMaxScaler()
            self.scaler.fit(df[self.BASE_COLS])
        else:
            self.scaler = scaler

        df[self.BASE_COLS] = self.scaler.transform(df[self.BASE_COLS])

        # Delta features (GT) computed AFTER scaling. The training rollout
        # OVERRIDES idx 1/2/3 with AR-derived deltas (current - previous,
        # both from the AR chain), so the GT deltas stored here are only
        # used as scaffolding -- not as actual model inputs at runtime.
        # dInput_T is exogenous (Input_T is user-specified at deployment)
        # so it is kept GT and fed unchanged.
        df["dT_outer (C)"] = df["T_outer (C)"].diff().fillna(0)
        df["dT_inner (C)"] = df["T_inner (C)"].diff().fillna(0)
        df["dT_avg (C)"] = df["T_avg (C)"].diff().fillna(0)
        df["dInput Temperature (C)"] = df["Input Temperature (C)"].diff().fillna(0)

        # 5-input refactor (Arnold 2026-05-20). T_inner is now an input
        # (in addition to remaining a prediction target). Feature ordering:
        #   abs / abs_window / abs_sliding:
        #     idx 0 = Time, 1 = T_outer, 2 = T_inner, 3 = T_avg, 4 = Input_T
        #   delta:
        #     idx 0 = Time, 1 = dT_outer, 2 = dT_inner, 3 = dT_avg, 4 = dInput_T
        #   abs+delta:
        #     idx 0 = Time, 1..4 = abs (outer/inner/avg/Input_T),
        #     idx 5..8 = deltas in same channel order.
        if variant == 'delta':
            feat_cols = [
                "Time (s)",
                "dT_outer (C)", "dT_inner (C)", "dT_avg (C)",
                "dInput Temperature (C)",
            ]
        elif variant == 'abs+delta':
            feat_cols = [
                "Time (s)",
                "T_outer (C)", "T_inner (C)", "T_avg (C)", "Input Temperature (C)",
                "dT_outer (C)", "dT_inner (C)", "dT_avg (C)",
                "dInput Temperature (C)",
            ]
        elif variant == 'inverse_delta':
            # Only Time + dT_avg (per Arnold spec: T_avg is the input).
            feat_cols = ["Time (s)", "dT_avg (C)"]
        elif variant == 'inverse_abs+delta':
            feat_cols = ["Time (s)", "T_avg (C)", "dT_avg (C)"]
        elif variant in ('inverse_abs', 'inverse_abs_sliding'):
            # inverse_abs_sliding uses the same 2-d per-step features; the
            # W-length T_avg lookback window is built at rollout time
            # directly from inputs (T_avg is GT exogenous, no AR feedback).
            feat_cols = ["Time (s)", "T_avg (C)"]
        else:  # abs / abs_window / abs_sliding
            feat_cols = [
                "Time (s)",
                "T_outer (C)", "T_inner (C)", "T_avg (C)", "Input Temperature (C)",
            ]

        self.X = []
        self.Y = []
        self.time_values = []
        self.init_conditions = []
        self.window_obs = []
        self.full_time = []
        self.full_t_outer = []
        self.full_t_inner = []
        self.full_t_avg = []
        self.full_input_temp = []

        # The model's target depends on whether this is a forward variant
        # (predict the 3 media temperatures) or an inverse variant (predict
        # the driving Input_T + the two outer/inner temperatures, taking
        # T_avg as input instead of output).
        if is_inverse(variant):
            output_cols = ["Input Temperature (C)", "T_inner (C)", "T_outer (C)"]
        else:
            output_cols = ["T_inner (C)", "T_outer (C)", "T_avg (C)"]

        for _, group in df.groupby("FileName"):
            X_seq = group[feat_cols].values[:-1]
            Y_seq = group[output_cols].values[1:]
            time_vals = group["Time (s)"].values[1:]

            init_cond = group[["T_outer (C)", "T_inner (C)", "T_avg (C)",
                               "Input Temperature (C)"]].values[0]

            # Window of first W absolute observations (T_outer, T_inner, T_avg, Input_T).
            # Pads by repeating the last available row if the case is shorter than W.
            full_obs = group[["T_outer (C)", "T_inner (C)", "T_avg (C)",
                              "Input Temperature (C)"]].values
            W = window_size
            if full_obs.shape[0] >= W:
                window_obs_arr = full_obs[:W]
            else:
                pad = np.tile(full_obs[-1:], (W - full_obs.shape[0], 1))
                window_obs_arr = np.concatenate([full_obs, pad], axis=0)

            self.X.append(X_seq)
            self.Y.append(Y_seq)
            self.time_values.append(time_vals)
            self.init_conditions.append(init_cond)
            self.window_obs.append(window_obs_arr)
            self.full_time.append(group["Time (s)"].values)
            self.full_t_outer.append(group["T_outer (C)"].values)
            self.full_t_inner.append(group["T_inner (C)"].values)
            self.full_t_avg.append(group["T_avg (C)"].values)
            self.full_input_temp.append(group["Input Temperature (C)"].values)

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.Y = torch.tensor(np.array(self.Y), dtype=torch.float32)
        self.time_values = np.array(self.time_values)
        self.init_conditions = torch.tensor(np.array(self.init_conditions),
                                            dtype=torch.float32)
        self.window_obs = torch.tensor(np.array(self.window_obs), dtype=torch.float32)
        self.full_time = np.array(self.full_time)
        self.full_t_outer = np.array(self.full_t_outer)
        self.full_t_inner = np.array(self.full_t_inner)
        self.full_t_avg = np.array(self.full_t_avg)
        self.full_input_temp = np.array(self.full_input_temp)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return (
            self.X[i], self.Y[i], self.time_values[i],
            self.init_conditions[i], self.window_obs[i],
            self.full_time[i],
            self.full_t_outer[i], self.full_t_inner[i],
            self.full_t_avg[i], self.full_input_temp[i],
        )


# ============================================================
# 3. Encoders
# ============================================================
class InitStateEncoder(nn.Module):
    """4-d t=0 obs -> h0 (L, B, H). Same as V3/V4."""

    def __init__(self, obs_size=4, hidden_size=128, num_layers=5):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
            nn.Linear(obs_size, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, num_layers * hidden_size),
        )

    def forward(self, x0):
        h = self.encoder(x0).view(-1, self.num_layers, self.hidden_size)
        return h.permute(1, 0, 2).contiguous()


class WindowInitStateEncoder(nn.Module):
    """Updated InitStateEncoder: maps a fixed initial window (W, 4) -> h0 (L, B, H).

    Unlike the V3/V4 point-wise InitStateEncoder (which uses only the t=0
    observation), this version processes the first W observations through a
    small GRU encoder before producing h0 for the main GRU.

    NOTE: this is a FIXED initial window -- it is applied exactly once at the
    start of the rollout to produce h0, not recomputed at each step.
    Conceptually it's a trainable counterpart of V2's burn-in (an internal
    GRU warm-up over the first W steps) bolted onto the V3 InitStateEncoder
    contract: returns an h0 of the same shape and is used only at the start
    of the autoregressive rollout.
    """

    def __init__(self, obs_size=4, window_size=WINDOW_SIZE,
                 hidden_size=128, num_layers=5,
                 enc_layers=2, enc_hidden=128):
        super().__init__()
        self.window_size = window_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.encoder_gru = nn.GRU(input_size=obs_size, hidden_size=enc_hidden,
                                  num_layers=enc_layers, batch_first=True)
        self.proj = nn.Linear(enc_hidden, num_layers * hidden_size)

    def forward(self, x_window):
        out, _ = self.encoder_gru(x_window)
        last = out[:, -1, :]
        h = self.proj(last).view(-1, self.num_layers, self.hidden_size)
        return h.permute(1, 0, 2).contiguous()


# ============================================================
# 4. Model
# ============================================================
class ThermalGRU(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=3,
                 num_layers=5, dropout=0.3,
                 encoder='point', window_size=WINDOW_SIZE):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.encoder_kind = encoder

        self.gru = nn.GRU(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

        if encoder == 'point':
            self.init_encoder = InitStateEncoder(
                obs_size=4, hidden_size=hidden_size, num_layers=num_layers,
            )
        elif encoder == 'window':
            self.init_encoder = WindowInitStateEncoder(
                obs_size=4, window_size=window_size,
                hidden_size=hidden_size, num_layers=num_layers,
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder}")

    def init_hidden(self, batch_size, x0=None, x_window=None):
        device = next(self.parameters()).device
        if self.encoder_kind == 'point':
            assert x0 is not None, "point encoder needs x0"
            return self.init_encoder(x0.to(device))
        else:
            assert x_window is not None, "window encoder needs x_window"
            return self.init_encoder(x_window.to(device))

    def forward(self, x, hidden):
        out, hidden = self.gru(x, hidden)
        return self.fc(out), hidden


# ============================================================
# 5. Loss
# ============================================================
_DEFAULT_WEIGHTS = torch.tensor([1.0, 1.0, 1.0])


def weighted_loss(predictions, targets, weights=None):
    if weights is None:
        weights = _DEFAULT_WEIGHTS
    weights = weights.to(predictions.device)
    return torch.mean(torch.abs(predictions - targets) * weights)


# ============================================================
# 6. Data loading (resolves to the 2026-04-26 newly uploaded dataset)
# ============================================================
def _resolve_new_data_root(script_dir):
    """Return the first NEW_DATA_ROOT_CANDIDATES path that exists, or None."""
    for rel in NEW_DATA_ROOT_CANDIDATES:
        p = os.path.normpath(os.path.join(script_dir, rel))
        if os.path.isdir(p):
            return p
    if NEW_DATA_ROOT_ABSOLUTE and os.path.isdir(NEW_DATA_ROOT_ABSOLUTE):
        return NEW_DATA_ROOT_ABSOLUTE
    return None


def _resolve_legacy_data_root(script_dir):
    """Old layout: raw/ML-model-for-thermal-predictions/data/."""
    project_root = os.path.dirname(script_dir)
    candidates = [
        os.path.join(project_root, "data"),
        os.path.join(script_dir, "..", "..", "data"),
        os.path.join(script_dir, "data"),
        "data",
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    return None


def load_all_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    new_root = _resolve_new_data_root(script_dir)
    data_root = None
    if new_root is not None:
        train_dir = os.path.join(new_root, "training_data", TRAIN_SUBDIR)
        test_dir = os.path.join(new_root, "tests", TEST_SUBDIR)
        if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
            print(f"WARNING: new dataset root resolved to:\n    {new_root}")
            print(f"  but TRAIN_SUBDIR='{TRAIN_SUBDIR}' or TEST_SUBDIR='{TEST_SUBDIR}'")
            print(f"  do not exist there. Falling back to legacy layout.")
            new_root = None
        else:
            data_root = new_root

    if new_root is not None:
        data_source = "NEW (2026-04-26 upload)"
    else:
        legacy_root = _resolve_legacy_data_root(script_dir)
        if legacy_root is None:
            print("ERROR: neither new nor legacy data directory found.")
            return None, None, None, None
        train_dir = os.path.join(legacy_root, "data_in_10s")
        test_dir = os.path.join(legacy_root, "test_in_10s")
        data_root = legacy_root
        data_source = "LEGACY (old data_in_10s/test_in_10s)"

    print(f"Data source: {data_source}")
    print(f"  train_dir = {train_dir}")
    print(f"  test_dir  = {test_dir}")

    raw_train_paths = sorted(glob.glob(os.path.join(train_dir, "**", "*.csv"),
                                       recursive=True))
    raw_test_paths = sorted(glob.glob(os.path.join(test_dir, "**", "*.csv"),
                                      recursive=True))
    print(f"  CSVs in train_dir : {len(raw_train_paths)}")
    print(f"  CSVs in test_dir  : {len(raw_test_paths)}")
    if not raw_train_paths:
        print(f"No training files found in {train_dir}")
        return None, None, None, None

    # ------------------------------------------------------------
    # Decide on the split
    # ------------------------------------------------------------
    if MANUAL_SPLIT_ENABLED:
        # Strategy: keep ALL files in tests/<TEST_SUBDIR>/ as the core test
        # set, then top up to OUTER_TEST_FRAC of the pooled total by sampling
        # additional cases from training_data/<TRAIN_SUBDIR>/. The remaining
        # training_data files form the train+val pool, then 95/5 inner split.
        n_total = len(raw_train_paths) + len(raw_test_paths)
        n_test_target = int(round(n_total * OUTER_TEST_FRAC))
        n_supplement = max(0, n_test_target - len(raw_test_paths))

        if n_supplement > len(raw_train_paths):
            raise RuntimeError(
                f"Cannot build a {n_test_target}-file test set: "
                f"{len(raw_test_paths)} core + only {len(raw_train_paths)} "
                f"available in train_dir."
            )

        rng = random.Random(SPLIT_SEED)
        train_pool_shuffled = list(raw_train_paths)
        rng.shuffle(train_pool_shuffled)

        test_supplement = train_pool_shuffled[:n_supplement]
        trainval_pool = train_pool_shuffled[n_supplement:]

        # Sort each output list for stable ordering (independent of shuffle).
        test_paths = sorted(list(raw_test_paths) + test_supplement)

        # Inner 95/5 split (re-shuffle the surviving train+val pool first
        # so val isn't biased toward early-glob-order files).
        rng.shuffle(trainval_pool)
        n_val = int(round(len(trainval_pool) * INNER_VAL_FRAC))
        val_paths = sorted(trainval_pool[:n_val])
        actual_train_paths = sorted(trainval_pool[n_val:])

        n_train = len(actual_train_paths)
        n_val_actual = len(val_paths)
        n_test_actual = len(test_paths)
        n_trainval = n_train + n_val_actual

        print(f"\nMANUAL split (40 core + supplement to "
              f"{OUTER_TEST_FRAC * 100:.0f}%, seed={SPLIT_SEED}):")
        print(f"  pooled total          : {n_total}")
        print(f"  test     ({n_test_actual:>3}) = "
              f"{n_test_actual / n_total * 100:5.1f}%   "
              f"(target {OUTER_TEST_FRAC * 100:.0f}%)")
        print(f"      core      (tests/{TEST_SUBDIR})        : "
              f"{len(raw_test_paths)}")
        print(f"      supplement (from training_data/{TRAIN_SUBDIR}): "
              f"{len(test_supplement)}")
        print(f"  train+val({n_trainval:>3}) = "
              f"{n_trainval / n_total * 100:5.1f}%   "
              f"(target {(1 - OUTER_TEST_FRAC) * 100:.0f}%)")
        print(f"      val   ({n_val_actual:>3}) = "
              f"{n_val_actual / n_trainval * 100:5.1f}% of train+val   "
              f"(target {INNER_VAL_FRAC * 100:.0f}%)")
        print(f"      train ({n_train:>3}) = "
              f"{n_train / n_trainval * 100:5.1f}% of train+val   "
              f"(target {(1 - INNER_VAL_FRAC) * 100:.0f}%)")
    else:
        # Folder-as-given (legacy behaviour). 95/5 split inside train_dir;
        # test_dir is taken whole.
        val_split = int(0.05 * len(raw_train_paths))
        val_paths = raw_train_paths[:val_split]
        actual_train_paths = raw_train_paths[val_split:]
        test_paths = raw_test_paths

        print(f"\nFOLDER-AS-GIVEN split (legacy):")
        print(f"  train: {len(actual_train_paths)}, val: {len(val_paths)}, "
              f"test: {len(test_paths)}")

    # Build a unique, traceable case_id per file. Pure basenames collide
    # (e.g. 'Case 1.csv' exists under multiple case-type folders), which would
    # corrupt the per-case test summary and overwrite plot files. Use the
    # path relative to the resolved data root, posix-style.
    def _case_id(path):
        try:
            rel = os.path.relpath(path, data_root)
        except ValueError:
            rel = os.path.basename(path)
        return rel.replace(os.sep, "/")

    print("\nLoading training data...")
    train_dfs = [(pd.read_csv(f), _case_id(f)) for f in actual_train_paths]
    print("Loading validation data...")
    val_dfs = [(pd.read_csv(f), _case_id(f)) for f in val_paths]
    print("Loading test data...")
    test_dfs = [(pd.read_csv(f), _case_id(f)) for f in test_paths]

    # Sanity: case_ids must be globally unique across train/val/test
    all_ids = [fid for _, fid in train_dfs] + [fid for _, fid in val_dfs] \
              + [fid for _, fid in test_dfs]
    if len(set(all_ids)) != len(all_ids):
        from collections import Counter
        dups = [fid for fid, c in Counter(all_ids).items() if c > 1]
        raise RuntimeError(f"Duplicate case_ids after split: {dups[:5]}...")

    print("Fitting scaler on training data...")
    cleaned = []
    for df, _ in train_dfs:
        d = df.copy()
        d.rename(columns={"T_ave (C)": "T_avg (C)"}, inplace=True)
        if "Input Temperature (C)" not in d.columns:
            d["Input Temperature (C)"] = d["T_inner (C)"]
        # Drop the Convection column if it leaked in (Convection/ files have 6 cols)
        if "Convection (W/m2C)" in d.columns:
            d = d.drop(columns=["Convection (W/m2C)"])
        cleaned.append(d[ThermalDataset.BASE_COLS])
    scaler = MinMaxScaler()
    scaler.fit(pd.concat(cleaned)[ThermalDataset.BASE_COLS])

    return train_dfs, val_dfs, test_dfs, scaler


# ============================================================
# 7. Variant rollout helpers
# ============================================================
def variant_to_encoder(variant):
    # abs_sliding still uses the point InitStateEncoder (h0 built from
    # the single t=0 observation, which is production-available).
    # The "sliding window" in abs_sliding is in the per-step INPUT, not
    # in the h0 computation. Inverse variants likewise use the point
    # encoder (init_cond is 4-d, h0 from t=0 absolute state).
    return 'window' if variant == 'abs_window' else 'point'


def _rewrite_step_input(variant, x_t,
                        cur_outer, cur_inner, cur_avg,
                        prev_outer, prev_inner, prev_avg):
    """
    Variant-aware input rewrite at one AR step (5-input refactor, 2026-05-20).

    Conventions across variants (column indices in x_t):
      delta     (5-d): [Time, dT_outer, dT_inner, dT_avg, dInput_T]
                       idx 1, 2, 3 overwritten with AR-derived deltas.
                       idx 4 (dInput_T) stays GT exogenous.
      abs+delta (9-d): [Time, T_outer, T_inner, T_avg, Input_T,
                       dT_outer, dT_inner, dT_avg, dInput_T]
                       idx 1, 2, 3 = AR-fed abs T_outer / T_inner / T_avg.
                       idx 5, 6, 7 = AR-derived deltas.
                       idx 4 Input_T and idx 8 dInput_T stay GT exogenous.
      abs       (5-d): [Time, T_outer, T_inner, T_avg, Input_T]
      abs_window(5-d): same as abs (the window enters via h0, not the input)
                       idx 1, 2, 3 = AR-fed abs values.

    cur_X / prev_X are the AR-tracked absolute T_outer / T_inner / T_avg
    at the current and previous step. Deltas = cur - prev, computed from
    the AR chain only (never future GT).
    """
    x_new = x_t.clone()
    dT_outer = (cur_outer - prev_outer)
    dT_inner = (cur_inner - prev_inner)
    dT_avg = (cur_avg - prev_avg)

    if variant == 'delta':
        x_new[:, 1] = dT_outer
        x_new[:, 2] = dT_inner
        x_new[:, 3] = dT_avg
        # idx 4 dInput_T stays GT
    elif variant == 'abs+delta':
        x_new[:, 1] = cur_outer
        x_new[:, 2] = cur_inner
        x_new[:, 3] = cur_avg
        # idx 4 Input_T stays GT
        x_new[:, 5] = dT_outer
        x_new[:, 6] = dT_inner
        x_new[:, 7] = dT_avg
        # idx 8 dInput_T stays GT
    else:  # abs / abs_window
        x_new[:, 1] = cur_outer
        x_new[:, 2] = cur_inner
        x_new[:, 3] = cur_avg
    return x_new


def run_rollout_train(model, variant, inputs, targets, init_conds, win_obs,
                      tf_prob=0.0):
    """Forward pass during training/validation, variant-aware.

    abs_sliding has its own logic in _rollout_sliding. The other 4 variants
    share a unified per-step loop that maintains a two-step AR history for
    T_outer / T_inner / T_avg, so dT slots can be derived from the AR chain
    without any future GT.

    tf_prob: probability of using the GROUND-TRUTH absolute state instead
    of the model's previous-step prediction when stepping forward in the
    AR chain. Linear-decay schedule is computed by the caller (typically
    `teacher_forcing_prob(epoch)`). At test/inference time pass tf_prob=0.

    targets: (B, N, 3) tensor of GT (T_inner, T_outer, T_avg) at steps 1..N.
    Used only when tf_prob > 0.
    """
    batch_size = inputs.shape[0]
    if model.encoder_kind == 'point':
        hidden = model.init_hidden(batch_size, x0=init_conds)
    else:
        hidden = model.init_hidden(batch_size, x_window=win_obs)

    if variant == 'abs_sliding':
        return _rollout_sliding(model, inputs, targets, hidden, batch_size,
                                tf_prob=tf_prob)

    if is_inverse(variant):
        # Inverse rollout: T_avg is GT exogenous (in inputs[:, :, 1] for
        # the no-AR inverse variants), no AR feedback in the input.
        # Single-shot forward for non-sliding flavors; per-step window
        # build for inverse_abs_sliding. TF gate is a no-op (no AR chain).
        return _rollout_inverse(model, variant, inputs, hidden, batch_size)

    # --- Initialize the AR-history triple (T_outer, T_inner, T_avg) ---
    if variant == 'delta':
        cur_outer = init_conds[:, 0].clone()    # T_outer(0)
        cur_inner = init_conds[:, 1].clone()    # T_inner(0)
        cur_avg = init_conds[:, 2].clone()      # T_avg(0)
    else:  # abs+delta / abs / abs_window
        cur_outer = inputs[:, 0, 1].clone()
        cur_inner = inputs[:, 0, 2].clone()
        cur_avg = inputs[:, 0, 3].clone()
    prev_outer = cur_outer.clone()
    prev_inner = cur_inner.clone()
    prev_avg = cur_avg.clone()

    preds = []
    cur_h = hidden
    for t in range(inputs.size(1)):
        # --- Teacher-forcing gate ---
        # At t > 0, with probability tf_prob, override the AR-fed cur_X
        # with the corresponding GT abs value at step t (= targets[t-1]).
        # Per-batch-element Bernoulli decision.
        if t > 0 and tf_prob > 0.0:
            tf_mask = (torch.rand(batch_size, device=inputs.device) < tf_prob)
            gt_inner = targets[:, t - 1, 0]
            gt_outer = targets[:, t - 1, 1]
            gt_avg = targets[:, t - 1, 2]
            cur_outer = torch.where(tf_mask, gt_outer, cur_outer)
            cur_inner = torch.where(tf_mask, gt_inner, cur_inner)
            cur_avg = torch.where(tf_mask, gt_avg, cur_avg)

        x_t = _rewrite_step_input(
            variant,
            inputs[:, t, :],
            cur_outer.detach(), cur_inner.detach(), cur_avg.detach(),
            prev_outer.detach(), prev_inner.detach(), prev_avg.detach(),
        )
        out, cur_h = model(x_t.unsqueeze(1), cur_h)
        pred_t = out[:, 0, :]    # (B, 3) = (T_inner_pred, T_outer_pred, T_avg_pred)
        preds.append(pred_t)

        # Shift AR history forward: new "previous" is the old "current",
        # new "current" is this step's prediction.
        prev_outer = cur_outer
        prev_inner = cur_inner
        prev_avg = cur_avg
        cur_outer = pred_t[:, 1]    # model output index 1 = T_outer
        cur_inner = pred_t[:, 0]    # index 0 = T_inner
        cur_avg = pred_t[:, 2]      # index 2 = T_avg
    return torch.stack(preds, dim=1)


def _rollout_sliding(model, inputs, targets, hidden, batch_size,
                     tf_prob=0.0, W=None):
    """abs_sliding rollout (sequence-input, 2026-05-25 -- per Arnold).

    Earlier (now removed) version flattened the past W steps into a
    (5 + 3*W)-d FEATURE vector at each rollout step (seq_len = 1 for the
    GRU; the lookback was a static feature stack). Per Arnold's spec on
    2026-05-25, this is wrong: the model should see the past W steps as
    an actual TEMPORAL sequence so the GRU's recurrence can model them.

    New rollout shape:
        per rollout step t, GRU input = (B, W, 5):
            window[k] = [Time(t-W+1+k),
                         T_outer_AR(t-W+1+k),
                         T_inner_AR(t-W+1+k),
                         T_avg_AR(t-W+1+k),
                         Input_T(t-W+1+k)]
        for k = 0 .. W-1. Steps before 0 are zero-padded.

    Decisions (Arnold + user 2026-05-25):
      * Hidden state CARRIES OVER between rollout steps (the GRU's final
        hidden state after the W-step window becomes the starting hidden
        state for the next rollout step's window). This means the GRU is
        advanced W steps per rollout iteration; with stride=1 the windows
        overlap by W-1.
      * Stride = 1: one prediction per input step (output[..., -1, :]
        of each W-step window is the prediction for step t+1).
      * AR in window: the T_outer / T_inner / T_avg values at each step
        in the window come from the model's previous predictions (or
        GT-via-TF at the time those steps were produced), NOT from
        ground truth. Time and Input_T at each step ARE GT (Time is
        always exogenous, Input_T is exogenous in the forward problem).

    Teacher-forcing schedule: with probability tf_prob (per batch element)
    at step t > 0, the most-recent AR value (used as window[..., -1, :]
    for THIS step's prediction AND saved into the history for future
    windows) is replaced by GT. Earlier history entries are not
    retroactively altered -- TF affects each step only at the moment it
    is added to history, matching the per-step semantics used by the
    forward `abs` variant.
    """
    if W is None:
        W = WINDOW_SIZE
    seq_len = inputs.size(1)
    device = inputs.device

    # AR history per step k = 0 .. seq_len (one extra slot for the final
    # prediction we never use). Initialised from the GT t=0 row.
    outer_hist = [inputs[:, 0, 1]]   # k=0: T_outer GT initial
    inner_hist = [inputs[:, 0, 2]]
    avg_hist = [inputs[:, 0, 3]]

    preds = []
    cur_h = hidden

    for t in range(seq_len):
        # Pull the current step's AR values (set by the previous iteration
        # or by the t=0 init above).
        cur_outer = outer_hist[t]
        cur_inner = inner_hist[t]
        cur_avg = avg_hist[t]

        # Teacher-forcing gate (t > 0). Replace the t-step AR values with
        # GT for the masked batch elements. This is what gets written to
        # history AND used in this iteration's window.
        if t > 0 and tf_prob > 0.0:
            tf_mask = (torch.rand(batch_size, device=device) < tf_prob)
            gt_inner = targets[:, t - 1, 0]
            gt_outer = targets[:, t - 1, 1]
            gt_avg = targets[:, t - 1, 2]
            cur_outer = torch.where(tf_mask, gt_outer, cur_outer)
            cur_inner = torch.where(tf_mask, gt_inner, cur_inner)
            cur_avg = torch.where(tf_mask, gt_avg, cur_avg)
            # Update history so future windows see the TF-modified value.
            outer_hist[t] = cur_outer.detach()
            inner_hist[t] = cur_inner.detach()
            avg_hist[t] = cur_avg.detach()

        # Build the W-step window: steps [t-W+1, t-W+2, ..., t].
        # Steps with k < 0 are zero-padded.
        window_steps = []
        for offset in range(-(W - 1), 1):   # offset = -(W-1) .. 0 inclusive
            k = t + offset
            if k < 0:
                pad = torch.zeros(batch_size, 5, device=device)
                window_steps.append(pad)
            else:
                # AR slots at step k come from history (detached, so the
                # window does NOT backprop through the AR chain -- mirrors
                # forward `abs` semantics).
                step_5d = torch.stack([
                    inputs[:, k, 0],          # Time(k)        GT
                    outer_hist[k].detach(),   # T_outer(k)     AR (or GT-via-TF)
                    inner_hist[k].detach(),   # T_inner(k)     AR
                    avg_hist[k].detach(),     # T_avg(k)       AR
                    inputs[:, k, 4],          # Input_T(k)     GT exogenous
                ], dim=1)
                window_steps.append(step_5d)
        window = torch.stack(window_steps, dim=1)   # (B, W, 5)

        # GRU forward over the W-step window. Hidden state advances W
        # steps and carries to the next rollout iteration.
        out, cur_h = model(window, cur_h)
        # Last step's output is the prediction for step t+1.
        pred_t = out[:, -1, :]                       # (B, 3) = (T_inner, T_outer, T_avg)
        preds.append(pred_t)

        # Save prediction as AR value at step t+1.
        outer_hist.append(pred_t[:, 1])
        inner_hist.append(pred_t[:, 0])
        avg_hist.append(pred_t[:, 2])

    return torch.stack(preds, dim=1)


def _rollout_inverse(model, variant, inputs, hidden, batch_size, W=None):
    """Inverse rollout (Arnold 2026-05-21).

    For all inverse variants, the per-step input contains only Time +
    T_avg (and dT_avg / T_avg-window for delta / sliding flavors). T_avg
    is GT EXOGENOUS at every step (the model assumes the user can
    measure T_avg in deployment, like a sensor reading at the centre of
    the media). No AR feedback in the input -- the model just maps the
    observed T_avg trajectory to the unknown driving Input_T plus T_inner
    and T_outer.

    Inputs already contain the full feat_cols values (Time + T_avg [+
    dT_avg]). For inverse_abs_sliding we additionally build a per-step
    window over the past W T_avg values (current included at the last
    slot), zero-padded at the oldest positions when t < W-1.

    Returns predictions of shape (batch, seq_len, 3). The 3 channels are
    (Input_T, T_inner, T_outer) for inverse variants (see
    VARIANT_OUTPUT_CHANNELS).
    """
    if variant != 'inverse_abs_sliding':
        # Non-sliding inverse variants: all inputs are GT, single forward.
        out, _ = model(inputs, hidden)
        return out

    # inverse_abs_sliding (sequence-input, 2026-05-25 -- per Arnold):
    # at each rollout step t, the GRU sees a W-step sequence
    #     window[k] = [Time(t-W+1+k), T_avg(t-W+1+k)]   for k = 0..W-1
    # Steps before 0 are zero-padded. All values are GT (no AR in the
    # inverse problem). Hidden state CARRIES OVER between rollout steps;
    # the GRU's last-step output is the prediction for step t+1.
    if W is None:
        W = WINDOW_SIZE
    seq_len = inputs.size(1)
    device = inputs.device

    preds = []
    cur_h = hidden

    for t in range(seq_len):
        window_steps = []
        for offset in range(-(W - 1), 1):   # offset = -(W-1) .. 0
            k = t + offset
            if k < 0:
                pad = torch.zeros(batch_size, 2, device=device)
                window_steps.append(pad)
            else:
                # inputs has 2 features per step for inverse_abs_sliding:
                # idx 0 = Time, idx 1 = T_avg. Both GT.
                window_steps.append(inputs[:, k, :2])
        window = torch.stack(window_steps, dim=1)   # (B, W, 2)

        out, cur_h = model(window, cur_h)
        preds.append(out[:, -1, :])   # last step's output = pred for t+1

    return torch.stack(preds, dim=1)


# ============================================================
# 8. Training (variant-aware)
# ============================================================
def train_model(variant, train_dfs, val_dfs, test_dfs, scaler, params):
    print(f"\n--- Building datasets for variant '{variant}' ---")
    train_datasets = [ThermalDataset(df, variant, scaler=scaler, file_name=fn)
                      for df, fn in train_dfs]
    val_datasets = [ThermalDataset(df, variant, scaler=scaler, file_name=fn)
                    for df, fn in val_dfs]
    test_datasets = [ThermalDataset(df, variant, scaler=scaler, file_name=fn)
                     for df, fn in test_dfs]

    pin = (DEVICE.type == "cuda")
    train_loader = DataLoader(ConcatDataset(train_datasets),
                              batch_size=params['batch_size'], shuffle=True,
                              pin_memory=pin, num_workers=0)
    val_loader = DataLoader(ConcatDataset(val_datasets),
                            batch_size=params['batch_size'],
                            pin_memory=pin, num_workers=0)

    model = ThermalGRU(
        input_size=INPUT_DIMS[variant],
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        dropout=params['dropout'],
        encoder=variant_to_encoder(variant),
        window_size=WINDOW_SIZE,
    ).to(DEVICE)

    # ---- Modern RNN init: orthogonal weight_hh for the main GRU
    # (Chen 2019). PyTorch's default uniform init is suboptimal for
    # stacked depth >= 3. Done BEFORE any resume-from-checkpoint so a
    # resumed run uses the saved (already-trained) weights.
    if ORTHOGONAL_INIT_RECURRENT:
        apply_orthogonal_init_gru(model.gru)
        # Also apply to the small window-encoder GRU if this variant uses it.
        if getattr(model, 'encoder_kind', 'point') == 'window':
            apply_orthogonal_init_gru(model.init_encoder.encoder_gru)
        print(f"[{variant}] orthogonal init applied to main GRU "
              f"(and window encoder if present)")

    # ---- Trainable parameter counts (broken down) ----
    def _count(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    n_params_total = _count(model)
    n_params_gru = _count(model.gru)
    n_params_fc = _count(model.fc)
    n_params_encoder = _count(model.init_encoder)
    print(f"[{variant}] trainable params: total={n_params_total:,}  "
          f"gru={n_params_gru:,}  fc={n_params_fc:,}  "
          f"encoder({model.encoder_kind})={n_params_encoder:,}")

    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        factor=params['sched_factor'], patience=params['sched_patience'],
    )

    best_val = float('inf')
    best_epoch = 0           # 1-based epoch at which best_val was achieved
    total_epoch = 0          # 1-based final epoch reached (early-stop or max_epochs)
    early = 0
    os.makedirs(checkpoints_dir(), exist_ok=True)
    ckpt_path = os.path.join(checkpoints_dir(), f"best_gru_variant_{variant}.pth")

    train_hist, val_hist, lr_hist = [], [], []
    start_epoch = 0          # epoch index from which to (re)start the loop

    # ---- Resume from saved state if available ----
    if RESUME_ENABLED:
        resume = load_resume_state(variant)
        if resume is not None:
            try:
                model.load_state_dict(resume['model_state'])
                optimizer.load_state_dict(resume['optimizer_state'])
                scheduler.load_state_dict(resume['scheduler_state'])
                best_val = resume['best_val']
                best_epoch = resume['best_epoch']
                early = resume['early_stop_counter']
                total_epoch = resume['total_epoch']
                train_hist = list(resume['train_hist'])
                val_hist = list(resume['val_hist'])
                lr_hist = list(resume['lr_hist'])
                start_epoch = resume['epoch']  # epochs already completed
                # Restore RNG state for deterministic continuation.
                random.setstate(resume['rng_python'])
                np.random.set_state(resume['rng_numpy'])
                torch.set_rng_state(resume['rng_torch'].cpu()
                                    if torch.is_tensor(resume['rng_torch'])
                                    else resume['rng_torch'])
                if (resume.get('rng_torch_cuda') is not None
                        and torch.cuda.is_available()):
                    try:
                        torch.cuda.set_rng_state_all(resume['rng_torch_cuda'])
                    except Exception as e:
                        print(f"[{variant}] WARNING: could not restore CUDA RNG: {e}")
                print(f"[{variant}] RESUMED from epoch {start_epoch} "
                      f"(best_val={best_val:.6f} @ ep{best_epoch})")
            except Exception as e:
                print(f"[{variant}] WARNING: resume_state.pt found but could not be "
                      f"loaded ({e}). Starting fresh.")
                start_epoch = 0
                best_val = float('inf')
                best_epoch = 0
                total_epoch = 0
                early = 0
                train_hist, val_hist, lr_hist = [], [], []

    for epoch in range(start_epoch, params['max_epochs']):
        # Teacher-forcing probability for this epoch (decay 0.5 -> 0 over
        # the first TF_DECAY_EPOCHS, then 0 thereafter).
        tf_prob_train = teacher_forcing_prob(epoch)

        # ---- LR warmup (Liu 2019 RAdam motivation) ----
        # If still inside the warmup window, override the optimizer's LR
        # with a linearly-ramped value. ReduceLROnPlateau is suspended
        # during warmup (we don't call scheduler.step) so it can't
        # interfere with the ramp. After warmup completes the scheduler
        # takes over from base_lr.
        wlr = warmup_lr(epoch, params['lr'])
        in_warmup = wlr is not None
        if in_warmup:
            for pg in optimizer.param_groups:
                pg['lr'] = wlr

        # ---- train ----
        model.train()
        tloss = 0.0
        for batch in train_loader:
            inputs, targets, _, init_conds, win_obs, *_ = [
                b.to(DEVICE, non_blocking=True) if torch.is_tensor(b) else b
                for b in batch
            ]
            optimizer.zero_grad()
            preds = run_rollout_train(model, variant, inputs, targets,
                                       init_conds, win_obs,
                                       tf_prob=tf_prob_train)
            loss = weighted_loss(preds, targets)
            loss.backward()
            # ---- Gradient clipping (Pascanu 2013) ----
            if GRADIENT_CLIP_NORM is not None and GRADIENT_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=GRADIENT_CLIP_NORM,
                )
            optimizer.step()
            tloss += loss.item()
        avg_train = tloss / max(1, len(train_loader))
        train_hist.append(avg_train)

        # ---- val (always full autoregressive, tf_prob=0) ----
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets, _, init_conds, win_obs, *_ = [
                    b.to(DEVICE, non_blocking=True) if torch.is_tensor(b) else b
                    for b in batch
                ]
                preds = run_rollout_train(model, variant, inputs, targets,
                                          init_conds, win_obs,
                                          tf_prob=0.0)
                vloss += weighted_loss(preds, targets).item()
        avg_val = vloss / max(1, len(val_loader))
        val_hist.append(avg_val)

        # ReduceLROnPlateau only runs AFTER warmup is over.
        if not in_warmup:
            scheduler.step(avg_val)

        cur_lr = optimizer.param_groups[0]['lr']
        lr_hist.append(cur_lr)

        total_epoch = epoch + 1
        if avg_val < best_val:
            best_val = avg_val
            best_epoch = epoch + 1
            early = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            early += 1

        if (epoch + 1) % 10 == 0:
            phase = 'warmup' if in_warmup else 'main  '
            print(f"[{variant}] epoch {epoch + 1:>4} ({phase}): "
                  f"train={avg_train:.4f}  val={avg_val:.4f}  "
                  f"lr={cur_lr:.2e}  tf={tf_prob_train:.3f}  "
                  f"best={best_val:.4f} @ ep{best_epoch}")

        # ---- Periodic resume-state save (atomic) ----
        if (epoch + 1) % CHECKPOINT_EVERY_N_EPOCHS == 0:
            try:
                save_resume_state(variant, {
                    'variant': variant,
                    'epoch': epoch + 1,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'best_val': best_val,
                    'best_epoch': best_epoch,
                    'early_stop_counter': early,
                    'total_epoch': total_epoch,
                    'train_hist': train_hist,
                    'val_hist': val_hist,
                    'lr_hist': lr_hist,
                    'rng_python': random.getstate(),
                    'rng_numpy': np.random.get_state(),
                    'rng_torch': torch.get_rng_state(),
                    'rng_torch_cuda': (torch.cuda.get_rng_state_all()
                                       if torch.cuda.is_available() else None),
                })
            except Exception as e:
                print(f"[{variant}] WARNING: resume save failed at epoch "
                      f"{epoch + 1}: {e}")

        if early >= params['early_stop_patience']:
            print(f"[{variant}] early stop at epoch {epoch + 1}")
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    print(f"[{variant}] restored best (val={best_val:.6f} @ ep{best_epoch} / "
          f"{total_epoch} total) from {ckpt_path}")

    param_breakdown = {
        'total': n_params_total,
        'gru': n_params_gru,
        'fc': n_params_fc,
        'encoder': n_params_encoder,
    }

    # ---- Persist full training history (always overwrites; small file) ----
    save_train_history(
        variant, train_hist, val_hist, lr_hist,
        best_epoch=best_epoch, total_epoch=total_epoch,
        val_mae=best_val, param_breakdown=param_breakdown,
    )

    return (model, test_datasets, best_val, best_epoch, total_epoch,
            param_breakdown, train_hist, val_hist, lr_hist)


# ============================================================
# 9. Testing (variant-aware)
# ============================================================
def _r2(y_true, y_pred):
    """Coefficient of determination. Returns NaN if SS_tot == 0."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 0:
        return float('nan')
    return 1.0 - ss_res / ss_tot


def _mape(y_true, y_pred, eps=1e-6):
    """Mean Absolute Percentage Error (in %). Skips |y_true| < eps to avoid blow-up."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.abs(y_true) >= eps
    if not np.any(mask):
        return float('nan')
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def test_model(variant, model, test_datasets, output_dir, set_name):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    # Per-case plots live in a `plots/` subfolder so the variant directory
    # itself stays uncluttered (~70 PNGs per variant).
    case_plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(case_plots_dir, exist_ok=True)
    summary = []
    # Per-case inference time (ms). Excludes data loading / inverse_transform /
    # plotting -- pure model forward time only.
    per_case_infer_ms = []
    total_steps_predicted = 0
    # Raw predictions (collected for predictions.npz so plots can be regenerated
    # later without re-running inference).
    raw_predictions = []

    print(f"\n{'=' * 60}")
    print(f"[{variant}] Testing on {set_name}: {len(test_datasets)} cases")
    print(f"{'=' * 60}")

    for idx, ds in enumerate(test_datasets):
        case_id = ds.file_name  # may be a relative path "subdir/Case 1.csv"
        file_disp = os.path.basename(case_id)
        # Sanitize for filesystem (the relative path can contain '/' '\' or
        # other separators that aren't safe for filenames).
        safe_id = (case_id.replace("/", "__")
                          .replace("\\", "__")
                          .replace(":", "_"))
        print(f"  [{variant}] {idx + 1}/{len(test_datasets)}: {case_id}")

        x, _, _, init_cond, win_obs, ft, ft_out, ft_in, ft_avg, ft_inp = ds[0]
        x = x.to(DEVICE)
        init_cond_b = init_cond.unsqueeze(0).to(DEVICE)
        win_obs_b = win_obs.unsqueeze(0).to(DEVICE)

        # Sync GPU before timing so the previous case's kernels don't bleed in.
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        case_t0 = time.perf_counter()

        if model.encoder_kind == 'point':
            hidden = model.init_hidden(1, x0=init_cond_b)
        else:
            hidden = model.init_hidden(1, x_window=win_obs_b)

        seq_len = x.shape[0]

        if is_inverse(variant):
            # Inverse rollout (single-case mirror of _rollout_inverse).
            # All inputs are GT exogenous (no AR); single forward pass for
            # non-sliding inverse variants, per-step window build for
            # inverse_abs_sliding.
            if variant != 'inverse_abs_sliding':
                with torch.no_grad():
                    out, _ = model(x.unsqueeze(0), hidden)
                pred_seq = out[0].cpu().numpy()
            else:
                # Sequence-input sliding (single-case mirror of
                # _rollout_inverse for inverse_abs_sliding -- Arnold's
                # 2026-05-25 spec). Per rollout step, feed GRU a (1, W, 2)
                # sequence of [Time(k), T_avg(k)] for k in [t-W+1, t];
                # zero-pad k < 0. Carry hidden state. Last output = pred t+1.
                W = WINDOW_SIZE
                preds = []
                cur_h = hidden
                for t in range(seq_len):
                    window_steps = []
                    for offset in range(-(W - 1), 1):
                        k = t + offset
                        if k < 0:
                            window_steps.append([0.0, 0.0])
                        else:
                            window_steps.append(
                                [x[k, 0].item(), x[k, 1].item()]
                            )
                    window = torch.tensor(
                        window_steps, dtype=torch.float32
                    ).unsqueeze(0).to(DEVICE)   # (1, W, 2)
                    with torch.no_grad():
                        out, cur_h = model(window, cur_h)
                    preds.append(out[0, -1].cpu().numpy())   # last step
                pred_seq = np.array(preds)
        elif variant == 'abs_sliding':
            # Sequence-input sliding (single-case mirror of _rollout_sliding
            # -- Arnold's 2026-05-25 spec). Per rollout step t, feed GRU a
            # (1, W, 5) sequence:
            #   window[k] = [Time(t-W+1+k),
            #                T_outer_AR(t-W+1+k),
            #                T_inner_AR(t-W+1+k),
            #                T_avg_AR(t-W+1+k),
            #                Input_T(t-W+1+k)]
            # Zero-pad k < 0. Hidden state carries over. Last-step output =
            # prediction for t+1. Test time: tf_prob = 0 (no teacher forcing).
            W = WINDOW_SIZE
            outer_hist = [float(x[0, 1].item())]   # k=0: GT initial
            inner_hist = [float(x[0, 2].item())]
            avg_hist = [float(x[0, 3].item())]

            preds = []
            cur_h = hidden
            for t in range(seq_len):
                window_steps = []
                for offset in range(-(W - 1), 1):
                    k = t + offset
                    if k < 0:
                        window_steps.append([0.0, 0.0, 0.0, 0.0, 0.0])
                    else:
                        window_steps.append([
                            x[k, 0].item(),     # Time(k)         GT
                            outer_hist[k],      # T_outer(k)      AR
                            inner_hist[k],      # T_inner(k)      AR
                            avg_hist[k],        # T_avg(k)        AR
                            x[k, 4].item(),     # Input_T(k)      GT exogenous
                        ])
                window = torch.tensor(
                    window_steps, dtype=torch.float32
                ).unsqueeze(0).to(DEVICE)   # (1, W, 5)

                with torch.no_grad():
                    out, cur_h = model(window, cur_h)
                pred = out[0, -1].cpu().numpy()   # last step's output
                preds.append(pred)

                # pred channels (per VARIANT_OUTPUT_CHANNELS) = [T_inner, T_outer, T_avg]
                outer_hist.append(float(pred[1]))
                inner_hist.append(float(pred[0]))
                avg_hist.append(float(pred[2]))
            pred_seq = np.array(preds)
        else:
            # Unified per-step rollout for delta / abs+delta / abs / abs_window.
            # Maintains two-step AR history for T_outer / T_inner / T_avg so
            # dT slots can be derived from the AR chain (never future GT).
            # Test time: tf_prob = 0 (no teacher forcing).
            #
            #   cur_X    : AR-fed T_X at the current step t
            #              (= prediction from step t-1, or initial GT at t=0)
            #   prev_X   : AR-fed T_X at step t-1
            #              (= cur_X from previous iteration)
            #
            # The 'delta' variant has no abs T_X slots in x, so initial
            # values come from init_cond rather than x[0].
            if variant == 'delta':
                cur_outer = float(init_cond[0].item())   # T_outer(0)
                cur_inner = float(init_cond[1].item())   # T_inner(0)
                cur_avg = float(init_cond[2].item())     # T_avg(0)
            else:  # abs+delta / abs / abs_window
                cur_outer = float(x[0, 1].item())
                cur_inner = float(x[0, 2].item())
                cur_avg = float(x[0, 3].item())
            prev_outer, prev_inner, prev_avg = cur_outer, cur_inner, cur_avg

            preds = []
            cur_h = hidden
            for t in range(seq_len):
                x_t = x[t].clone()
                dT_outer = cur_outer - prev_outer
                dT_inner = cur_inner - prev_inner
                dT_avg = cur_avg - prev_avg

                if variant == 'delta':
                    # 5-d: [Time, dT_outer, dT_inner, dT_avg, dInput_T]
                    x_t[1] = dT_outer
                    x_t[2] = dT_inner
                    x_t[3] = dT_avg
                    # idx 4 dInput_T stays GT
                elif variant == 'abs+delta':
                    # 9-d: [Time, T_outer, T_inner, T_avg, Input_T,
                    #       dT_outer, dT_inner, dT_avg, dInput_T]
                    x_t[1] = cur_outer
                    x_t[2] = cur_inner
                    x_t[3] = cur_avg
                    # idx 4 Input_T stays GT
                    x_t[5] = dT_outer
                    x_t[6] = dT_inner
                    x_t[7] = dT_avg
                    # idx 8 dInput_T stays GT
                else:  # abs / abs_window
                    # 5-d: [Time, T_outer, T_inner, T_avg, Input_T]
                    x_t[1] = cur_outer
                    x_t[2] = cur_inner
                    x_t[3] = cur_avg

                x_in = x_t.unsqueeze(0).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    out, cur_h = model(x_in, cur_h)
                pred = out[0, 0].cpu().numpy()
                preds.append(pred)

                # Shift AR history forward.
                prev_outer, prev_inner, prev_avg = cur_outer, cur_inner, cur_avg
                cur_outer = float(pred[1])
                cur_inner = float(pred[0])
                cur_avg = float(pred[2])
            pred_seq = np.array(preds)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        case_infer_ms = (time.perf_counter() - case_t0) * 1000.0
        per_case_infer_ms.append(case_infer_ms)

        n = len(pred_seq)
        total_steps_predicted += n

        # Predictions correspond to steps t=1..n. Targets / actuals also at t=1..n.
        # ft[1:n+1] is the time at those steps (length n).
        sl = slice(1, n + 1) if len(ft) >= n + 1 else slice(0, n)

        time_axis = ft[sl]
        actual_outer = ft_out[sl]
        actual_inner = ft_in[sl]
        actual_avg = ft_avg[sl]
        actual_input_temp = ft_inp[sl]

        # Scaler columns layout (set in ThermalDataset.BASE_COLS):
        # 0=Time, 1=T_outer, 2=T_inner, 3=T_avg, 4=Input_T.
        SCALER_COL = {
            'Time': 0, 'T_outer': 1, 'T_inner': 2,
            'T_avg': 3, 'Input_T': 4,
        }
        ALL_CHANNELS = ['T_inner', 'T_outer', 'T_avg', 'Input_T']

        # Build dummy_actual (all GT) and dummy_pred (GT in non-predicted
        # columns, model outputs in predicted columns). Variant determines
        # which of the 4 channels is predicted (3 of them).
        dummy_actual = np.column_stack([
            time_axis, actual_outer, actual_inner, actual_avg,
            actual_input_temp,
        ])
        dummy_pred = dummy_actual.copy()

        channel_names = VARIANT_OUTPUT_CHANNELS[variant]   # 3 strings
        for ch_idx, ch_name in enumerate(channel_names):
            dummy_pred[:, SCALER_COL[ch_name]] = pred_seq[:, ch_idx]

        inv_pred = ds.scaler.inverse_transform(dummy_pred)
        inv_actual = ds.scaler.inverse_transform(dummy_actual)

        # ---- Metrics (in deg C; per channel; NaN for non-predicted) ----
        win = max(10, n // 10)
        metric_row = {
            'Case': case_id,
            'CaseFile': file_disp,
            'N_steps': n,
            'Infer_ms_per_case': case_infer_ms,
            'OutputChannels': '|'.join(channel_names),
        }
        for ch_name in ALL_CHANNELS:
            col = SCALER_COL[ch_name]
            if ch_name in channel_names:
                g = inv_actual[:, col]
                p = inv_pred[:, col]
                diff = np.abs(g - p)
                sq = (g - p) ** 2
                metric_row[f'MAE_{ch_name} (C)']    = float(np.mean(diff))
                metric_row[f'RMSE_{ch_name} (C)']   = float(np.sqrt(np.mean(sq)))
                metric_row[f'MaxErr_{ch_name} (C)'] = float(np.max(diff))
                metric_row[f'EarlyMAE_{ch_name} (C)'] = float(
                    np.mean(np.abs(g[:win] - p[:win])))
                metric_row[f'LateMAE_{ch_name} (C)']  = float(
                    np.mean(np.abs(g[-win:] - p[-win:])))
                metric_row[f'MAPE_{ch_name} (%)'] = _mape(g, p)
                metric_row[f'R2_{ch_name}']       = _r2(g, p)
            else:
                # Channel not predicted by this variant -> NaN
                for k in (f'MAE_{ch_name} (C)', f'RMSE_{ch_name} (C)',
                          f'MaxErr_{ch_name} (C)',
                          f'EarlyMAE_{ch_name} (C)', f'LateMAE_{ch_name} (C)',
                          f'MAPE_{ch_name} (%)', f'R2_{ch_name}'):
                    metric_row[k] = float('nan')

        # Overall MAPE / R^2 (concatenate ONLY predicted channels)
        concat_g = np.concatenate(
            [inv_actual[:, SCALER_COL[c]] for c in channel_names])
        concat_p = np.concatenate(
            [inv_pred[:, SCALER_COL[c]] for c in channel_names])
        metric_row['MAPE_Overall (%)'] = _mape(concat_g, concat_p)
        metric_row['R2_Overall']       = _r2(concat_g, concat_p)

        # Stash raw arrays for replotting (saved to predictions.npz at end).
        raw_predictions.append({
            'case_id': case_id,
            'inv_actual': inv_actual.astype(np.float32),
            'inv_pred': inv_pred.astype(np.float32),
            'channels': channel_names,
        })

        summary.append(metric_row)

        # ---- Plot ----
        # Plot only the channels this variant predicts (actual vs pred).
        # Channel -> color map for visual consistency across variants.
        CH_COLOR = {
            'T_inner': 'red',     'T_outer': 'blue',
            'T_avg':   'green',   'Input_T': 'purple',
        }
        plt.figure(figsize=(12, 6))
        time_plot = inv_actual[:, 0]
        for ch_name in channel_names:
            col = SCALER_COL[ch_name]
            c = CH_COLOR.get(ch_name, 'black')
            plt.plot(time_plot, inv_actual[:, col],
                     label=f"{ch_name} Actual", color=c)
            plt.plot(time_plot, inv_pred[:, col], '--',
                     label=f"{ch_name} Pred", color=c)
        plt.xlabel("Time (s)")
        plt.ylabel("Temperature (C)")
        plt.title(f"[{variant}] Prediction - {case_id}")
        plt.legend(fontsize=8)
        # Composite info text from whatever channels exist
        info_lines = [f"Channels: {'|'.join(channel_names)}"]
        for ch in channel_names:
            mae_v = metric_row[f'MAE_{ch} (C)']
            rmse_v = metric_row[f'RMSE_{ch} (C)']
            info_lines.append(f"  {ch}: MAE={mae_v:.3f}  RMSE={rmse_v:.3f}")
        info_lines.append(
            f"Overall: MAPE={metric_row['MAPE_Overall (%)']:.2f}%  "
            f"R^2={metric_row['R2_Overall']:.4f}")
        plt.text(0.02, 0.98, '\n'.join(info_lines),
                 transform=plt.gca().transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                 fontsize=8)
        plt.tight_layout()
        out_path = os.path.join(case_plots_dir,
                                f"plot_{safe_id.replace('.csv', '.png')}")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()

    summary_df = pd.DataFrame(summary)
    avg_row = {'Case': 'AVERAGE', 'CaseFile': ''}
    # Build AVERAGE row column-by-column. Only numeric columns get .mean();
    # string columns (e.g. 'OutputChannels' = 'T_inner|T_outer|T_avg', added
    # 2026-05-20 to label which channels the variant predicts) would raise
    # TypeError on .mean(). For non-numeric columns we copy the first row's
    # value, which is the canonical value for this variant since all 70 test
    # cases share the same channel layout. (Bug found by santhosh on 2026-05-25.)
    for col in summary_df.columns:
        if col in ('Case', 'CaseFile'):
            continue
        if pd.api.types.is_numeric_dtype(summary_df[col]):
            avg_row[col] = summary_df[col].mean()
        else:
            avg_row[col] = summary_df[col].iloc[0]
    summary_df = pd.concat([summary_df, pd.DataFrame([avg_row])], ignore_index=True)

    # Filename simplified to `summary_errors.csv` since the directory is
    # already variant-scoped under runs/<RUN_NAME>/variants/<variant>/.
    csv_path = os.path.join(output_dir, "summary_errors.csv")
    summary_df.to_csv(csv_path, index=False, float_format='%.4f')

    # Inference-speed aggregates (excludes the AVERAGE row).
    avg_ms_per_case = float(np.mean(per_case_infer_ms))
    total_infer_s = float(np.sum(per_case_infer_ms)) / 1000.0
    cases_per_sec = (len(per_case_infer_ms) / total_infer_s) if total_infer_s > 0 else float('nan')
    avg_ms_per_step = (sum(per_case_infer_ms) / total_steps_predicted
                       if total_steps_predicted > 0 else float('nan'))

    print(f"\n[{variant}] summary -> {csv_path}")
    # Only print metrics for the channels this variant actually predicts.
    channel_names = VARIANT_OUTPUT_CHANNELS[variant]

    def _fmt(val, fmt='.4f'):
        try:
            return format(val, fmt) if not (isinstance(val, float) and
                                            (val != val)) else '   nan'
        except (TypeError, ValueError):
            return '   nan'

    mae_parts = '  '.join(
        f"{ch}={_fmt(avg_row[f'MAE_{ch} (C)'])}" for ch in channel_names)
    rmse_parts = '  '.join(
        f"{ch}={_fmt(avg_row[f'RMSE_{ch} (C)'])}" for ch in channel_names)
    mape_parts = '  '.join(
        f"{ch}={_fmt(avg_row[f'MAPE_{ch} (%)'], '.3f')}%" for ch in channel_names)
    r2_parts = '  '.join(
        f"{ch}={_fmt(avg_row[f'R2_{ch}'])}" for ch in channel_names)
    print(f"[{variant}] avg MAE   {mae_parts}")
    print(f"[{variant}] avg RMSE  {rmse_parts}")
    print(f"[{variant}] avg MAPE  {mape_parts}  Overall="
          f"{_fmt(avg_row['MAPE_Overall (%)'], '.3f')}%")
    print(f"[{variant}] avg R^2   {r2_parts}  Overall="
          f"{_fmt(avg_row['R2_Overall'])}")
    # Early / Late MAE is reported on the FIRST predicted channel (most
    # informative for AR-drift diagnosis on the hardest channel).
    primary_ch = channel_names[0] if not is_inverse(variant) else 'T_inner'
    if primary_ch in channel_names:
        print(f"[{variant}] avg Early/Late MAE  {primary_ch} "
              f"{_fmt(avg_row[f'EarlyMAE_{primary_ch} (C)'])} / "
              f"{_fmt(avg_row[f'LateMAE_{primary_ch} (C)'])}")
    print(f"[{variant}] inference: {avg_ms_per_case:.1f} ms/case  |  "
          f"{cases_per_sec:.2f} cases/s  |  "
          f"{avg_ms_per_step * 1000:.2f} us/step")

    speed_stats = {
        'avg_ms_per_case': avg_ms_per_case,
        'total_infer_s': total_infer_s,
        'cases_per_sec': cases_per_sec,
        'avg_us_per_step': avg_ms_per_step * 1000,
        'total_steps_predicted': total_steps_predicted,
    }

    # Persist raw predictions so plots can be regenerated without re-running inference
    if SAVE_PREDICTIONS:
        try:
            save_predictions_npz(variant, raw_predictions)
            print(f"[{variant}] predictions saved -> "
                  f"{os.path.join(output_dir, 'predictions.npz')}")
        except Exception as e:
            print(f"[{variant}] WARNING: predictions.npz save failed: {e}")

    return summary_df, speed_stats


# ============================================================
# 10. Helpers
# ============================================================
def format_duration(seconds):
    if seconds is None or (isinstance(seconds, float) and np.isnan(seconds)):
        return "N/A"
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {s:.1f}s"
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}h {int(m)}m {s:.0f}s"


# ============================================================
# 10b. Resume / persistence helpers
#
# Per variant we maintain a directory results_variant_<name>/ containing:
#   resume_state.pt        in-progress optimizer/model/RNG state (atomic write)
#   train_history.json     train/val/lr histories + best/total epoch + params
#   predictions.npz        raw inv_pred / inv_actual per test case
#   summary_errors_*.csv   per-case metrics  (existing)
#   meta.json              all_results-style scalar dict for cross-variant table
#   done.flag              sentinel written only after train+test fully succeed
#   plot_*.png             per-case plots  (existing)
#
# If done.flag exists  -> the variant is skipped on a re-run.
# Else if resume_state.pt exists -> training resumes from that epoch.
# Else -> train fresh.
# ============================================================
def _to_jsonable(obj):
    """Recursively convert numpy / torch scalars to native Python for JSON."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    return obj


def run_dir():
    """Top-level directory for this run. e.g. src/runs/<RUN_NAME>/."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, RUNS_ROOT_DIR, RUN_NAME)


def checkpoints_dir():
    return os.path.join(run_dir(), "checkpoints")


def comparison_dir():
    return os.path.join(run_dir(), "comparison")


def variant_dir(variant):
    """Per-variant subdir: runs/<RUN_NAME>/variants/<variant>/."""
    return os.path.join(run_dir(), "variants", variant)


def plots_dir(variant):
    """All per-case plots for this variant go here."""
    return os.path.join(variant_dir(variant), "plots")


def ensure_run_layout():
    """Create the canonical empty directory structure for the run."""
    for d in (run_dir(), checkpoints_dir(), comparison_dir(),
              os.path.join(run_dir(), "variants")):
        os.makedirs(d, exist_ok=True)
    for v in VARIANTS:
        os.makedirs(variant_dir(v), exist_ok=True)
        os.makedirs(plots_dir(v), exist_ok=True)


def save_run_config_snapshot():
    """One-time JSON dump of the config used for this run, written to
    runs/<RUN_NAME>/run_config.json. Helpful provenance for any reviewer
    who later inspects the output folder."""
    cfg = {
        'run_name': RUN_NAME,
        'started_or_resumed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'seed': SEED,
        'window_size': WINDOW_SIZE,
        'variants': VARIANTS,
        'input_dims': INPUT_DIMS,
        'latest_params': LATEST_PARAMS,
        'data': {
            'train_subdir': TRAIN_SUBDIR,
            'test_subdir': TEST_SUBDIR,
            'manual_split_enabled': MANUAL_SPLIT_ENABLED,
            'outer_test_frac': OUTER_TEST_FRAC,
            'inner_val_frac': INNER_VAL_FRAC,
            'split_seed': SPLIT_SEED,
        },
        'persistence': {
            'resume_enabled': RESUME_ENABLED,
            'save_predictions': SAVE_PREDICTIONS,
            'checkpoint_every_n_epochs': CHECKPOINT_EVERY_N_EPOCHS,
        },
    }
    path = os.path.join(run_dir(), "run_config.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path


def is_variant_complete(variant):
    return os.path.isfile(os.path.join(variant_dir(variant), "done.flag"))


def has_resume_state(variant):
    return os.path.isfile(os.path.join(variant_dir(variant), "resume_state.pt"))


def save_resume_state(variant, payload):
    """Atomic save of the in-progress training state."""
    var_dir = variant_dir(variant)
    os.makedirs(var_dir, exist_ok=True)
    final = os.path.join(var_dir, "resume_state.pt")
    tmp = final + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, final)


def load_resume_state(variant):
    path = os.path.join(variant_dir(variant), "resume_state.pt")
    if not os.path.isfile(path):
        return None
    try:
        # weights_only=False: we are explicitly loading our own RNG / optimizer
        # objects (PyTorch >=2.6 default would block these as 'unsafe').
        return torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        # weights_only kwarg unsupported on older PyTorch versions.
        return torch.load(path, map_location=DEVICE)


def remove_resume_state(variant):
    path = os.path.join(variant_dir(variant), "resume_state.pt")
    if os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            pass


def save_train_history(variant, train_hist, val_hist, lr_hist,
                       best_epoch, total_epoch, val_mae, param_breakdown):
    var_dir = variant_dir(variant)
    os.makedirs(var_dir, exist_ok=True)
    payload = {
        'variant': variant,
        'train_hist': list(train_hist),
        'val_hist': list(val_hist),
        'lr_hist': list(lr_hist),
        'best_epoch': int(best_epoch),
        'total_epoch': int(total_epoch),
        'best_val_mae': float(val_mae),
        'param_breakdown': param_breakdown,
        'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    path = os.path.join(var_dir, "train_history.json")
    with open(path, "w") as f:
        json.dump(_to_jsonable(payload), f, indent=2)


def load_train_history(variant):
    path = os.path.join(variant_dir(variant), "train_history.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_predictions_npz(variant, raw_predictions):
    """raw_predictions: list of dicts with keys 'case_id', 'inv_actual', 'inv_pred'."""
    if not SAVE_PREDICTIONS:
        return
    var_dir = variant_dir(variant)
    os.makedirs(var_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(var_dir, "predictions.npz"),
        case_ids=np.array([p['case_id'] for p in raw_predictions], dtype=object),
        inv_actuals=np.array([p['inv_actual'] for p in raw_predictions],
                             dtype=object),
        inv_preds=np.array([p['inv_pred'] for p in raw_predictions], dtype=object),
    )


def save_variant_meta(variant, meta):
    var_dir = variant_dir(variant)
    os.makedirs(var_dir, exist_ok=True)
    with open(os.path.join(var_dir, "meta.json"), "w") as f:
        json.dump(_to_jsonable(meta), f, indent=2)


def load_variant_meta(variant):
    path = os.path.join(variant_dir(variant), "meta.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def mark_variant_done(variant):
    var_dir = variant_dir(variant)
    os.makedirs(var_dir, exist_ok=True)
    with open(os.path.join(var_dir, "done.flag"), "w") as f:
        f.write(f"completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    # Resume state no longer needed once the variant is fully done.
    remove_resume_state(variant)


# ============================================================
# 11. Main -- run all 4 variants and produce comparison
# ============================================================
if __name__ == "__main__":
    total_start = time.time()

    print("=" * 60)
    print("LOADING ALL DATA (ONE TIME ONLY)")
    print("=" * 60)
    train_dfs, val_dfs, test_dfs, scaler = load_all_data()
    if train_dfs is None:
        print("Failed to load data. Exiting.")
        raise SystemExit(1)

    print(f"\nData loaded successfully.")
    print(f"  Training samples  : {len(train_dfs)}")
    print(f"  Validation samples: {len(val_dfs)}")
    print(f"  Test samples      : {len(test_dfs)}")
    print("=" * 60)

    # ========================================================
    # OUTER LOOP: iterate over SEEDS. Each iteration mutates the
    # module-global RUN_NAME so all path helpers (run_dir,
    # variant_dir, comparison_dir) produce per-seed paths
    # runs/<RUN_NAME_BASE>_seed<N>/...  Each (seed, variant) has
    # its own resume / done flag so a crash mid-experiment picks
    # up exactly where it left off.
    # ========================================================
    for seed_idx, current_seed in enumerate(SEEDS):
        RUN_NAME = f"{RUN_NAME_BASE}_seed{current_seed}"
        print()
        print("#" * 70)
        print(f"# SEED {seed_idx + 1}/{len(SEEDS)}  seed={current_seed}  "
              f"RUN_NAME={RUN_NAME}")
        print("#" * 70)

        print("\n" + "#" * 60)
        print("INPUT-FEATURE ABLATION  -  4 variants, latest hyperparameters")
        print(f"  hidden={LATEST_PARAMS['hidden_size']}  "
              f"layers={LATEST_PARAMS['num_layers']}  "
              f"dropout={LATEST_PARAMS['dropout']}  "
              f"lr={LATEST_PARAMS['lr']}")
        print(f"  batch={LATEST_PARAMS['batch_size']}  "
              f"max_epochs={LATEST_PARAMS['max_epochs']}  "
              f"early_stop={LATEST_PARAMS['early_stop_patience']}")
        print(f"  Variants: {VARIANTS}")
        print(f"  Window size (abs_window only): {WINDOW_SIZE}")
        print(f"  Run name: {RUN_NAME}")
        print(f"  All artifacts will be written under: {run_dir()}")
        print("#" * 60)

        # Create the canonical run directory layout up-front, then dump the
        # config snapshot so any later inspection of the run folder is
        # self-describing (no need to read the script).
        ensure_run_layout()
        cfg_path = save_run_config_snapshot()
        print(f"Run config snapshot -> {cfg_path}")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        all_results = []

        for variant in VARIANTS:
            print(f"\n{'=' * 60}")
            print(f"VARIANT: {variant}  (input_size={INPUT_DIMS[variant]}, "
                  f"encoder={variant_to_encoder(variant)})")
            print(f"{'=' * 60}")

            # ---- Skip if this variant already finished previously ----
            if RESUME_ENABLED and is_variant_complete(variant):
                cached_meta = load_variant_meta(variant)
                cached_hist = load_train_history(variant)
                if cached_meta is not None and cached_hist is not None:
                    cached = dict(cached_meta)
                    cached['train_history'] = cached_hist['train_hist']
                    cached['val_history'] = cached_hist['val_hist']
                    cached['lr_history'] = cached_hist['lr_hist']
                    all_results.append(cached)
                    print(f"[{variant}] DONE flag found -> skipping. Loaded "
                          f"cached MAE={cached.get('Test_MAE_Overall', float('nan')):.4f}C "
                          f"from meta.json")
                    print(f"  (delete {os.path.join(variant_dir(variant), 'done.flag')} "
                          f"to retrain)")
                    continue
                else:
                    print(f"[{variant}] done.flag exists but meta/history missing -- retraining.")

            # Reset seed before each variant for fairness across runs.
            # If we're resuming, train_model will restore RNG from resume_state.pt
            # (overriding this seed); seeding here ensures variants started from
            # scratch are deterministic regardless of how many earlier variants ran.
            set_seed(current_seed)

            run_start = time.time()
            (model, test_sets, val_mae, best_epoch, total_epoch, param_breakdown,
             train_hist, val_hist, lr_hist) = train_model(
                variant, train_dfs, val_dfs, test_dfs, scaler, LATEST_PARAMS,
            )
            train_time = time.time() - run_start

            # Save final-state checkpoint as well (best is already saved during training).
            os.makedirs(checkpoints_dir(), exist_ok=True)
            final_path = os.path.join(checkpoints_dir(), f"gru_variant_{variant}.pth")
            torch.save(model.state_dict(), final_path)

            test_start = time.time()
            if test_sets and len(test_sets) > 0:
                summary_df, speed_stats = test_model(
                    variant, model, test_sets,
                    output_dir=variant_dir(variant),
                    set_name=f"variant_{variant}",
                )
                avg = summary_df[summary_df['Case'] == 'AVERAGE'].iloc[0]
                test_time = time.time() - test_start
                total_time = train_time + test_time

                channel_names = VARIANT_OUTPUT_CHANNELS[variant]
                # Overall MAE = mean of MAE on the channels this variant predicts
                # (3 channels in all variants, but which 3 depends on variant).
                overall_mae = float(np.mean([
                    avg[f'MAE_{ch} (C)'] for ch in channel_names
                ]))

                early_inner = avg.get('EarlyMAE_T_inner (C)', float('nan'))
                late_inner = avg.get('LateMAE_T_inner (C)', float('nan'))

                result_dict = {
                    'Variant': variant,
                    'Input_Size': INPUT_DIMS[variant],
                    'Encoder': variant_to_encoder(variant),
                    'OutputChannels': '|'.join(channel_names),
                    # Capacity / training efficiency
                    'Total_Params': param_breakdown['total'],
                    'GRU_Params': param_breakdown['gru'],
                    'FC_Params': param_breakdown['fc'],
                    'Encoder_Params': param_breakdown['encoder'],
                    'Best_Epoch': best_epoch,
                    'Total_Epoch': total_epoch,
                    'Best_Epoch_Frac': (best_epoch / total_epoch
                                        if total_epoch > 0 else float('nan')),
                    'Validation_MAE': val_mae,
                    # Headline overall
                    'Test_MAE_Overall': overall_mae,
                    'Test_MAPE_Overall': avg['MAPE_Overall (%)'],
                    'Test_R2_Overall': avg['R2_Overall'],
                    # Inference speed
                    'Infer_ms_per_case': speed_stats['avg_ms_per_case'],
                    'Infer_us_per_step': speed_stats['avg_us_per_step'],
                    'Infer_cases_per_sec': speed_stats['cases_per_sec'],
                    # Wallclock
                    'Train_Time_s': train_time,
                    'Test_Time_s': test_time,
                    'Total_Time_s': total_time,
                    'train_history': train_hist,
                    'val_history': val_hist,
                    'lr_history': lr_hist,
                }
                # Per-channel metrics for ALL 4 possible channels (T_inner,
                # T_outer, T_avg, Input_T). Non-predicted channels stay NaN.
                for ch in ['T_inner', 'T_outer', 'T_avg', 'Input_T']:
                    for metric, suffix in [
                        ('MAE', ' (C)'), ('RMSE', ' (C)'), ('MaxErr', ' (C)'),
                        ('EarlyMAE', ' (C)'), ('LateMAE', ' (C)'),
                        ('MAPE', ' (%)'), ('R2', ''),
                    ]:
                        src_key = f'{metric}_{ch}{suffix}'
                        dst_key = f'Test_{metric}_{ch}'
                        result_dict[dst_key] = avg.get(src_key, float('nan'))
                # Late/Early ratio on T_inner (drift indicator). NaN for variants
                # that don't predict T_inner.
                if (isinstance(early_inner, float) and
                        early_inner == early_inner and early_inner > 0):
                    result_dict['Test_LateOverEarly_T_inner'] = float(late_inner) / float(early_inner)
                else:
                    result_dict['Test_LateOverEarly_T_inner'] = float('nan')

                all_results.append(result_dict)

                # Persist the all_results-style scalar dict and mark the variant
                # as fully complete (predictions.npz + train_history.json have
                # already been saved inside test_model / train_model).
                meta_to_save = {k: v for k, v in all_results[-1].items()
                                if k not in ('train_history', 'val_history',
                                             'lr_history')}
                save_variant_meta(variant, meta_to_save)
                mark_variant_done(variant)

                print(f"[{variant}] DONE  overall_MAE={overall_mae:.4f}C  "
                      f"time={format_duration(total_time)}")
                print(f"  meta -> {os.path.join(variant_dir(variant), 'meta.json')}")
                print(f"  done.flag written; resume_state.pt removed.")
            else:
                print(f"[{variant}] WARNING: no test data")

        # ============================================================
        # 12. Cross-variant comparison
        # ============================================================
        if all_results:
            print("\n" + "=" * 60)
            print("VARIANT COMPARISON SUMMARY")
            print("=" * 60)

            comparison_df = pd.DataFrame([
                {k: v for k, v in r.items()
                 if k not in ('train_history', 'val_history', 'lr_history')}
                for r in all_results
            ])
            comparison_df = comparison_df.sort_values('Test_MAE_Overall', ascending=True)
            comparison_df['Rank'] = range(1, len(comparison_df) + 1)

            cols = [
                'Rank', 'Variant', 'Input_Size', 'Encoder', 'OutputChannels',
                # Capacity / training efficiency
                'Total_Params', 'GRU_Params', 'FC_Params', 'Encoder_Params',
                'Best_Epoch', 'Total_Epoch', 'Best_Epoch_Frac',
                # Headline metrics
                'Validation_MAE', 'Test_MAE_Overall',
                'Test_MAPE_Overall', 'Test_R2_Overall',
                # Per-channel metrics for all 4 possible channels. Forward
                # variants populate T_inner/T_outer/T_avg (Input_T is NaN).
                # Inverse variants populate T_inner/T_outer/Input_T (T_avg is NaN).
                'Test_MAE_T_inner', 'Test_MAE_T_outer',
                'Test_MAE_T_avg', 'Test_MAE_Input_T',
                'Test_RMSE_T_inner', 'Test_RMSE_T_outer',
                'Test_RMSE_T_avg', 'Test_RMSE_Input_T',
                'Test_MaxErr_T_inner', 'Test_MaxErr_T_outer',
                'Test_MaxErr_T_avg', 'Test_MaxErr_Input_T',
                'Test_EarlyMAE_T_inner', 'Test_EarlyMAE_T_outer',
                'Test_EarlyMAE_T_avg', 'Test_EarlyMAE_Input_T',
                'Test_LateMAE_T_inner', 'Test_LateMAE_T_outer',
                'Test_LateMAE_T_avg', 'Test_LateMAE_Input_T',
                'Test_LateOverEarly_T_inner',
                'Test_MAPE_T_inner', 'Test_MAPE_T_outer',
                'Test_MAPE_T_avg', 'Test_MAPE_Input_T',
                'Test_R2_T_inner', 'Test_R2_T_outer',
                'Test_R2_T_avg', 'Test_R2_Input_T',
                # Inference speed
                'Infer_ms_per_case', 'Infer_us_per_step', 'Infer_cases_per_sec',
                # Wallclock
                'Train_Time_s', 'Test_Time_s', 'Total_Time_s',
            ]
            comparison_df = comparison_df[[c for c in cols if c in comparison_df.columns]]

            os.makedirs(comparison_dir(), exist_ok=True)
            comparison_path = os.path.join(comparison_dir(), "variant_comparison.csv")
            comparison_df.to_csv(comparison_path, index=False, float_format='%.4f')

            # Aggregate JSON snapshot for fast figure regeneration without
            # re-loading every per-variant meta + history file.
            agg_path = os.path.join(comparison_dir(), "variant_results.json")
            with open(agg_path, "w") as f:
                json.dump({
                    'config': {
                        'variants': VARIANTS,
                        'input_dims': INPUT_DIMS,
                        'window_size': WINDOW_SIZE,
                        'latest_params': LATEST_PARAMS,
                        'manual_split_enabled': MANUAL_SPLIT_ENABLED,
                        'outer_test_frac': OUTER_TEST_FRAC,
                        'inner_val_frac': INNER_VAL_FRAC,
                        'split_seed': SPLIT_SEED,
                        'train_subdir': TRAIN_SUBDIR,
                        'test_subdir': TEST_SUBDIR,
                    },
                    'all_results': _to_jsonable(all_results),
                    'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                }, f, indent=2)
            print(f"Aggregate JSON saved -> {agg_path}")

            # Headline ranked table -- everything a reviewer asks first.
            # Some channels (e.g. T_inner early/late MAE for inverse variants
            # that predict T_inner) may be NaN; handle with safe formatting.
            def _safe(val, fmt='.4f'):
                try:
                    if isinstance(val, float) and val != val:
                        return 'nan'
                    return format(val, fmt)
                except (TypeError, ValueError):
                    return 'nan'

            print(f"\n{'Rank':<5}{'Variant':<22}{'In':<4}{'Enc':<8}"
                  f"{'Params':<10}{'Best/Tot':<11}"
                  f"{'MAE':<9}{'MAPE':<9}{'R^2':<8}"
                  f"{'EarlyInn':<10}{'LateInn':<10}"
                  f"{'ms/case':<10}{'Time':<10}")
            print("-" * 140)
            for _, r in comparison_df.iterrows():
                print(f"{int(r['Rank']):<5}{r['Variant']:<22}"
                      f"{int(r['Input_Size']):<4}{r['Encoder']:<8}"
                      f"{int(r['Total_Params']):>8,}  "
                      f"{int(r['Best_Epoch']):>3}/{int(r['Total_Epoch']):<3}   "
                      f"{_safe(r['Test_MAE_Overall']):<9}"
                      f"{_safe(r['Test_MAPE_Overall'], '.3f'):<9}"
                      f"{_safe(r['Test_R2_Overall']):<8}"
                      f"{_safe(r['Test_EarlyMAE_T_inner']):<10}"
                      f"{_safe(r['Test_LateMAE_T_inner']):<10}"
                      f"{_safe(r['Infer_ms_per_case'], '.1f'):<10}"
                      f"{format_duration(r['Total_Time_s']):<10}")
            print("-" * 140)

            best = comparison_df.iloc[0]
            print(f"\nBEST VARIANT: {best['Variant']}  "
                  f"(MAE={best['Test_MAE_Overall']:.4f}C  "
                  f"MAPE={best['Test_MAPE_Overall']:.3f}%  "
                  f"R^2={best['Test_R2_Overall']:.4f}  "
                  f"params={int(best['Total_Params']):,})")
            print(f"\nComparison CSV saved -> {comparison_path}")

            # ---- Loss / val / LR curves ----
            # 1x3 panel; ALL panels share the same x-axis (epoch) so a horizontal
            # line through them shows "what was happening at epoch e in each
            # metric". Per user request (2026-05-20): the LR panel also has the
            # train + val loss overlaid as faint background curves so a drop in
            # loss right after a LR step is visually obvious.
            fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True)
            ax1, ax2, ax3 = axes
            variant_colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(all_results))))
            for i, r in enumerate(all_results):
                label = r['Variant']
                col = variant_colors[i % len(variant_colors)]
                ax1.plot(r['train_history'], label=label, linewidth=2, color=col)
                ax2.plot(r['val_history'], label=label, linewidth=2, color=col)
                ax3.plot(r['lr_history'], label=label, linewidth=2, color=col)
                # Mark best epoch (the one the saved best_model.pt corresponds
                # to) on train + val curves. best_epoch is 1-based.
                be = int(r.get('Best_Epoch', 0) or 0)
                vh = r['val_history']
                th = r['train_history']
                if 0 < be <= len(vh):
                    ax1.scatter([be - 1], [th[be - 1]],
                                color=col, marker='o', s=55,
                                edgecolor='black', linewidth=0.8, zorder=10)
                    ax2.scatter([be - 1], [vh[be - 1]],
                                color=col, marker='o', s=55,
                                edgecolor='black', linewidth=0.8, zorder=10)
            ax1.set(xlabel='Epoch', ylabel='Train loss (scaled MAE, log)',
                    title='Train Loss vs Epoch (per variant; o = best epoch)')
            ax2.set(xlabel='Epoch', ylabel='Val loss (scaled MAE, log)',
                    title='Validation Loss vs Epoch (per variant; o = best epoch)')
            ax1.set_yscale('log')
            ax2.set_yscale('log')
            ax3.set(xlabel='Epoch', ylabel='Learning rate (log)',
                    title='LR Schedule (with loss faintly overlaid)')
            ax3.set_yscale('log')

            # Faint loss overlay on LR panel (twin y-axis so two log scales coexist).
            ax3b = ax3.twinx()
            ax3b.set_yscale('log')
            ax3b.set_ylabel('Loss (faint overlay)', color='grey', fontsize=9)
            ax3b.tick_params(axis='y', colors='grey', labelsize=8)
            for i, r in enumerate(all_results):
                col = variant_colors[i % len(variant_colors)]
                # train loss = solid faint, val loss = dashed faint
                ax3b.plot(r['train_history'], linewidth=1.0, alpha=0.25,
                          color=col, linestyle='-')
                ax3b.plot(r['val_history'], linewidth=1.0, alpha=0.25,
                          color=col, linestyle='--')
            # Bring LR lines visually in front of the faint loss overlay
            ax3.set_zorder(ax3b.get_zorder() + 1)
            ax3.patch.set_visible(False)

            for ax in axes:
                ax.legend(fontsize=10, loc='best')
                ax.grid(True, alpha=0.3, which='both')

            plt.tight_layout()
            loss_path = os.path.join(comparison_dir(), "variant_loss_curves.png")
            plt.savefig(loss_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Loss curves saved -> {loss_path}")

            # ---- Bar chart per variant ----
            # 4 possible channels (T_inner, T_outer, T_avg, Input_T); each
            # variant predicts 3 of them. NaN bars are simply zero-height.
            fig, ax = plt.subplots(figsize=(13, 6.5))
            # Keep variants in their definition order for readability.
            ordered = [r for v in VARIANTS for r in all_results if r['Variant'] == v]
            labels = [r['Variant'] for r in ordered]
            mae_overall = [r['Test_MAE_Overall'] for r in ordered]
            # Replace NaN with 0 for plotting (bar disappears for not-predicted ch).
            def _nz(v):
                return 0.0 if (isinstance(v, float) and v != v) else v
            mae_inner = [_nz(r['Test_MAE_T_inner']) for r in ordered]
            mae_outer = [_nz(r['Test_MAE_T_outer']) for r in ordered]
            mae_avg = [_nz(r['Test_MAE_T_avg']) for r in ordered]
            mae_input = [_nz(r['Test_MAE_Input_T']) for r in ordered]
            time_vals = [r['Total_Time_s'] for r in ordered]

            xs = np.arange(len(labels))
            width = 0.17
            bars_overall = ax.bar(xs - 2 * width, mae_overall, width,
                                  label='Overall', color='#444444',
                                  edgecolor='black')
            ax.bar(xs - 1 * width, mae_inner, width,
                   label='T_inner', color='#d62728', edgecolor='black')
            ax.bar(xs + 0 * width, mae_outer, width,
                   label='T_outer', color='#1f77b4', edgecolor='black')
            ax.bar(xs + 1 * width, mae_avg, width,
                   label='T_avg', color='#2ca02c', edgecolor='black')
            ax.bar(xs + 2 * width, mae_input, width,
                   label='Input_T (inverse only)', color='#9467bd', edgecolor='black')

            ax.set_xticks(xs)
            ax.set_xticklabels(labels, fontsize=9, rotation=20, ha='right')
            ax.set_xlabel('Variant', fontsize=12)
            ax.set_ylabel('Test MAE (C)', fontsize=12)
            ax.set_title('GRU Input-Feature Ablation -- Test MAE per Variant\n'
                         f'(latest params: lr={LATEST_PARAMS["lr"]}, '
                         f'h={LATEST_PARAMS["hidden_size"]}, '
                         f'L={LATEST_PARAMS["num_layers"]}, '
                         f'd={LATEST_PARAMS["dropout"]})',
                         fontsize=13, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')

            max_v = max(mae_overall + mae_inner + mae_outer + mae_avg + mae_input
                         + [1e-6])
            if max_v <= 5:
                ax.yaxis.set_major_locator(MultipleLocator(0.5))
                ax.yaxis.set_minor_locator(MultipleLocator(0.1))
            elif max_v <= 20:
                ax.yaxis.set_major_locator(MultipleLocator(2))
                ax.yaxis.set_minor_locator(MultipleLocator(0.5))
            else:
                ax.yaxis.set_major_locator(MultipleLocator(5))
                ax.yaxis.set_minor_locator(MultipleLocator(1))
            ax.grid(True, alpha=0.3, which='major', axis='y')
            ax.grid(True, alpha=0.15, which='minor', axis='y')
            ax.set_ylim(0, max_v * 1.15)

            # Overall MAE labels + params + time
            params_list = [r['Total_Params'] for r in ordered]
            for i, (mo, tt, pp) in enumerate(zip(mae_overall, time_vals, params_list)):
                ax.text(xs[i] - 2 * width, mo + max_v * 0.02,
                        f'{mo:.2f}\n{pp / 1000:.0f}k\n({format_duration(tt)})',
                        ha='center', va='bottom', fontsize=7, fontweight='bold')

            plt.tight_layout()
            bar_path = os.path.join(comparison_dir(), "variant_test_mae.png")
            plt.savefig(bar_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Bar chart saved -> {bar_path}")

            # ---- Capacity / efficiency / drift / convergence summary (5-panel) ----
            # Layout: 2x3 grid; axF is hidden (placeholder for future expansion).
            fig2, ((axA, axB, axE), (axC, axD, axF)) = plt.subplots(
                2, 3, figsize=(20, 10)
            )
            axF.axis('off')   # no 6th panel for now.

            # Dynamic per-variant palette (replaces a 4-color hardcoded list
            # that would crash now that we have >4 variants).
            n_var = len(ordered)
            panel_palette = plt.cm.tab10(
                np.linspace(0, 1, max(10, n_var))
            )[:n_var]

            infer_ms = [r['Infer_ms_per_case'] for r in ordered]
            mape_overall = [r['Test_MAPE_Overall'] for r in ordered]
            r2_overall = [r['Test_R2_Overall'] for r in ordered]
            early_inner = [r['Test_EarlyMAE_T_inner'] for r in ordered]
            late_inner = [r['Test_LateMAE_T_inner'] for r in ordered]
            best_ep = [int(r['Best_Epoch']) for r in ordered]
            total_ep = [int(r['Total_Epoch']) for r in ordered]
            best_frac = [
                (be / te) if te > 0 else float('nan')
                for be, te in zip(best_ep, total_ep)
            ]

            # Panel A: trainable params per variant
            bars_p = axA.bar(xs, [p / 1000 for p in params_list],
                             color=panel_palette,
                             edgecolor='black')
            for i, (p, rr) in enumerate(zip(params_list, ordered)):
                axA.text(xs[i], p / 1000 + max(params_list) / 1000 * 0.02,
                         f'{p:,}\n(In={rr["Input_Size"]})',
                         ha='center', va='bottom', fontsize=8, fontweight='bold')
            axA.set_xticks(xs)
            axA.set_xticklabels(labels, fontsize=10)
            axA.set_ylabel('Trainable parameters (k)', fontsize=11)
            axA.set_title('A. Model capacity per variant', fontsize=12, fontweight='bold')
            axA.grid(True, alpha=0.3, axis='y')

            # Panel B: Test MAE vs Total Params (scatter, "more params = better?")
            for i, rr in enumerate(ordered):
                axB.scatter(rr['Total_Params'], rr['Test_MAE_Overall'],
                            s=140, edgecolors='black', linewidths=1.2,
                            color=panel_palette[i])
                axB.annotate(rr['Variant'],
                             (rr['Total_Params'], rr['Test_MAE_Overall']),
                             textcoords='offset points', xytext=(8, 4), fontsize=10)
            axB.set_xlabel('Trainable parameters', fontsize=11)
            axB.set_ylabel('Test MAE Overall (C)', fontsize=11)
            axB.set_title('B. Accuracy vs capacity\n'
                          '(does the winner just have more params?)',
                          fontsize=12, fontweight='bold')
            axB.grid(True, alpha=0.3)

            # Panel C: Inference speed (ms/case) -- surrogate-vs-CFD justification
            bars_t = axC.bar(xs, infer_ms,
                             color=panel_palette,
                             edgecolor='black')
            for i, (ms, cps) in enumerate(zip(infer_ms,
                                              [r['Infer_cases_per_sec'] for r in ordered])):
                axC.text(xs[i], ms + max(infer_ms) * 0.02,
                         f'{ms:.1f} ms\n({cps:.1f} cases/s)',
                         ha='center', va='bottom', fontsize=8, fontweight='bold')
            axC.set_xticks(xs)
            axC.set_xticklabels(labels, fontsize=10)
            axC.set_ylabel('Inference time per case (ms)', fontsize=11)
            axC.set_title('C. Inference speed (ms / case)\n'
                          'surrogate-vs-CFD speedup justification',
                          fontsize=12, fontweight='bold')
            axC.grid(True, alpha=0.3, axis='y')

            # Panel D: Early vs Late MAE on T_inner -- AR drift
            wd = 0.35
            axD.bar(xs - wd / 2, early_inner, wd, label='Early MAE (first 10%)',
                    color='#5DADE2', edgecolor='black')
            axD.bar(xs + wd / 2, late_inner, wd, label='Late MAE (last 10%)',
                    color='#E74C3C', edgecolor='black')
            for i, (e, l) in enumerate(zip(early_inner, late_inner)):
                ratio = (l / e) if e > 0 else float('nan')
                axD.text(xs[i], max(e, l) + max(max(early_inner), max(late_inner)) * 0.03,
                         f'late/early\n={ratio:.2f}x',
                         ha='center', va='bottom', fontsize=8, fontweight='bold')
            axD.set_xticks(xs)
            axD.set_xticklabels(labels, fontsize=10)
            axD.set_ylabel('MAE on T_inner (C)', fontsize=11)
            axD.set_title('D. Autoregressive drift on T_inner\n'
                          '(error growth from early -> late steps)',
                          fontsize=12, fontweight='bold')
            axD.legend(fontsize=9)
            axD.grid(True, alpha=0.3, axis='y')

            # Panel E: Best epoch vs Total epoch (convergence efficiency).
            # Two paired bars per variant:
            #   filled    = Best_Epoch       (epoch the best_model.pt is from)
            #   hatched   = Total_Epoch      (last epoch trained -- max_epochs
            #                                 unless training crashed early)
            # Annotation shows best/total fraction. A fraction near 1 means the
            # model was still improving at the end (more epochs might help).
            # A fraction near 0 means the model peaked very early and never
            # recovered (a sign of the cliff / bad-basin trap).
            wd_e = 0.35
            axE.bar(xs - wd_e / 2, total_ep, wd_e,
                    label='Total epoch trained',
                    color='#CCCCCC', edgecolor='black')
            axE.bar(xs + wd_e / 2, best_ep, wd_e,
                    label='Best epoch (best_model.pt)',
                    color=panel_palette, edgecolor='black')
            for i, (be, te, frac) in enumerate(zip(best_ep, total_ep, best_frac)):
                axE.text(xs[i] - wd_e / 2, te + max(total_ep) * 0.01,
                         f'{te}', ha='center', va='bottom',
                         fontsize=7, color='#444444')
                axE.text(xs[i] + wd_e / 2, be + max(total_ep) * 0.01,
                         f'{be}\n({frac * 100:.0f}%)',
                         ha='center', va='bottom',
                         fontsize=7, fontweight='bold')
            axE.set_xticks(xs)
            axE.set_xticklabels(labels, fontsize=9, rotation=20, ha='right')
            axE.set_ylabel('Epoch', fontsize=11)
            axE.set_title('E. Convergence efficiency\n'
                          '(best epoch / total trained; o on loss curves)',
                          fontsize=12, fontweight='bold')
            axE.legend(fontsize=8, loc='upper right')
            axE.grid(True, alpha=0.3, axis='y')

            fig2.suptitle(
                'GRU Input-Feature Ablation -- '
                'Capacity / Efficiency / Drift / Convergence',
                fontsize=14, fontweight='bold')
            plt.tight_layout()
            cap_path = os.path.join(comparison_dir(), "variant_capacity_speed.png")
            plt.savefig(cap_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Capacity/speed/drift summary saved -> {cap_path}")

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"All variants done. Total runtime: {format_duration(total_elapsed)}")
    print("=" * 60)
