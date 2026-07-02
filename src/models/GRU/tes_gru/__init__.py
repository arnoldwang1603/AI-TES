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

