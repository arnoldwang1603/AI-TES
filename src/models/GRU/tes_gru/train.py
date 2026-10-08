"""Variant-aware training loop with resume support, dynamic tracking, and PINN analysis."""
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt

from .config import *
from .data import *
from .models import *
from .rollout import *
from .runio import *


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

    # ---- Modern RNN init: orthogonal weight_hh for the main GRU (Chen 2019) ----
    if ORTHOGONAL_INIT_RECURRENT:
        apply_orthogonal_init_gru(model.gru)
        if getattr(model, 'encoder_kind', 'point') == 'window':
            apply_orthogonal_init_gru(model.init_encoder.encoder_gru)
        print(f"[{variant}] orthogonal init applied to main GRU "
              f"(and window encoder if present)")

    # ---- Register Ablation / Arm Attributes on Model ----
    model.use_dwa = USE_DWA
    model.warmup_steps = WARMUP_STEPS
    model.use_spatial_anchor = USE_SPATIAL_ANCHOR

    if SOFT_START_TAU > 0.0:
        model.soft_start_tau = SOFT_START_TAU
        print(f"[{variant}] soft-start blending enabled (tau={SOFT_START_TAU:g})")
    else:
        model.soft_start_tau = None

    if WARMUP_STEPS > 0:
        print(f"[{variant}] steady-state recurrent warmup active ({WARMUP_STEPS} steps)")

    if EARLY_CH_WEIGHTS is not None and EARLY_CH_TAUS is not None:
        print(f"[{variant}] channel-decoupled early loss active (Arm L): "
              f"weights={EARLY_CH_WEIGHTS}, taus={EARLY_CH_TAUS}")
    elif EARLY_LOSS_WEIGHT > 1.0:
        print(f"[{variant}] early-horizon loss multiplier active: "
              f"weight={EARLY_LOSS_WEIGHT:g}, tau={EARLY_LOSS_TAU:g}")

    if EARLY_DERIV_WEIGHT > 0.0:
        print(f"[{variant}] early transient derivative loss active (Arm M): "
              f"weight={EARLY_DERIV_WEIGHT:g}, tau={EARLY_DERIV_TAU:g}")

    if USE_SPATIAL_ANCHOR:
        print(f"[{variant}] convex spatial bounding head active for T_avg (Arm N)")

    if EARLY_SMOOTH_TO_WEIGHT > 0.0 or EARLY_SMOOTH_TA_WEIGHT > 0.0:
        print(f"[{variant}] transient-localized PINN smoothness active: "
              f"to_weight={EARLY_SMOOTH_TO_WEIGHT:g}, ta_weight={EARLY_SMOOTH_TA_WEIGHT:g}, "
              f"tau={EARLY_SMOOTH_TAU:g}")

    if CURRICULUM_WARMUP_EPOCHS > 0:
        print(f"[{variant}] progressive rollout curriculum active (Arm P): "
              f"epochs={CURRICULUM_WARMUP_EPOCHS}, min_horizon={CURRICULUM_MIN_HORIZON}")

    if getattr(model, 'use_dwa', False):
        model.epoch_task_losses_history = []
        model.dwa_w1 = 1.0
        model.dwa_w2 = 1.0
        model.dwa_w3 = 1.0

    # ---- Raw-space affine constants for physical transformations ----
    any_pinn_active = (SMOOTH_TO_WEIGHT > 0.0 or SMOOTH_TA_WEIGHT > 0.0 or
                       EARLY_SMOOTH_TO_WEIGHT > 0.0 or EARLY_SMOOTH_TA_WEIGHT > 0.0 or
                       PINN_ENERGY_WEIGHT > 0.0)
    if any_pinn_active and not is_inverse(variant):
        _i = {c: ThermalDataset.BASE_COLS.index(c) for c in
              ("T_inner (C)", "T_outer (C)", "T_avg (C)")}
        model.raw_affine = tuple(
            (float(scaler.scale_[_i[c]]), float(scaler.min_[_i[c]]))
            for c in ("T_inner (C)", "T_outer (C)", "T_avg (C)"))
        print(f"[{variant}] physics affine registered for PINN regularizations.")
    else:
        model.raw_affine = None

    # ---- Arms Z2-Z5: lumped energy-balance coefficients (train set only) ----
    model.energy_coef = None
    if PINN_ENERGY_WEIGHT > 0.0 and model.raw_affine is not None:
        ef = fit_energy_balance(train_dfs)
        model.energy_coef = (ef['a'], ef['b'], ef['c'], ef['sigma'])
        print(f"[{variant}] energy-balance fit on train: a={ef['a']:.5f} "
              f"b={ef['b']:.5f} c={ef['c']:+.5f} sigma={ef['sigma']:.4f} C "
              f"R2={ef['r2']:.4f} (n={ef['n']})")
        if not (ef['r2'] >= 0.5):
            print(f"[{variant}] WARNING: energy-balance R2={ef['r2']:.3f} < 0.5 -- "
                  f"the heat-flow rule explains little of dT_avg; see check_energy_fit.py")

    # ---- PINN magnitude controls (Arms Z1-Z5) ----
    # pinn_scale multiplies every PINN term inside weighted_loss. Fixed at 1.0
    # unless loss-ratio balancing is on, in which case it is calibrated on the
    # first training batch and re-set each epoch from PINN_BALANCE_START.
    model.pinn_scale = 1.0
    pinn_scale_calibrated = False
    if PINN_NORM == "weighted":
        print(f"[{variant}] PINN_NORM=weighted: decayed penalties averaged over "
              f"the decay window, not the full horizon")
    if PINN_BALANCE_RATIO > 0.0:
        print(f"[{variant}] PINN loss-ratio balancing: target L_PINN/L_data="
              f"{PINN_BALANCE_RATIO:g}, epoch updates from ep{PINN_BALANCE_START}")
    if VAL_METRIC == "data":
        print(f"[{variant}] validation / checkpointing on the DATA loss only")

    print(f"[{variant}] loss weights (T_inner,T_outer,T_avg) = {LOSS_WEIGHTS}")

    # ---- T_outer / T_avg persistence parameterization (Arm D) ----
    if variant == 'abs_sliding' and OTHER_CH_MODE == 'persistence':
        step_stats = {}
        for name, col in (("T_outer", "T_outer (C)"), ("T_avg", "T_avg (C)")):
            d = []
            for df, _ in train_dfs:
                dd = df.copy()
                dd.rename(columns={"T_ave (C)": "T_avg (C)"}, inplace=True)
                d.append(np.diff(dd[col].values))
            d = np.concatenate(d)
            mu, sd = float(d.mean()), float(max(d.std(), 1e-3))
            s = float(scaler.scale_[ThermalDataset.BASE_COLS.index(col)])
            step_stats[name] = (s * mu, s * sd)   # (KB, KC) in scaled space
            print(f"[{variant}] {name} persistence: step mean={mu:+.4f} C "
                  f"std={sd:.4f} C -> KB={s*mu:+.6f} KC={s*sd:.6f}")
        model.step_anchor = (step_stats["T_outer"], step_stats["T_avg"])
    else:
        model.step_anchor = None

    # ---- T_inner residual anchor on Input_T ----
    if variant == 'abs_sliding' and TINNER_MODE == 'anchor':
        deltas = []
        for df, _ in train_dfs:
            d = df.copy()
            d.rename(columns={"T_ave (C)": "T_avg (C)"}, inplace=True)
            if "Input Temperature (C)" not in d.columns:
                d["Input Temperature (C)"] = d["T_inner (C)"]
            tin = d["T_inner (C)"].values
            inp = d["Input Temperature (C)"].values
            deltas.append(tin[1:] - (inp[1:] if ANCHOR_LEAD else inp[:-1]))
        deltas = np.concatenate(deltas)
        d_mean = float(deltas.mean())
        d_std = float(max(deltas.std(), 0.05))
        i_tin = ThermalDataset.BASE_COLS.index("T_inner (C)")
        i_inp = ThermalDataset.BASE_COLS.index("Input Temperature (C)")
        s_tin, m_tin = float(scaler.scale_[i_tin]), float(scaler.min_[i_tin])
        s_inp, m_inp = float(scaler.scale_[i_inp]), float(scaler.min_[i_inp])
        ka = s_tin / s_inp
        kb = -ka * m_inp + s_tin * d_mean + m_tin
        kc = s_tin * d_std
        model.tinner_anchor = (ka, kb, kc)
        print(f"[{variant}] T_inner anchor on Input_T: delta mean={d_mean:+.3f} C "
              f"std={d_std:.3f} C  ->  KA={ka:.5f} KB={kb:+.5f} KC={kc:.6f}")

    # ---- Trainable parameter counts ----
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
    best_epoch = 0
    total_epoch = 0
    early = 0
    os.makedirs(checkpoints_dir(), exist_ok=True)
    ckpt_path = os.path.join(checkpoints_dir(), f"best_gru_variant_{variant}.pth")

    train_hist, val_hist, lr_hist = [], [], []
    data_loss_hist, pinn_loss_hist = [], []
    start_epoch = 0

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
                data_loss_hist = list(resume.get('data_loss_hist', []))
                pinn_loss_hist = list(resume.get('pinn_loss_hist', []))
                start_epoch = resume['epoch']

                model.pinn_scale = float(resume.get('pinn_scale', 1.0))
                pinn_scale_calibrated = bool(resume.get('pinn_scale_calibrated', False))

                if getattr(model, 'use_dwa', False):
                    model.epoch_task_losses_history = list(resume.get('epoch_task_losses_history', []))
                    model.dwa_w1 = resume.get('dwa_w1', 1.0)
                    model.dwa_w2 = resume.get('dwa_w2', 1.0)
                    model.dwa_w3 = resume.get('dwa_w3', 1.0)

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
                data_loss_hist, pinn_loss_hist = [], []
                model.pinn_scale = 1.0
                pinn_scale_calibrated = False
                if getattr(model, 'use_dwa', False):
                    model.epoch_task_losses_history = []
                    model.dwa_w1 = 1.0
                    model.dwa_w2 = 1.0
                    model.dwa_w3 = 1.0

    # ============================================================
    # Epoch Training Loop
    # ============================================================
    for epoch in range(start_epoch, params['max_epochs']):
        tf_prob_train = teacher_forcing_prob(epoch)
        wlr = warmup_lr(epoch, params['lr'])
        in_warmup = wlr is not None
        if in_warmup:
            for pg in optimizer.param_groups:
                pg['lr'] = wlr

        # ---- Train Phase ----
        model.train()
        tloss = 0.0
        epoch_data_loss = 0.0
        epoch_phys_loss = 0.0
        epoch_smooth_to_loss = 0.0
        epoch_smooth_ta_loss = 0.0
        epoch_pinn_loss = 0.0        # actual weighted contribution (k * base)
        epoch_pinn_base = 0.0        # at base weights (before k)

        for batch in train_loader:
            inputs, targets, _, init_conds, win_obs, *_ = [
                b.to(DEVICE, non_blocking=True) if torch.is_tensor(b) else b
                for b in batch
            ]

            # Arm P: Calculate Progressive Curriculum Horizon per batch
            curr_horizon = None
            if CURRICULUM_WARMUP_EPOCHS > 0 and epoch < CURRICULUM_WARMUP_EPOCHS:
                full_len = inputs.size(1)
                frac = (epoch + 1) / float(CURRICULUM_WARMUP_EPOCHS)
                curr_horizon = max(CURRICULUM_MIN_HORIZON,
                                   int(CURRICULUM_MIN_HORIZON + frac * (full_len - CURRICULUM_MIN_HORIZON)))

            optimizer.zero_grad()
            preds = run_rollout_train(model, variant, inputs, targets,
                                      init_conds, win_obs,
                                      tf_prob=tf_prob_train,
                                      curriculum_horizon=curr_horizon)
            
            # Match target horizon if curriculum slicing occurred
            loss_targets = targets[:, :preds.size(1), :] if preds.size(1) < targets.size(1) else targets

            # First-batch calibration of the balancing scale, so the PINN term
            # starts at the target ratio instead of at its raw magnitude.
            if (PINN_BALANCE_RATIO > 0.0 and not pinn_scale_calibrated
                    and model.raw_affine is not None):
                with torch.no_grad():
                    probe = weighted_loss(preds, loss_targets,
                                          raw_affine=model.raw_affine, model=model)
                if probe.loss_pinn_base > 0.0:
                    model.pinn_scale = float(np.clip(
                        PINN_BALANCE_RATIO * probe.loss_data / probe.loss_pinn_base,
                        PINN_SCALE_MIN, PINN_SCALE_MAX))
                    pinn_scale_calibrated = True
                    print(f"[{variant}] PINN scale calibrated on first batch: "
                          f"k={model.pinn_scale:.4e} (data={probe.loss_data:.4e}, "
                          f"pinn_base={probe.loss_pinn_base:.4e})")

            loss = weighted_loss(preds, loss_targets, raw_affine=getattr(model, 'raw_affine', None), model=model)
            loss.backward()
            epoch_pinn_loss += getattr(loss, 'loss_pinn', 0.0)
            epoch_pinn_base += getattr(loss, 'loss_pinn_base', 0.0)
            
            if hasattr(loss, 'loss_data'):
                epoch_data_loss += loss.loss_data
            if hasattr(loss, 'loss_phys'):
                epoch_phys_loss += loss.loss_phys
            if hasattr(loss, 'loss_smooth_to'):
                epoch_smooth_to_loss += loss.loss_smooth_to
            if hasattr(loss, 'loss_smooth_ta'):
                epoch_smooth_ta_loss += loss.loss_smooth_ta

            if GRADIENT_CLIP_NORM is not None and GRADIENT_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=GRADIENT_CLIP_NORM,
                )
            optimizer.step()
            tloss += loss.item()
            
        num_batches = max(1, len(train_loader))
        avg_train = tloss / num_batches
        avg_data = epoch_data_loss / num_batches

        # Exact weighted PINN contribution, as actually added to the loss.
        # (Early-smooth / energy terms are tracked inside weighted_loss; the
        # legacy global smoothness of Arms F/G is added on top below.)
        weighted_pinn = epoch_pinn_loss / num_batches
        avg_pinn_base = epoch_pinn_base / num_batches
        if SMOOTH_TO_WEIGHT > 0.0:
            weighted_pinn += (epoch_smooth_to_loss / num_batches) * SMOOTH_TO_WEIGHT * getattr(model, 'dwa_w2', 1.0)
        if SMOOTH_TA_WEIGHT > 0.0:
            weighted_pinn += (epoch_smooth_ta_loss / num_batches) * SMOOTH_TA_WEIGHT * getattr(model, 'dwa_w3', 1.0)
        pinn_ratio = weighted_pinn / max(avg_data, 1e-12)
        scale_used = model.pinn_scale    # the k that produced this epoch's losses

        train_hist.append(avg_train)
        data_loss_hist.append(avg_data)
        pinn_loss_hist.append(weighted_pinn)

        # ---- Dynamic Weight Annealing (DWA) update (Arm G) ----
        if getattr(model, 'use_dwa', False):
            avg_sto = epoch_smooth_to_loss / num_batches
            avg_sta = epoch_smooth_ta_loss / num_batches
            
            model.epoch_task_losses_history.append((avg_data, avg_sto, avg_sta))
            if len(model.epoch_task_losses_history) >= 2:
                L_t1 = model.epoch_task_losses_history[-1]
                L_t2 = model.epoch_task_losses_history[-2]
                
                eps = 1e-8
                r1 = np.clip(L_t1[0] / max(L_t2[0], eps), 0.0, 10.0)
                r2 = np.clip(L_t1[1] / max(L_t2[1], eps), 0.0, 10.0)
                r3 = np.clip(L_t1[2] / max(L_t2[2], eps), 0.0, 10.0)
                
                T = 2.0
                exp_r1 = np.exp(r1 / T)
                exp_r2 = np.exp(r2 / T)
                exp_r3 = np.exp(r3 / T)
                sum_exp = exp_r1 + exp_r2 + exp_r3
                
                model.dwa_w1 = 3.0 * exp_r1 / sum_exp
                model.dwa_w2 = 3.0 * exp_r2 / sum_exp
                model.dwa_w3 = 3.0 * exp_r3 / sum_exp
            else:
                model.dwa_w1 = 1.0
                model.dwa_w2 = 1.0
                model.dwa_w3 = 1.0

        # ---- Validation Phase (Autoregressive Rollout across Full Horizon) ----
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
                                          tf_prob=0.0,
                                          curriculum_horizon=None)
                vl = weighted_loss(preds, targets,
                                   raw_affine=getattr(model, 'raw_affine', None),
                                   model=model)
                vloss += vl.loss_data if VAL_METRIC == "data" else vl.item()
        avg_val = vloss / max(1, len(val_loader))
        val_hist.append(avg_val)

        # ---- PINN loss-ratio balancing (Arms Z2-Z5) ----
        # Log-space EMA toward k* = ratio * L_data / L_pinn_base, at most a 2x
        # change per epoch. Applies from the NEXT epoch.
        if (PINN_BALANCE_RATIO > 0.0 and epoch + 1 >= PINN_BALANCE_START
                and avg_pinn_base > 0.0):
            k_old = model.pinn_scale
            k_target = PINN_BALANCE_RATIO * avg_data / avg_pinn_base
            k_new = float(np.exp(0.5 * np.log(k_old) + 0.5 * np.log(k_target)))
            k_new = min(max(k_new, 0.5 * k_old), 2.0 * k_old)
            model.pinn_scale = float(np.clip(k_new, PINN_SCALE_MIN, PINN_SCALE_MAX))

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
            pinn_str = ""
            if any_pinn_active:
                avg_sto = epoch_smooth_to_loss / num_batches
                avg_sta = epoch_smooth_ta_loss / num_batches
                pinn_str = f" | PINN: smooth_to={avg_sto:.5f} smooth_ta={avg_sta:.5f} (pinn_w={weighted_pinn:.5f})"
                if PINN_ENERGY_WEIGHT > 0.0:
                    pinn_str += f" energy={epoch_phys_loss / num_batches:.4e}"
                # Machine-readable magnitudes (parsed by run_pinn_analysis):
                pinn_str += (f" | PHYS: data={avg_data:.4e} pinn={weighted_pinn:.4e} "
                             f"ratio={pinn_ratio:.4f} k={scale_used:.4e}")
                if getattr(model, 'use_dwa', False):
                    pinn_str += f" | DWA: w1={model.dwa_w1:.3f} w2={model.dwa_w2:.3f} w3={model.dwa_w3:.3f}"
                
            print(f"[{variant}] epoch {epoch + 1:>4} ({phase}): "
                  f"train={avg_train:.4f}  val={avg_val:.4f}{pinn_str}  "
                  f"lr={cur_lr:.2e}  tf={tf_prob_train:.3f}  "
                  f"best={best_val:.4f} @ ep{best_epoch}")

        # ---- Periodic Atomic Resume-State Save ----
        if (epoch + 1) % CHECKPOINT_EVERY_N_EPOCHS == 0:
            try:
                save_payload = {
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
                    'data_loss_hist': data_loss_hist,
                    'pinn_loss_hist': pinn_loss_hist,
                    'pinn_scale': model.pinn_scale,
                    'pinn_scale_calibrated': pinn_scale_calibrated,
                    'rng_python': random.getstate(),
                    'rng_numpy': np.random.get_state(),
                    'rng_torch': torch.get_rng_state(),
                    'rng_torch_cuda': (torch.cuda.get_rng_state_all()
                                       if torch.cuda.is_available() else None),
                }
                if getattr(model, 'use_dwa', False):
                    save_payload['epoch_task_losses_history'] = model.epoch_task_losses_history
                    save_payload['dwa_w1'] = model.dwa_w1
                    save_payload['dwa_w2'] = model.dwa_w2
                    save_payload['dwa_w3'] = model.dwa_w3

                save_resume_state(variant, save_payload)
            except Exception as e:
                print(f"[{variant}] WARNING: resume save failed at epoch "
                      f"{epoch + 1}: {e}")

        if early >= params['early_stop_patience']:
            print(f"[{variant}] early stop at epoch {epoch + 1}")
            break

    # Restore best model weights
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    print(f"[{variant}] restored best (val={best_val:.6f} @ ep{best_epoch} / "
          f"{total_epoch} total) from {ckpt_path}")

    param_breakdown = {
        'total': n_params_total,
        'gru': n_params_gru,
        'fc': n_params_fc,
        'encoder': n_params_encoder,
    }

    # ---- Diagnostic Graph: Physics Loss vs. Pure GRU Data Loss ----
    if any_pinn_active and len(pinn_loss_hist) > 0:
        try:
            plt.figure(figsize=(10, 5))
            epochs_x = range(1, len(train_hist) + 1)
            plt.plot(epochs_x, data_loss_hist, label="Pure GRU Data Loss ($L_{data}$)", color="royalblue", linewidth=2)
            plt.plot(epochs_x, pinn_loss_hist, label="Weighted Physics Penalty ($L_{PINN}$)", color="crimson", linewidth=2, linestyle="--")
            plt.plot(epochs_x, train_hist, label="Composite Loss ($L_{total}$)", color="black", alpha=0.3, linewidth=1)
            
            if 0 < best_epoch <= len(train_hist):
                plt.axvline(best_epoch, color="forestgreen", linestyle=":", label=f"Best Model (Epoch {best_epoch})")

            plt.yscale("log")
            plt.xlabel("Epoch", fontsize=11)
            plt.ylabel("Loss Magnitude (Log Scale)", fontsize=11)
            plt.title(f"[{variant}] Physics vs. Data Loss Trajectory (Best Val = {best_val:.5f} @ Ep {best_epoch})", fontsize=12, fontweight="bold")
            plt.grid(True, which="both", alpha=0.3)
            plt.legend(fontsize=10)
            plt.tight_layout()
            
            pinn_plot_path = os.path.join(variant_dir(variant), "pinn_vs_data_loss.png")
            plt.savefig(pinn_plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[{variant}] PINN diagnostic plot saved -> {pinn_plot_path}")
        except Exception as e:
            print(f"[{variant}] WARNING: Failed to plot PINN loss curve: {e}")

    save_train_history(
        variant, train_hist, val_hist, lr_hist,
        best_epoch=best_epoch, total_epoch=total_epoch,
        val_mae=best_val, param_breakdown=param_breakdown,
        data_loss_hist=data_loss_hist, pinn_loss_hist=pinn_loss_hist
    )

    return (model, test_datasets, best_val, best_epoch, total_epoch,
            param_breakdown, train_hist, val_hist, lr_hist)
