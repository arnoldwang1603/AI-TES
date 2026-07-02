"""Variant-aware training loop with resume support."""
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

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


