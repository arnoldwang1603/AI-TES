import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import copy
import csv
import random
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset

from config import (
    SEED, BATCH_SIZE, MAX_EPOCHS, PATIENCE,
    LR, WEIGHT_DECAY, GRAD_CLIP,
    CHECKPOINT_DIR, RESULTS_DIR, VAL_SPLIT,
    T_AVG_WEIGHT, TAVG_WARMUP_START, TAVG_WARMUP_LEN,
    MODEL_VERSION,
)
from dataset import ThermalDataset, load_data
from model import ThermalPredictor, count_params


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def tavg_weight_for_epoch(epoch: int) -> float:
    if epoch < TAVG_WARMUP_START:
        return 0.0
    ramp = epoch - TAVG_WARMUP_START
    if ramp < TAVG_WARMUP_LEN:
        return T_AVG_WEIGHT * (ramp + 1) / TAVG_WARMUP_LEN
    return T_AVG_WEIGHT


def run_epoch(model, loader, device, optimizer=None, train: bool = True, tavg_w=None):
    model.train(train)
    totals = dict(total=0.0, data=0.0, smooth=0.0, phys=0.0)
    mae_inner = mae_outer = mae_avg = 0.0
    n = 0

    with torch.set_grad_enabled(train):
        for batch in loader:
            (tin_sc, dtin_sc, init_sc,
             tin_ph, init_ph, y_ph) = [b.to(device) for b in batch]

            preds, params = model(tin_sc, dtin_sc, init_sc, tin_ph, init_ph)
            L_total, L_data, L_smooth, L_phys = model.compute_loss(
                preds, y_ph, params, tavg_w=tavg_w)

            if train and optimizer is not None:
                optimizer.zero_grad()
                L_total.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            B = tin_sc.size(0)
            totals['total']  += L_total.item() * B
            totals['data']   += L_data.item()  * B
            totals['smooth'] += L_smooth.item() * B
            totals['phys']   += L_phys.item()  * B

            with torch.no_grad():
                err = (preds - y_ph).abs()
                mae_inner += err[:, :, 0].mean().item() * B
                mae_outer += err[:, :, 1].mean().item() * B
                mae_avg   += err[:, :, 2].mean().item() * B

            n += B

    losses = {k: v / n for k, v in totals.items()}
    losses['mae_inner'] = mae_inner / n
    losses['mae_outer'] = mae_outer / n
    losses['mae_avg']   = mae_avg   / n
    return losses


def train():
    set_seed()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR = RESULTS_DIR / 'logs' / MODEL_VERSION
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"\n{'='*65}")
    print(f"  {MODEL_VERSION}  |  device: {device}")
    print(f"{'='*65}\n")

    print("Loading data...")
    train_dfs, val_dfs, test_dfs, scaler = load_data(val_split=VAL_SPLIT)

    train_ds = ConcatDataset([ThermalDataset(df, scaler=scaler, file_name=fn)
                               for df, fn in train_dfs])
    val_ds   = ConcatDataset([ThermalDataset(df, scaler=scaler, file_name=fn)
                               for df, fn in val_dfs])

    pin = (device.type == 'cuda')
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=pin)

    model = ThermalPredictor().to(device)
    model.register_scaler(scaler)
    print(f"  Parameters: {count_params(model):,}")
    print(f"  x init:     {model.ode.x.item():.5f} m")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)

    best_val     = float('inf')
    best_state   = None
    patience_ctr = 0
    start_epoch  = 0

    RESUME_PATH = CHECKPOINT_DIR / 'resume.pt'
    if RESUME_PATH.exists():
        print(f"  Resuming from checkpoint...")
        ckpt = torch.load(RESUME_PATH, map_location=device)
        try:
            model.load_state_dict(ckpt['state_dict'])
            model.register_scaler(scaler)
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            start_epoch  = ckpt['epoch'] + 1
            saved_phase  = ckpt.get('phase', None)
            resume_phase = (0 if start_epoch < TAVG_WARMUP_START
                            else 1 if start_epoch < TAVG_WARMUP_START + TAVG_WARMUP_LEN
                            else 2)
            patience_ctr = 0
            if saved_phase is None or saved_phase != resume_phase:
                best_val = float('inf')
                print(f"  Resuming at epoch {start_epoch} (phase changed — reset)\n")
            else:
                best_val = ckpt['best_val']
                print(f"  Resuming at epoch {start_epoch} (best val = {best_val:.4f})\n")
        except (RuntimeError, KeyError) as e:
            print(f"  Checkpoint incompatible ({e}). Starting fresh.\n")
            RESUME_PATH.unlink(missing_ok=True)
            (CHECKPOINT_DIR / 'best_model.pt').unlink(missing_ok=True)
    else:
        print(f"  Starting fresh\n")

    log_path   = LOG_DIR / 'training_log.csv'
    log_fields = ['epoch', 'train_total', 'train_data', 'train_smooth', 'train_phys',
                  'val_total', 'val_data', 'val_smooth', 'val_phys',
                  'val_mae_inner', 'val_mae_outer', 'val_mae_avg',
                  'tavg_w', 'lr', 'x_m', 'epoch_time_s']
    log_mode   = 'a' if start_epoch > 0 else 'w'
    log_file   = open(log_path, log_mode, newline='')
    log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
    if log_mode == 'w':
        log_writer.writeheader()

    print(f"  Ta warmup: silent {TAVG_WARMUP_START} ep, ramp {TAVG_WARMUP_LEN} ep\n")
    print(f"  {'Ep':>4}  {'train':>9}  {'val_data':>9}  "
          f"{'Ti':>7}  {'To':>7}  {'Ta':>7}  {'Ta_w':>6}  {'phys':>7}")
    print(f"  {'─'*72}")

    for epoch in range(start_epoch, MAX_EPOCHS):
        t0   = time.time()
        ta_w = tavg_weight_for_epoch(epoch)

        if epoch == TAVG_WARMUP_START:
            best_val, patience_ctr = float('inf'), 0
            print(f"\n  Phase 2: Ta loss ramping in")
        elif epoch == TAVG_WARMUP_START + TAVG_WARMUP_LEN:
            best_val, patience_ctr = float('inf'), 0
            print(f"\n  Phase 3: full supervision")

        train_losses = run_epoch(model, train_loader, device,
                                 optimizer=optimizer, train=True, tavg_w=ta_w)
        val_losses   = run_epoch(model, val_loader, device, train=False, tavg_w=ta_w)
        scheduler.step()

        epoch_time = time.time() - t0
        lr_now     = optimizer.param_groups[0]['lr']
        x_now      = model.ode.x.item()
        phase3     = epoch >= TAVG_WARMUP_START + TAVG_WARMUP_LEN
        monitor    = val_losses['data']

        if monitor < best_val:
            best_val     = monitor
            best_state   = copy.deepcopy(model.state_dict())
            patience_ctr = 0
            ckpt_data = {'epoch': epoch, 'state_dict': best_state, 'val_loss': best_val}
            torch.save(ckpt_data, CHECKPOINT_DIR / 'best_model.pt')
            torch.save(ckpt_data, LOG_DIR / 'best_model.pt')
            star = ' *'
        else:
            patience_ctr += 1 if phase3 else 0
            star = ''

        torch.save({
            'epoch':        epoch,
            'state_dict':   model.state_dict(),
            'optimizer':    optimizer.state_dict(),
            'scheduler':    scheduler.state_dict(),
            'best_val':     best_val,
            'patience_ctr': patience_ctr,
            'phase':        (0 if epoch < TAVG_WARMUP_START
                             else 1 if epoch < TAVG_WARMUP_START + TAVG_WARMUP_LEN
                             else 2),
        }, RESUME_PATH)

        patience_str = f"  p={patience_ctr}/{PATIENCE}" if phase3 else ""
        if epoch % 5 == 0 or star:
            print(
                f"  {epoch:4d}  {train_losses['total']:9.4f}  {val_losses['data']:9.4f}  "
                f"{val_losses['mae_inner']:7.3f}  {val_losses['mae_outer']:7.3f}  "
                f"{val_losses['mae_avg']:7.3f}  {ta_w:6.3f}  {val_losses['phys']:7.4f}"
                f"{star}{patience_str}"
            )

        log_writer.writerow({
            'epoch':         epoch,
            'train_total':   round(train_losses['total'],  5),
            'train_data':    round(train_losses['data'],   5),
            'train_smooth':  round(train_losses['smooth'], 8),
            'train_phys':    round(train_losses['phys'],   6),
            'val_total':     round(val_losses['total'],    5),
            'val_data':      round(val_losses['data'],     5),
            'val_smooth':    round(val_losses['smooth'],   8),
            'val_phys':      round(val_losses['phys'],     6),
            'val_mae_inner': round(val_losses['mae_inner'], 4),
            'val_mae_outer': round(val_losses['mae_outer'], 4),
            'val_mae_avg':   round(val_losses['mae_avg'],   4),
            'tavg_w':        round(ta_w, 4),
            'lr':            round(lr_now, 7),
            'x_m':           round(x_now, 5),
            'epoch_time_s':  round(epoch_time, 1),
        })
        log_file.flush()

        if phase3 and patience_ctr >= PATIENCE:
            print(f"\n  Early stop at epoch {epoch} (best val = {best_val:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n  Best model restored (val = {best_val:.4f})")
        print(f"  Learned x = {model.ode.x.item():.5f} m")

    log_file.close()
    print(f"\n  Log  -> {log_path}")
    print(f"  Best -> {LOG_DIR / 'best_model.pt'}\n")
    return model, scaler


if __name__ == '__main__':
    model, scaler = train()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    _, _, test_dfs, _ = load_data()
    if test_dfs:
        from evaluate import evaluate
        evaluate(model, test_dfs, scaler, device)
