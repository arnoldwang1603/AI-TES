"""
Full overnight training of PhysTFv3 on the new temperature dataset.

Train data : new data/Temperature input/training_data.../Latest Database.../  (311 CSVs)
Test  data : new data/Temperature input/tests.../70_cases/                     (70 CSVs)

Results land in:
  PhysTFv3/checkpoints_new_temp/best_model.pt   — best weights + scaler
  PhysTFv3/checkpoints_new_temp/resume.pt        — resume checkpoint
  PhysTFv3/results/logs/PhysTFv3_NewTemp/        — training log, plots, metrics
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# --- Path override BEFORE any module reads them ----------------------------
import config
import dataset as dataset_module

_BASE       = config.BASE_DIR.parent
_NEW_TRAIN  = (_BASE / "new data" / "Temperature input"
               / "training_data (temperature input)"
               / "Latest Database (Use this for training)")
_NEW_TEST   = (_BASE / "new data" / "Temperature input"
               / "tests (temperature input)" / "70_cases")

config.TRAIN_DATA_DIR         = _NEW_TRAIN
config.TEST_DATA_DIR          = _NEW_TEST
config.MODEL_VERSION          = "PhysTFv3_NewTemp"
dataset_module.TRAIN_DATA_DIR = _NEW_TRAIN
dataset_module.TEST_DATA_DIR  = _NEW_TEST

# --- Now safe to import the rest -------------------------------------------
import copy
import csv
import pickle
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
    RESULTS_DIR, VAL_SPLIT,
    T_AVG_WEIGHT, TAVG_WARMUP_START, TAVG_WARMUP_LEN,
    MODEL_VERSION,
)
from dataset import ThermalDataset, load_data
from model import ThermalPredictor, count_params
import model as model_module

# Separate checkpoint directory — keeps the original model untouched
CKPT_DIR   = config.BASE_DIR / "checkpoints_new_temp"
RESUME_PATH = CKPT_DIR / "resume.pt"
BEST_PATH   = CKPT_DIR / "best_model.pt"
SCALER_PATH = CKPT_DIR / "scaler.pkl"

# Use window=120 (best from original sweep)
model_module.ATTN_WINDOW = 120


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


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
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR = RESULTS_DIR / 'logs' / MODEL_VERSION
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()

    print(f"\n{'='*65}")
    print(f"  {MODEL_VERSION}  |  device: {device}")
    print(f"  Train: {_NEW_TRAIN}")
    print(f"  Test:  {_NEW_TEST}")
    print(f"{'='*65}\n")

    print("Loading data...")
    train_dfs, val_dfs, test_dfs, scaler = load_data(val_split=VAL_SPLIT)

    # Persist scaler — needed to reload for future evaluation
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  Scaler saved to {SCALER_PATH}")

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
    print(f"  Parameters:  {count_params(model):,}")
    print(f"  ATTN_WINDOW: {model_module.ATTN_WINDOW}")
    print(f"  x init:      {model.ode.x.item():.5f} m\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)

    best_val     = float('inf')
    best_state   = None
    patience_ctr = 0
    start_epoch  = 0

    if RESUME_PATH.exists():
        print("  Resuming from checkpoint...")
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
            BEST_PATH.unlink(missing_ok=True)
    else:
        print("  Starting fresh\n")

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

    print(f"  Ta warmup: silent {TAVG_WARMUP_START} ep, ramp {TAVG_WARMUP_LEN} ep")
    print(f"  Max epochs: {MAX_EPOCHS}  |  Patience: {PATIENCE}\n")
    print(f"  {'Ep':>4}  {'train':>9}  {'val_data':>9}  "
          f"{'Ti':>7}  {'To':>7}  {'Ta':>7}  {'Ta_w':>6}  {'phys':>7}  {'s/ep':>6}")
    print(f"  {'─'*80}")

    t_train_start = time.time()

    for epoch in range(start_epoch, MAX_EPOCHS):
        t0   = time.time()
        ta_w = tavg_weight_for_epoch(epoch)

        if epoch == TAVG_WARMUP_START:
            best_val, patience_ctr = float('inf'), 0
            print(f"\n  Phase 2: Ta loss ramping in (epoch {epoch})")
        elif epoch == TAVG_WARMUP_START + TAVG_WARMUP_LEN:
            best_val, patience_ctr = float('inf'), 0
            print(f"\n  Phase 3: full supervision (epoch {epoch})")

        train_losses = run_epoch(model, train_loader, device,
                                 optimizer=optimizer, train=True, tavg_w=ta_w)
        val_losses   = run_epoch(model, val_loader,   device,
                                 train=False, tavg_w=ta_w)
        scheduler.step()

        epoch_time = time.time() - t0
        lr_now     = optimizer.param_groups[0]['lr']
        x_now      = model.ode.x.item()
        phase3     = epoch >= TAVG_WARMUP_START + TAVG_WARMUP_LEN
        monitor    = val_losses['data']

        if monitor < best_val:
            best_val   = monitor
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
            ckpt_data = {
                'epoch': epoch, 'state_dict': best_state,
                'val_loss': best_val, 'x_m': x_now,
            }
            torch.save(ckpt_data, BEST_PATH)
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
                f"  {epoch_time:5.1f}s{star}{patience_str}"
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

    total_train_time = (time.time() - t_train_start) / 60
    print(f"\n  Training complete in {total_train_time:.1f} min")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  Best model restored (val = {best_val:.4f})")
        print(f"  Learned x = {model.ode.x.item():.5f} m")

    log_file.close()
    print(f"  Log  -> {log_path}")
    print(f"  Best -> {BEST_PATH}\n")
    return model, scaler, test_dfs, device


def evaluate_full(model, test_dfs, scaler, device):
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score
    from datetime import datetime

    LOG_DIR  = RESULTS_DIR / 'logs' / MODEL_VERSION
    plot_dir = LOG_DIR / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    results = []

    print(f"\n{'='*65}")
    print(f"  {MODEL_VERSION} — {len(test_dfs)} test cases")
    print(f"{'='*65}\n")
    print(f"  {'Case':42s}  {'Ti MAE':>7}  {'To MAE':>7}  {'Ta MAE':>7}")
    print(f"  {'─'*70}")

    for df, fname in test_dfs:
        ds = ThermalDataset(df, scaler=scaler, file_name=fname)
        (tin_sc, dtin_sc, init_sc,
         tin_ph, init_ph, y_ph) = [t.unsqueeze(0).to(device) for t in ds[0]]

        with torch.no_grad():
            preds, _ = model(tin_sc, dtin_sc, init_sc, tin_ph, init_ph)

        pred_np   = preds[0].cpu().numpy()
        target_np = y_ph[0].cpu().numpy()
        err_np    = np.abs(target_np - pred_np)

        mae_i  = err_np[:, 0].mean()
        mae_o  = err_np[:, 1].mean()
        mae_a  = err_np[:, 2].mean()
        rmse_i = np.sqrt((err_np[:, 0] ** 2).mean())
        rmse_o = np.sqrt((err_np[:, 1] ** 2).mean())
        rmse_a = np.sqrt((err_np[:, 2] ** 2).mean())
        max_i  = err_np[:, 0].max()
        max_o  = err_np[:, 1].max()
        max_a  = err_np[:, 2].max()
        r2_i   = r2_score(target_np[:, 0], pred_np[:, 0])
        r2_o   = r2_score(target_np[:, 1], pred_np[:, 1])
        r2_a   = r2_score(target_np[:, 2], pred_np[:, 2])

        print(f"  {fname:42s}  {mae_i:7.3f}  {mae_o:7.3f}  {mae_a:7.3f}")
        results.append(dict(
            case=fname,
            mae_ti=mae_i,    mae_to=mae_o,    mae_ta=mae_a,
            rmse_ti=rmse_i,  rmse_to=rmse_o,  rmse_ta=rmse_a,
            maxerr_ti=max_i, maxerr_to=max_o, maxerr_ta=max_a,
            r2_ti=r2_i,     r2_to=r2_o,      r2_ta=r2_a,
        ))

        # Per-case plot (3 panels: Ti, To, Ta)
        times  = np.arange(len(pred_np)) * 10
        labels = ['T_inner (°C)', 'T_outer (°C)', 'T_avg (°C)']
        cols   = ['crimson', 'steelblue', 'forestgreen']
        maes   = [mae_i, mae_o, mae_a]

        fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=False)
        fig.suptitle(f'{MODEL_VERSION}  —  {fname}', fontsize=11)
        for k, ax in enumerate(axes):
            ax.plot(times, target_np[:, k], color=cols[k], lw=1.5, label='actual')
            ax.plot(times, pred_np[:, k],   color=cols[k], lw=1.5, ls='--',
                    alpha=0.85, label=f'pred  MAE={maes[k]:.2f}°C')
            ax.set_title(labels[k], fontsize=10)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('°C')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / (os.path.splitext(fname)[0] + '.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    import pandas as pd
    df_res = pd.DataFrame(results)
    means  = df_res.select_dtypes(float).mean().to_dict()
    means['case'] = 'MEAN'
    df_out = pd.concat([df_res, pd.DataFrame([means])], ignore_index=True)

    print(f"\n  {'─'*70}")
    print(f"  {'MEAN':42s}  {means['mae_ti']:7.3f}  {means['mae_to']:7.3f}  {means['mae_ta']:7.3f}")
    print(f"\n  RMSE   Ti: {means['rmse_ti']:.3f}°C  To: {means['rmse_to']:.3f}°C  Ta: {means['rmse_ta']:.3f}°C")
    print(f"  MaxErr Ti: {means['maxerr_ti']:.3f}°C  To: {means['maxerr_to']:.3f}°C  Ta: {means['maxerr_ta']:.3f}°C")
    print(f"  R²     Ti: {means['r2_ti']:.4f}       To: {means['r2_to']:.4f}       Ta: {means['r2_ta']:.4f}")

    df_out.to_csv(LOG_DIR / 'test_results.csv', index=False, float_format='%.4f')
    df_out[['case', 'mae_ti', 'mae_to', 'mae_ta']].to_csv(
        LOG_DIR / 'summary.csv', index=False, float_format='%.4f')

    # Summary bar chart across all 70 cases
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    for k, (col_name, label, color) in enumerate([
            ('mae_ti', 'T_inner MAE (°C)', 'crimson'),
            ('mae_to', 'T_outer MAE (°C)', 'steelblue'),
            ('mae_ta', 'T_avg MAE (°C)',   'forestgreen')]):
        ax  = axes[k]
        vals = df_res[col_name].values
        ax.bar(range(len(vals)), sorted(vals), color=color, alpha=0.75)
        ax.axhline(means[col_name], color='black', lw=1.5, ls='--',
                   label=f'mean={means[col_name]:.3f}°C')
        ax.set_title(label, fontsize=11)
        ax.set_xlabel('Test case (sorted)')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle(f'{MODEL_VERSION} — 70-case test summary', fontsize=12)
    fig.tight_layout()
    fig.savefig(LOG_DIR / 'test_summary_bars.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved to results/logs/{MODEL_VERSION}/")
    print(f"    test_results.csv    — full stats (70 cases)")
    print(f"    summary.csv         — MAE per case")
    print(f"    test_summary_bars.png")
    print(f"    plots/              — {len(results)} per-case PNGs")

    # Update model_comparison.csv
    comp_path = RESULTS_DIR / 'logs' / 'model_comparison.csv'
    row = {
        'model': MODEL_VERSION, 'date': datetime.now().strftime('%Y-%m-%d'),
        'n_cases': len(results),
        'mae_ti': round(means['mae_ti'], 4), 'mae_to': round(means['mae_to'], 4),
        'mae_ta': round(means['mae_ta'], 4), 'rmse_ti': round(means['rmse_ti'], 4),
        'rmse_to': round(means['rmse_to'], 4), 'rmse_ta': round(means['rmse_ta'], 4),
        'maxerr_ti': round(means['maxerr_ti'], 4), 'maxerr_to': round(means['maxerr_to'], 4),
        'maxerr_ta': round(means['maxerr_ta'], 4),
        'r2_ti': round(means['r2_ti'], 4), 'r2_to': round(means['r2_to'], 4),
        'r2_ta': round(means['r2_ta'], 4),
    }
    if comp_path.exists():
        df_comp = pd.read_csv(comp_path)
        df_comp = df_comp[df_comp['model'] != MODEL_VERSION]
        df_comp = pd.concat([df_comp, pd.DataFrame([row])], ignore_index=True)
    else:
        df_comp = pd.DataFrame([row])
    df_comp.to_csv(comp_path, index=False)
    print(f"    ../model_comparison.csv — updated\n")


if __name__ == '__main__':
    model, scaler, test_dfs, device = train()
    if test_dfs:
        evaluate_full(model, test_dfs, scaler, device)
    else:
        print("  WARNING: no test files found — check _NEW_TEST path above")
