import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import config
import dataset as dataset_module

_BASE = config.BASE_DIR.parent
_NEW_TRAIN = (_BASE / "new data" / "Temperature input"
              / "training_data (temperature input)"
              / "Latest Database (Use this for training)")
_NEW_TEST  = (_BASE / "new data" / "Temperature input"
              / "tests (temperature input)" / "70_cases")

config.TRAIN_DATA_DIR       = _NEW_TRAIN
config.TEST_DATA_DIR        = _NEW_TEST
dataset_module.TRAIN_DATA_DIR = _NEW_TRAIN
dataset_module.TEST_DATA_DIR  = _NEW_TEST

import copy
import random
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import r2_score

import model as model_module
from dataset import ThermalDataset, load_data
from model import ThermalPredictor, count_params

SEED_FIXED   = 42
WINDOW_FIXED = 120

SEEDS   = [7, 21, 42, 123]
WINDOWS = [60, 120, 240, 720, 1440]

EXP_MAX_EPOCHS = 150
EXP_PATIENCE   = 40

if os.environ.get('FAST_RUN'):
    SEEDS          = [7, 21, 42, 123]   # seed sweep only, skip window sweep
    WINDOWS        = [120]              # single window keeps grid to 4 runs
    EXP_MAX_EPOCHS = 60
    EXP_PATIENCE   = 15
    config.TAVG_WARMUP_START = 10
    config.TAVG_WARMUP_LEN   = 10

EXP_DIR = config.RESULTS_DIR / 'experiments_new_temp'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def tavg_weight(epoch):
    ws, wl = config.TAVG_WARMUP_START, config.TAVG_WARMUP_LEN  # reads live config values
    if epoch < ws:
        return 0.0
    ramp = epoch - ws
    if ramp < wl:
        return config.T_AVG_WEIGHT * (ramp + 1) / wl
    return config.T_AVG_WEIGHT


def run_epoch(model, loader, device, optimizer=None, train=True, tavg_w=None):
    model.train(train)
    totals = dict(total=0.0, data=0.0)
    maes   = [0.0, 0.0, 0.0]
    n = 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            tin_sc, dtin_sc, init_sc, tin_ph, init_ph, y_ph = [b.to(device) for b in batch]
            preds, params = model(tin_sc, dtin_sc, init_sc, tin_ph, init_ph)
            L_total, L_data, _, _ = model.compute_loss(preds, y_ph, params, tavg_w=tavg_w)
            if train and optimizer:
                optimizer.zero_grad()
                L_total.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                optimizer.step()
            B = tin_sc.size(0)
            totals['total'] += L_total.item() * B
            totals['data']  += L_data.item()  * B
            with torch.no_grad():
                err = (preds - y_ph).abs()
                for k in range(3):
                    maes[k] += err[:, :, k].mean().item() * B
            n += B
    return {k: v / n for k, v in totals.items()}, [m / n for m in maes]


def run_experiment(seed, window, train_loader, val_loader, test_dfs, scaler, device, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    model_module.ATTN_WINDOW = window

    model = ThermalPredictor().to(device)
    model.register_scaler(scaler)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EXP_MAX_EPOCHS, eta_min=1e-5)

    best_val, best_state, patience_ctr = float('inf'), None, 0
    log_rows = []

    print(f"\n  seed={seed}  window={window}  [{count_params(model):,} params]")
    print(f"  {'ep':>4}  {'train':>8}  {'val':>8}  {'Ti':>6}  {'To':>6}  {'Ta':>6}")

    for epoch in range(EXP_MAX_EPOCHS):
        ta_w = tavg_weight(epoch)

        ws = config.TAVG_WARMUP_START
        if epoch == ws:
            best_val, patience_ctr = float('inf'), 0
        elif epoch == ws + config.TAVG_WARMUP_LEN:
            best_val, patience_ctr = float('inf'), 0

        train_loss, _ = run_epoch(model, train_loader, device,
                                   optimizer=optimizer, train=True, tavg_w=ta_w)
        val_loss, val_maes = run_epoch(model, val_loader, device,
                                        train=False, tavg_w=ta_w)
        scheduler.step()

        phase3  = epoch >= config.TAVG_WARMUP_START + config.TAVG_WARMUP_LEN  # uses live config
        monitor = val_loss['data']

        if monitor < best_val:
            best_val     = monitor
            best_state   = copy.deepcopy(model.state_dict())
            patience_ctr = 0
            star = ' *'
        else:
            patience_ctr += 1 if phase3 else 0
            star = ''

        log_rows.append({
            'epoch': epoch, 'train': round(train_loss['total'], 5),
            'val': round(val_loss['data'], 5),
            'mae_ti': round(val_maes[0], 4),
            'mae_to': round(val_maes[1], 4),
            'mae_ta': round(val_maes[2], 4),
            'ta_w':   round(ta_w, 3),
        })

        if epoch % 10 == 0 or star:
            print(f"  {epoch:4d}  {train_loss['total']:8.4f}  {val_loss['data']:8.4f}  "
                  f"{val_maes[0]:6.3f}  {val_maes[1]:6.3f}  {val_maes[2]:6.3f}{star}")

        if phase3 and patience_ctr >= EXP_PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break

    pd.DataFrame(log_rows).to_csv(out_dir / 'training_log.csv', index=False)

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    results = []
    for df, fname in test_dfs:
        ds = ThermalDataset(df, scaler=scaler, file_name=fname)
        inputs = [t.unsqueeze(0).to(device) for t in ds[0]]
        tin_sc, dtin_sc, init_sc, tin_ph, init_ph, y_ph = inputs
        with torch.no_grad():
            preds, _ = model(tin_sc, dtin_sc, init_sc, tin_ph, init_ph)
        pred_np   = preds[0].cpu().numpy()
        target_np = y_ph[0].cpu().numpy()
        err_np    = np.abs(target_np - pred_np)
        results.append({
            'case':    fname,
            'mae_ti':  err_np[:, 0].mean(),
            'mae_to':  err_np[:, 1].mean(),
            'mae_ta':  err_np[:, 2].mean(),
            'rmse_ti': np.sqrt((err_np[:, 0] ** 2).mean()),
            'rmse_to': np.sqrt((err_np[:, 1] ** 2).mean()),
            'rmse_ta': np.sqrt((err_np[:, 2] ** 2).mean()),
            'r2_ti':   r2_score(target_np[:, 0], pred_np[:, 0]),
            'r2_to':   r2_score(target_np[:, 1], pred_np[:, 1]),
            'r2_ta':   r2_score(target_np[:, 2], pred_np[:, 2]),
        })

    df_res = pd.DataFrame(results)
    means  = df_res.select_dtypes(float).mean().to_dict()
    means['case'] = 'MEAN'
    pd.concat([df_res, pd.DataFrame([means])]).to_csv(
        out_dir / 'test_results.csv', index=False, float_format='%.4f')

    print(f"  Test -> Ti:{means['mae_ti']:.3f}  To:{means['mae_to']:.3f}  Ta:{means['mae_ta']:.3f}")
    return means, log_rows


def bar_chart(groups, values_ti, values_to, values_ta, xlabel, title, path):
    x     = np.arange(len(groups))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(6, len(groups) * 1.5), 4))
    ax.bar(x - width, values_ti, width, label='Ti', color='crimson',     alpha=0.85)
    ax.bar(x,         values_to, width, label='To', color='steelblue',   alpha=0.85)
    ax.bar(x + width, values_ta, width, label='Ta', color='forestgreen', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('MAE (°C)')
    ax.set_title(title)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot -> {path}")


def loss_curves(experiments, label_fn, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for tag, log in experiments:
        df  = pd.DataFrame(log)
        lbl = label_fn(tag)
        axes[0].plot(df['epoch'], df['train'], label=lbl, lw=1.5)
        axes[1].plot(df['epoch'], df['val'],   label=lbl, lw=1.5)
    for ax, ttl in zip(axes, ['Train loss', 'Val loss (data)']):
        ax.set_title(ttl)
        ax.set_xlabel('Epoch')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot -> {path}")


def main():
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"\nDevice: {device}")
    print(f"Train: {_NEW_TRAIN}")
    print(f"Test:  {_NEW_TEST}")

    set_seed(SEED_FIXED)
    train_dfs, val_dfs, test_dfs, scaler = load_data(val_split=config.VAL_SPLIT)

    pin = (device.type == 'cuda')
    train_loader = DataLoader(
        ConcatDataset([ThermalDataset(df, scaler=scaler, file_name=fn) for df, fn in train_dfs]),
        batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(
        ConcatDataset([ThermalDataset(df, scaler=scaler, file_name=fn) for df, fn in val_dfs]),
        batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin)

    seen, grid = set(), []
    for s in SEEDS:
        grid.append((s, WINDOW_FIXED))
        seen.add((s, WINDOW_FIXED))
    for w in WINDOWS:
        if (SEED_FIXED, w) not in seen:
            grid.append((SEED_FIXED, w))
            seen.add((SEED_FIXED, w))

    all_results, all_logs = {}, {}

    t_start = time.time()
    for i, (seed, window) in enumerate(grid):
        print(f"\n[{i+1}/{len(grid)}]  seed={seed}  window={window}")
        out_dir = EXP_DIR / f'seed{seed}_window{window}'
        if (out_dir / 'test_results.csv').exists():
            print(f"  Already done — loading cached results")
            df    = pd.read_csv(out_dir / 'test_results.csv')
            means = df[df['case'] == 'MEAN'].iloc[0].to_dict()
            log   = pd.read_csv(out_dir / 'training_log.csv').to_dict('records')
        else:
            means, log = run_experiment(
                seed, window, train_loader, val_loader, test_dfs, scaler, device, out_dir)
        all_results[(seed, window)] = means
        all_logs[(seed, window)]    = log

    elapsed = time.time() - t_start
    print(f"\nAll experiments done in {elapsed/3600:.1f} h")

    rows = []
    for (seed, window), means in all_results.items():
        rows.append({'seed': seed, 'window': window,
                     'mae_ti':  round(means['mae_ti'],  4),
                     'mae_to':  round(means['mae_to'],  4),
                     'mae_ta':  round(means['mae_ta'],  4),
                     'rmse_ti': round(means['rmse_ti'], 4),
                     'rmse_to': round(means['rmse_to'], 4),
                     'rmse_ta': round(means['rmse_ta'], 4)})
    pd.DataFrame(rows).to_csv(EXP_DIR / 'summary.csv', index=False)

    seed_keys = [(s, WINDOW_FIXED) for s in SEEDS]
    bar_chart(
        groups    = [str(s) for s in SEEDS],
        values_ti = [all_results[k]['mae_ti'] for k in seed_keys],
        values_to = [all_results[k]['mae_to'] for k in seed_keys],
        values_ta = [all_results[k]['mae_ta'] for k in seed_keys],
        xlabel    = 'Seed',
        title     = f'MAE by Seed — new temp data  (window={WINDOW_FIXED})',
        path      = EXP_DIR / 'comparison_seeds.png',
    )
    loss_curves(
        experiments = [(k, all_logs[k]) for k in seed_keys],
        label_fn    = lambda k: f'seed={k[0]}',
        title       = f'Training curves by seed — new temp data  (window={WINDOW_FIXED})',
        path        = EXP_DIR / 'curves_seeds.png',
    )

    window_keys   = [(SEED_FIXED, w) for w in WINDOWS]
    window_labels = [str(w) if w < 1440 else 'full' for w in WINDOWS]
    bar_chart(
        groups    = window_labels,
        values_ti = [all_results[k]['mae_ti'] for k in window_keys],
        values_to = [all_results[k]['mae_to'] for k in window_keys],
        values_ta = [all_results[k]['mae_ta'] for k in window_keys],
        xlabel    = 'Attention window (steps)',
        title     = f'MAE by Attention Window — new temp data  (seed={SEED_FIXED})',
        path      = EXP_DIR / 'comparison_windows.png',
    )
    loss_curves(
        experiments = [(k, all_logs[k]) for k in window_keys],
        label_fn    = lambda k: f'window={k[1]}' if k[1] < 1440 else 'full',
        title       = f'Training curves by window — new temp data  (seed={SEED_FIXED})',
        path        = EXP_DIR / 'curves_windows.png',
    )

    print(f"\n{'seed':>6}  {'window':>7}  {'Ti MAE':>7}  {'To MAE':>7}  {'Ta MAE':>7}")
    print('─' * 44)
    for r in rows:
        print(f"  {r['seed']:>4}  {r['window']:>7}  {r['mae_ti']:>7.3f}  "
              f"{r['mae_to']:>7.3f}  {r['mae_ta']:>7.3f}")
    print(f"\nResults -> {EXP_DIR}")


if __name__ == '__main__':
    main()
