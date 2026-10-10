import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
from pathlib import Path
from datetime import datetime

from config import CHECKPOINT_DIR, RESULTS_DIR, VAL_SPLIT, MODEL_VERSION
from dataset import ThermalDataset, load_data
from model import ThermalPredictor, count_params


def _version_dir() -> Path:
    return RESULTS_DIR / 'logs' / MODEL_VERSION


def evaluate(model, test_dfs, scaler, device):
    ver_dir  = _version_dir()
    plot_dir = ver_dir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / 'logs').mkdir(parents=True, exist_ok=True)

    model.eval()
    results = []

    print(f"\n{'='*65}")
    print(f"  {MODEL_VERSION} — {len(test_dfs)} test cases")
    print(f"{'='*65}\n")
    print(f"  {'Case':40s}  {'Ti MAE':>7}  {'To MAE':>7}  {'Ta MAE':>7}")
    print(f"  {'─'*68}")

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

        print(f"  {fname:40s}  {mae_i:7.3f}  {mae_o:7.3f}  {mae_a:7.3f}")
        results.append(dict(
            case=fname,
            mae_ti=mae_i,    mae_to=mae_o,    mae_ta=mae_a,
            rmse_ti=rmse_i,  rmse_to=rmse_o,  rmse_ta=rmse_a,
            maxerr_ti=max_i, maxerr_to=max_o, maxerr_ta=max_a,
            r2_ti=r2_i,     r2_to=r2_o,      r2_ta=r2_a,
        ))

        times  = np.arange(len(pred_np)) * 10
        labels = ['T_inner', 'T_outer', 'T_avg']
        cols   = ['crimson', 'steelblue', 'forestgreen']
        maes   = [mae_i, mae_o, mae_a]

        fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=False)
        fig.suptitle(f'{MODEL_VERSION}  —  {fname}', fontsize=11)
        for k, ax in enumerate(axes):
            ax.plot(times, target_np[:, k], color=cols[k], lw=1.5, label='actual')
            ax.plot(times, pred_np[:, k], color=cols[k], lw=1.5, ls='--',
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

    df_res = pd.DataFrame(results)
    means  = df_res.select_dtypes(float).mean().to_dict()
    means['case'] = 'MEAN'
    df_res = pd.concat([df_res, pd.DataFrame([means])], ignore_index=True)

    print(f"\n  {'─'*68}")
    print(f"  {'MEAN':40s}  {means['mae_ti']:7.3f}  {means['mae_to']:7.3f}  {means['mae_ta']:7.3f}")
    print(f"\n  RMSE   Ti: {means['rmse_ti']:.3f}°C  To: {means['rmse_to']:.3f}°C  Ta: {means['rmse_ta']:.3f}°C")
    print(f"  MaxErr Ti: {means['maxerr_ti']:.3f}°C  To: {means['maxerr_to']:.3f}°C  Ta: {means['maxerr_ta']:.3f}°C")
    print(f"  R²     Ti: {means['r2_ti']:.4f}       To: {means['r2_to']:.4f}       Ta: {means['r2_ta']:.4f}")

    df_res[['case', 'mae_ti', 'mae_to', 'mae_ta']].to_csv(
        ver_dir / 'summary.csv', index=False, float_format='%.4f')
    df_res.to_csv(ver_dir / 'test_results.csv', index=False, float_format='%.4f')

    print(f"\n  Saved to results/logs/{MODEL_VERSION}/")
    print(f"    plots/           — {len(results)} PNG files")
    print(f"    summary.csv      — MAE per case")
    print(f"    test_results.csv — full stats")

    comparison_path = RESULTS_DIR / 'logs' / 'model_comparison.csv'
    row = {
        'model':       MODEL_VERSION,
        'date':        datetime.now().strftime('%Y-%m-%d'),
        'n_cases':     len(results),
        'mae_ti':      round(means['mae_ti'],    4),
        'mae_to':      round(means['mae_to'],    4),
        'mae_ta':      round(means['mae_ta'],    4),
        'rmse_ti':     round(means['rmse_ti'],   4),
        'rmse_to':     round(means['rmse_to'],   4),
        'rmse_ta':     round(means['rmse_ta'],   4),
        'maxerr_ti':   round(means['maxerr_ti'], 4),
        'maxerr_to':   round(means['maxerr_to'], 4),
        'maxerr_ta':   round(means['maxerr_ta'], 4),
        'r2_ti':       round(means['r2_ti'],     4),
        'r2_to':       round(means['r2_to'],     4),
        'r2_ta':       round(means['r2_ta'],     4),
    }
    if comparison_path.exists():
        df_comp = pd.read_csv(comparison_path)
        df_comp = df_comp[df_comp['model'] != MODEL_VERSION]
        df_comp = pd.concat([df_comp, pd.DataFrame([row])], ignore_index=True)
    else:
        df_comp = pd.DataFrame([row])
    df_comp.to_csv(comparison_path, index=False)
    print(f"    ../model_comparison.csv — updated\n")


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"\n  {MODEL_VERSION} Evaluation  |  device: {device}")

    ckpt_path = CHECKPOINT_DIR / 'best_model.pt'
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}. Run train.py first.")

    _, _, test_dfs, scaler = load_data(val_split=VAL_SPLIT)
    model = ThermalPredictor().to(device)
    model.register_scaler(scaler)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    print(f"  Loaded epoch {ckpt['epoch']}  (val = {ckpt['val_loss']:.4f})")
    print(f"  Parameters: {count_params(model):,}")
    print(f"  Learned x  = {model.ode.x.item():.5f} m")

    if not test_dfs:
        print("  No test files found — check TEST_DATA_DIR in config.py")
    else:
        evaluate(model, test_dfs, scaler, device)
