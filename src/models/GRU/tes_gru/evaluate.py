"""Per-case testing, metrics, and per-case prediction plots."""
import os
import time

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from .config import *
from .runio import *


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
                            if SLIDING_PAD_MODE == "variable":
                                continue
                            elif SLIDING_PAD_MODE == "zero":
                                window_steps.append([0.0, 0.0])
                            else:  # "init": repeat t=0 observation (Time, T_avg)
                                window_steps.append([x[0, 0].item(), x[0, 1].item()])
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
            # Column layout mirrors data.py / _rollout_sliding:
            #   arfed/anchor: [Time, T_outer, T_inner, T_avg, Input_T] (5-d)
            #   output_only : [Time, T_outer, T_avg, Input_T]          (4-d)
            out_only = (TINNER_MODE == "output_only")
            if out_only:
                IDX_AVG, IDX_INP = 2, 3
            else:
                IDX_AVG, IDX_INP = 3, 4
            feat_dim = x.shape[1]
            outer_hist = [float(x[0, 1].item())]   # k=0: GT initial
            inner_hist = None if out_only else [float(x[0, 2].item())]
            avg_hist = [float(x[0, IDX_AVG].item())]
            anchor = model.tinner_anchor if TINNER_MODE == "anchor" else None
            step_anchor = getattr(model, 'step_anchor', None) \
                if OTHER_CH_MODE == "persistence" else None

            preds = []
            cur_h = hidden
            for t in range(seq_len):
                window_steps = []
                for offset in range(-(W - 1), 1):
                    k = t + offset
                    if k < 0:
                        if SLIDING_PAD_MODE == "variable":
                            continue
                        elif SLIDING_PAD_MODE == "zero":
                            window_steps.append([0.0] * feat_dim)
                        else:  # "init": repeat the t=0 steady-state row
                            window_steps.append(
                                [x[0, j].item() for j in range(feat_dim)])
                    else:
                        if out_only:
                            window_steps.append([
                                x[k, 0].item(),     # Time(k)      GT
                                outer_hist[k],      # T_outer(k)   AR
                                avg_hist[k],        # T_avg(k)     AR
                                x[k, IDX_INP].item(),  # Input_T(k) GT
                            ])
                        else:
                            window_steps.append([
                                x[k, 0].item(),     # Time(k)      GT
                                outer_hist[k],      # T_outer(k)   AR
                                inner_hist[k],      # T_inner(k)   AR
                                avg_hist[k],        # T_avg(k)     AR
                                x[k, IDX_INP].item(),  # Input_T(k) GT
                            ])
                window = torch.tensor(
                    window_steps, dtype=torch.float32
                ).unsqueeze(0).to(DEVICE)   # (1, W, feat_dim)

                with torch.no_grad():
                    out, cur_h = model(window, cur_h)
                pred = out[0, -1].cpu().numpy().copy()   # last step's output
                if step_anchor is not None:
                    (kb_o, kc_o), (kb_a, kc_a) = step_anchor
                    pred[1] = outer_hist[t] + kb_o + kc_o * pred[1]
                    pred[2] = avg_hist[t] + kb_a + kc_a * pred[2]
                if anchor is not None:
                    # Reconstruct T_inner from the z-scored residual head,
                    # anchored on GT Input_T(t) -- mirrors _rollout_sliding.
                    ka, kb, kc = anchor
                    t_a = min(t + 1, seq_len - 1) if ANCHOR_LEAD else t
                    pred[0] = ka * float(x[t_a, IDX_INP].item()) + kb + kc * pred[0]
                preds.append(pred)

                # pred channels (per VARIANT_OUTPUT_CHANNELS) = [T_inner, T_outer, T_avg]
                outer_hist.append(float(pred[1]))
                avg_hist.append(float(pred[2]))
                if inner_hist is not None:
                    inner_hist.append(float(pred[0]))
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
        # Prepend the t=0 row (Arnold 2026-07-23): predictions cover t=1..n,
        # so plotting from t=1 hid the shared starting point and made a
        # one-step-lagged prediction look like it started somewhere else.
        # t=0 is the GIVEN initial condition, identical for actual and pred.
        # Plot-only -- inv_actual / inv_pred (and every metric) are untouched.
        row0 = ds.scaler.inverse_transform(np.column_stack([
            ft[:1], ft_out[:1], ft_in[:1], ft_avg[:1], ft_inp[:1],
        ]).astype(np.float64))
        plot_actual = np.vstack([row0, inv_actual])
        plot_pred = np.vstack([row0, inv_pred])
        time_plot = plot_actual[:, 0]
        for ch_name in channel_names:
            col = SCALER_COL[ch_name]
            c = CH_COLOR.get(ch_name, 'black')
            plt.plot(time_plot, plot_actual[:, col],
                     label=f"{ch_name} Actual", color=c)
            plt.plot(time_plot, plot_pred[:, col], '--',
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


