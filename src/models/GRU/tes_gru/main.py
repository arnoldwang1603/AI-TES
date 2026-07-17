"""Orchestration: sweep all seeds x variants, then build comparison figures.

The original `if __name__ == "__main__":` block is now `def main():`; the
thin launcher GRU_input_ablation.py calls it.
"""
import os
import glob
import json
import shutil
import time

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from . import config            # live access to the mutable config.RUN_NAME
from .config import *
from .utils import _to_jsonable, format_duration
from .data import *
from .models import *
from .rollout import *
from .runio import *
from .train import *
from .evaluate import *


def main():
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
        config.RUN_NAME = f"{RUN_NAME_BASE}_seed{current_seed}"
        # Keep the live config.SEED in sync so run_config.json records the
        # seed actually being run (it used to freeze the legacy SEED=SEEDS[0]
        # constant, stamping "seed": 7 into every seed's snapshot).
        config.SEED = current_seed
        print()
        print("#" * 70)
        print(f"# SEED {seed_idx + 1}/{len(SEEDS)}  seed={current_seed}  "
              f"RUN_NAME={config.RUN_NAME}")
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
        print(f"  Sliding pad mode: {SLIDING_PAD_MODE}")
        print(f"  Run name: {config.RUN_NAME}")
        print(f"  All artifacts will be written under: {run_dir()}")
        print("#" * 60)

        # Create the canonical run directory layout up-front, then dump the
        # config snapshot so any later inspection of the run folder is
        # self-describing (no need to read the script).
        ensure_run_layout()
        cfg_path = save_run_config_snapshot()
        print(f"Run config snapshot -> {cfg_path}")

        script_dir = BASE_DIR
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
                    # meta.json round-trips NaN -> JSON null -> None (see
                    # _to_jsonable, which converts non-finite floats to None
                    # because JSON has no NaN). Freshly-trained variants hold
                    # real float('nan') in memory for non-predicted channels
                    # (e.g. Test_MAE_Input_T on a forward variant), so restore
                    # None -> NaN here to keep cached and fresh results
                    # consistent. Without this, downstream plotting does
                    # int(0) + None and crashes (TypeError: int + NoneType).
                    # Only scalar meta fields are present at this point (the
                    # *_history lists are added below), so a blanket coercion
                    # is safe. (Bug found 2026-05-25 after checkpoint reuse.)
                    for _k, _v in cached.items():
                        if _v is None:
                            cached[_k] = float('nan')
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
                        'sliding_pad_mode': SLIDING_PAD_MODE,
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
            # Replace NaN / None with 0 for plotting (bar disappears for a
            # not-predicted channel). None can reach here from a cached
            # meta.json (JSON null); NaN from a freshly-trained variant.
            def _nz(v):
                if v is None:
                    return 0.0
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
