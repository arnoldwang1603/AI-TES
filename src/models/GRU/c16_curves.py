"""C16 reference-quality, horizon and raw-head range analyses (no training).

CLI: validation only. F11 uses test_points with precomputed training reference
statistics; it must not refit reference statistics on the sealed test set.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from analysis_common import (CHANNELS, COLS, bundle_arrays, cache_for_keys,
                             load_manifest, load_val_bundles, normalized_key,
                             parser, predict_frame, read_frames, read_json,
                             save_plot, setup_run, split_spec, write_table, manifest_context)


def reference_statistics(records, training, physics, masked_keys, prereg=None):
    from round10_refit import channel_refit_metadata
    rows = []
    seen = set()
    arm_metadata = {}
    for record in records:
        config = record["config"]
        arm = record["arm"]
        metadata = channel_refit_metadata(config, prereg)
        if arm in seen:
            if metadata != arm_metadata[arm]:
                raise RuntimeError("Output form differs across seeds of the same arm: " + arm)
            continue
        seen.add(arm)
        arm_metadata[arm] = metadata
        ratio_mode = config.get("other_ch_mode") == "ratio_input"
        ratio_channels = {"R1": (1,), "R2": (1, 2), "R3": (0, 1, 2)}.get(config.get("ratio_channels"), ())
        for index, (channel, column) in enumerate(zip(CHANNELS, COLS)):
            residuals = []
            targets = []
            selected = []
            for key, frame in training.items():
                if channel == "T_avg" and key in masked_keys:
                    continue
                y = frame[column].to_numpy(float)[1:]
                inlet = frame["Input Temperature (C)"].to_numpy(float)
                if ratio_mode and index in ratio_channels:
                    ref = inlet[1:] + 273.15 if config.get("ratio_ref", "kelvin") == "kelvin" else np.maximum(inlet[1:], 20.0)
                    selected.append((y, ref))
                    continue
                if config.get("phys_mode", "") in ("residual", "input_residual") and index in (1, 2):
                    ref = physics[key]["T_outer_phys" if index == 1 else "T_avg_line_phys"][1:]
                    kind = "physics"
                elif channel == "T_inner" and config.get("tinner_mode", "anchor") == "anchor":
                    lead = bool(config.get("anchor_lead", True))
                    ref = inlet[1:] if lead else inlet[:-1]
                    kind = "inlet"
                elif channel == "T_outer" and config.get("touter_mode", "abs") == "anchor_input":
                    ref = inlet[1:] if config.get("anchor_lead", True) else inlet[:-1]
                    kind = "inlet"
                elif index in (1, 2) and config.get("other_ch_mode", "abs") == "anchor_initial":
                    ref = np.full(len(y), float(frame[column].iloc[0]))
                    kind = "initial"
                else:
                    ref = np.zeros(len(y))
                    kind = "none"
                residuals.append(y - ref)
                targets.append(y)
            if selected:
                mu = float(np.mean(np.concatenate([y / ref for y, ref in selected])))
                residuals = [y - mu * ref for y, ref in selected]
                targets = [y for y, ref in selected]
                kind = "ratio_kelvin" if config.get("ratio_ref", "kelvin") == "kelvin" else "ratio_celsius20"
            if not residuals:
                continue
            residual = np.concatenate(residuals)
            rows.append({"arm": arm, "channel": channel, "reference": kind,
                         "reference_residual_std_C": float(residual.std()),
                         "reference_residual_mean_C": float(residual.mean()),
                         "target_std_C": float(np.concatenate(targets).std()),
                         "n_training_files": len(residuals),
                         "n_training_steps": len(residual), **metadata[channel]})
    return pd.DataFrame(rows).drop_duplicates(["arm", "channel"])


def points(bundles, reference_stats, masked_keys=(), which_set="val"):
    rows = []
    for bundle in bundles:
        ids, actuals, predictions = bundle_arrays(bundle)
        for key, truth, pred in zip(ids, actuals, predictions):
            key = normalized_key(key)
            for index, channel in enumerate(CHANNELS):
                if channel == "T_avg" and key in masked_keys:
                    continue
                rows.append({"arm": bundle["arm"], "seed": bundle["seed"], "file": key,
                             "channel": channel, "MAE": float(np.mean(np.abs(pred[:, index] - truth[:, index]))),
                             "set": which_set})
    values = pd.DataFrame(rows)
    if not len(values):
        return values
    values = values.groupby(["arm", "channel", "set"])["MAE"].agg(["mean", "std", "count"]).reset_index()
    values.rename(columns={"mean": "model_MAE_C", "std": "seed_file_sd_C", "count": "n_seed_files"}, inplace=True)
    return values.merge(reference_stats, on=["arm", "channel"], how="left")


def test_points(bundles, reference_stats):
    """F11-only caller adapter; frames and predictions are already loaded."""
    if isinstance(reference_stats, (str, Path)):
        reference_stats = pd.read_csv(reference_stats)
    return points(bundles, reference_stats, which_set="test")


def horizon_statistics(bundles, frames, masked_keys=()):
    rows = []
    for bundle in bundles:
        ids, actuals, predictions = bundle_arrays(bundle)
        for key, truth, pred in zip(ids, actuals, predictions):
            key = normalized_key(key)
            time = frames[key]["Time (s)"].to_numpy(float)[1:]
            elapsed = time - float(frames[key]["Time (s)"].iloc[0])
            for seconds in (600, 3600, 7200, 14400):
                available = elapsed <= seconds + 1e-6
                if not available.any():
                    continue
                nearest = int(np.argmin(np.abs(elapsed - seconds)))
                for index, channel in enumerate(CHANNELS):
                    if channel == "T_avg" and key in masked_keys:
                        continue
                    error = pred[:, index] - truth[:, index]
                    rows.append({"arm": bundle["arm"], "seed": bundle["seed"], "file": key,
                                 "channel": channel, "horizon_seconds": seconds,
                                 "cumulative_MAE_C": float(np.mean(np.abs(error[available]))),
                                 "instant_abs_error_C": float(abs(error[nearest])),
                                 "bias_C": float(np.mean(error[available]))})
    return pd.DataFrame(rows)


def render_plots(values, horizons, out):
    """Render saved result tables without rerunning checkpoint inference."""
    arms = sorted(values.arm.unique())
    def styling(plt):
        colors = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)
        return {arm: colors[index % len(colors)] for index, arm in enumerate(arms)}
    def quality(plt):
        colors = styling(plt)
        figure, axes = plt.subplots(1, 3, figsize=(16, 5.5))
        for axis, channel in zip(axes, CHANNELS):
            subset = values[values.channel == channel]
            for arm, rows in subset.groupby("arm"):
                axis.scatter(rows.reference_residual_std_C, rows.model_MAE_C,
                             color=colors[arm], label=arm, alpha=.85)
            axis.set_xscale("symlog", linthresh=1)
            axis.set_xlabel("Training reference residual std (C)")
            axis.set_ylabel("Validation MAE (C)")
            axis.set_title(channel)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncol=7, fontsize=8,
                      bbox_to_anchor=(.5, .01), frameon=False)
        return figure
    save_plot(Path(out) / "c16_reference_quality.png", quality, rect=(0, .22, 1, 1))
    def horizon_plot(plt):
        colors = styling(plt)
        figure, axes = plt.subplots(1, 3, figsize=(16, 5.5))
        grouped = horizons.groupby(["arm", "channel", "horizon_seconds"]).cumulative_MAE_C.mean().reset_index()
        for axis, channel in zip(axes, CHANNELS):
            for arm, subset in grouped[grouped.channel == channel].groupby("arm"):
                axis.plot(subset.horizon_seconds / 3600, subset.cumulative_MAE_C,
                          marker="o", color=colors[arm], label=arm)
            axis.set_xlabel("Elapsed hours")
            axis.set_ylabel("Cumulative MAE (C)")
            axis.set_title(channel)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncol=7, fontsize=8,
                      bbox_to_anchor=(.5, .01), frameon=False)
        return figure
    save_plot(Path(out) / "c16_horizons.png", horizon_plot, rect=(0, .22, 1, 1))


def run_analysis(manifest, out, data_root=None, physics_cache_dir=None, **kwargs):
    from round10_refit import REFIT_SCOPE, write_refit_guide
    data_root, physics_cache_dir, _ = manifest_context(manifest, data_root, physics_cache_dir)
    records = load_manifest(manifest)
    bundles = load_val_bundles(manifest)
    spec = split_spec()
    frames = read_frames(sorted(set(spec["train"]) | set(spec["val"])), data_root, spec)
    training = {key: frames[key] for key in spec["train"]}
    physics = cache_for_keys(physics_cache_dir, list(training))
    masked = set(spec["tavg_mask"])
    reference_stats = reference_statistics(records, training, physics, masked)
    values = points(bundles, reference_stats, masked)
    horizons = horizon_statistics(bundles, frames, masked)
    heads = []
    for record in records:
        session = setup_run(record["run_dir"])
        selected = read_json(Path(record["run_dir"]) / "run_config.json")["data"].get("split_name", "round10_vs3_s42.json")
        own_spec = split_spec(Path(selected) if Path(selected).is_absolute() else Path(__file__).parent / "splits" / selected)
        for key in own_spec["val"]:
            if key not in frames:
                continue
            _, head = predict_frame(session, frames[key], key, capture_head=True)
            if head is None:
                continue
            for index, channel in enumerate(CHANNELS):
                if index >= head.shape[1] or channel == "T_avg" and key in masked:
                    continue
                values_head = head[:, index]
                heads.append({"arm": record["arm"], "seed": record["seed"], "file": key,
                              "channel": channel, "head_min": float(values_head.min()),
                              "head_max": float(values_head.max()),
                              "head_std": float(values_head.std()),
                              "head_abs_gt3_share": float(np.mean(np.abs(values_head) > 3)),
                              "head_units": "native fc output (z for reparameterized heads, scaled temperature for absolute heads)"})
    write_table(reference_stats, out, "c16_reference_statistics",
                "Statistics fitted on full s42 training data only; T_avg excludes118maskedfiles. Ratio spread is converted back to C. " + REFIT_SCOPE)
    write_table(values, out, "c16_validation_points",
                "Reference quality is descriptive. A large-residual reference helping the model falsifies the simple proposed reference-quality rule. " + REFIT_SCOPE)
    write_refit_guide(out)
    write_table(horizons, out, "c16_horizons",
                "Cumulative MAE from step1 through horizon; instant error at nearest elapsed time is reported separately.")
    write_table(pd.DataFrame(heads), out, "c16_head_ranges",
                "Native raw fc outputs captured during inference; z units only where a z-scored head is used.")
    render_plots(values, horizons, out)
    return {"reference_stats": reference_stats, "points": values, "horizons": horizons}


def main():
    args = parser(__doc__).parse_args()
    if not args.manifest:
        raise ValueError("--manifest is required")
    run_analysis(args.manifest, args.out, args.data_root, args.physics_cache_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

