"""Validation-side MSE error budgets and separate label-floor yardsticks.

MAE terms cannot be added. The exact MSE identity includes the convention/
residual cross term. F11 calls analyze_bundles on preloaded test frames only.
A 1-D discrepancy is a yardstick or upper bound, never a measured noise floor.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
from analysis_common import (CHANNELS, COLS, bundle_arrays, cache_for_keys,
                             convention_map, load_val_bundles, normalized_key,
                             parser, physics_array, read_frames, save_plot,
                             split_spec, write_table, manifest_context)

# Existing eval_checkpoint's registered switched/both-phase stratum.
SWITCHED = set(range(32, 41)) | {50, 51, 52, 53, 58} | set(range(59, 66))


def analyze_bundles(bundles, frames, physics, conventions=None, which_set="val", masked_keys=()):
    """Pure preloaded-array analysis, usable by the sole F11 test reader."""
    conventions = convention_map(conventions)
    masked_keys = set(masked_keys)
    grouped = {}
    for bundle in bundles:
        ids, actuals, predictions = bundle_arrays(bundle)
        for key, truth, pred in zip(ids, actuals, predictions):
            key = normalized_key(key)
            grouped.setdefault((bundle["arm"], key), []).append((bundle["seed"], truth, pred))
    budget, floors, oracle = [], [], []
    for key, frame in frames.items():
        key = normalized_key(key)
        truth = frame[list(COLS)].to_numpy(float)[1:]
        phys = physics_array(physics[key])[1:]
        metadata = conventions.get(key, {})
        label = str(metadata.get("label", "unknown"))
        inner_share = float(metadata.get("inner_share_pct", 0) or 0) / 100.0
        for index, channel in enumerate(CHANNELS):
            if channel == "T_avg" and key in masked_keys:
                continue
            error = phys[:, index] - truth[:, index]
            floor_row = {"set": which_set, "file": key, "channel": channel,
                         "label": label, "strict_line_average": label == "line average",
                         "physics_yardstick_MAE_C": float(np.mean(np.abs(error))),
                         "physics_yardstick_RMSE_C": float(np.sqrt(np.mean(error ** 2))),
                         "label_upper_bound_RMSE_C": (metadata.get("tavg_rmse_best_mix_C")
                                                     if channel == "T_avg" else np.nan),
                         "floor_interpretation": "1-D model discrepancy; no independent FE repeat noise measured"}
            floors.append(floor_row)
        match = re.search(r"\((\d+)\)", key)
        case_id = int(match.group(1)) if match else None
        inlet = frame["Input Temperature (C)"].to_numpy(float)
        if case_id is not None and which_set == "test":
            stratum = "both_phase21" if case_id in SWITCHED else "other49"
        else:
            peak = int(np.argmax(inlet))
            stratum = "both_phase" if 0 < peak < len(inlet) - 1 and inlet[-1] < inlet[peak] else "other"
        clipped = np.clip(truth[:, 2], np.minimum(truth[:, 0], truth[:, 1]), np.maximum(truth[:, 0], truth[:, 1]))
        if key not in masked_keys:
            oracle.append({"set": which_set, "file": key, "case_id": case_id, "stratum": stratum,
                           "bracket_clip_oracle_MAE_C": float(np.mean(np.abs(clipped - truth[:, 2]))),
                           "bracket_violation_share": float(np.mean(clipped != truth[:, 2])),
                           "interpretation": "minimum error imposed by clipping true T_avg between true walls"})
    for (arm, key), items in sorted(grouped.items()):
        truth = items[0][1]
        predictions = np.stack([item[2] for item in items])
        if any(not np.allclose(item[1], truth, atol=1e-5, rtol=0) for item in items):
            raise ValueError("truth mismatch across seeds: %s/%s" % (arm, key))
        mean = predictions.mean(axis=0)
        bias = mean - truth
        deviations = predictions - mean
        physics_line = np.asarray(physics[key]["T_avg_line_phys"], dtype=float)[1:]
        physics_inner = np.asarray(physics[key]["T_inner_phys"], dtype=float)[1:]
        delta = physics_inner - physics_line
        metadata = conventions.get(key, {})
        eligible = metadata.get("label") in ("inner-weighted", "borderline")
        # Same pre-registered convention table selects the fitting scope;
        # no new temperature threshold or deciding rule is introduced.
        fit_scope = "Case9 convention-affected pure discharge" if eligible and key.startswith("Case 9 ") else "convention-affected file" if eligible else "none"
        for index, channel in enumerate(CHANNELS):
            if channel == "T_avg" and key in masked_keys:
                continue
            beta = float(np.dot(delta, bias[:, index]) / np.dot(delta, delta)) if channel == "T_avg" and eligible and np.dot(delta, delta) > 1e-12 else 0.0
            convention = beta * delta
            residual = bias[:, index] - convention
            convention_mse = float(np.mean(convention ** 2))
            residual_mse = float(np.mean(residual ** 2))
            cross = float(2 * np.mean(convention * residual))
            seed_mse = float(np.mean(deviations[:, :, index] ** 2))
            total = float(np.mean((predictions[:, :, index] - truth[:, index]) ** 2))
            decomposition = convention_mse + residual_mse + cross + seed_mse
            if abs(total - decomposition) > 1e-8 * max(1, total):
                raise AssertionError("MSE identity failed")
            budget.append({"set": which_set, "arm": arm, "file": key, "channel": channel,
                           "n_seeds": len(items), "beta": beta, "beta_fit_scope": fit_scope,
                           "convention_MSE_C2": convention_mse, "residual_bias_MSE_C2": residual_mse,
                           "cross_term_C2": cross, "seed_variance_C2": seed_mse, "total_MSE_C2": total,
                           "identity_error_C2": total - decomposition,
                           "ensemble_MAE_C": float(np.mean(np.abs(bias[:, index]))),
                           "single_model_MAE_C": float(np.mean(np.abs(predictions[:, :, index] - truth[:, index]))),
                           "interpretation": "descriptive MSE decomposition; MAE is reported separately and is not additive"})
    return {"budget": pd.DataFrame(budget), "floors": pd.DataFrame(floors), "oracle": pd.DataFrame(oracle)}


def run_analysis(manifest, out, data_root=None, physics_cache_dir=None, **kwargs):
    data_root, physics_cache_dir, _ = manifest_context(manifest, data_root, physics_cache_dir)
    bundles = load_val_bundles(manifest)
    spec = split_spec()
    keys = sorted({normalized_key(key) for bundle in bundles for key in bundle["case_ids"]})
    frames = read_frames(keys, data_root, spec)
    physics = cache_for_keys(physics_cache_dir, keys)
    result = analyze_bundles(bundles, frames, physics, which_set="val", masked_keys=spec["tavg_mask"])
    for name, table in result.items():
        write_table(table, out, "error_" + name, __doc__)
    if len(result["budget"]):
        summary = result["budget"].groupby(["arm", "channel"])[
            ["convention_MSE_C2", "residual_bias_MSE_C2", "cross_term_C2", "seed_variance_C2", "total_MSE_C2",
             "ensemble_MAE_C", "single_model_MAE_C"]].mean().reset_index()
        write_table(summary, out, "error_budget_summary")
        def chart(plt):
            figure, axes = plt.subplots(1, 3, figsize=(13, 4))
            for axis, channel in zip(axes, CHANNELS):
                subset = summary[summary.channel == channel].set_index("arm")
                subset[["convention_MSE_C2", "residual_bias_MSE_C2", "seed_variance_C2"]].plot.bar(stacked=True, ax=axis)
                axis.set_ylabel("MSE terms (C^2); cross term in table")
                axis.set_title(channel)
                axis.tick_params(axis="x", rotation=70, labelsize=7)
                axis.legend(fontsize=6)
            return figure
        save_plot(Path(out) / "error_budget.png", chart)
    return result


def main():
    args = parser(__doc__).parse_args()
    if not args.manifest:
        raise ValueError("--manifest is required")
    run_analysis(args.manifest, args.out, args.data_root, args.physics_cache_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

