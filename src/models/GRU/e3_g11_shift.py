"""E3/G11 inference-only material shifts on 1-D-generated synthetic truth.

Truth is k(T)x0.8/1.2, not a new FE or measured material dataset. All three
prediction paths use the same synthetic truth and uniform t0 state. G11 fits
one diffusivity using only10/30/60minute prefixes, then scores the tail.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from analysis_common import (CHANNELS, COLS, load_manifest, parser, physics_array, cache_for_keys, read_json,
                             predict_frame, read_frames, save_plot, setup_run,
                             split_spec, write_table, manifest_context)
from physics_reference import from_frame, fit_prefix


def _score(prediction, truth, time, prefix_rows=1):
    result = {}
    for index, channel in enumerate(CHANNELS):
        difference = prediction - truth
        result[channel] = float(np.mean(np.abs(difference[prefix_rows - 1:, index])))
    return result


def run_analysis(manifest, out, data_root=None, physics_cache_dir=None, max_files=0, **kwargs):
    data_root, physics_cache_dir, _ = manifest_context(manifest, data_root, physics_cache_dir)
    records = load_manifest(manifest)
    direct = [record for record in records if record["arm"] in ("B2_mask", "C1")]
    hybrid = [record for record in records if record["arm"] == "C11"]
    if not direct or not hybrid:
        raise ValueError("E3/G11 requires C1/B2_mask and C11 checkpoints")
    spec = split_spec()
    keys = spec["val"][:int(max_files)] if max_files else spec["val"]
    frames = read_frames(keys, data_root, spec)
    cache_dir = Path(physics_cache_dir or "")
    params_path = cache_dir / "params.json"
    if not params_path.exists():
        params_path = cache_dir.parent / "params.json"
    cached_params = read_json(params_path)
    params = {"r_o": cached_params.get("r_o", 0.127), "n_layers": cached_params.get("n_layers", 60),
              "sub_steps": cached_params.get("sub_steps", 2)}
    cached = cache_for_keys(physics_cache_dir, keys)
    generated = {}
    fits = {}
    rows = []
    # CPU simulator computations are shared across seeds and checkpoint arms.
    for key, original in frames.items():
        nominal = cached[key]
        recomputed = from_frame(original, params=params)
        if np.max(np.abs(physics_array(nominal) - physics_array(recomputed))) > 1e-8:
            raise ValueError("portable solver disagrees with nominal cache:%s" % key)
        generated[(key, 1.0)] = nominal
        for scale in (0.8, 1.2):
            truth = from_frame(original, scale, params)
            generated[(key, scale)] = truth
            for minutes in (10, 30, 60):
                fit = fit_prefix(original, truth, minutes, params=params)
                fits[(key, scale, minutes)] = (fit, from_frame(original, fit["k_scale"], params))
            synthetic = original.copy()
            synthetic[list(COLS)] = physics_array(truth)
            times = synthetic["Time (s)"].to_numpy(float)[1:]
            ground = physics_array(truth)[1:]
            row = {"arm": "pure1d_nominal", "seed": -1, "file": key, "k_truth_scale": scale,
                   "prefix_minutes": 0, "k_fitted_scale": 1.0, "scope": "full"}
            row.update(_score(physics_array(nominal)[1:], ground, times))
            rows.append(row)
            for minutes in (10, 30, 60):
                fit, calibrated = fits[(key, scale, minutes)]
                row = {"arm": "pure1d_prefix_fit", "seed": -1, "file": key, "k_truth_scale": scale,
                       "prefix_minutes": minutes, "k_fitted_scale": fit["k_scale"], "scope": "unseen_tail",
                       "prefix_rows": fit["prefix_rows"]}
                row.update(_score(physics_array(calibrated)[1:], ground, times, fit["prefix_rows"]))
                rows.append(row)
                row = {"arm": "pure1d_nominal", "seed": -1, "file": key, "k_truth_scale": scale,
                       "prefix_minutes": minutes, "k_fitted_scale": 1.0, "scope": "unseen_tail",
                       "prefix_rows": fit["prefix_rows"]}
                row.update(_score(physics_array(nominal)[1:], ground, times, fit["prefix_rows"]))
                rows.append(row)
    for record in direct + hybrid:
        session = setup_run(record["run_dir"])
        for key, original in frames.items():
            for scale in (0.8, 1.2):
                truth = generated[(key, scale)]
                synthetic = original.copy()
                synthetic[list(COLS)] = physics_array(truth)
                ground = physics_array(truth)[1:]
                is_hybrid = record["arm"] == "C11"
                pred = predict_frame(session, synthetic, key,
                                     physics_override=truth if is_hybrid else None)
                row = {"arm": "C11_known_shifted_base" if is_hybrid else "direct_zero_shot",
                       "seed": record["seed"], "file": key, "k_truth_scale": scale,
                       "prefix_minutes": 0, "k_fitted_scale": scale if is_hybrid else np.nan,
                       "scope": "full"}
                row.update(_score(pred, ground, synthetic["Time (s)"].to_numpy(float)[1:]))
                rows.append(row)
                if not is_hybrid:
                    for minutes in (10, 30, 60):
                        fit, _ = fits[(key, scale, minutes)]
                        row = {"arm": "direct_zero_shot", "seed": record["seed"], "file": key,
                               "k_truth_scale": scale, "prefix_minutes": minutes,
                               "k_fitted_scale": np.nan, "scope": "unseen_tail",
                               "prefix_rows": fit["prefix_rows"]}
                        row.update(_score(pred, ground, synthetic["Time (s)"].to_numpy(float)[1:], fit["prefix_rows"]))
                        rows.append(row)
                if is_hybrid:
                    for minutes in (10, 30, 60):
                        fit, calibrated = fits[(key, scale, minutes)]
                        pred = predict_frame(session, synthetic, key, physics_override=calibrated)
                        row = {"arm": "C11_prefix_fit", "seed": record["seed"], "file": key,
                               "k_truth_scale": scale, "prefix_minutes": minutes,
                               "k_fitted_scale": fit["k_scale"], "prefix_rows": fit["prefix_rows"],
                               "scope": "unseen_tail"}
                        row.update(_score(pred, ground, synthetic["Time (s)"].to_numpy(float)[1:], fit["prefix_rows"]))
                        rows.append(row)
        print("E3/G11 complete %s seed%d" % (record["arm"], record["seed"]), flush=True)
    table = pd.DataFrame(rows)
    summary = table.groupby(["arm", "k_truth_scale", "prefix_minutes", "scope"])[list(CHANNELS)].mean().reset_index()
    note = __doc__ + ("\nPartial functional smoke:only%dvalidation files; no transfer claim." % len(keys) if max_files else "")
    write_table(table, out, "e3_g11_per_file", note)
    write_table(summary, out, "e3_g11_summary",
                "Synthetic1-D material shifts only. Direct zero-shot includes nominal1-D-vs-FE mismatch; absolute errors are not FE transfer evidence.")
    render_plot(summary, out)
    return {"scores": table, "summary": summary}


def render_plot(summary, out):
    """Render the saved synthetic-shift summary without model inference."""
    def chart(plt):
        figure, axes = plt.subplots(1, 3, figsize=(16, 5.5))
        for axis, channel in zip(axes, CHANNELS):
            for (arm, minutes), subset in summary.groupby(["arm", "prefix_minutes"]):
                axis.plot(subset.k_truth_scale, subset[channel], marker="o", label="%s/%dmin" % (arm, minutes))
            axis.set_xlabel("Synthetic k multiplier")
            axis.set_ylabel("MAE (C)")
            axis.set_title(channel)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncol=4, fontsize=7,
                      bbox_to_anchor=(.5, .01), frameon=False)
        return figure
    save_plot(Path(out) / "e3_g11_shift.png", chart, rect=(0, .24, 1, 1))


def main():
    ap = parser(__doc__)
    ap.add_argument("--max-files", type=int, default=0, help="functional smoke only; default usesall20validationfiles")
    args = ap.parse_args()
    if not args.manifest:
        raise ValueError("--manifest is required")
    run_analysis(args.manifest, args.out, args.data_root, args.physics_cache_dir, max_files=args.max_files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

