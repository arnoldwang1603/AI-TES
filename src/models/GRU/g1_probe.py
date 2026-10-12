"""G1 constant-shift probe on validation-derived inlet shapes only.

Round 9's original scripts used sealed test shapes. Round 10 adapts their
constant-shift/h0 protocol to 14 deterministic validation shapes, seven starts
300:600:50; this change of probe population is explicitly disclosed.
s_hat is a least-squares slope on a 1-D line reference driven by the SAME
shifted inlet. This diagnostic makes no new recipe-selection decision.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from analysis_common import (CHANNELS, COLS, load_manifest, parser, predict_frame,
                             read_frames, save_plot, setup_run, split_spec, write_table, manifest_context)
from physics_reference import from_frame


def select_shapes(spec):
    """One per inlet family, plus second shapes for1/4/8/9: 14 total."""
    by_family = {}
    for key in sorted(spec["val"]):
        by_family.setdefault(key.split("/")[0], []).append(key)
    chosen = [values[0] for _, values in sorted(by_family.items())]
    for number in (1, 4, 8, 9):
        family = next(name for name in by_family if name.startswith("Case %d " % number))
        if len(by_family[family]) > 1:
            chosen.append(by_family[family][1])
    return sorted(chosen)


def run_analysis(manifest, out, data_root=None, physics_cache_dir=None, max_shapes=0, **kwargs):
    data_root, physics_cache_dir, _ = manifest_context(manifest, data_root, physics_cache_dir)
    records = load_manifest(manifest)
    allowed = {"B1_mask", "B1_orig", "B2_mask", "B2_orig", "B2_relabel", "B2_drop9", "B7", "B8"}
    records = [record for record in records if record["arm"] in allowed or record["arm"].startswith("Round9")]
    if not records:
        raise ValueError("G1 needs completed B1/B2/B7/B8 checkpoints")
    spec = split_spec()
    keys = select_shapes(spec)
    if max_shapes:
        keys = keys[:int(max_shapes)]
    frames = read_frames(keys, data_root, spec)
    masked = set(spec["tavg_mask"])
    rows = []
    shifted_physics = {}
    for record in records:
        session = setup_run(record["run_dir"])
        for key, original in frames.items():
            initial = float(original["Input Temperature (C)"].iloc[0])
            for start in range(300, 601, 50):
                shift = float(start) - initial
                shifted = original.copy()
                shifted[list(COLS) + ["Input Temperature (C)"]] += shift
                if shifted[list(COLS) + ["Input Temperature (C)"]].to_numpy().min() < 1.0:
                    rows.append({"arm": record["arm"], "seed": record["seed"], "shape": key,
                                 "T0": start, "status": "excluded: shifted temperature below1C"})
                    continue
                if (key, start) not in shifted_physics:
                    shifted_physics[(key, start)] = from_frame(shifted)
                physics = shifted_physics[(key, start)]
                pred = predict_frame(session, shifted, key)
                line = np.asarray(physics["T_avg_line_phys"])[1:]
                x = pred[:, 0] - line
                y = pred[:, 2] - line
                usable = np.isfinite(x) & np.isfinite(y)
                denominator = float(np.dot(x[usable], x[usable]))
                s_hat = float(np.dot(x[usable], y[usable]) / denominator) if denominator > 1e-12 else np.nan
                truth = original[list(COLS)].to_numpy(float)[1:]
                errors = pred - shift - truth
                row = {"arm": record["arm"], "seed": record["seed"], "shape": key,
                       "T0": start, "shift": shift, "status": "scored", "s_hat": s_hat,
                       "n_slope_steps": int(usable.sum()),
                       "physics_Tavg_bias_C": float(np.mean(pred[:, 2] - line)),
                       "physics_Tavg_MAE_C": float(np.mean(np.abs(pred[:, 2] - line)))}
                for index, channel in enumerate(CHANNELS):
                    row["shift_back_MAE_" + channel] = (np.nan if channel == "T_avg" and key in masked
                                                        else float(np.mean(np.abs(errors[:, index]))))
                rows.append(row)
        print("G1 complete %s seed%d" % (record["arm"], record["seed"]), flush=True)
    table = pd.DataFrame(rows)
    scored = table[table.status == "scored"]
    summaries = []
    for (arm, shape), subset in scored.groupby(["arm", "shape"]):
        curve = subset.groupby("T0").s_hat.mean()
        switch = curve[curve > 0.03]
        summaries.append({"arm": arm, "shape": shape,
                          "switch_T0_gt3pct": float(switch.index.min()) if len(switch) else np.nan,
                          "step_height": float(curve.max() - curve.min()),
                          "s_hat_low": float(curve.iloc[0]), "s_hat_high": float(curve.iloc[-1]),
                          "n_levels": len(curve)})
    summary = pd.DataFrame(summaries)
    note = __doc__ + ("\nPartial functional smoke:only%dvalidation shapes; no mechanism claim." % len(keys) if max_shapes else "")
    write_table(table, out, "g1_probe", note)
    write_table(summary, out, "g1_switch_summary",
                "Descriptive slopes; 3% is the existing protocol's switch readout. Missing low starts are disclosed, not extrapolated.")
    def chart(plt):
        figure, axis = plt.subplots(figsize=(9, 5))
        for arm, subset in scored.groupby("arm"):
            curve = subset.groupby("T0").s_hat.mean()
            axis.plot(curve.index, 100 * curve.values, marker="o", label=arm)
        axis.axhline(3, color="0.6", linestyle=":")
        axis.set_xlabel("Shifted inlet start (C)")
        axis.set_ylabel("Implied inner share (%)")
        axis.legend(fontsize=7)
        return figure
    save_plot(Path(out) / "g1_shift.png", chart)
    return {"probe": table, "summary": summary}


def main():
    ap = parser(__doc__)
    ap.add_argument("--max-shapes", type=int, default=0, help="functional smoke only; default usesall14shapes")
    args = ap.parse_args()
    if not args.manifest:
        raise ValueError("--manifest is required")
    run_analysis(args.manifest, args.out, args.data_root, args.physics_cache_dir, max_shapes=args.max_shapes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

