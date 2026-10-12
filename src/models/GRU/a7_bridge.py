"""A7 Round9->Round10 validation bridge (60 preregistered checkpoints).

Use the12 unmasked common validation files, report Case9 walls separately,
and list new s42 validation files as in-sample for Round9. No test arrays or
test CSVs are opened. Legacy constants come from each checkpoint's own split.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from analysis_common import (CHANNELS, COLS, cache_for_keys, normalized_key, load_manifest,
                             parser, predict_frame, read_frames, read_json,
                             setup_run, split_spec, write_table, manifest_context)

FAMILIES = (
    ("r9_direct160", "2026-08-06_forward_direct_W10_Tin-anchor_l1_w1-6-3_oth-abs_h160x1dp0.2_ep2000_vs2_seed*", 10),
    ("r9_direct128", "2026-08-06_forward_direct_W10_Tin-anchor_l1_w1-6-3_oth-abs_h128x1dp0.2_ep2000_vs2_seed*", 10),
    ("r9_direct128x2", "2026-08-06_forward_direct_W10_Tin-anchor_l1_w1-6-3_oth-abs_h128x2dp0.3_ep2000_vs2_seed*", 20),
    ("r9_fl40_lg", "2026-08-06_forward_direct_W10_Tin-anchor_l1_w1-6-3_oth-pos_head_h128x2dp0.3_ep2000_fl40_lg_vs2_seed*", 20),
)


def discover_runs(round9_root):
    result = []
    for arm, pattern, expected in FAMILIES:
        paths = sorted(Path(round9_root).glob(pattern))
        paths = [path for path in paths if (path / "run_config.json").is_file()]
        if len(paths) != expected:
            raise ValueError("A7 %s expected%d runs, found%d under%s" % (arm, expected, len(paths), round9_root))
        for path in paths:
            config = read_json(path / "run_config.json")
            result.append({"arm": arm, "seed": config["seed"], "run_dir": str(path)})
    return result


def legacy_validation_scalar(session, frames):
    """Reproduce the archived, unmasked weighted scaled validation MAE."""
    config = session["run_config"]
    weights = np.asarray(config["loss_weights"], dtype=float)
    scales = np.asarray([session["scaler"].scale_[list(session["scaler"].feature_names_in_).index(column)]
                         for column in COLS], dtype=float)
    errors = []
    for key, frame in frames.items():
        prediction = predict_frame(session, frame, key)
        truth = frame[list(COLS)].to_numpy(float)[1:]
        errors.append(float(np.mean(np.abs(prediction - truth) * scales * weights)))
    return float(np.mean(errors))


def run_analysis(manifest, out, data_root=None, physics_cache_dir=None, round9_root=None, **kwargs):
    data_root, physics_cache_dir, round9_root = manifest_context(manifest, data_root, physics_cache_dir, round9_root)
    if not round9_root:
        raise ValueError("A7 requires --round9-root containing the60 selected archived checkpoints")
    spec = split_spec()
    legacy = split_spec(Path(__file__).parent / "splits/stratified_v2_s42.json")
    shared = sorted((set(spec["val"]) & set(legacy["val"])) - set(spec["tavg_mask"]))
    if len(shared) != 12:
        raise ValueError("A7 shared unmasked validation set is%d, expected12" % len(shared))
    walls = sorted(key for key in set(spec["val"]) & set(legacy["val"]) if key.startswith("Case 9 "))
    changed = sorted(set(spec["val"]) - set(legacy["val"]))
    frames = read_frames(sorted(set(shared) | set(walls) | set(changed)), data_root, spec)
    legacy_frames = read_frames(legacy["val"], data_root, legacy)
    rows = []
    sanity_rows = []
    records = discover_runs(round9_root)
    if manifest:
        records.extend(record for record in load_manifest(manifest) if record["arm"] == "RETRAIN")
    for record in records:
        session = setup_run(record["run_dir"])
        scalar = legacy_validation_scalar(session, legacy_frames)
        meta = read_json(Path(record["run_dir"]) / "variants" / session["variant"] / "meta.json")
        archived = float(meta["Validation_MAE"])
        delta = scalar - archived
        # H9 deliberately changes the very last T_inner step; the existing
        # bridge rule asks for reproduction to3decimalplaces, not bitwise.
        passed = abs(delta) < 0.001
        sanity_rows.append({"arm": record["arm"], "seed": record["seed"],
                            "recomputed_vs2_weighted_scaled_MAE": scalar,
                            "archived_Validation_MAE": archived, "delta": delta,
                            "tf32": session["run_config"].get("tf32", "legacy"),
                            "pass_3decimal_tolerance": passed,
                            "note": "H9 last-step correction applied; no test statistic used"})
        if not passed:
            raise ValueError("A7 legacy validation scalar mismatch:%s seed%d delta%.6g" % (record["arm"], record["seed"], delta))
        for key, frame in frames.items():
            predictions = predict_frame(session, frame, key)
            truth = frame[list(COLS)].to_numpy(float)[1:]
            scope = "shared12" if key in shared else "Case9_walls" if key in walls else "new_vs3_in_sample_Round9"
            for index, channel in enumerate(CHANNELS):
                if channel == "T_avg" and key in spec["tavg_mask"]:
                    continue
                rows.append({"arm": record["arm"], "seed": record["seed"], "file": key,
                             "channel": channel, "scope": scope,
                             "MAE": float(np.mean(np.abs(predictions[:, index] - truth[:, index]))),
                             "constants_source": session["constants_source"],
                             "tf32": session["run_config"].get("tf32", "legacy"), "note": "Round10H9 last-step correction applied"})
        print("A7 complete %s seed%d" % (record["arm"], record["seed"]), flush=True)
    table = pd.DataFrame(rows)
    summary = table.groupby(["arm", "channel", "scope"]).MAE.agg(["mean", "std", "count"]).reset_index()
    write_table(table, out, "a7_bridge_per_file", __doc__)
    write_table(summary, out, "a7_bridge_summary",
                "R9 one-layer dp0.2/dp0.3 twins are not both used. Newvalidation files were trained on in R9 and must not support generalisation.")
    sanity = pd.DataFrame(sanity_rows)
    write_table(sanity, out, "a7_vs2_sanity", "Own20file validation scalar reproduced undercheckpoint-ownTF32 (legacyforRound9,0forRETRAIN); intentionalH9 last-step change disclosed.")
    machine = []
    shared_scores = table[table.scope == "shared12"].groupby(["arm", "seed", "channel"]).MAE.mean().unstack("arm")
    if "RETRAIN" in shared_scores and "r9_direct160" in shared_scores:
        from paper_stats import paired_stats
        for channel, subset in shared_scores.groupby(level="channel"):
            paired = subset[["RETRAIN", "r9_direct160"]].dropna()
            if len(paired) < 3:
                raise ValueError("A7 machine bridge needs at least3 paired seeds")
            stats = paired_stats((paired.RETRAIN - paired.r9_direct160).to_numpy(), np.random.default_rng(20261012))
            machine.append(dict(channel=channel, comparison="RETRAIN-Round9 onshared12", **stats))
    machine = pd.DataFrame(machine)
    write_table(machine, out, "a7_machine_bridge", "Paired machine difference onidentical12validationfiles; nevercompareoldTestMAEwithnewValMAE.")
    return {"scores": table, "summary": summary, "shared_keys": shared, "machine_bridge": machine, "vs2_sanity": sanity}


def main():
    ap = parser(__doc__)
    ap.add_argument("--round9-root")
    args = ap.parse_args()
    run_analysis(args.manifest, args.out, args.data_root, args.physics_cache_dir, round9_root=args.round9_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

