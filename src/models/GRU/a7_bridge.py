"""A7 Round9->Round10 validation bridge (60 preregistered checkpoints).

Use the12 unmasked common validation files, report Case9 walls separately,
and list new s42 validation files as in-sample for Round9. No test arrays or
test CSVs are opened. Legacy constants come from each checkpoint's own split.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from analysis_common import (BASE_COLS, CHANNELS, COLS, cache_for_keys, normalized_key, load_manifest,
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
        seeds = [int(record["seed"]) for record in result if record["arm"] == arm]
        if len(set(seeds)) != expected:
            raise ValueError("A7 duplicate seed in archived family " + arm)
    return result


def validate_bridge_records(records, seeds):
    """The machine comparison is the complete registered S10 panel."""
    seeds = list(map(int, seeds))
    if len(seeds) != 10 or len(set(seeds)) != 10:
        raise ValueError("A7 machine bridge requires exactly 10 unique registered S10 seeds")
    for arm in ("RETRAIN", "r9_direct160"):
        actual = [int(record["seed"]) for record in records if record["arm"] == arm]
        if len(actual) != len(set(actual)) or set(actual) != set(seeds):
            raise ValueError("A7 machine bridge seed membership mismatch for " + arm)
    return sorted(seeds)


def machine_bridge(table, seeds, shared_keys):
    """Compare every S10 pair on the same 12 files; never drop incomplete pairs."""
    from paper_stats import paired_stats, signflip_metadata
    seeds = sorted(map(int, seeds))
    keys = set(shared_keys)
    if len(seeds) != 10 or len(set(seeds)) != 10 or len(keys) != 12:
        raise ValueError("A7 machine bridge requires registered S10 and 12 shared validation files")
    arms = ("RETRAIN", "r9_direct160")
    selected = table[(table.scope == "shared12") & table.arm.isin(arms)].copy()
    columns = ["arm", "seed", "channel", "file"]
    expected = {(arm, seed, channel, key) for arm in arms for seed in seeds for channel in CHANNELS for key in keys}
    actual = set(selected[columns].itertuples(index=False, name=None))
    if selected.duplicated(columns).any() or actual != expected or not np.isfinite(selected.MAE.to_numpy(float)).all():
        raise ValueError("A7 machine bridge incomplete/duplicate seed-file membership or nonfinite score")
    levels = selected.groupby(["arm", "seed", "channel"]).MAE.mean().unstack("arm")
    output = []
    for channel in sorted(CHANNELS):
        paired = levels.xs(channel, level="channel").loc[seeds]
        stats = paired_stats((paired.RETRAIN - paired.r9_direct160).to_numpy(), np.random.default_rng(20261012))
        output.append(dict(channel=channel, comparison="RETRAIN-Round9 onshared12", seed_group="S10",
                           seed_ids=" ".join(map(str, seeds)), **stats, **signflip_metadata(stats["n"])))
    return pd.DataFrame(output)


def legacy_validation_scalar(session, frames):
    """Reproduce the archived, unmasked weighted scaled validation MAE."""
    config = session["run_config"]
    weights = np.asarray(config["loss_weights"], dtype=float)
    scaler = session["scaler"]
    # Both own-split fitting and sidecar restoration use BASE_COLS. Older
    # sklearn scalers may omit feature_names_in_; this optional metadata
    # must not block replay or change the frozen scaler/model constants.
    scales = np.asarray(scaler.scale_, dtype=float)
    if (scales.shape != (len(BASE_COLS),)
            or not np.isfinite(scales).all() or np.any(scales <= 0)
            or getattr(scaler, "n_features_in_", len(BASE_COLS)) != len(BASE_COLS)):
        raise ValueError("A7 scaler must have five finite positive scales in frozen BASE_COLS order")
    if hasattr(scaler, "feature_names_in_"):
        names = np.asarray(scaler.feature_names_in_, dtype=object)
        if names.shape != (len(BASE_COLS),) or names.tolist() != list(BASE_COLS):
            raise ValueError("A7 scaler feature names differ from frozen BASE_COLS order")
    scales = scales[[BASE_COLS.index(column) for column in COLS]]
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
    protocol = read_json(Path(__file__).parent / "prereg/round10_v4.json")
    records = discover_runs(round9_root)
    if manifest:
        records.extend(record for record in load_manifest(manifest) if record["arm"] == "RETRAIN")
    bridge_seeds = validate_bridge_records(records, protocol["seeds"]["S10"])
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
    machine = machine_bridge(table, bridge_seeds, shared)
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

