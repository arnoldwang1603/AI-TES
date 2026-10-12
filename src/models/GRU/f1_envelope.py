"""F1 leave-one-family envelope and full-train constants provenance.

Pipeline calls freeze_full_train_constants before any fold is trained.
Envelope uses five paired seeds and per-family in-distribution validation
readouts, not in-sample training predictions. No significance claim at n=5.
"""
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from analysis_common import (CHANNELS, bundle_arrays, load_val_bundles,
                             normalized_key, parser, read_json, save_plot,
                             sha256, split_spec, write_json, write_table)


def freeze_full_train_constants(source_run, output_path, allow_smoke=False):
    """Copy a validated C1 full284-fit sidecar, shared by all10F1 folds."""
    source_run = Path(source_run)
    config = read_json(source_run / "run_config.json")
    epochs = int(config.get("latest_params", {}).get("max_epochs", 2000))
    if epochs <= 2 and not allow_smoke:
        raise ValueError("smoke constants cannot seed a formal F1 run")
    data = config.get("data", {})
    own_split = data.get("split_name")
    if own_split != "round10_vs3_s42.json" or config.get("tavg_mask") != "split":
        raise ValueError("F1 constants source must be full s42 masked C1")
    if config.get("other_ch_mode", "abs") != "abs" or config.get("phys_mode", "") :
        raise ValueError("F1 constants source must be direct absolute C1")
    variant = config["variants"][0]
    path = source_run / "checkpoints" / ("best_gru_variant_%s.constants.json" % variant)
    sidecar = read_json(path)
    spec = split_spec()
    if sidecar.get("split", {}).get("n_train", len(spec["train"])) != len(spec["train"]):
        raise ValueError("F1 constants not fit on284 training files")
    if not sidecar.get("constants", {}).get("tinner_anchor"):
        raise ValueError("F1 source has no T_inner anchor constants")
    sidecar["frozen_full_train"] = {"split": "round10_vs3_s42.json", "train": spec["train"],
                                   "source_run": str(source_run.resolve()), "source_sha256": sha256(path),
                                   "source_max_epochs": config.get("latest_params", {}).get("max_epochs"),
                                   "scope": "scaler+anchor constants frozen onfull284; optimisation excludesheldoutfamily"}
    output_path = Path(output_path)
    if output_path.exists():
        current = read_json(output_path)
        if (current.get("scaler") != sidecar.get("scaler") or
                current.get("constants") != sidecar.get("constants") or
                current.get("loss_weights_realised") != sidecar.get("loss_weights_realised")):
            raise ValueError("different scaler/model constants already frozen at%s" % output_path)
        # Sidecar timestamps, run metadata and checkpoints differ across
        # seeds/epochs; the frozen data-derived constants must remain exact.
        # Never overwrite the shipped resource's provenance or hash.
        return str(output_path.resolve())
    write_json(output_path, sidecar)
    return str(output_path.resolve())


def run_analysis(manifest, out, data_root=None, physics_cache_dir=None, **kwargs):
    bundles = load_val_bundles(manifest)
    spec = split_spec()
    masked = set(spec["tavg_mask"])
    rows = []
    for bundle in bundles:
        config = bundle["config"]
        split_name = config.get("data", {}).get("split_name", "")
        is_fold = "loto_vs3_fold" in split_name or str(bundle["arm"]).startswith("F1")
        if not is_fold and bundle["arm"] not in ("B2_mask", "C1"):
            continue
        heldout = None
        if is_fold:
            fold_path = Path(split_name) if Path(split_name).is_absolute() else Path(__file__).parent / "splits" / split_name
            heldout = split_spec(fold_path).get("heldout_family")
        ids, actuals, predictions = bundle_arrays(bundle)
        for key, truth, pred in zip(ids, actuals, predictions):
            key = normalized_key(key)
            family = key.split("/")[0]
            if is_fold and family != heldout:
                continue
            for index, channel in enumerate(CHANNELS):
                if channel == "T_avg" and key in masked:
                    continue
                rows.append({"design": "LOTO" if is_fold else "ID", "seed": bundle["seed"],
                             "family": family, "channel": channel, "file": key,
                             "MAE": float(np.mean(np.abs(pred[:, index] - truth[:, index])))})
    scores = pd.DataFrame(rows)
    if not len(scores):
        raise ValueError("F1 needs fold predictions and B2_mask in-distribution predictions")
    means = scores.groupby(["design", "seed", "family", "channel"]).MAE.mean().unstack("design").reset_index()
    if "LOTO" not in means or "ID" not in means:
        raise ValueError("F1 missing LOTO or ID paired controls")
    means["ratio_LOTO_ID"] = means["LOTO"] / means["ID"].replace(0, np.nan)
    envelope = []
    for family, subset in means.groupby("family"):
        selected = subset[subset.channel.isin(("T_outer", "T_avg"))]
        paired = selected.groupby("seed")[["LOTO", "ID"]].mean().dropna()
        if not len(paired):
            continue
        ratios = paired.LOTO / paired.ID.replace(0, np.nan)
        envelope.append({"family": family, "n_seeds": len(paired),
                         "LOTO_ToTa": float(paired.LOTO.mean()), "ID_ToTa": float(paired.ID.mean()),
                         "ratio_LOTO_ID": float(ratios.mean()), "within2x": bool(ratios.mean() <= 2),
                         "channels": ",".join(sorted(set(selected.channel))),
                         "reference": "paired in-distribution validation files; Case9 T_avg masked"})
    envelope = pd.DataFrame(envelope)
    write_table(scores, out, "f1_per_file")
    write_table(means, out, "f1_channel_ratios", "No significance claims atfive seeds; ratios are descriptive.")
    write_table(envelope, out, "f1_envelope",
                "Envelope statement:8 of10families within2x. ToTa available channels; Case9 usesT_outer only.")
    passing = int(envelope.within2x.sum())
    complete = len(envelope) == 10 and bool((envelope.n_seeds == 5).all())
    write_json(Path(out) / "f1_envelope_statement.json",
               {"n_families": len(envelope), "n_within2x": passing,
                "criterion": "8 of10 families within2x", "complete_five_seed_protocol": complete,
                "passes": complete and passing >= 8,
                "meaning": "descriptive family envelope; no n=5 significance claim"})
    def chart(plt):
        figure, axis = plt.subplots(figsize=(11, 5))
        axis.bar(envelope.family, envelope.ratio_LOTO_ID)
        axis.axhline(2, color="black", linestyle="--")
        axis.set_ylabel("LOTO/ID error ratio")
        axis.tick_params(axis="x", rotation=60, labelsize=7)
        axis.set_title("%d/%d families within2x" % (passing, len(envelope)))
        return figure
    save_plot(Path(out) / "f1_envelope.png", chart)
    return {"scores": scores, "ratios": means, "envelope": envelope}


def main():
    ap = parser(__doc__)
    ap.add_argument("--freeze-constants-from")
    args = ap.parse_args()
    if args.freeze_constants_from:
        freeze_full_train_constants(args.freeze_constants_from, Path(args.out) / "full_train.constants.json")
    if args.manifest:
        run_analysis(args.manifest, args.out, args.data_root, args.physics_cache_dir)
    if not args.manifest and not args.freeze_constants_from:
        raise ValueError("--manifest or --freeze-constants-from is required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

