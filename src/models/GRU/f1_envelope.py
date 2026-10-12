"""F1 v4: strict held-out-family scoring for direct and the frozen recipe.

Network fitting and constants use fold training; checkpoint selection uses
inner validation. The separate held-out family is scored after training.
"""
from pathlib import Path
import json
import re
import tempfile
import numpy as np
import pandas as pd
from analysis_common import (CHANNELS, bundle_arrays, load_manifest, manifest_context,
                             normalized_key, parser, read_json, save_plot,
                             sha256, split_spec, write_json, write_table)


def freeze_full_train_constants(source_run, output_path, allow_smoke=False):
    """Historical v3 packaging helper; v4 folds reject this full-train fit."""
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


F1_STATEMENT = ("Held-out-family error with inner-validation checkpoint selection and "
                "fold-train-only fitted statistics; ratios are descriptive, paired ten seeds.")


def _fold_model(arm):
    match = re.fullmatch(r"F1_(?:(.+)_)?(\d+)", str(arm))
    return (match.group(1) or "direct", int(match.group(2))) if match else None


def score_fold_holdouts(records, out=None):
    """Score completed folds once; retain a checkpoint/sidecar/split receipt.

    With out=None, files live in each run's variant directory. Pipeline calls
    this after all fold training. A matching receipt permits safe resume.
    """
    import eval_checkpoint
    from paper_report import atomic_json
    scored = []
    for record in records:
        if not _fold_model(record["arm"]):
            continue
        run = Path(record["run_dir"])
        config = read_json(run / "run_config.json")
        variant = config["variants"][0]
        if not (run / "variants" / variant / "done.flag").is_file():
            raise ValueError("F1 holdout scoring requires completed training: " + str(run))
        split = Path(config["data"]["split_name"])
        split = split if split.is_absolute() else Path(__file__).parent / "splits" / split
        spec = split_spec(split)
        if not spec.get("holdout"):
            raise ValueError("F1 v4 scorer refuses a legacy validation-as-holdout fold")
        dest = (Path(out) / record["arm"] / ("seed%d" % record["seed"])) if out else run / "variants" / variant
        receipt = dest / "f1_holdout_score.json"
        signature = {"checkpoint_sha256": sha256(eval_checkpoint.checkpoint_path(str(run), variant, "best")),
                     "run_config_sha256": sha256(run / "run_config.json"),
                     "constants_sha256": sha256(eval_checkpoint.constants_sidecar_path(str(run), variant)),
                     "split_sha256": sha256(split), "arm": record["arm"], "seed": int(record["seed"]),
                     "scoring_scope": F1_STATEMENT}
        artifacts = [dest / "predictions_holdout.npz", dest / "summary_errors_holdout.csv"]
        if receipt.exists():
            saved = read_json(receipt)
            if saved.get("signature") != signature:
                raise ValueError("F1 holdout inputs changed after scoring: " + str(run))
            if not all(path.is_file() and saved.get("artifacts", {}).get(path.name) == sha256(path) for path in artifacts):
                raise ValueError("F1 holdout scoring artifacts are missing or changed: " + str(run))
            result = saved["result"]
        else:
            result = eval_checkpoint.evaluate_one(str(run), "best", which_set="holdout", out_dir=str(dest))
            atomic_json(str(receipt), {"signature": signature, "result": result,
                                      "artifacts": {path.name: sha256(path) for path in artifacts}})
        scored.append(dict(record, holdout_dir=str(dest), holdout_result=result))
    return scored


def check_holdout_roundtrip(run_dir, out):
    """Smoke helper: restore the same fold twice and compare all predictions."""
    import eval_checkpoint
    parent = Path(out)
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="holdout_roundtrip_", dir=str(parent)) as temporary:
        first, second = Path(temporary) / "first", Path(temporary) / "second"
        a = eval_checkpoint.evaluate_one(str(run_dir), "best", "holdout", str(first))
        b = eval_checkpoint.evaluate_one(str(run_dir), "best", "holdout", str(second))
        maximum = 0.0
        with np.load(first / "predictions_holdout.npz", allow_pickle=True) as xa, np.load(second / "predictions_holdout.npz", allow_pickle=True) as xb:
            if list(xa["case_ids"]) != list(xb["case_ids"]):
                raise ValueError("F1 holdout round-trip changed case IDs")
            for ta, tb, pa, pb in zip(xa["inv_actuals"], xb["inv_actuals"], xa["inv_preds"], xb["inv_preds"]):
                if not all(np.isfinite(np.asarray(value, float)).all() for value in (ta, tb, pa, pb)):
                    raise ValueError("F1 holdout round-trip contains nonfinite truth or predictions")
                if not np.array_equal(np.asarray(ta, float), np.asarray(tb, float)):
                    raise ValueError("F1 holdout round-trip changed truth")
                maximum = max(maximum, float(np.max(np.abs(np.asarray(pa, float) - np.asarray(pb, float)))))
        if a["n_files"] != b["n_files"] or maximum > 1e-6:
            raise ValueError("F1 holdout prediction round-trip exceeds 1e-6 C")
        return {"n_files": a["n_files"], "max_prediction_delta_C": maximum, "test_access": False}


def _prediction_bundle(record, which):
    variant = record["config"]["variants"][0]
    directory = Path(record.get("holdout_dir") or Path(record["run_dir"]) / "variants" / variant)
    with np.load(directory / ("predictions_%s.npz" % which), allow_pickle=True) as archive:
        ids = [str(key) for key in archive["case_ids"]]
        if any("test_70_cases" in key.lower() for key in ids):
            raise ValueError("F1 analysis refuses sealed Test70 predictions")
        return dict(record, case_ids=ids, actuals=list(archive["inv_actuals"]), preds=list(archive["inv_preds"]))


def analyze_bundles(bundles, frozen_recipe, seeds, spec=None):
    """Pure aggregation with explicit model identity and exact seed coverage."""
    spec = spec or split_spec()
    seeds = list(map(int, seeds))
    if len(seeds) != 10 or len(set(seeds)) != 10:
        raise ValueError("F1 v4 requires exactly ten distinct registered seeds")
    candidate = frozen_recipe.get("arm")
    if not candidate:
        raise ValueError("F1 needs the recipe frozen before holdout access")
    models = {"direct": "B2_mask"}
    if candidate != "direct":
        models[candidate] = candidate
    families = sorted(spec["types"])
    masked = set(spec["tavg_mask"])
    rows, seen = [], set()
    for bundle in bundles:
        arm = bundle["arm"]
        folded = _fold_model(arm)
        if folded:
            model, fold = folded
            if model not in models:
                raise ValueError("F1 contains a model outside the frozen recipe: " + model)
            own = bundle.get("split_spec")
            if own is None:
                name = Path(bundle["config"]["data"]["split_name"])
                own = split_spec(name if name.is_absolute() else Path(__file__).parent / "splits" / name)
            if not own.get("holdout") or own.get("constants_fit_scope") != "fold_train_only" or own.get("checkpoint_selection_scope") != "inner_validation_only":
                raise ValueError("F1 analysis requires independently held-out v4 folds")
            if fold != own.get("fold"):
                raise ValueError("F1 arm and split fold differ")
            heldout = own["heldout_family"]
            expected_keys = set(own["holdout"])
            design = "LOTO"
        else:
            model = next((name for name, source in models.items() if arm == source), None)
            if model is None:
                continue
            heldout, fold, design = None, None, "ID"
            expected_keys = set(spec["val"])
        seed = int(bundle["seed"])
        if seed not in seeds:
            if folded:
                raise ValueError("F1 has an unregistered seed")
            continue
        identity = (model, design, fold, seed)
        if identity in seen:
            raise ValueError("duplicate F1 model/design/fold/seed")
        seen.add(identity)
        ids, actuals, predictions = bundle_arrays(bundle)
        keys = [normalized_key(key) for key in ids]
        if len(keys) != len(set(keys)) or set(keys) != expected_keys:
            raise ValueError("F1 prediction IDs do not match the registered scoring pool")
        for key, truth, pred in zip(keys, actuals, predictions):
            family = key.split("/")[0]
            if folded and family != heldout:
                continue
            for index, channel in enumerate(CHANNELS):
                if channel == "T_avg" and key in masked:
                    continue
                value = float(np.mean(np.abs(pred[:, index] - truth[:, index])))
                if not np.isfinite(value):
                    raise ValueError("nonfinite F1 prediction error")
                rows.append({"model": model, "arm": arm, "reference_arm": models[model], "design": design,
                             "seed": seed, "family": family, "channel": channel, "file": key, "MAE": value})
    expected = {(model, "ID", None, seed) for model in models for seed in seeds}
    expected |= {(model, "LOTO", fold, seed) for model in models for fold in range(1, 11) for seed in seeds}
    if seen != expected:
        raise ValueError("F1 incomplete ten-seed/model/fold protocol: missing %d, extra %d" % (len(expected-seen), len(seen-expected)))
    scores = pd.DataFrame(rows)
    means = scores.groupby(["model", "seed", "family", "channel", "design"]).MAE.mean().unstack("design").reset_index()
    if means[["LOTO", "ID"]].isna().any().any() or (means.ID <= 0).any():
        raise ValueError("F1 requires positive paired ID references for every scored channel")
    means["ratio_LOTO_ID"] = means.LOTO / means.ID
    envelope = []
    for (model, family), subset in means.groupby(["model", "family"]):
        selected = subset[subset.channel.isin(("T_outer", "T_avg"))]
        paired = selected.groupby("seed")[["LOTO", "ID"]].mean()
        if set(paired.index) != set(seeds):
            raise ValueError("F1 family lost a paired seed")
        ratios = paired.LOTO / paired.ID
        envelope.append({"model": model, "reference_arm": models[model], "family": family, "n_seeds": len(paired),
                         "LOTO_ToTa": float(paired.LOTO.mean()), "ID_ToTa": float(paired.ID.mean()),
                         "ratio_LOTO_ID": float(ratios.mean()), "within2x": bool(ratios.mean() <= 2),
                         "channels": ",".join(sorted(set(selected.channel))),
                         "reference": "own-model paired ID validation; Case9 T_avg masked"})
    envelope = pd.DataFrame(envelope)
    if any(set(group.family) != set(families) for _, group in envelope.groupby("model")):
        raise ValueError("F1 requires every registered inlet family for every model")
    paired_rows = []
    if candidate != "direct":
        import paper_stats
        values = means.pivot(index=["seed", "family", "channel"], columns="model", values="LOTO")
        for (family, channel), subset in values.groupby(level=["family", "channel"]):
            difference = subset[candidate] - subset["direct"]
            result = paper_stats.paired_stats(difference.to_numpy(), np.random.default_rng(0))
            paired_rows.append(dict(model=candidate, baseline="direct", family=family, channel=channel,
                                    n_seeds=result["n"], mean_diff_C=result["mean"], ci_lo_C=result["lo"],
                                    ci_hi_C=result["hi"], p_signflip=result["p"], selection_use=False))
    return {"scores": scores, "ratios": means, "envelope": envelope, "paired": pd.DataFrame(paired_rows)}


def run_analysis(manifest, out, data_root=None, physics_cache_dir=None, **kwargs):
    metadata = read_json(manifest) if isinstance(manifest, (str, Path)) else manifest
    if not isinstance(metadata, dict) or not metadata.get("frozen_recipe"):
        raise ValueError("F1 manifest must carry frozen_recipe")
    manifest_context(metadata, data_root=data_root, physics_cache_dir=physics_cache_dir)
    execution = read_json(Path(__file__).parent / "prereg/round10_execution_v4.json")
    records = load_manifest(metadata)
    folds = score_fold_holdouts(records)
    scored = {(item["arm"], int(item["seed"])): item for item in folds}
    candidate = metadata["frozen_recipe"]["arm"]
    ids = {"B2_mask"} | ({candidate} if candidate != "direct" else set())
    bundles = []
    for record in records:
        if _fold_model(record["arm"]):
            record = dict(record, holdout_dir=scored[(record["arm"], int(record["seed"]))]["holdout_dir"])
            bundles.append(_prediction_bundle(record, "holdout"))
        elif record["arm"] in ids:
            bundles.append(_prediction_bundle(record, "val"))
    result = analyze_bundles(bundles, metadata["frozen_recipe"], execution["F1"]["seeds"])
    for key, name in (("scores", "f1_per_file"), ("ratios", "f1_channel_ratios"), ("envelope", "f1_envelope")):
        write_table(result[key], out, name, F1_STATEMENT)
    if len(result["paired"]):
        write_table(result["paired"], out, "f1_candidate_direct_paired", "Descriptive paired holdout contrasts; never used to select the recipe.")
    statements = []
    for model, group in result["envelope"].groupby("model"):
        passing = int(group.within2x.sum())
        statements.append({"model": model, "n_families": len(group), "n_within2x": passing,
                           "criterion": "8 of 10 families within 2x own-model ID error",
                           "complete_ten_seed_protocol": True, "passes": passing >= 8, "meaning": F1_STATEMENT})
    write_json(Path(out) / "f1_envelope_statement.json", {"models": statements, "selection_use": False})
    def chart(plt):
        models = list(result["envelope"].model.unique())
        figure, axes = plt.subplots(len(models), 1, figsize=(11, 5 * len(models)), squeeze=False)
        for axis, model in zip(axes[:, 0], models):
            group = result["envelope"][result["envelope"].model == model]
            axis.bar(group.family, group.ratio_LOTO_ID)
            axis.axhline(2, color="black", linestyle="--")
            axis.set_ylabel("Holdout / own-model ID error")
            axis.tick_params(axis="x", rotation=60, labelsize=7)
            axis.set_title("%s: %d/10 families within 2x" % (model, int(group.within2x.sum())))
        figure.tight_layout()
        return figure
    save_plot(Path(out) / "f1_envelope.png", chart)
    return result


def main():
    ap = parser(__doc__)
    args = ap.parse_args()
    if not args.manifest:
        raise ValueError("--manifest is required")
    run_analysis(args.manifest, args.out, args.data_root, args.physics_cache_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
