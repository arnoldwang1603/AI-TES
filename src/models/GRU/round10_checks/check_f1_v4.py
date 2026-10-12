"""F1 isolation, model/seed accounting and holdout scoring; no training/Test70."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest import mock
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(Path(__file__).parent))
from check_h9 import clean_env, fresh_import, block_test_csv_reads
import eval_checkpoint as ec
import f1_envelope as f1

checks = []
def check(name, condition):
    checks.append(bool(condition)); print(("PASS " if condition else "FAIL ") + name, flush=True)

def rejected(fn, fragment):
    try:
        fn()
    except (ValueError, RuntimeError) as error:
        return fragment in str(error)
    return False


def aggregation_checks():
    seeds = [1, 2, 3, 5, 7, 11, 13, 21, 42, 123]
    families = ["Case %d (synthetic)" % n for n in range(1, 11)]
    ids = [name + "/id.csv" for name in families]
    held = [name + "/hold.csv" for name in families]
    spec = dict(types=families, val=ids, tavg_mask=[ids[8], held[8]])
    bundles = []
    for model, reference, id_error, hold_error in (("direct", "B2_mask", 1., 2.), ("R3", "R3", 2., 3.)):
        for seed in seeds:
            bundles.append(dict(arm=reference, seed=seed, case_ids=ids,
                                actuals=[np.zeros((4, 3)) for _ in ids],
                                preds=[np.full((4, 3), id_error) for _ in ids]))
            for fold, (family, key) in enumerate(zip(families, held), 1):
                own = dict(holdout=[key], heldout_family=family, fold=fold,
                           constants_fit_scope="fold_train_only", checkpoint_selection_scope="inner_validation_only")
                arm = "F1_%d" % fold if model == "direct" else "F1_%s_%d" % (model, fold)
                bundles.append(dict(arm=arm, seed=seed, split_spec=own, case_ids=[key],
                                    actuals=[np.zeros((4, 3))], preds=[np.full((4, 3), hold_error)]))
    result = f1.analyze_bundles(bundles, {"arm": "R3"}, seeds, spec)
    envelope = result["envelope"]
    check("two model envelopes use their own ID denominator", len(envelope) == 20 and
          np.allclose(envelope[envelope.model == "direct"].ratio_LOTO_ID, 2.) and
          np.allclose(envelope[envelope.model == "R3"].ratio_LOTO_ID, 1.5))
    check("candidate minus direct paired errors do not mix models", len(result["paired"]) == 29 and
          np.allclose(result["paired"].mean_diff_C, 1.) and set(result["paired"].n_seeds) == {10})
    check("Case9 remains wall-only in both model envelopes", all(envelope[envelope.family == families[8]].channels == "T_outer"))
    check("missing paired seed rejected", rejected(lambda: f1.analyze_bundles(bundles[:-1], {"arm": "R3"}, seeds, spec), "incomplete"))
    check("duplicate model/fold/seed rejected", rejected(lambda: f1.analyze_bundles(bundles + [bundles[1]], {"arm": "R3"}, seeds, spec), "duplicate"))
    check("unfrozen model rejected", rejected(lambda: f1.analyze_bundles(bundles, {"arm": "direct"}, seeds, spec), "outside"))
    parent = HERE / "runs/_round10_check_scratch"; parent.mkdir(parents=True, exist_ok=True)
    def nonfinite_evaluation(run, checkpoint, which_set, out):
        dest = Path(out); dest.mkdir(parents=True, exist_ok=True)
        np.savez(dest / "predictions_holdout.npz", case_ids=np.array(["synthetic/f.csv"]),
                 inv_actuals=np.zeros((1, 4, 3)), inv_preds=np.full((1, 4, 3), np.nan))
        return {"n_files": 1}
    with tempfile.TemporaryDirectory(prefix="f1_nan_", dir=str(parent)) as temporary:
        with mock.patch.object(ec, "evaluate_one", side_effect=nonfinite_evaluation):
            check("NaN predictions cannot pass a zero-delta smoke roundtrip", rejected(
                lambda: f1.check_holdout_roundtrip("unused", temporary), "nonfinite"))


def fitting_and_scoring_checks():
    import torch
    base = dict(VARIANTS="forward_direct", VAL_SPLIT="stratified", SPLIT_NAME="loto_v4_fold9.json",
                TAVG_MASK="split", NUM_LAYERS="1", HIDDEN_SIZE="16", DROPOUT="0", SEAL_TEST="1", TF32="0", SEEDS="7", MAX_EPOCHS="2")
    original = pd.read_csv
    seen = []
    def forbid_holdout(path, *args, **kwargs):
        name = str(getattr(path, "name", path)).replace("\\", "/")
        if "/Case 9 (" in name:
            raise RuntimeError("held-out CSV reached optimisation or fitted statistics")
        seen.append(name)
        return original(path, *args, **kwargs)
    cache = str(HERE / "round10_resources/physics")
    # A nominal/ subdirectory is used by the packaged physics cache.
    if (Path(cache) / "nominal").is_dir():
        cache = str(Path(cache) / "nominal")
    for label, knobs in (("direct", {}), ("ratio", {"OTHER_CH_MODE": "ratio_input", "RATIO_CHANNELS": "R3", "TINNER_MODE": "arfed"}),
                         ("physics", {"PHYS_MODE": "residual", "PHYSICS_CACHE_DIR": cache})):
        clean_env(ec); os.environ.pop("CONSTANTS_FROM", None)
        cfg = fresh_import(dict(base, **knobs))
        from tes_gru import data, models, train, runio
        with mock.patch.object(pd, "read_csv", side_effect=forbid_holdout):
            tr, va, te, scaler = data.load_all_data(include_test=False)
            model = models.build_model("forward_direct", cfg.LATEST_PARAMS, "point").to(cfg.DEVICE)
            train.fit_training_constants(model, "forward_direct", tr, scaler)
        spec = data.load_split_spec()
        check(label + " scaler/all constants never read heldout family", not te and
              {data.split_key(key) for _, key in scaler._round10_fit_records} == set(spec["train"]) and
              not set(spec["holdout"]) & {data.split_key(key) for _, key in tr + va})
        if label == "direct":
            parent = HERE / "runs/_round10_check_scratch"; parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="f1_v4_api_", dir=str(parent)) as temporary:
                root = Path(temporary); run = root / "run"; (run / "checkpoints").mkdir(parents=True)
                config = runio.run_config_dict(); config["seed"] = 7
                (run / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
                ckpt = Path(ec.checkpoint_path(str(run), "forward_direct", "best"))
                torch.save(model.state_dict(), ckpt)
                runio.write_constants_sidecar(str(run / "checkpoints/best_gru_variant_forward_direct.constants.json"), model, scaler,
                                              extra={"variant": "forward_direct", "checkpoint": ckpt.name})
                # Genuine evaluator restores an untrained, synthetic-checkpoint
                # model. It reads only the training-source heldout family.
                result = f1.check_holdout_roundtrip(str(run), str(root / "roundtrip"))
                check("holdout scorer round-trips its own sidecar without training", result["n_files"] == len(spec["holdout"]) and result["max_prediction_delta_C"] <= 1e-6)
                scored = ec.evaluate_one(str(run), "best", "holdout", str(root / "scored"))
                check("Case9 holdout scoring applies the holdout Tavg mask", scored["n_files_T_avg"] == 0 and not np.isfinite(scored["T_avg"]))
                record = {"arm": "F1_9", "seed": 7, "run_dir": str(run)}
                check("partial fold cannot enter the holdout report", rejected(lambda: f1.score_fold_holdouts([record]), "completed training"))
                variant_dir = run / "variants/forward_direct"; variant_dir.mkdir(parents=True)
                (variant_dir / "done.flag").write_text("synthetic test fixture", encoding="utf-8")
                with mock.patch.object(ec, "evaluate_one", wraps=ec.evaluate_one) as evaluate:
                    f1.score_fold_holdouts([record])
                    f1.score_fold_holdouts([record])
                    check("identical resumed fold reuses verified holdout artifacts", evaluate.call_count == 1)
                prediction_path = variant_dir / "predictions_holdout.npz"
                prediction_path.write_bytes(prediction_path.read_bytes() + b"altered")
                check("changed holdout artifact refused on resume", rejected(lambda: f1.score_fold_holdouts([record]), "artifacts"))
    clean_env(ec); os.environ.pop("CONSTANTS_FROM", None)
    cfg = fresh_import(dict(base, FROZEN_CONSTANTS=str(HERE / "round10_resources/full_train.constants.json")))
    from tes_gru.data import load_all_data
    check("strict fold refuses external/full-train constants", rejected(lambda: load_all_data(include_test=False), "prohibits"))


def main():
    block_test_csv_reads()
    aggregation_checks()
    fitting_and_scoring_checks()
    print("check_f1_v4: %d/%d PASS; no training or Test70 reads" % (sum(checks), len(checks)))
    return 0 if all(checks) else 1

if __name__ == "__main__":
    raise SystemExit(main())
