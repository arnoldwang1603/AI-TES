"""Compare every validation step against the separately frozen P0 package.

The archived P0 source already includes H9. Both independently built models
receive the same champion weights, training/validation frames and scaler.
Neither package is allowed to load the real test CSVs in this gate.
"""
import argparse
import importlib
import importlib.util
import os
import sys

from paper_report import atomic_json, read_json, sha256
from round10_integrity import matches_frozen_sha256

HERE = os.path.dirname(os.path.abspath(__file__))
LEGACY = os.path.join(HERE, "round10_resources", "legacy_code")


def verify_baseline(verify_source=False):
    manifest = read_json(os.path.join(LEGACY, "manifest.json"))
    for relative, expected in manifest["files"].items():
        if not matches_frozen_sha256(os.path.join(LEGACY, relative), expected):
            raise RuntimeError("Frozen P0 baseline changed: " + relative)
        if verify_source and not matches_frozen_sha256(os.path.join(manifest["source_root"], relative), expected):
            raise RuntimeError("Original P0 source changed after baseline capture: " + relative)
    return manifest


def import_legacy():
    alias = "round10_legacy_gru"
    for name in [n for n in list(sys.modules) if n == alias or n.startswith(alias + ".")]:
        del sys.modules[name]
    package = os.path.join(LEGACY, "tes_gru")
    spec = importlib.util.spec_from_file_location(alias, os.path.join(package, "__init__.py"),
                                                submodule_search_locations=[package])
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return {name: importlib.import_module(alias + "." + name)
            for name in ("config", "data", "models", "rollout", "train")}


def run_gate(verify_source=False):
    import torch
    import eval_checkpoint as ec
    from round10_checks.check_h9 import CHAMP, block_test_csv_reads

    baseline = verify_baseline(verify_source)
    block_test_csv_reads()
    config = read_json(os.path.join(CHAMP, "run_config.json"))
    env = ec.env_from_config(config)
    for key in set(ec.KEYS.values()) | set(ec.ENV_VARS_EXTRA):
        os.environ.pop(key, None)
    os.environ.update(env)
    os.environ.update(TF32="legacy", SEAL_TEST="1", ROUND10_NO_PLOTS="1")
    for name in [n for n in list(sys.modules) if n.startswith("tes_gru")]:
        del sys.modules[name]
    new = {name: importlib.import_module("tes_gru." + name)
           for name in ("config", "data", "models", "rollout", "train")}
    old = import_legacy()

    def forbid_old_loader(*_args, **_kwargs):
        raise RuntimeError("The reference package may not invoke its data loader")
    old["data"].load_all_data = forbid_old_loader
    old["train"].load_all_data = forbid_old_loader
    new["config"].set_seed(config["seed"])
    train, validation, tests, scaler = new["data"].load_all_data(include_test=False)
    if tests or len(validation) != 20:
        raise RuntimeError("Gate requires 20 validation frames and zero test frames")
    variant = config["variants"][0]
    device = new["config"].DEVICE
    checkpoint = ec.checkpoint_path(CHAMP, variant, "best")
    try:
        weights = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        weights = torch.load(checkpoint, map_location=device)
    models = {}
    for label, modules in (("old", old), ("new", new)):
        modules["config"].set_seed(config["seed"])
        model = modules["models"].build_model(variant, modules["config"].LATEST_PARAMS,
                                              modules["rollout"].variant_to_encoder(variant)).to(device)
        # Independent fits expose changes to defaults instead of concealing
        # them by copying the new constants into the frozen implementation.
        modules["train"].fit_model_constants(model, variant, train, scaler)
        model.load_state_dict(weights, strict=True)
        model.eval()
        models[label] = model
    if sum(p.numel() for p in models["old"].parameters()) != sum(p.numel() for p in models["new"].parameters()):
        raise RuntimeError("Default parameter count changed from the frozen P0 baseline")
    rows = []
    max_delta = max_loss_delta = 0.0
    with torch.no_grad():
        for frame, file_name in validation:
            old_ds = old["data"].ThermalDataset(frame, variant, scaler=scaler, file_name=file_name)
            new_ds = new["data"].ThermalDataset(frame, variant, scaler=scaler, file_name=file_name)
            old_item, new_item = old_ds[0], new_ds[0]
            dataset_delta = max(float((old_item[i] - new_item[i]).abs().max()) for i in (0, 1, 3, 4, 10))
            if dataset_delta != 0:
                raise RuntimeError("Default validation input/target construction changed: " + file_name)
            predictions = {}
            losses = {}
            for label, modules, item in (("old", old, old_item), ("new", new, new_item)):
                pred = modules["rollout"].run_rollout_train(models[label], variant,
                    item[0][None].to(device), item[1][None].to(device), item[3][None].to(device),
                    item[4][None].to(device), tf_prob=0.0, inp_last=item[10].reshape(1).to(device))
                predictions[label] = pred
                w = torch.tensor(modules["config"].LOSS_WEIGHTS, dtype=pred.dtype, device=device)
                losses[label] = modules["models"].weighted_loss(pred, item[1][None].to(device),
                    weights=w, raw_affine=models[label].raw_affine).item()
            delta = (predictions["old"] - predictions["new"]).abs()
            if not torch.isfinite(delta).all():
                raise RuntimeError("Non-finite validation regression: " + file_name)
            difference = float(delta.max())
            per_channel = delta.reshape(-1, 3).max(dim=0).values.cpu().tolist()
            loss_delta = abs(losses["old"] - losses["new"])
            max_delta, max_loss_delta = max(max_delta, difference), max(max_loss_delta, loss_delta)
            rows.append(dict(file=file_name, steps=int(predictions["new"].shape[1]),
                             max_delta_scaled=difference, channel_deltas_scaled=per_channel,
                             weighted_loss_delta=loss_delta))
    return dict(ok=max_delta <= 1e-6 and max_loss_delta <= 1e-7,
        tf32="legacy", set="validation", test_csv_access=False,
        gate="Frozen original P0 package versus new defaults; independent constants fits and complete rollouts",
        source_verified=bool(verify_source), baseline_files_verified=len(baseline["files"]),
        baseline_manifest_sha256=sha256(os.path.join(LEGACY, "manifest.json")),
        checkpoint_sha256=sha256(checkpoint), n_files=len(rows),
        max_delta_scaled=max_delta, tolerance_scaled=1e-6,
        max_weighted_loss_delta=max_loss_delta, tolerance_loss=1e-7, files=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify-source", action="store_true", help="also verify the original local P0 source before package installation")
    a = ap.parse_args()
    result = run_gate(a.verify_source)
    atomic_json(a.out, result)
    if not result["ok"]:
        raise RuntimeError("Independent P0 validation regression failed: max scaled delta %.3g, loss delta %.3g" %
                           (result["max_delta_scaled"], result["max_weighted_loss_delta"]))
    print("Independent P0 validation gate PASS: 20 files, all steps; scaled delta %.3g, loss delta %.3g; no test CSV access" %
          (result["max_delta_scaled"], result["max_weighted_loss_delta"]))


if __name__ == "__main__":
    main()
