"""Shared validation-only I/O and checkpoint inference for Round 10.

Only paper_report loads real test files. Pure array helpers below can be called
by its F11 adapter, but analysis CLIs refuse test paths and test predictions.
"""
import argparse
import contextlib
import hashlib
import json
import os
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CHANNELS = ("T_inner", "T_outer", "T_avg")
COLS = ("T_inner (C)", "T_outer (C)", "T_avg (C)")
BASE_COLS = ("Time (s)", "T_outer (C)", "T_inner (C)", "T_avg (C)", "Input Temperature (C)")


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def split_spec(path=None):
    return read_json(path or HERE / "splits/round10_vs3_s42.json")


def resolve_data_root(path=None):
    root = Path(path or os.environ.get("DATA_ROOT", HERE.parents[2] / "data")).resolve()
    if not (root / "Latest Database (Use this for training)").is_dir():
        raise FileNotFoundError("DATA_ROOT has no training directory: %s" % root)
    return root


def read_frames(keys, data_root=None, spec=None):
    """Read explicit training/validation keys only; reject sealed test paths."""
    spec = spec or split_spec()
    allowed = set(spec["train"]) | set(spec["val"])
    validated = []
    for key in keys:
        key = str(key).replace("\\", "/")
        if "test_70_cases" in key.lower() or key not in allowed or ".." in Path(key).parts:
            raise ValueError("analysis input is not an allowed training/validation key: %s" % key)
        validated.append(key)
    root = resolve_data_root(data_root)
    train = root / spec["train_subdir"]
    frames = {}
    for key in validated:
        frame = pd.read_csv(train / key)
        frame.columns = [str(c).strip() for c in frame.columns]
        frame.rename(columns={"T_ave (C)": "T_avg (C)"}, inplace=True)
        if "Input Temperature (C)" not in frame:
            frame["Input Temperature (C)"] = frame["T_inner (C)"]
        if frame[list(BASE_COLS)].isna().any().any():
            raise ValueError("non-finite frame: %s" % key)
        frames[key] = frame
    return frames


def load_manifest(path_or_value):
    value = read_json(path_or_value) if isinstance(path_or_value, (str, Path)) else path_or_value
    runs = value.get("runs", []) if isinstance(value, dict) else value
    if not isinstance(runs, list):
        raise ValueError("manifest must be a list or {'runs': [...]} object")
    out = []
    for record in runs:
        record = dict(record)
        record["run_dir"] = str(Path(record["run_dir"]).resolve())
        config = read_json(Path(record["run_dir"]) / "run_config.json")
        record.setdefault("seed", config["seed"])
        record.setdefault("arm", config.get("round10_arm", Path(record["run_dir"]).name))
        record["config"] = config
        out.append(record)
    return out


def manifest_context(manifest, data_root=None, physics_cache_dir=None, round9_root=None):
    value = read_json(manifest) if isinstance(manifest, (str, Path)) else manifest
    value = value if isinstance(value, dict) else {}
    data_root = data_root or value.get("data_root") or os.environ.get("DATA_ROOT")
    physics_cache_dir = physics_cache_dir or value.get("physics_cache_dir") or os.environ.get("PHYSICS_CACHE_DIR")
    round9_root = round9_root or value.get("round9_root") or os.environ.get("ROUND9_RUNS_ROOT")
    if data_root:
        os.environ["DATA_ROOT"] = str(data_root)
    if physics_cache_dir:
        os.environ["PHYSICS_CACHE_DIR"] = str(physics_cache_dir)
    return data_root, physics_cache_dir, round9_root


def bundle_arrays(bundle):
    actuals = bundle.get("actuals", bundle.get("inv_actuals"))
    preds = bundle.get("preds", bundle.get("inv_preds"))
    ids = [str(k) for k in bundle["case_ids"]]
    if actuals is None or preds is None or len(ids) != len(actuals) or len(ids) != len(preds):
        raise ValueError("incomplete prediction bundle")
    return ids, [temperature_array(a) for a in actuals], [temperature_array(a) for a in preds]


def temperature_array(array):
    """Accept legacy five-column [time,To,Ti,Ta,input] or new [Ti,To,Ta]."""
    array = np.asarray(array, dtype=float)
    if array.ndim != 2:
        raise ValueError("prediction array must have two dimensions")
    if array.shape[1] == 5:
        return array[:, [2, 1, 3]]
    if array.shape[1] == 3:
        return array
    raise ValueError("prediction array needs 3 or 5 columns")


def load_val_bundles(manifest, arms=None):
    bundles = []
    for record in load_manifest(manifest):
        if arms and record["arm"] not in arms:
            continue
        variant = record["config"]["variants"][0]
        path = Path(record["run_dir"]) / "variants" / variant / "predictions_val.npz"
        with np.load(path, allow_pickle=True) as archive:
            ids = [str(k) for k in archive["case_ids"]]
            if any("test_70_cases" in k.lower() for k in ids):
                raise ValueError("sealed test predictions supplied to validation analysis")
            bundle = dict(record, case_ids=ids, actuals=list(archive["inv_actuals"]),
                          preds=list(archive["inv_preds"]))
        bundles.append(bundle)
    if not bundles:
        raise ValueError("manifest contains no completed validation predictions")
    return bundles


def cache_for_keys(directory, keys):
    """Open only explicitly named training/validation cache entries."""
    if not directory:
        raise ValueError("PHYSICS_CACHE_DIR is required")
    directory = Path(directory)
    if (directory / "nominal").is_dir():
        directory = directory / "nominal"
    result = {}
    for key in keys:
        if "test_70_cases" in str(key).lower():
            raise ValueError("test cache may only be generated inside F11 from preloaded frames")
        path = directory / (str(key).replace("/", "__") + ".npz")
        with np.load(path) as archive:
            result[key] = {k: archive[k].copy() for k in archive.files}
    return result


def physics_array(value):
    return np.column_stack([value["T_inner_phys"], value["T_outer_phys"], value["T_avg_line_phys"]])


def frame_targets(frame):
    return frame[list(COLS)].to_numpy(float)[1:]


def normalized_key(key):
    key = str(key).replace("\\", "/")
    marker = "Latest Database (Use this for training)/"
    return key.split(marker, 1)[-1] if marker in key else key


def convention_map(value=None):
    if value is None or isinstance(value, (str, Path)):
        value = pd.read_csv(value or HERE / "splits/tavg_convention_by_file.csv")
    if isinstance(value, dict):
        return value
    result = {}
    for _, row in value.iterrows():
        key = normalized_key(str(row["folder"]) + "/" + str(row["file"]))
        result[key] = row.to_dict()
    return result


def write_table(frame, output, name, note=""):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / (name + ".csv"), index=False)
    lines = ["# " + name.replace("_", " "), "", note, ""]
    if len(frame):
        columns = list(frame.columns)
        lines.extend(["| " + " | ".join(map(str, columns)) + " |",
                      "|" + "|".join("---" for _ in columns) + "|"])
        for row in frame.itertuples(index=False, name=None):
            cells = [("%.6g" % x if isinstance(x, (float, np.floating)) else str(x))
                     for x in row]
            lines.append("| " + " | ".join(cells) + " |")
    else:
        lines.append("No applicable rows.")
    (output / (name + ".md")).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parser(description):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--manifest")
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT"))
    ap.add_argument("--physics-cache-dir", default=os.environ.get("PHYSICS_CACHE_DIR"))
    ap.add_argument("--out", required=True)
    return ap


def setup_run(run_dir):
    import eval_checkpoint as ec
    # Loader details remain available through exceptions/provenance; avoid
    # repeating the whole311file split banner for every archived checkpoint.
    with contextlib.redirect_stdout(io.StringIO()):
        model, scaler, config = ec.load_run_model(str(run_dir), which="best")
    from tes_gru import config as live_config
    constants_source = "sidecar" if Path(ec.constants_sidecar_path(str(run_dir), config["variants"][0])).is_file() else "own-split-refit"
    return {"model": model, "scaler": scaler, "run_config": config, "constants_source": constants_source,
            "config": live_config, "variant": config["variants"][0]}


def predict_frame(session, frame, key, physics_override=None, capture_head=False):
    """Inference on an already-loaded frame. No data file is opened here."""
    import torch
    from tes_gru.data import ThermalDataset, ITEM_INP_LAST
    import tes_gru.data as data
    from tes_gru.rollout import run_rollout_train
    ds = ThermalDataset(frame, session["variant"], scaler=session["scaler"],
                        file_name=key, physics_override=physics_override)
    item = ds[0]
    device = session["config"].DEVICE
    model = session["model"]
    heads = []
    handle = None
    if capture_head and hasattr(model, "fc"):
        handle = model.fc.register_forward_hook(
            lambda module, args, output: heads.append(output.detach().cpu().numpy()))
    kwargs = {"inp_last": item[ITEM_INP_LAST].unsqueeze(0).to(device)}
    if hasattr(data, "ITEM_PHYS_TARGETS") and len(item) > data.ITEM_PHYS_TARGETS:
        kwargs["phys_targets"] = item[data.ITEM_PHYS_TARGETS].unsqueeze(0).to(device)
    try:
        with torch.no_grad():
            predictions = run_rollout_train(
                model, session["variant"], item[0].unsqueeze(0).to(device),
                item[1].unsqueeze(0).to(device), item[3].unsqueeze(0).to(device),
                item[4].unsqueeze(0).to(device), tf_prob=0.0, **kwargs)
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        scaled = predictions[0].detach().cpu().numpy()
        raw = np.empty_like(scaled, dtype=float)
        for index, column in enumerate(COLS):
            col = list(BASE_COLS).index(column)
            raw[:, index] = (scaled[:, index] - session["scaler"].min_[col]) / session["scaler"].scale_[col]
    finally:
        if handle is not None:
            handle.remove()
    if capture_head:
        # G7(c) emits a20step burn-in before the actual forward_direct
        # sequence. The last fc call is the scored sequence in that mode.
        selected_heads = heads[-1:] if session["variant"] == "forward_direct" else heads
        flat_heads = np.concatenate([np.asarray(h).reshape(-1, np.asarray(h).shape[-1]) for h in selected_heads]) if selected_heads else None
        return raw, flat_heads
    return raw


def save_plot(path, callback, rect=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure = callback(plt)
    figure.tight_layout(rect=rect)
    figure.savefig(path, dpi=170)
    plt.close(figure)

