"""Training/validation-only linear system-identification baselines.

F11 calls predict/score_frames with its preloaded frames. This CLI never opens
the sealed test directory. Kernel order and ridge are chosen on validation.
"""
import argparse
import weakref
from pathlib import Path
import numpy as np
import pandas as pd
from analysis_common import (CHANNELS, COLS, frame_targets, parser, read_frames,
                             split_spec, write_json, write_table, save_plot, manifest_context)


_FEATURE_CACHE = {}


def kernel_features(frame, order):
    cache_key = (id(frame), int(order), "kernel")
    if cache_key in _FEATURE_CACHE and _FEATURE_CACHE[cache_key][0]() is frame:
        return _FEATURE_CACHE[cache_key][1]
    inlet = frame["Input Temperature (C)"].to_numpy(float)
    delta = inlet - inlet[0]
    taus = np.geomspace(10.0, 3000.0, int(order))
    filters = []
    for tau in taus:
        rate = np.exp(-1.0 / tau)
        filtered = np.zeros(len(inlet))
        for index in range(1, len(inlet)):
            filtered[index] = rate * filtered[index - 1] + (1.0 - rate) * delta[index]
        filters.append(filtered[1:])
    result = np.column_stack([np.ones(len(inlet) - 1), delta[1:]] + filters)
    _FEATURE_CACHE[cache_key] = (weakref.ref(frame), result)
    return result


def inner_features(frame, order):
    cache_key = (id(frame), int(order), "inner")
    if cache_key in _FEATURE_CACHE and _FEATURE_CACHE[cache_key][0]() is frame:
        return _FEATURE_CACHE[cache_key][1]
    inlet = frame["Input Temperature (C)"].to_numpy(float)
    differences = np.diff(inlet, prepend=inlet[0])
    columns = [np.ones(len(inlet) - 1)]
    for lag in range(int(order)):
        shifted = np.zeros(len(inlet))
        if lag == 0:
            shifted[:] = differences
        else:
            shifted[lag:] = differences[:-lag]
        columns.append(shifted[1:])
    result = np.column_stack(columns)
    _FEATURE_CACHE[cache_key] = (weakref.ref(frame), result)
    return result


def _fit_coefficients(frames, masked_keys, order, ridge, fallback=None):
    products = {}
    for channel, col in zip(CHANNELS, COLS):
        dim = order + 1 if channel == "T_inner" else order + 2
        gram = np.zeros((dim, dim))
        rhs = np.zeros(dim)
        rows = 0
        for key, frame in frames.items():
            if channel == "T_avg" and key in masked_keys:
                continue
            features = inner_features(frame, order) if channel == "T_inner" else kernel_features(frame, order)
            target = frame[col].to_numpy(float)[1:]
            if channel == "T_inner":
                target = target - frame["Input Temperature (C)"].to_numpy(float)[1:]
            else:
                target = target - float(frame[col].iloc[0])
            gram += features.T @ features
            rhs += features.T @ target
            rows += len(target)
        if not rows:
            if fallback is not None:
                products[channel] = fallback[channel]
                continue
            raise ValueError("no unmasked training labels for %s" % channel)
        penalty = np.eye(dim) * float(ridge)
        penalty[0, 0] = 0.0
        coefficients = np.linalg.lstsq(gram / rows + penalty, rhs / rows, rcond=None)[0]
        products[channel] = coefficients.tolist()
    return products


def fit_model(frames, masked_keys, order, ridge, binned=False):
    model = {"schema": 1, "kind": "linear_sysid", "order": int(order), "ridge": float(ridge),
             "bin_width_degC": 100, "binned": bool(binned),
             "global": _fit_coefficients(frames, masked_keys, order, ridge),
             "bins": {}, "train_files": sorted(frames),
             "tavg_scored_train_files": sorted(set(frames) - set(masked_keys)),
             "definition": "Ti=inlet(t+1)+FIR(inlet differences); To/Ta=initial+fixed-weight exponential kernels"}
    if binned:
        by_bin = {}
        for key, frame in frames.items():
            label = str(int(np.floor(float(frame["Input Temperature (C)"].iloc[0]) / 100)))
            by_bin.setdefault(label, {})[key] = frame
        for label, subset in sorted(by_bin.items()):
            # A bin without unmasked T_avg labels uses global T_avg coefficients.
            model["bins"][label] = _fit_coefficients(subset, masked_keys, order, ridge, fallback=model["global"])
    return model


def predict(model, frame):
    """Return steps1..N raw temperature predictions ordered Ti, To, Ta."""
    order = int(model["order"])
    inlet = frame["Input Temperature (C)"].to_numpy(float)
    label = str(int(np.floor(float(inlet[0]) / float(model.get("bin_width_degC", 100)))))
    coefficients = model.get("bins", {}).get(label, model["global"]) if model.get("binned") else model["global"]
    inner = inner_features(frame, order)
    kernels = kernel_features(frame, order)
    columns = []
    for channel, col in zip(CHANNELS, COLS):
        features = inner if channel == "T_inner" else kernels
        ref = inlet[1:] if channel == "T_inner" else float(frame[col].iloc[0])
        columns.append(ref + features @ np.asarray(coefficients[channel], dtype=float))
    return np.column_stack(columns)


def score_frames(model, frames_dict, masked_keys=()):
    masked_keys = set(masked_keys)
    rows = []
    for key, frame in frames_dict.items():
        error = predict(model, frame) - frame_targets(frame)
        for index, channel in enumerate(CHANNELS):
            if channel == "T_avg" and key in masked_keys:
                continue
            rows.append({"file": key, "family": key.split("/")[0], "channel": channel,
                         "MAE": float(np.mean(np.abs(error[:, index]))),
                         "RMSE": float(np.sqrt(np.mean(error[:, index] ** 2))),
                         "MaxErr": float(np.max(np.abs(error[:, index])))})
    return pd.DataFrame(rows)


def load_models(path):
    from analysis_common import read_json
    value = read_json(path)
    return value["models"] if "models" in value else value


def run_analysis(manifest, out, data_root=None, physics_cache_dir=None, **kwargs):
    data_root, physics_cache_dir, _ = manifest_context(manifest, data_root, physics_cache_dir)
    spec = split_spec()
    training = read_frames(spec["train"], data_root, spec)
    validation = read_frames(spec["val"], data_root, spec)
    masked = set(spec["tavg_mask"])
    candidates, chosen, best = [], {}, {}
    for binned in (False, True):
        name = "linear_binned" if binned else "linear_global"
        for order in (8, 12, 16):
            for ridge in (1e-6, 1e-4, 1e-2):
                model = fit_model(training, masked, order, ridge, binned)
                scores = score_frames(model, validation, masked)
                means = scores.groupby("channel")["MAE"].mean().to_dict()
                objective = (means["T_outer"] + means["T_avg"]) / 2.0
                row = dict(model=name, order=order, ridge=ridge, selection_ToTa=objective)
                row.update(means)
                candidates.append(row)
                if name not in best or objective < best[name]:
                    best[name] = objective
                    chosen[name] = model
    for name, model in chosen.items():
        model["selection_validation_keys"] = list(spec["val"])
        model["selection_objective"] = "(T_outer+T_avg)/2 per-file MAE on validation"
        model["selection_value"] = best[name]
    write_json(Path(out) / "linear_sysid_models.json", {"models": chosen})
    write_table(pd.DataFrame(candidates), out, "linear_selection",
                "Descriptive baseline; ridge and order selected on validation only. No sealed test file loaded.")
    rows = []
    for name, model in chosen.items():
        scores = score_frames(model, validation, masked)
        scores["model"] = name
        rows.append(scores)
    scores = pd.concat(rows, ignore_index=True)
    write_table(scores, out, "linear_validation_per_file")
    summary = scores.groupby(["model", "channel"])["MAE"].agg(["mean", "std", "count"]).reset_index()
    write_table(summary, out, "linear_validation_summary",
                "Per-file statistics. A binned linear T_outer within0.5C of GRU requires narrowing the paper's ML-value sentence.")
    def chart(plt):
        figure, axis = plt.subplots(figsize=(7, 4))
        pivot = summary.pivot(index="channel", columns="model", values="mean")
        pivot.plot.bar(ax=axis)
        axis.set_ylabel("Validation MAE (C)")
        axis.tick_params(axis="x", rotation=0)
        return figure
    save_plot(Path(out) / "linear_validation.png", chart)
    return {"models": chosen, "scores": scores, "summary": summary}


def main():
    ap = parser(__doc__)
    args = ap.parse_args()
    run_analysis(args.manifest, args.out, args.data_root, args.physics_cache_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

