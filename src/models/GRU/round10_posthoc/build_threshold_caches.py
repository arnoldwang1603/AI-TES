"""Build the degraded physics caches for the reference-threshold study.

One cache directory per level, same npz layout as round10_resources/physics/nominal
(keys time, T_inner_phys, T_outer_phys, T_avg_line_phys). Only the 284 training
and 20 validation files of round10_vs3_s42 are written; the test set is never read.

Levels (see PREREG_reference_threshold.md):
  s080 s060 s040 s025 : 1-D model with k(T) multiplied by 0.80 / 0.60 / 0.40 / 0.25
  b005 b010 b020 b035 b050 : (1 - a) * nominal + a * T0, a = 0.05 ... 0.50

Also writes level_stats.csv: training residual share and reference MAE per level
(T_outer over all training files, T_avg over the line-average training files),
and the same MAEs on the validation files.

Usage (from src/models/GRU):
  python round10_posthoc/build_threshold_caches.py [--out runs/_threshold_caches] [--levels s080,b010]
Python 3.8+; numpy, pandas.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import physics_reference  # noqa: E402

LEVELS = [
    ("s080", "diffusivity", 0.80), ("s060", "diffusivity", 0.60),
    ("s040", "diffusivity", 0.40), ("s025", "diffusivity", 0.25),
    ("b005", "blend", 0.05), ("b010", "blend", 0.10), ("b020", "blend", 0.20),
    ("b035", "blend", 0.35), ("b050", "blend", 0.50),
]
COLS = ("Time (s)", "T_outer (C)", "T_avg (C)", "Input Temperature (C)")


def data_root():
    return os.path.abspath(os.environ.get("DATA_ROOT") or os.path.join(HERE, "..", "..", "..", "data"))


def load_frame(path):
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    if "T_avg (C)" not in df.columns and "T_ave (C)" in df.columns:
        df = df.rename(columns={"T_ave (C)": "T_avg (C)"})
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        raise RuntimeError("%s lacks %s" % (path, missing))
    return {c: df[c].to_numpy(float) for c in COLS}


def cache_name(key):
    return key.replace("\\", "/").replace("/", "__") + ".npz"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "_threshold_caches"))
    ap.add_argument("--split", default=os.path.join(HERE, "splits", "round10_vs3_s42.json"))
    ap.add_argument("--nominal", default=os.path.join(HERE, "round10_resources", "physics", "nominal"))
    ap.add_argument("--levels", default="")
    args = ap.parse_args(argv)

    with open(args.split, encoding="utf-8") as f:
        split = json.load(f)
    train, val, masked = list(split["train"]), list(split["val"]), set(split["tavg_mask"])
    root = os.path.join(data_root(), split.get("train_subdir", "Latest Database (Use this for training)"))
    wanted = [l for l in LEVELS if not args.levels or l[0] in args.levels.split(",")]
    if not wanted:
        raise SystemExit("no level selected")
    os.makedirs(args.out, exist_ok=True)

    print("loading %d files from %s" % (len(train) + len(val), root), flush=True)
    frames, nominal = {}, {}
    for key in train + val:
        frames[key] = load_frame(os.path.join(root, key))
        with np.load(os.path.join(args.nominal, cache_name(key)), allow_pickle=False) as z:
            nominal[key] = {k: np.asarray(z[k], float) for k in ("time", "T_outer_phys", "T_avg_line_phys")}
        if not np.allclose(nominal[key]["time"], frames[key]["Time (s)"]):
            raise RuntimeError("nominal cache time grid differs for " + key)

    # sanity: the nominal cache must reproduce physics_reference at k_scale 1 on a few files
    for key in (train[0], val[0]):
        fr = frames[key]
        sim = physics_reference.simulate(fr["Input Temperature (C)"], fr["Time (s)"], k_scale=1.0)
        dmax = max(np.abs(sim["T_outer_phys"] - nominal[key]["T_outer_phys"]).max(),
                   np.abs(sim["T_avg_line_phys"] - nominal[key]["T_avg_line_phys"]).max())
        if dmax > 1e-6:
            raise RuntimeError("physics_reference disagrees with the nominal cache by %.3g C on %s" % (dmax, key))

    rows = []
    for name, family, value in wanted:
        t0 = time.time()
        ldir = os.path.join(args.out, name)
        os.makedirs(ldir, exist_ok=True)
        refs = {}
        for key in train + val:
            fr, nom = frames[key], nominal[key]
            if family == "diffusivity":
                sim = physics_reference.simulate(fr["Input Temperature (C)"], fr["Time (s)"], k_scale=value)
                to, ta = sim["T_outer_phys"], sim["T_avg_line_phys"]
            else:
                t_start = nom["T_outer_phys"][0]          # = first inlet value = T0 (isothermal start)
                to = (1.0 - value) * nom["T_outer_phys"] + value * t_start
                ta = (1.0 - value) * nom["T_avg_line_phys"] + value * nom["T_avg_line_phys"][0]
            refs[key] = (to, ta)
            np.savez(os.path.join(ldir, cache_name(key)), time=nom["time"],
                     T_inner_phys=fr["Input Temperature (C)"], T_outer_phys=to, T_avg_line_phys=ta,
                     level=np.array(name), family=np.array(family), value=np.array(value))

        def share_and_mae(keys, use_mask):
            to_t, to_r, ta_t, ta_r = [], [], [], []
            for key in keys:
                to_t.append(frames[key]["T_outer (C)"]); to_r.append(refs[key][0])
                if not (use_mask and key in masked):
                    ta_t.append(frames[key]["T_avg (C)"]); ta_r.append(refs[key][1])
            to_t, to_r, ta_t, ta_r = map(np.concatenate, (to_t, to_r, ta_t, ta_r))
            return (100 * np.std(to_t - to_r) / np.std(to_t), np.mean(np.abs(to_t - to_r)),
                    100 * np.std(ta_t - ta_r) / np.std(ta_t), np.mean(np.abs(ta_t - ta_r)))

        tr = share_and_mae(train, True)
        # validation MAE per file then mean over files, matching summary_errors_val.csv
        vo = np.mean([np.mean(np.abs(frames[k]["T_outer (C)"] - refs[k][0])) for k in val])
        va = np.mean([np.mean(np.abs(frames[k]["T_avg (C)"] - refs[k][1])) for k in val if k not in masked])
        rows.append(dict(level=name, family=family, value=value,
                         train_share_T_outer_pct=tr[0], train_mae_T_outer=tr[1],
                         train_share_T_avg_pct=tr[2], train_mae_T_avg=tr[3],
                         val_mae_T_outer=vo, val_mae_T_avg=va, n_files=len(refs)))
        with open(os.path.join(ldir, "params.json"), "w", encoding="utf-8") as f:
            json.dump(dict(level=name, family=family, value=value, split=os.path.basename(args.split),
                           n_files=len(refs), nominal_dir=os.path.abspath(args.nominal),
                           physics_reference_sha256=physics_reference.SOURCE_SHA256,
                           created=time.strftime("%Y-%m-%d %H:%M:%S")), f, indent=1)
        print("%-5s %-11s %.2f  share To %.2f%%  Ta %.2f%%  val MAE To %.2f  Ta %.2f  (%.0f s)"
              % (name, family, value, tr[0], tr[2], vo, va, time.time() - t0), flush=True)

    stats = pd.DataFrame(rows)
    path = os.path.join(args.out, "level_stats.csv")
    if os.path.exists(path) and args.levels:
        old = pd.read_csv(path)
        stats = pd.concat([old[~old.level.isin(stats.level)], stats]).sort_values("level")
    stats.to_csv(path, index=False)
    print("wrote", path)


if __name__ == "__main__":
    main()
