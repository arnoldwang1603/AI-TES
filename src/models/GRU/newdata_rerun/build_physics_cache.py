"""Nominal 1-D physics references for the new-data rerun (2026-11-18).

Same solver, parameters and file format as round10_resources/physics/nominal/
(physics_reference.from_frame with round10_resources/physics/params.json: r_o 0.127 m,
s 1.0, 60 layers, 2 sub-steps; driven by Input only) for every training and validation
file of splits/newdata_1115_s42.json, written to
round10_resources/physics/nominal_nd1115/<folder>__<file>.npz plus index.csv.
Input is unchanged for every file the two databases share, so those references must
equal the Round 10 cache; the script checks this and stops if they differ by > 1e-6 C (observed 4e-8 C, float rounding).

Usage (from src/models/GRU):  python newdata_rerun/build_physics_cache.py [--data-root PATH]
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import physics_reference  # noqa: E402

SPLIT = os.path.join(HERE, "splits", "newdata_1115_s42.json")
OLD = os.path.join(HERE, "round10_resources", "physics", "nominal")
OUT = os.path.join(HERE, "round10_resources", "physics", "nominal_nd1115")


def cache_name(key):
    return key.replace("\\", "/").replace("/", "__") + ".npz"


def default_data_root():
    return os.path.abspath(os.path.join(HERE, "..", "..", "..", "data_2026-11-15"))


def is_complete(out=OUT, split=SPLIT):
    spec = json.load(open(split, encoding="utf-8"))
    return all(os.path.isfile(os.path.join(out, cache_name(k))) for k in spec["train"] + spec["val"])


def build(data_root, out=OUT, split=SPLIT):
    spec = json.load(open(split, encoding="utf-8"))
    params = json.load(open(os.path.join(HERE, "round10_resources", "physics", "params.json"), encoding="utf-8"))
    train_dir = os.path.join(data_root, spec["train_subdir"])
    os.makedirs(out, exist_ok=True)
    masked = set(spec.get("tinner_mask") or [])
    rows, worst, n_cmp = [], 0.0, 0
    for which in ("train", "val"):
        for key in spec[which]:
            df = pd.read_csv(os.path.join(train_dir, *key.split("/")))
            ph = physics_reference.from_frame(df, k_scale=float(params["s"]), params=params)
            name = cache_name(key)
            np.savez(os.path.join(out, name), time=ph["time"], T_inner_phys=ph["T_inner_phys"],
                     T_outer_phys=ph["T_outer_phys"], T_avg_line_phys=ph["T_avg_line_phys"],
                     T_avg_volume_phys=ph["T_avg_volume_phys"], r_o=np.float64(params["r_o"]),
                     s=np.float64(params["s"]), n_layers=np.int64(params["n_layers"]),
                     sub_steps=np.int64(params["sub_steps"]), key=np.str_(key))
            old = os.path.join(OLD, name)
            if os.path.isfile(old):
                with np.load(old, allow_pickle=False) as z:
                    d = max(float(np.abs(z[c] - ph[c]).max()) for c in ("T_outer_phys", "T_avg_line_phys"))
                worst, n_cmp = max(worst, d), n_cmp + 1
            folder, file = key.split("/", 1)
            rows.append(dict(key=key, folder=folder, file=file, set=which, param_set="nominal",
                             r_o=params["r_o"], s=params["s"], tinner_masked=key in masked,
                             path=os.path.basename(out) + "/" + name))
    with open(os.path.join(os.path.dirname(out), "index_%s.csv" % os.path.basename(out)), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("physics cache %s: %d files; %d shared with the Round 10 cache, max difference %.2e C"
          % (out, len(rows), n_cmp, worst), flush=True)
    if worst > 1e-6:
        raise SystemExit("references differ from the Round 10 cache although Input is unchanged")
    return len(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=default_data_root())
    args = ap.parse_args(argv)
    build(os.path.abspath(args.data_root))


if __name__ == "__main__":
    main()
