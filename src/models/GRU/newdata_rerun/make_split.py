"""Split for the new-data rerun (2026-11-18): splits/newdata_1115_s42.json.

Starts from the Round 10 split round10_vs3_s42.json and changes only what the
2026-11-15 database forces (audit: calit2-thermal-wiki outputs/data-audit-2026-11-15):
  drop         the two '2nd Run' files (copies of 'Case 1/Case new 18' and 'Case 4/Case new 10'
               apart from an HX-face T_inner), Case 1/Case new 19 (= Case new 7 within 0.03 C)
               and Case 8/Discharge 13 (byte-identical to Discharge 11). Round 10's other drops
               (Case 1/new 1-3, Case 10/Case 8) are fixed in this database and come back.
  tinner_mask  every kept file whose T_inner equals Input: slope s of (T_inner - Input) on
               dT_avg/dt through 0, |s| < 10 s (concrete-surface files: -60 to -92 s).
  val          Round 10's 20 validation files; a masked one is swapped for a draw with
               random.Random(42) among the unmasked files of the same folder.
  tavg_mask    none: every training T_avg is a line average in this database.
The json records the md5 digest of the training folder; run_newdata.py checks it.

Usage (from src/models/GRU):  python newdata_rerun/make_split.py [--data-root PATH]
"""
import argparse
import hashlib
import json
import os
import random

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_SUBDIR = "Latest Database (Use this for training)"
BASE_SPLIT = "round10_vs3_s42.json"
OUT_NAME = "newdata_1115_s42.json"
TAG = "nd1115"
DROP = [
    "Case 1 (linear charging)/2nd Run Case new 18 (1 second s.csv",
    "Case 4 (Linear Charging+Discharging)/2nd Run Case new 10.csv",
    "Case 1 (linear charging)/Case new 19 (1 second step).csv",
    "Case 8 (linear discharging)/Discharge 13- Erick wKCp.csv",
]
# The audit's list; the script recomputes it and refuses to write if the data disagree.
EXPECTED_TINNER_MASK = ["Case 10 (Discharging + Constant)/%s.csv" % n for n in
                        ("Case 9", "Case 10", "Case 11", "Case 12", "Case 13", "Case 14", "Case 15", "Case 16 ",
                         "Case 17", "Case 18", "Case 19", "Case 20", "Case 21", "Case 22", "Case 24", "Case 25")]
SLOPE_A_MAX_S = 10.0


def default_data_root():
    return os.path.abspath(os.path.join(HERE, "..", "..", "..", "data_2026-11-15"))


def train_keys(train_dir):
    keys = []
    for dp, _, fs in os.walk(train_dir):
        for f in fs:
            if f.lower().endswith(".csv"):
                keys.append(os.path.relpath(os.path.join(dp, f), train_dir).replace(os.sep, "/"))
    return sorted(keys)


def content_md5(path):
    """md5 of the file with CRLF turned into LF: git (core.autocrlf) stores the CSVs with LF
    and a Windows checkout shows CRLF, so raw bytes differ between machines; the numbers do not."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read().replace(b"\r\n", b"\n")).hexdigest()


def manifest_sha256(train_dir, keys):
    lines = ["%s\t%s" % (k, content_md5(os.path.join(train_dir, *k.split("/")))) for k in keys]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def tinner_slope(path):
    df = pd.read_csv(path)
    t, tin = df["Time (s)"].to_numpy(float), df["Input Temperature (C)"].to_numpy(float)
    ti, ta = df["T_inner (C)"].to_numpy(float), df["T_avg (C)"].to_numpy(float)
    g, r = (ti - tin)[3:], np.gradient(ta, t)[3:]
    return float(np.dot(g, r) / max(np.dot(r, r), 1e-12))


def build(data_root):
    train_dir = os.path.join(data_root, TRAIN_SUBDIR)
    keys = train_keys(train_dir)
    if len(keys) != 312:
        raise SystemExit("expected 312 training files under %s, found %d" % (train_dir, len(keys)))
    missing = [k for k in DROP if k not in keys]
    if missing:
        raise SystemExit("drop list names files not on disk: %s" % missing)
    kept = [k for k in keys if k not in DROP]
    slopes = {k: tinner_slope(os.path.join(train_dir, *k.split("/"))) for k in kept}
    mask = sorted(k for k, s in slopes.items() if abs(s) < SLOPE_A_MAX_S)
    if mask != sorted(EXPECTED_TINNER_MASK):
        raise SystemExit("T_inner = Input files differ from the audit: extra %s, missing %s"
                         % (sorted(set(mask) - set(EXPECTED_TINNER_MASK)), sorted(set(EXPECTED_TINNER_MASK) - set(mask))))
    base = json.load(open(os.path.join(HERE, "splits", BASE_SPLIT), encoding="utf-8"))
    val, swaps, rng = [], {}, random.Random(42)
    for v in base["val"]:
        if v not in kept:
            raise SystemExit("Round 10 validation file missing from the new data: " + v)
        if v in mask:
            folder = v.split("/")[0]
            pool = sorted(k for k in kept if k.split("/")[0] == folder and k not in mask and k not in base["val"])
            new = rng.choice(pool)
            swaps[v] = new
            val.append(new)
        else:
            val.append(v)
    val = sorted(val)
    train = sorted(k for k in kept if k not in val)
    spec = {
        "val_split": "stratified", "val_per_type": 2, "split_seed": 42, "train_subdir": TRAIN_SUBDIR,
        "types": sorted({k.split("/")[0] for k in kept}), "val": val, "train": train,
        "drop": DROP, "tinner_mask": mask, "tag": TAG, "split_name": OUT_NAME, "base_split": BASE_SPLIT,
        "val_swaps": swaps,
        "data_source": "2026-11-15 download of the Drive folder 'Temperature model_ most-recent-data (auto-updated)' "
                       "(AI-TES/data_2026-11-15)",
        "data_manifest_sha256": manifest_sha256(train_dir, keys),
        "n_train": len(train), "n_val": len(val), "n_drop": len(DROP),
        "n_tinner_mask_train": len(set(mask) & set(train)), "n_tinner_mask_val": len(set(mask) & set(val)),
        "note": "new-data rerun 2026-11-18; no T_avg mask; T_inner masked on the HX-face files; "
                "validation = Round 10's with masked files swapped (seed 42)",
    }
    return spec, slopes


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=default_data_root())
    args = ap.parse_args(argv)
    spec, slopes = build(os.path.abspath(args.data_root))
    out = os.path.join(HERE, "splits", OUT_NAME)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=1, ensure_ascii=False)
    b = [s for k, s in slopes.items() if k not in spec["tinner_mask"]]
    print("wrote %s: train %d, val %d, drop %d, T_inner masked %d (train %d, val %d); swaps %s"
          % (out, spec["n_train"], spec["n_val"], spec["n_drop"], len(spec["tinner_mask"]),
             spec["n_tinner_mask_train"], spec["n_tinner_mask_val"], spec["val_swaps"]))
    print("unmasked T_inner slopes %.1f to %.1f s; data digest %s" % (min(b), max(b), spec["data_manifest_sha256"][:16]))


if __name__ == "__main__":
    main()
