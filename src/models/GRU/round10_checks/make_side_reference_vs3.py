"""Round 10 / A4 static artefacts (run from src/models/GRU):

    python round10_checks/make_side_reference_vs3.py [--force]

1. Copies the label table tavg_convention_by_file.csv from the wiki
   (outputs/email_data_group/tavg_convention_check/) into splits/, so
   RELABEL_CSV=tavg_convention_by_file.csv resolves on any machine the repo
   is checked out on (override the source with RELABEL_SRC_CSV).
2. Writes splits/hot_discharge_side_reference_vs3.csv: the s-inverse
   line-average T_avg series, T_line = (T_avg - s*T_inner)/(1 - s), 1441 rows
   per file, for the three masked validation files of round10_vs3_s42
   (Case 1/Case 16, Case 9/New 1, Case 9/New 4) -- the hot-discharge side
   reference the P1 batch scores the model's T_avg against. The physics line
   average for the same files lives in the wiki
   (physics_cache_vs3/hot_discharge_side_reference.csv).
Refuses to overwrite either file unless --force.
"""
import os
import shutil
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # src/models/GRU
sys.path.insert(0, HERE)

WIKI = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "calit2-thermal-wiki", ".claude",
                                     "worktrees", "pensive-cohen-d3486a", "outputs"))
SRC_CSV = os.environ.get("RELABEL_SRC_CSV") or os.path.join(
    WIKI, "email_data_group", "tavg_convention_check", "tavg_convention_by_file.csv")
DST_CSV = os.path.join(HERE, "splits", "tavg_convention_by_file.csv")
OUT = os.path.join(HERE, "splits", "hot_discharge_side_reference_vs3.csv")
SIDE_KEYS = ["Case 1 (linear charging)/Case 16.csv",
             "Case 9 (non-linear discharging)/Case New 1 (Tejas).csv",
             "Case 9 (non-linear discharging)/Case New 4 (Tejas).csv"]


def main():
    force = "--force" in sys.argv
    if os.path.isfile(DST_CSV) and not force:
        print("keeping existing", DST_CSV)
    else:
        assert os.path.isfile(SRC_CSV), SRC_CSV
        shutil.copy2(SRC_CSV, DST_CSV)
        print("copied label table ->", DST_CSV)

    from tes_gru.config import BASE_DIR, TRAIN_SUBDIR
    from tes_gru.data import _resolve_new_data_root, load_relabel_table
    root = _resolve_new_data_root(BASE_DIR)
    assert root, "data root not found"
    train_dir = os.path.join(root, TRAIN_SUBDIR)
    table = load_relabel_table(DST_CSV)
    rows = []
    for key in SIDE_KEYS:
        s = table[key]
        df = pd.read_csv(os.path.join(train_dir, key))
        df.rename(columns={"T_ave (C)": "T_avg (C)"}, inplace=True)
        ta = df["T_avg (C)"].to_numpy(dtype=np.float64)
        ti = df["T_inner (C)"].to_numpy(dtype=np.float64)
        to = df["T_outer (C)"].to_numpy(dtype=np.float64)
        line = (ta - s * ti) / (1.0 - s)
        rows.append(pd.DataFrame({
            "key": key, "time_s": df["Time (s)"].to_numpy(dtype=np.float64),
            "T_inner": ti, "T_outer": to, "T_avg_label": ta,
            "inner_share_pct": round(100.0 * s, 4), "T_avg_line_sinv": line}))
        print("  %-60s s=%.4f  rows %d  mean(T_avg_label - T_line) %+.3f C" % (
            key, s, len(df), float((ta - line).mean())))
    out = pd.concat(rows, ignore_index=True)
    if os.path.isfile(OUT) and not force:
        old = pd.read_csv(OUT)
        same = (len(old) == len(out) and np.allclose(old["T_avg_line_sinv"], out["T_avg_line_sinv"]))
        print("keeping existing", OUT, "(identical)" if same else "(DIFFERS -- rerun with --force)")
        return 0 if same else 1
    out.to_csv(OUT, index=False, float_format="%.6f")
    print("wrote", OUT, len(out), "rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
