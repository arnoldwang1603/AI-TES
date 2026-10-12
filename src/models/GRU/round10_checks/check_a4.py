"""Round 10 / A4 check: algebraic T_avg relabel hook (RELABEL_CSV). Run from
src/models/GRU:  python round10_checks/check_a4.py

No training. PASS / FAIL per item, non-zero exit on any FAIL. Needs the data
under AI-TES/data, splits/tavg_convention_by_file.csv and
splits/hot_discharge_side_reference_vs3.csv (both made by
round10_checks/make_side_reference_vs3.py) and the wiki physics cache
(physics_cache_vs3/nominal; override with PHYSICS_CACHE_DIR).
"""
import glob
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # src/models/GRU
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(HERE)

from check_h9 import clean_env, fresh_import   # noqa: E402

RESULTS = []
VS3 = "round10_vs3_s42.json"
LABELS = "tavg_convention_by_file.csv"
CACHE = os.environ.get("PHYSICS_CACHE_DIR") or os.path.normpath(os.path.join(
    HERE, "..", "..", "..", "..", "calit2-thermal-wiki", ".claude", "worktrees", "pensive-cohen-d3486a",
    "outputs", "round10_plan", "physics_cache_vs3", "nominal"))
NEW18 = "Case 1 (linear charging)/Case new 18 (1 second step).csv"
NEW18_C9 = "Case 9 (non-linear discharging)/Case New 18 (Tejas).csv"
SIDE_KEYS = ["Case 1 (linear charging)/Case 16.csv",
             "Case 9 (non-linear discharging)/Case New 1 (Tejas).csv",
             "Case 9 (non-linear discharging)/Case New 4 (Tejas).csv"]


def report(item, ok, detail=""):
    RESULTS.append(bool(ok))
    print("%s  %s  %s" % ("PASS" if ok else "FAIL", item, detail), flush=True)


def base_env(extra):
    env = {"VARIANTS": "forward_direct", "VAL_SPLIT": "stratified", "VAL_PER_TYPE": "2",
           "HIDDEN_SIZE": "128", "NUM_LAYERS": "1", "DROPOUT": "0.0", "TRAIN_EXCLUDE_TYPE": "",
           "TINNER_MODE": "anchor", "ANCHOR_LEAD": "1", "OTHER_CH_MODE": "abs", "LOSS_WEIGHTS": "1,6,3",
           "PHYSICS_BOUND_WEIGHT": "0", "TF32": "0", "SPLIT_NAME": VS3}
    env.update(extra)
    return env


def tavg_of(dfs, key):
    from tes_gru.data import split_key
    for df, cid in dfs:
        if split_key(cid) == key:
            d = df.rename(columns={"T_ave (C)": "T_avg (C)"})
            return d["T_avg (C)"].to_numpy(dtype=np.float64), d["T_inner (C)"].to_numpy(dtype=np.float64)
    raise KeyError(key)


def main():
    import eval_checkpoint as ec
    clean_env(ec)

    # ------------------------------------------------------------ baseline (no relabel)
    config0 = fresh_import(base_env({}))
    from tes_gru.data import load_all_data, load_split_spec, split_key, load_relabel_table
    tr0, va0, te0, sc0 = load_all_data(include_test=False)
    spec = load_split_spec(config0.split_file_path())
    mask = set(spec["tavg_mask"])
    report("(d) no RELABEL_CSV: run name carries no 'rl' tag, config.RELABEL_CSV_PATH is None",
           "_rl" not in config0.RUN_NAME_BASE and config0.RELABEL_CSV_PATH is None and config0.RELABEL_TAG == "")

    # ------------------------------------------------------------ relabel on
    clean_env(ec)
    config = fresh_import(base_env({"RELABEL_CSV": LABELS}))
    from tes_gru.data import load_all_data, load_relabel_table, get_relabel_sets, split_key  # noqa: F811
    table = load_relabel_table()
    rl_tr, rl_va = get_relabel_sets()
    in_split = set(table) & (set(spec["train"]) | set(spec["val"]))
    report("(table) label table in splits/: its inner-weighted / borderline keys in train+val == the vs3 tavg_mask (118)",
           in_split == mask and len(rl_tr) == 115 and len(rl_va) == 3 and rl_va == set(SIDE_KEYS)
           and not any(k.startswith("Test_70_cases") for k in table)
           and (set(table) - in_split) <= set(spec["drop"]),
           "listed %d (%d in train+val, %d dropped), train %d val %d" % (
               len(table), len(in_split), len(set(table) - in_split), len(rl_tr), len(rl_va)))
    tr1, va1, te1, sc1 = load_all_data(include_test=False)

    # (a) new 18 vs the physics line average
    ta_new, ti_new = tavg_of(tr1, NEW18)
    ta_old, _ = tavg_of(tr0, NEW18)
    s = table[NEW18]
    expect = (ta_old - s * ti_new) / (1.0 - s)
    npz = os.path.join(CACHE, NEW18.replace("/", "__") + ".npz")
    z = np.load(npz)
    phys = z["T_avg_line_phys"].astype(np.float64)
    rms_new = float(np.sqrt(np.mean((ta_new - phys) ** 2)))
    rms_old = float(np.sqrt(np.mean((ta_old - phys) ** 2)))
    report("(a) relabelled Case 1/new 18 within 1.0 C RMS of the physics line average (cache T_avg_line_phys)",
           rms_new < 1.0 and len(ta_new) == 1441 and np.allclose(ta_new, expect, atol=1e-9),
           "RMS %.3f C after relabel (%.3f C before, s=%.4f)" % (rms_new, rms_old, s))
    try:
        ta9, _ = tavg_of(tr1, NEW18_C9)
        z9 = np.load(os.path.join(CACHE, NEW18_C9.replace("/", "__") + ".npz"))
        rms9 = float(np.sqrt(np.mean((ta9 - z9["T_avg_line_phys"]) ** 2)))
        ta9o, _ = tavg_of(tr0, NEW18_C9)
        rms9o = float(np.sqrt(np.mean((ta9o - z9["T_avg_line_phys"]) ** 2)))
        print("      (info) Case 9/New 18: RMS vs physics line %.3f C after relabel (%.3f C before)" % (rms9, rms9o))
    except Exception as e:
        print("      (info) Case 9/New 18 not compared:", e)

    # (b) a line-average file is bit-identical; test files untouched; relabel count
    line_key = next(k for k in sorted(spec["train"]) if k not in table)
    la_new, _ = tavg_of(tr1, line_key)
    la_old, _ = tavg_of(tr0, line_key)
    report("(b) a line-average training file (%s) is bit-identical after the hook" % line_key,
           np.array_equal(la_new, la_old))
    n_changed = sum(1 for (d1, c1), (d0, c0) in zip(tr1 + va1, tr0 + va0)
                    if not np.array_equal(d1["T_avg (C)"].to_numpy(), d0["T_avg (C)"].to_numpy()))
    report("(b) exactly 118 train+val files changed (115 train + 3 val), walls untouched",
           n_changed == 118 and all(np.array_equal(d1["T_inner (C)"].to_numpy(), d0["T_inner (C)"].to_numpy())
                                    and np.array_equal(d1["T_outer (C)"].to_numpy(), d0["T_outer (C)"].to_numpy())
                                    for (d1, _), (d0, _) in zip(tr1 + va1, tr0 + va0)),
           "changed %d" % n_changed)
    report("(b) sealed test CSVs excluded from both baseline and relabel loading",
           not te0 and not te1)
    # (the global T_avg min / max is set by line-average files, so the scaler
    #  range itself does not discriminate; check the order in the source and
    #  that the fitted range is exactly that of the RELABELLED training data)
    import inspect
    src = inspect.getsource(load_all_data)
    ta_all = np.concatenate([d["T_avg (C)"].to_numpy(dtype=np.float64) for d, _ in tr1])
    report("(b) relabel happens before the scaler fit (source order; fitted T_avg range == relabelled training data)",
           src.index("relabel_tavg(") < src.index("Fitting scaler")
           and sc1.data_min_[3] == ta_all.min() and sc1.data_max_[3] == ta_all.max(),
           "T_avg range %.2f..%.2f (unrelabelled %.2f..%.2f)" % (sc1.data_min_[3], sc1.data_max_[3], sc0.data_min_[3], sc0.data_max_[3]))
    report("(b) the other scaler columns are untouched; mask routing stays keyed by file",
           config.TAVG_MASK == "" and np.array_equal(sc1.data_min_[[0, 1, 2, 4]], sc0.data_min_[[0, 1, 2, 4]])
           and np.array_equal(sc1.data_max_[[0, 1, 2, 4]], sc0.data_max_[[0, 1, 2, 4]]))

    # (c) side-reference artefact
    side_p = os.path.join(HERE, "splits", "hot_discharge_side_reference_vs3.csv")
    ok_c = os.path.isfile(side_p)
    detail = "missing"
    if ok_c:
        side = pd.read_csv(side_p)
        counts = side.groupby("key").size().to_dict()
        ok_c = (set(counts) == set(SIDE_KEYS) and all(v == 1441 for v in counts.values())
                and {"time_s", "T_inner", "T_outer", "T_avg_label", "inner_share_pct", "T_avg_line_sinv"} <= set(side.columns))
        # recompute from the relabelled validation dfs
        worst = 0.0
        for k in SIDE_KEYS:
            ta_k, _ = tavg_of(va1, k)
            worst = max(worst, float(np.abs(side[side["key"] == k]["T_avg_line_sinv"].to_numpy() - ta_k).max()))
        ok_c = ok_c and worst < 1e-5
        detail = "rows %s, max |csv - hook| %.1e C" % (counts, worst)
    report("(c) splits/hot_discharge_side_reference_vs3.csv: the three series exist, 1441 rows each, equal to the hook's T_line", ok_c, detail)

    # (d) tag, run_config, KEYS, export family
    report("(d) RELABEL_CSV set: run name carries 'rl'", "_rl" in config.RUN_NAME_BASE and config.RELABEL_TAG == "rl",
           config.RUN_NAME_BASE[-40:])
    from tes_gru import runio as _runio
    from tes_gru.utils import sha256_file
    config.RUN_NAME = "_check_a4_tmp"
    try:
        _runio.ensure_run_layout()
        rc = json.load(open(_runio.save_run_config_snapshot()))
        report("(d) run_config.json: relabel_csv, relabel_sha256 (matches the file), relabel_n_train 115 / n_val 3",
               rc.get("relabel_csv") == LABELS and rc.get("relabel_sha256") == sha256_file(config.RELABEL_CSV_PATH)
               and rc.get("relabel_n_train") == 115 and rc.get("relabel_n_val") == 3)
    finally:
        shutil.rmtree(_runio.run_dir(), ignore_errors=True)
    env = ec.env_from_config(rc)
    report("(d) eval_checkpoint KEYS / env_from_config pass RELABEL_CSV through; absent key -> not set",
           ec.KEYS.get("relabel_csv") == "RELABEL_CSV" and env.get("RELABEL_CSV") == LABELS
           and "RELABEL_CSV" not in ec.env_from_config(dict(rc, relabel_csv=None)))
    import export_results as ex
    fam = ex.family(rc)
    report("(d) export_results family carries rl (and vs3)", "vs3" in fam and "_rl" in fam, fam)
    report("(d) legacy run_config (no relabel key) -> no rl in family", "rl" not in ex.family(dict(rc, relabel_csv=None)))

    n_ok = sum(RESULTS)
    print("\ncheck_a4: %d/%d PASS" % (n_ok, len(RESULTS)))
    sys.exit(0 if n_ok == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
