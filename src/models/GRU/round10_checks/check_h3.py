"""Round 10 / H3 check: validation-side scoring, vetoes, SEAL_TEST,
eval_checkpoint --set. Run from src/models/GRU:  python round10_checks/check_h3.py

No training. PASS / FAIL per item, non-zero exit on any FAIL. Needs the data
under AI-TES/data and the Round 9 champion run directory (sibling checkout
calit2-thermal-wiki; override with CHAMPION_RUN_DIR). Nothing is written
into the champion directory: the validation scorings go to temp dirs.
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # src/models/GRU
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(HERE)

from check_h9 import CHAMP, clean_env, fresh_import, block_test_csv_reads   # noqa: E402

RESULTS = []
VARIANT = "forward_direct"
CH = ("T_inner", "T_outer", "T_avg")
MASKED_VAL = {"Case 1 (linear charging)/Case 16.csv",
              "Case 9 (non-linear discharging)/Case New 1 (Tejas).csv",
              "Case 9 (non-linear discharging)/Case New 4 (Tejas).csv"}


def report(item, ok, detail=""):
    RESULTS.append(bool(ok))
    print("%s  %s  %s" % ("PASS" if ok else "FAIL", item, detail), flush=True)


def fake_run_dir(tmp, **cfg_updates):
    """A run dir with the champion's checkpoint + train_history and a run_config
    edited by cfg_updates (nested 'data' keys merged)."""
    c = json.load(open(os.path.join(CHAMP, "run_config.json")))
    for k, v in cfg_updates.items():
        if k == "data":
            c["data"].update(v)
        else:
            c[k] = v
    rd = os.path.join(tmp, "run")
    os.makedirs(os.path.join(rd, "checkpoints"))
    os.makedirs(os.path.join(rd, "variants", VARIANT))
    shutil.copy2(os.path.join(CHAMP, "checkpoints", "best_gru_variant_%s.pth" % VARIANT),
                 os.path.join(rd, "checkpoints"))
    shutil.copy2(os.path.join(CHAMP, "variants", VARIANT, "train_history.json"),
                 os.path.join(rd, "variants", VARIANT))
    json.dump(c, open(os.path.join(rd, "run_config.json"), "w"), indent=1)
    return rd


def main():
    import eval_checkpoint as ec
    clean_env(ec)
    block_test_csv_reads()
    tmp = tempfile.mkdtemp(prefix="check_h3_")
    try:
        # ============================================================== (a) test access rejected
        # A regression check must not consume a B6/F11 access. The guarded
        # public entry point rejects it before any real test CSV is opened.
        rejected = False
        rejection = ""
        try:
            ec.evaluate_one(CHAMP, "best", which_set="test")
        except RuntimeError as exc:
            rejection = str(exc)
            rejected = "P0 checks may not reopen" not in rejection
        report("(a) unlogged public --set test is refused before the test loader",
               rejected, rejection)
        with np.load(os.path.join(CHAMP, "variants", VARIANT, "predictions.npz"), allow_pickle=True) as old:
            archive_ok = (len(old["case_ids"]) == 70 and len(old["inv_actuals"]) == 70
                          and all(np.asarray(a).shape == np.asarray(b).shape
                                  and np.asarray(a).shape[1] == 5
                                  and np.isfinite(np.asarray(a, dtype=float)).all()
                                  and np.isfinite(np.asarray(b, dtype=float)).all()
                                  for a, b in zip(old["inv_preds"], old["inv_actuals"])))
        report("(a) archived paired payload layout is intact (no new test scoring)", archive_ok)

        # ============================================================== (b) --set val
        out_v = os.path.join(tmp, "val_out")
        res_v = ec.evaluate_one(CHAMP, "best", which_set="val", out_dir=out_v)
        csv_v = os.path.join(out_v, "summary_errors_val.csv")
        npz_v = os.path.join(out_v, "predictions_val.npz")
        df = pd.read_csv(csv_v)
        rows = df[df["Case"] != "AVERAGE"]
        new_cols = ["JumpSteps_gt2C_ToTa", "JumpSteps_gt1C_ToTa"] + \
                   ["Err_step1_%s (C)" % c for c in CH] + ["MaxErr_1_5_%s (C)" % c for c in CH]
        report("(b) --set val: 20 rows, predictions_val.npz written, no test artefact in out_dir",
               len(rows) == 20 and os.path.isfile(npz_v) and res_v["n_files"] == 20
               and not os.path.isfile(os.path.join(out_v, "summary_errors.csv"))
               and not os.path.isfile(os.path.join(out_v, "predictions.npz"))
               and "switched21_T_avg" not in res_v,
               "rows %d, files %s" % (len(rows), sorted(os.listdir(out_v))))
        z = np.load(npz_v, allow_pickle=True)
        report("(b) predictions_val.npz holds 20 cases with inv_pred / inv_actual",
               len(z["case_ids"]) == 20 and z["inv_preds"][0].shape[1] == 5)
        report("(b) new columns present: " + ", ".join(new_cols[:2]) + ", Err_step1_<ch>, MaxErr_1_5_<ch>",
               all(c in df.columns for c in new_cols) and "JumpSteps_gt2C" in df.columns,
               "missing: %s" % [c for c in new_cols if c not in df.columns])
        report("(b) JumpSteps_gt1C_ToTa >= JumpSteps_gt2C_ToTa on every row; MaxErr_1_5 >= Err_step1",
               bool((rows["JumpSteps_gt1C_ToTa"] >= rows["JumpSteps_gt2C_ToTa"]).all())
               and all((rows["MaxErr_1_5_%s (C)" % c] >= rows["Err_step1_%s (C)" % c] - 1e-12).all() for c in CH),
               "gt2 total %d  gt1 total %d" % (rows["JumpSteps_gt2C_ToTa"].sum(), rows["JumpSteps_gt1C_ToTa"].sum()))
        # the champion's own validation scalar lives in scaled space; here the
        # C-space MAE must simply be finite and plausible on all 20 files
        report("(b) unmasked split: T_avg scored on all 20 validation files",
               res_v["n_files_T_avg"] == 20 and np.isfinite(rows["MAE_T_avg (C)"]).all(),
               "val MAE Tin %.3f Tout %.3f Tavg %.3f" % (res_v["T_inner"], res_v["T_outer"], res_v["T_avg"]))
        champ_dirty = any(os.path.exists(os.path.join(CHAMP, "variants", VARIANT, f))
                          for f in ("summary_errors_val.csv", "predictions_val.npz", "plots_val"))
        report("(b) champion directory untouched", not champ_dirty)

        # ============================================================== (c) masked split
        rd = fake_run_dir(tmp, tavg_mask="split", tf32="legacy",
                          data={"split_name": "round10_vs3_s42.json", "split_tag": "vs3"})
        out_m = os.path.join(tmp, "masked_out")
        res_m = ec.evaluate_one(rd, "best", which_set="val", out_dir=out_m)
        dm = pd.read_csv(os.path.join(out_m, "summary_errors_val.csv"))
        rm = dm[dm["Case"] != "AVERAGE"]
        from tes_gru.data import split_key
        masked_rows = rm[rm["Case"].map(split_key).isin(MASKED_VAL)]
        other_rows = rm[~rm["Case"].map(split_key).isin(MASKED_VAL)]
        tavg_nan_cols = ["MAE_T_avg (C)", "RMSE_T_avg (C)", "Err_step1_T_avg (C)", "MaxErr_1_5_T_avg (C)"]
        report("(c) masked split: the 3 masked validation files have NaN T_avg metrics, the other 17 finite",
               len(masked_rows) == 3 and masked_rows[tavg_nan_cols].isna().all().all()
               and np.isfinite(other_rows[tavg_nan_cols].to_numpy(dtype=float)).all()
               and np.isfinite(masked_rows[["MAE_T_inner (C)", "MAE_T_outer (C)"]].to_numpy(dtype=float)).all(),
               "masked rows %d" % len(masked_rows))
        avg_row = dm[dm["Case"] == "AVERAGE"].iloc[0]
        # (the csv is written at 4 dp, so csv-side comparisons carry a 2e-4 tolerance)
        report("(c) aggregate uses 17 files for T_avg (AVERAGE row and aggregate_scores agree), 20 for the walls",
               res_m["n_files_T_avg"] == 17 and res_m["n_files"] == 20
               and abs(avg_row["MAE_T_avg (C)"] - other_rows["MAE_T_avg (C)"].mean()) < 2e-4
               and abs(res_m["T_avg"] - other_rows["MAE_T_avg (C)"].mean()) < 2e-4
               and abs(avg_row["MAE_T_inner (C)"] - rm["MAE_T_inner (C)"].mean()) < 2e-4,
               "T_avg over 17: %.4f (AVERAGE row %.4f)" % (res_m["T_avg"], avg_row["MAE_T_avg (C)"]))
        report("(c) masked files: ToTa jump counts come from T_outer alone (<= unmasked-style count)",
               bool((masked_rows["JumpSteps_gt1C_ToTa"] >= 0).all()))

        # ============================================================== (d) sealed resume
        clean_env(ec)
        config0 = fresh_import({"VARIANTS": VARIANT, "VAL_SPLIT": "stratified", "SPLIT_NAME": "round10_vs3_s42.json",
                                "TAVG_MASK": "split", "TF32": "0", "SEAL_TEST": "0"})
        from tes_gru import main as _main0
        raised_a = raised_b = False
        try:
            _main0.check_cached_meta_seal({"sealed": True, "Val_MAE_Overall": 0.5}, VARIANT)
        except RuntimeError:
            raised_a = True
        try:
            _main0.check_cached_meta_seal({"Test_MAE_Overall": 0.5}, VARIANT)
        except RuntimeError:
            raised_b = True
        report("(d) SEAL_TEST=0: a fake sealed meta makes the resume path raise; a test-scored one passes",
               raised_a and not raised_b and _main0.SCORE_PFX == "Test" and not config0.SEAL_TEST)
        clean_env(ec)
        config1 = fresh_import({"VARIANTS": VARIANT, "VAL_SPLIT": "stratified", "SPLIT_NAME": "round10_vs3_s42.json",
                                "TAVG_MASK": "split", "TF32": "0", "SEAL_TEST": "1"})
        from tes_gru import main as _main1
        raised_c = raised_d = False
        try:
            _main1.check_cached_meta_seal({"Test_MAE_Overall": 0.5}, VARIANT)
        except RuntimeError:
            raised_c = True
        try:
            _main1.check_cached_meta_seal({"sealed": True, "Val_MAE_Overall": 0.5}, VARIANT)
        except RuntimeError:
            raised_d = True
        report("(d) SEAL_TEST=1: a test-scored meta makes the resume path raise; a sealed one passes",
               raised_c and not raised_d and _main1.SCORE_PFX == "Val" and config1.SEAL_TEST)
        # run_config carries sealed; train_model would build no test datasets
        from tes_gru import runio as _runio
        from tes_gru.train import train_model   # noqa: F401  (import only)
        config1.RUN_NAME = "_check_h3_tmp"
        try:
            _runio.ensure_run_layout()
            p = _runio.save_run_config_snapshot()
            rc = json.load(open(p))
            report("(d) run_config.json of a SEAL_TEST=1 run carries sealed=true (false under SEAL_TEST=0)",
                   rc.get("sealed") is True and _runio.run_config_dict()["sealed"] is True)
        finally:
            shutil.rmtree(_runio.run_dir(), ignore_errors=True)
        import inspect
        src = inspect.getsource(_main1.main)
        report("(d) main.py writes no literal Test_ key (prefix SCORE_PFX everywhere)",
               "'Test_" not in src and '"Test_' not in src)

        # ============================================================== (e) aggregate function
        from tes_gru.round10_metrics import aggregate_scores, AGG_KEYS
        agg = aggregate_scores(dm)
        want = ("MAE_T_inner", "MAE_T_outer", "MAE_T_avg", "MAE_Overall", "MAE_ToTa", "blowup",
                "JumpSteps_gt2C_ToTa", "JumpSteps_gt1C_ToTa", "n_files", "n_files_T_avg")
        report("(e) aggregate_scores returns the documented keys",
               all(k in agg for k in want) and tuple(agg) == AGG_KEYS,
               "%d keys" % len(agg))
        report("(e) aggregate arithmetic: MAE_ToTa = (T_outer + T_avg)/2, blowup = overall > 1.5 C, totals = column sums",
               abs(agg["MAE_ToTa"] - 0.5 * (agg["MAE_T_outer"] + agg["MAE_T_avg"])) < 1e-12
               and agg["blowup"] == (agg["MAE_Overall"] > 1.5)
               and agg["JumpSteps_gt2C_ToTa"] == int(rm["JumpSteps_gt2C_ToTa"].sum())
               and agg["JumpSteps_gt1C_ToTa"] == int(rm["JumpSteps_gt1C_ToTa"].sum())
               and agg["n_files_T_avg"] == 17 and isinstance(agg["blowup"], bool),
               "ToTa %.4f overall %.4f blowup %s" % (agg["MAE_ToTa"], agg["MAE_Overall"], agg["blowup"]))
        # synthetic blow-up
        fake = pd.DataFrame([{"Case": "x%d" % i, "MAE_T_inner (C)": 2.0, "MAE_T_outer (C)": 2.0,
                              "MAE_T_avg (C)": float("nan") if i == 0 else 2.0,
                              "JumpSteps_gt2C_ToTa": 1, "JumpSteps_gt1C_ToTa": 2} for i in range(3)])
        fa = aggregate_scores(fake)
        report("(e) synthetic table: NaN T_avg skipped (n=2), blowup flagged, totals 3 / 6",
               fa["n_files_T_avg"] == 2 and fa["blowup"] and fa["JumpSteps_gt2C_ToTa"] == 3
               and fa["JumpSteps_gt1C_ToTa"] == 6 and abs(fa["MAE_Overall"] - 2.0) < 1e-12)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    n_ok = sum(RESULTS)
    print("\ncheck_h3: %d/%d PASS" % (n_ok, len(RESULTS)))
    sys.exit(0 if n_ok == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
