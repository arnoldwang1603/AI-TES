"""Evaluate a saved checkpoint of an existing run on the fixed test set --
no retraining. Written (2026-09-19) to test the early-stopping hypothesis
behind round 8's T_outer-anchor failure by scoring the final-epoch weights.

That test turned out to be impossible with the saved files: `train_model`
restores the best checkpoint before returning, and main.py writes
`gru_variant_<v>.pth` after that, so the "final" file is byte-identical to
`best_gru_variant_<v>.pth` in every run so far (checked on round 8: max
|diff| = 0, and all 20 EMA-arm seeds re-score to exactly their meta.json
numbers). No final-epoch state exists on disk. The script stays useful for
re-scoring a run under a changed test set or evaluation code, and for the
self-check that `fit_model_constants` + a checkpoint reproduce a run's
recorded metrics (it does, to 3 decimals).

Usage (from this folder):
    python eval_checkpoint.py <run_dir> [--which final|best] [--out results.json]
    python eval_checkpoint.py "<runs root>" --glob "*_To-em1200_cgate_ah_seed*" --which final

The run's own run_config.json supplies every knob: the script re-exports them
into the environment and re-imports tes_gru, so the model, the data split and
the fitted constants (fit_model_constants) are exactly those of the original
run. It never writes into the run directory; per-case metrics go to --out.
"""
import argparse
import csv
import glob
import importlib
import json
import os
import re
import statistics as st
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
FLAGGED = set(range(32, 41)) | {51, 52, 53}
SWITCHED = FLAGGED | {50, 58} | set(range(59, 66))

# run_config.json key -> env var name (only knobs that config.py reads)
KEYS = {
    "tinner_mode": "TINNER_MODE", "anchor_lead": "ANCHOR_LEAD",
    "physics_bound_weight": "PHYSICS_BOUND_WEIGHT", "other_ch_mode": "OTHER_CH_MODE",
    "input_lookahead": "INPUT_LOOKAHEAD", "anchor_scale": "ANCHOR_SCALE",
    "pos_gap_floor": "POS_GAP_FLOOR", "pos_case_gate": "POS_CASE_GATE",
    "pos_floor_soft": "POS_FLOOR_SOFT", "pos_temp_gate": "POS_TEMP_GATE",
    "pos_temp_soft": "POS_TEMP_SOFT", "pos_abs_head": "POS_ABS_HEAD",
    "pos_learned_gate": "POS_LEARNED_GATE", "pos_gate_bias": "POS_GATE_BIAS",
    "pos_learned_pool": "POS_LEARNED_POOL", "pos_anchored_fallback": "POS_ANCHORED_FALLBACK",
    "pos_fit_clean": "POS_FIT_CLEAN", "case_flag_input": "CASE_FLAG_INPUT",
    "exclude_both_phase": "EXCLUDE_BOTH_PHASE", "touter_mode": "TOUTER_MODE",
    "touter_tau": "TOUTER_TAU", "touter_scale": "TOUTER_SCALE", "touter_ema_form": "TOUTER_EMA_FORM",
    "window_size": "WINDOW_SIZE", "sliding_pad_mode": "SLIDING_PAD_MODE",
    "model_split": "MODEL_SPLIT", "pos_time_prior": "POS_TIME_PRIOR",
}


def env_from_config(c):
    env = {}
    for k, var in KEYS.items():
        if k in c:
            v = c[k]
            env[var] = ("1" if v else "0") if isinstance(v, bool) else str(v)
    env["LOSS_WEIGHTS"] = ",".join("%g" % w for w in c["loss_weights"])
    env["VARIANTS"] = ",".join(c["variants"])
    lp = c["latest_params"]
    env["HIDDEN_SIZE"] = str(lp["hidden_size"]); env["NUM_LAYERS"] = str(lp["num_layers"])
    env["DROPOUT"] = str(lp["dropout"]); env["MAX_EPOCHS"] = str(lp.get("max_epochs", 1000))
    d = c.get("data", {})
    env["VAL_SPLIT"] = d.get("val_split", "legacy"); env["VAL_PER_TYPE"] = str(d.get("val_per_type", 2))
    env["TRAIN_EXCLUDE_TYPE"] = str(d.get("train_exclude_type", "") or "")
    env["SEEDS"] = str(c["seed"])
    return env


def evaluate_one(run_dir, which):
    c = json.load(open(os.path.join(run_dir, "run_config.json")))
    env = env_from_config(c)
    # Re-import tes_gru under this run's knobs (config.py reads env at import).
    # Knobs the run's config does not name (older runs predate them) are
    # REMOVED from the environment first, so they fall back to config.py's
    # defaults instead of whatever the shell or the previous --glob run set.
    for var in list(KEYS.values()) + ["LOSS_WEIGHTS", "VARIANTS", "HIDDEN_SIZE", "NUM_LAYERS",
                                      "DROPOUT", "MAX_EPOCHS", "VAL_SPLIT", "VAL_PER_TYPE", "TRAIN_EXCLUDE_TYPE", "SEEDS"]:
        os.environ.pop(var, None)
    for k, v in env.items():
        os.environ[k] = v
    for m in [m for m in list(sys.modules) if m.startswith("tes_gru")]:
        del sys.modules[m]
    sys.path.insert(0, HERE)
    import torch
    from tes_gru import config
    from tes_gru.data import load_all_data, ThermalDataset
    from tes_gru.models import build_model
    from tes_gru.rollout import variant_to_encoder
    from tes_gru.train import fit_model_constants
    from tes_gru.evaluate import test_model
    import tes_gru.evaluate as _ev
    from tes_gru.config import DEVICE, LATEST_PARAMS, set_seed
    variant = c["variants"][0]
    set_seed(c["seed"])
    train_dfs, val_dfs, test_dfs, scaler = load_all_data()
    model = build_model(variant, LATEST_PARAMS, variant_to_encoder(variant)).to(DEVICE)
    fit_model_constants(model, variant, train_dfs, scaler)
    ck = os.path.join(run_dir, "checkpoints",
                      ("gru_variant_%s.pth" if which == "final" else "best_gru_variant_%s.pth") % variant)
    model.load_state_dict(torch.load(ck, map_location=DEVICE))
    model.eval()
    test_sets = [ThermalDataset(df, variant, scaler=scaler, file_name=fn) for df, fn in test_dfs]
    out_dir = tempfile.mkdtemp(prefix="evalckpt_")
    # test_model writes plots/CSV into output_dir but predictions.npz into
    # runs/<config.RUN_NAME>/variants/<v>/ -- switch that write off and point
    # RUN_NAME at a throwaway name so the original run dir is never touched
    # (it was, before 2026-09-19: the re-score overwrote predictions.npz when
    # the run lived under runs/, and left stray runs/<name>/ dirs otherwise).
    _ev.SAVE_PREDICTIONS = False
    config.RUN_NAME = "_evalckpt_" + os.path.basename(out_dir)
    summary_df, _ = test_model(variant, model, test_sets, output_dir=out_dir,
                               set_name="ckpt_%s" % which)
    rows = summary_df[summary_df["Case"] != "AVERAGE"]
    per = {}
    for _, r in rows.iterrows():
        m = re.search(r"\((\d+)\)", str(r["CaseFile"]))
        if m:
            per[int(m.group(1))] = {ch: float(r["MAE_%s (C)" % ch]) for ch in ("T_inner", "T_outer", "T_avg")}
    def sub(ids): return st.mean(per[i]["T_avg"] for i in per if i in ids)
    res = dict(run=os.path.basename(run_dir), seed=c["seed"], which=which,
               T_inner=st.mean(v["T_inner"] for v in per.values()),
               T_outer=st.mean(v["T_outer"] for v in per.values()),
               T_avg=st.mean(v["T_avg"] for v in per.values()))
    res["overall"] = (res["T_inner"] + res["T_outer"] + res["T_avg"]) / 3
    res["flagged12_T_avg"] = sub(FLAGGED); res["switched21_T_avg"] = sub(SWITCHED)
    res["rest49_T_avg"] = st.mean(per[i]["T_avg"] for i in per if i not in SWITCHED)
    th = json.load(open(os.path.join(run_dir, "variants", variant, "train_history.json")))
    res["best_epoch"] = th["best_epoch"]; res["total_epoch"] = th["total_epoch"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="a run dir, or a runs root when --glob is given")
    ap.add_argument("--glob", default=None)
    ap.add_argument("--which", default="final", choices=["final", "best"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dirs = sorted(glob.glob(os.path.join(a.path, a.glob))) if a.glob else [a.path]
    results = []
    for d in dirs:
        r = evaluate_one(d, a.which)
        results.append(r)
        print("%-90s seed %3d %s: overall %.3f  Tout %.3f  Tavg %.3f  sw21 %.2f  r49 %.2f  (best ep %d / total %d)" % (
            r["run"][-90:], r["seed"], a.which, r["overall"], r["T_outer"], r["T_avg"],
            r["switched21_T_avg"], r["rest49_T_avg"], r["best_epoch"], r["total_epoch"]), flush=True)
    if a.out:
        json.dump(results, open(a.out, "w"), indent=1)
    if len(results) > 1:
        print("\nMEAN over %d seeds (%s checkpoint): overall %.3f  T_outer %.3f  T_avg %.3f  switched21 %.2f  rest49 %.2f" % (
            len(results), a.which, st.mean(r["overall"] for r in results), st.mean(r["T_outer"] for r in results),
            st.mean(r["T_avg"] for r in results), st.mean(r["switched21_T_avg"] for r in results),
            st.mean(r["rest49_T_avg"] for r in results)))


if __name__ == "__main__":
    main()
