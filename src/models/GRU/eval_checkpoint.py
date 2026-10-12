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
    python eval_checkpoint.py <run_dir> [--which final|best] [--set test|val] [--out results.json]
    python eval_checkpoint.py "<runs root>" --glob "*_To-em1200_cgate_ah_seed*" --which final

The run's own run_config.json supplies every knob: the script re-exports them
into the environment and re-imports tes_gru, so the model, the data split and
the fitted constants are exactly those of the original run. Round 10:
  --set test   (default) the fixed test set; nothing is written into the run
               directory (a scratch dir unless --out-dir).
  --set val    the run's OWN validation files (its SPLIT_NAME / TAVG_MASK;
               masked files are not scored on T_avg) -> summary_errors_val.csv
               + predictions_val.npz next to the run's test artefacts (never
               overwriting them), or under --out-dir. The FLAGGED / SWITCHED
               test-id subsets are never applied to validation files.
  constants    the H4 sidecar checkpoints/best_gru_variant_<v>.constants.json
               is preferred (scaler rebuilt, constants set, no refit from the
               training CSVs); without it the constants are refitted with a
               one-line warning. Either way they come from the checkpoint's
               own split, never from the caller's.
Aggregates per run go to --out (json).
"""
import argparse
import csv
import glob
import importlib
import json
import os
import re
import shutil
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
    "tf32": "TF32",                                     # H9 (Round 10)
    "tavg_mask": "TAVG_MASK",                           # H2 (Round 10)
    "relabel_csv": "RELABEL_CSV",                       # A4 (Round 10)
    "loss_w0_mode": "LOSS_W0_MODE",
    "enc_widths": "ENC_WIDTHS", "h0_mode": "H0_MODE", "init_encoder": "INIT_ENCODER",
    "feat_cols": "FEAT_COLS",
    "early_step_weight": "EARLY_STEP_WEIGHT",
    "ratio_channels": "RATIO_CHANNELS", "ratio_ref": "RATIO_REF",
    "phys_mode": "PHYS_MODE", "physics_cache_dir": "PHYSICS_CACHE_DIR",
    "init_burnin_steps": "INIT_BURNIN_STEPS", "synthetic_rest_files": "SYNTHETIC_REST_FILES",
    "constants_from": "FROZEN_CONSTANTS",
}


def env_from_config(c):
    env = {}
    for k, var in KEYS.items():
        if k in c and c[k] is not None:
            v = c[k]
            if k == 'physics_cache_dir' and not os.path.isdir(str(v)) and os.path.isdir(os.environ.get('PHYSICS_CACHE_DIR', '')):
                v = os.environ['PHYSICS_CACHE_DIR']
            env[var] = ("1" if v else "0") if isinstance(v, bool) else str(v)
    env["LOSS_WEIGHTS"] = ",".join("%g" % w for w in c["loss_weights"])
    env["VARIANTS"] = ",".join(c["variants"])
    lp = c["latest_params"]
    env["HIDDEN_SIZE"] = str(lp["hidden_size"]); env["NUM_LAYERS"] = str(lp["num_layers"])
    env["DROPOUT"] = str(lp["dropout"]); env["MAX_EPOCHS"] = str(lp.get("max_epochs", 1000))
    if "tf32" not in c:
        # Pre-Round-10 run: trained and scored at the torch default (matmul
        # fp32, cuDNN TF32), which TF32=legacy reproduces bit-for-bit (H9).
        env["TF32"] = "legacy"
    d = c.get("data", {})
    env["VAL_SPLIT"] = d.get("val_split", "legacy"); env["VAL_PER_TYPE"] = str(d.get("val_per_type", 2))
    env["TRAIN_EXCLUDE_TYPE"] = str(d.get("train_exclude_type", "") or "")
    if d.get("split_name"):                      # H1: the run's own split json
        env["SPLIT_NAME"] = str(d["split_name"])
    env["SEEDS"] = str(c["seed"])
    return env


ENV_VARS_EXTRA = ["LOSS_WEIGHTS", "VARIANTS", "HIDDEN_SIZE", "NUM_LAYERS", "DROPOUT", "MAX_EPOCHS",
                  "VAL_SPLIT", "VAL_PER_TYPE", "TRAIN_EXCLUDE_TYPE", "SEEDS", "SPLIT_NAME", "TAVG_MASK",
                  "SEAL_TEST", "RELABEL_CSV", "CONSTANTS_FROM"]


def checkpoint_path(run_dir, variant, which):
    return os.path.join(run_dir, "checkpoints",
                        ("gru_variant_%s.pth" if which == "final" else "best_gru_variant_%s.pth") % variant)


def constants_sidecar_path(run_dir, variant):
    """H4: the constants sidecar written next to the best checkpoint. The
    final-state checkpoint shares it (the constants do not depend on the
    weights)."""
    return os.path.join(run_dir, "checkpoints", "best_gru_variant_%s.constants.json" % variant)


def load_or_refit_constants(model, variant, run_dir, train_dfs=None, scaler=None):
    """Attach the data-derived constants to `model` and return
    (scaler, source). Prefers the H4 sidecar (scaler rebuilt from its numbers,
    constants set, the training CSVs never read); falls back to
    fit_model_constants on train_dfs / scaler -- the checkpoint's OWN split,
    which the caller obtained under the run's SPLIT_NAME -- with a warning."""
    from tes_gru.runio import load_constants_sidecar
    from tes_gru.data import scaler_from_sidecar
    from tes_gru.train import fit_training_constants, set_model_constants
    sc_path = constants_sidecar_path(run_dir, variant)
    if os.path.isfile(sc_path):
        d = load_constants_sidecar(sc_path)
        set_model_constants(model, d["constants"])
        print("[constants] sidecar %s (written %s)" % (sc_path, d.get("written_at")))
        return scaler_from_sidecar(d), "sidecar"
    if train_dfs is None or scaler is None:
        raise RuntimeError("no constants sidecar at %s and no training data to refit from" % sc_path)
    print("WARNING: no constants sidecar at %s -- refitting the constants from the training CSVs "
          "of the run's own split" % sc_path)
    fit_training_constants(model, variant, train_dfs, scaler)
    return scaler, "refit"


def load_run_model(run_dir, which="best"):
    """Inference API: restore a checkpoint and its own constants.

    A sidecar needs no data reads. Legacy checkpoints refit on their own
    training/validation split, with the test set strictly excluded.
    """
    c = json.load(open(os.path.join(run_dir, "run_config.json")))
    env = env_from_config(c)
    for var in list(KEYS.values()) + ENV_VARS_EXTRA:
        os.environ.pop(var, None)
    os.environ.update(env)
    os.environ["SEAL_TEST"] = "1"
    for name in [n for n in list(sys.modules) if n.startswith("tes_gru")]:
        del sys.modules[name]
    import torch
    from tes_gru.config import DEVICE, LATEST_PARAMS, set_seed
    from tes_gru.models import build_model
    from tes_gru.rollout import variant_to_encoder
    from tes_gru.data import load_all_data
    variant = c["variants"][0]
    set_seed(c["seed"])
    model = build_model(variant, LATEST_PARAMS, variant_to_encoder(variant)).to(DEVICE)
    if os.path.isfile(constants_sidecar_path(run_dir, variant)):
        scaler, source = load_or_refit_constants(model, variant, run_dir)
    else:
        train, _, _, scaler = load_all_data(include_test=False)
        scaler, source = load_or_refit_constants(model, variant, run_dir, train, scaler)
    try:
        state = torch.load(checkpoint_path(run_dir, variant, which), map_location=DEVICE, weights_only=False)
    except TypeError:
        state = torch.load(checkpoint_path(run_dir, variant, which), map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model, scaler, c


def t0_sweep(run_dir, sweep="50:600:10", which="best", out_dir=None):
    """G7 synthetic constant-inlet probe, 60 predicted steps; no test reads."""
    import numpy as np
    import pandas as pd
    import torch
    model, scaler, c = load_run_model(run_dir, which)
    from tes_gru.config import DEVICE, PHYS_MODE
    from tes_gru.data import ThermalDataset
    from tes_gru.rollout import run_rollout_train
    parts = [float(v) for v in sweep.split(":")]
    if len(parts) != 3 or parts[2] <= 0 or parts[1] < parts[0]:
        raise ValueError("t0 sweep must be start:stop:positive_step")
    rows = []
    for t0 in np.arange(parts[0], parts[1] + parts[2] * .5, parts[2]):
        frame = pd.DataFrame({"Time (s)": np.arange(61) * 10.0})
        for name in ThermalDataset.BASE_COLS[1:]:
            frame[name] = t0
        physical = {'time': frame['Time (s)'].to_numpy(float), 'T_outer_phys': np.full(61, t0),
                    'T_avg_line_phys': np.full(61, t0)} if PHYS_MODE else None
        ds = ThermalDataset(frame, c["variants"][0], scaler=scaler, file_name="synthetic_t0_%g" % t0,
                            physics_override=physical)
        with torch.no_grad():
            p = run_rollout_train(model, c["variants"][0], ds.X.to(DEVICE), ds.Y.to(DEVICE),
                                  ds.init_conditions.to(DEVICE), ds.window_obs.to(DEVICE),
                                  inp_last=ds.inp_last.to(DEVICE),
                                  phys_targets=ds.physics_targets.to(DEVICE) if PHYS_MODE else None)[0].cpu().numpy()
        for j, ch in enumerate(("T_inner", "T_outer", "T_avg")):
            k = ThermalDataset.BASE_COLS.index(ch + " (C)")
            err = (p[:, j] - scaler.min_[k]) / scaler.scale_[k] - t0
            rows.append(dict(seed=c["seed"], T0=float(t0), channel=ch,
                             Err_step1=float(abs(err[0])), MaxErr_1_5=float(abs(err[:5]).max()),
                             MaxOffset_60=float(abs(err).max()), MeanOffset_60=float(err.mean()),
                             Spike_gt1C=float(abs(err[:5]).max() > 1)))
    df = pd.DataFrame(rows)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(os.path.join(out_dir, "t0_sweep.csv"), index=False)
    return df


def evaluate_one(run_dir, which, which_set="val", out_dir=None, physics_overrides=None):
    """Re-score one run's checkpoint on `which_set` ("test" | "val"); see the
    module docstring. Returns the aggregate dict (round10_metrics keys plus
    the test-id subsets for the test set).

    F11 physics checkpoints require physics_overrides keyed by the exact
    test dataset filename. Every value is a full-length from_frame dictionary
    with time/T_outer_phys/T_avg_line_phys. The shipping cache contains only
    training/validation files; this boundary never falls back to test caches.
    """
    assert which_set in ("test", "val", "holdout"), which_set
    lease = None
    if which_set == "test":
        from paper_report import assert_test_access
        lease = assert_test_access()
    c = json.load(open(os.path.join(run_dir, "run_config.json")))
    if which_set == 'test' and c.get('phys_mode'):
        if lease.get('mode') != 'F11':
            raise RuntimeError('Physics test scoring is restricted to the registered F11 window')
        if not isinstance(physics_overrides, dict) or not physics_overrides:
            raise RuntimeError('F11 physics test scoring requires complete preloaded physics_overrides')
    env = env_from_config(c)
    # Re-import tes_gru under this run's knobs (config.py reads env at import).
    # Knobs the run's config does not name (older runs predate them) are
    # REMOVED from the environment first, so they fall back to config.py's
    # defaults instead of whatever the shell or the previous --glob run set.
    for var in list(KEYS.values()) + ENV_VARS_EXTRA:
        os.environ.pop(var, None)
    for k, v in env.items():
        os.environ[k] = v
    if which_set == "holdout":
        os.environ["SEAL_TEST"] = "1"
    for m in [m for m in list(sys.modules) if m.startswith("tes_gru")]:
        del sys.modules[m]
    sys.path.insert(0, HERE)
    import torch
    from tes_gru import config
    from tes_gru.data import load_all_data, load_holdout_data, ThermalDataset, get_tavg_mask_sets, is_tavg_masked
    from tes_gru.models import build_model
    from tes_gru.rollout import variant_to_encoder
    from tes_gru.evaluate import test_model
    from tes_gru.round10_metrics import aggregate_scores
    import tes_gru.evaluate as _ev
    from tes_gru.config import DEVICE, LATEST_PARAMS, set_seed
    variant = c["variants"][0]
    set_seed(c["seed"])
    if which_set == "holdout":
        # A fold's family is scored only after its checkpoint was selected on
        # inner validation. Do not refit anything while opening the family.
        sidecar_path = constants_sidecar_path(run_dir, variant)
        if not c.get("sealed") or not os.path.isfile(sidecar_path):
            raise RuntimeError("F1 holdout scoring requires a sealed run and fitted constants sidecar")
        from paper_report import sha256
        from tes_gru.config import split_file_path
        with open(sidecar_path, encoding="utf-8") as stream:
            sidecar = json.load(stream)
        digest = sha256(split_file_path())
        if (sidecar.get("split", {}).get("split_sha256") != digest or
                c.get("data", {}).get("split_sha256") != digest):
            raise RuntimeError("F1 split changed since the checkpoint constants were fitted")
        holdout_dfs, holdout_mask = load_holdout_data()
        train_dfs, val_dfs, test_dfs, scaler = None, [], [], None
    else:
        train_dfs, val_dfs, test_dfs, scaler = load_all_data(include_test=which_set == "test")
    if which_set == 'test' and config.PHYS_MODE:
        expected = {fn for _, fn in test_dfs}
        if len(expected) != 70 or set(physics_overrides) != expected:
            raise RuntimeError('F11 physics_overrides must cover the identical full70 preloaded test IDs')
        for df, fn in test_dfs:
            physical = physics_overrides[fn]
            if not isinstance(physical, dict) or any(key not in physical for key in ('time', 'T_outer_phys', 'T_avg_line_phys')):
                raise RuntimeError('Incomplete preloaded physical reference for ' + fn)
            if any(len(physical[key]) != len(df) for key in ('time', 'T_outer_phys', 'T_avg_line_phys')):
                raise RuntimeError('Preloaded physics horizon differs from test frame for ' + fn)
    model = build_model(variant, LATEST_PARAMS, variant_to_encoder(variant)).to(DEVICE)
    scaler, const_src = load_or_refit_constants(model, variant, run_dir, train_dfs, scaler)
    model.load_state_dict(torch.load(checkpoint_path(run_dir, variant, which), map_location=DEVICE))
    model.eval()
    if which_set == "holdout":
        sets = [ThermalDataset(df, variant, scaler=scaler, file_name=fn,
                               tavg_masked=is_tavg_masked(fn, holdout_mask)) for df, fn in holdout_dfs]
        suffix = "_holdout"
        dest = out_dir or os.path.join(run_dir, "variants", variant)
    elif which_set == "val":
        # H3: the run's own validation files; masked ones (H2) are built with
        # tavg_masked so test_model leaves their T_avg metrics NaN.
        _tr_m, va_m = get_tavg_mask_sets()
        sets = [ThermalDataset(df, variant, scaler=scaler, file_name=fn,
                               tavg_masked=is_tavg_masked(fn, va_m)) for df, fn in val_dfs]
        suffix = "_val"
        dest = out_dir or os.path.join(run_dir, "variants", variant)
    else:
        sets = [ThermalDataset(df, variant, scaler=scaler, file_name=fn,
                               physics_override=(physics_overrides[fn] if config.PHYS_MODE else None))
                for df, fn in test_dfs]
        suffix = ""
        dest = out_dir                   # None: nothing of the test scoring is kept
    if dest:
        os.makedirs(dest, exist_ok=True)
    scratch = tempfile.mkdtemp(prefix="evalckpt_", dir=dest)
    # test_model's legacy npz write goes to runs/<config.RUN_NAME>/variants/<v>/
    # -- switch it off and point RUN_NAME at a throwaway name so the original
    # run dir is never touched by accident (it was, before 2026-09-19). The
    # csv (and, with dest, the npz) go where `dest` says; plots to scratch.
    _ev.SAVE_PREDICTIONS = False
    config.RUN_NAME = "_evalckpt_" + os.path.basename(scratch)
    if dest:
        os.makedirs(dest, exist_ok=True)
    try:
        summary_df, _ = test_model(
            variant, model, sets, output_dir=dest or scratch,
            set_name="ckpt_%s_%s" % (which, which_set), suffix=suffix,
            npz_path=(os.path.join(dest, "predictions%s.npz" % suffix) if dest else None),
            plots_dir=os.path.join(scratch, "plots"))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    rows = summary_df[summary_df["Case"] != "AVERAGE"]
    agg = aggregate_scores(rows)
    res = dict(run=os.path.basename(run_dir), seed=c["seed"], which=which, set=which_set,
               constants=const_src, n_files=agg["n_files"], n_files_T_avg=agg["n_files_T_avg"],
               T_inner=agg["MAE_T_inner"], T_outer=agg["MAE_T_outer"], T_avg=agg["MAE_T_avg"],
               overall=agg["MAE_Overall"], ToTa=agg["MAE_ToTa"], blowup=agg["blowup"],
               JumpSteps_gt2C_ToTa=agg["JumpSteps_gt2C_ToTa"],
               JumpSteps_gt1C_ToTa=agg["JumpSteps_gt1C_ToTa"])
    if which_set == "test":
        # The test-id subsets (never applied to validation files).
        per = {}
        for _, r in rows.iterrows():
            m = re.search(r"\((\d+)\)", str(r["CaseFile"]))
            if m:
                per[int(m.group(1))] = float(r["MAE_T_avg (C)"])
        def sub(ids): return st.mean(per[i] for i in per if i in ids)
        if per:
            res["flagged12_T_avg"] = sub(FLAGGED); res["switched21_T_avg"] = sub(SWITCHED)
            res["rest49_T_avg"] = st.mean(per[i] for i in per if i not in SWITCHED)
    th_path = os.path.join(run_dir, "variants", variant, "train_history.json")
    if os.path.isfile(th_path):
        th = json.load(open(th_path))
        res["best_epoch"] = th["best_epoch"]; res["total_epoch"] = th["total_epoch"]
    if dest:
        res["out_dir"] = dest
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="a run dir, or a runs root when --glob is given")
    ap.add_argument("--glob", default=None)
    ap.add_argument("--which", default="final", choices=["final", "best"])
    ap.add_argument("--set", dest="which_set", default="val", choices=["test", "val", "holdout"],
                    help="score own validation (default), independent F1 holdout, or access-gated Test70")
    ap.add_argument("--out-dir", default=None,
                    help="where summary_errors[_val].csv + predictions[_val].npz go "
                         "(val default: the run's variant dir; test default: nowhere)")
    ap.add_argument("--out", default=None, help="json with one aggregate dict per run")
    ap.add_argument("--t0-sweep", default=None, help="synthetic constant-inlet start:stop:step, 60 steps")
    a = ap.parse_args()
    dirs = sorted(glob.glob(os.path.join(a.path, a.glob))) if a.glob else [a.path]
    results = []
    for d in dirs:
        if a.t0_sweep:
            df = t0_sweep(d, a.t0_sweep, a.which, a.out_dir or os.path.join(d, "variants", "forward_direct"))
            print(df.to_string(index=False))
            continue
        r = evaluate_one(d, a.which, which_set=a.which_set, out_dir=a.out_dir)
        results.append(r)
        extra = ("  sw21 %.2f  r49 %.2f" % (r["switched21_T_avg"], r["rest49_T_avg"])
                 if "switched21_T_avg" in r else "  ToTa %.3f  J2 %d J1 %d" % (
                     r["ToTa"], r["JumpSteps_gt2C_ToTa"], r["JumpSteps_gt1C_ToTa"]))
        print("%-90s seed %3d %s/%s: overall %.3f  Tin %.3f  Tout %.3f  Tavg %.3f (n=%d/%d)%s  (best ep %s / total %s)" % (
            r["run"][-90:], r["seed"], a.which, a.which_set, r["overall"], r["T_inner"], r["T_outer"],
            r["T_avg"], r["n_files"], r["n_files_T_avg"], extra,
            r.get("best_epoch", "?"), r.get("total_epoch", "?")), flush=True)
    if a.out:
        json.dump(results, open(a.out, "w"), indent=1)
    if len(results) > 1:
        print("\nMEAN over %d runs (%s checkpoint, %s set): overall %.3f  T_inner %.3f  T_outer %.3f  T_avg %.3f" % (
            len(results), a.which, a.which_set, st.mean(r["overall"] for r in results),
            st.mean(r["T_inner"] for r in results), st.mean(r["T_outer"] for r in results),
            st.mean(r["T_avg"] for r in results)))
        if all("switched21_T_avg" in r for r in results):
            print("  switched21 %.2f  rest49 %.2f" % (st.mean(r["switched21_T_avg"] for r in results),
                                                     st.mean(r["rest49_T_avg"] for r in results)))


if __name__ == "__main__":
    main()
