"""POST-HOC test-set scoring and plotting of individual checkpoints (2026-10-22).

Why this exists: Round 10 runs are sealed (SEAL_TEST=1) and only plot the
validation set; the registered test windows (B6, F11 via paper_report.py) are
closed and the recipe is frozen. The owner requires complete test plots for
delivered runs. This script therefore reads Test_70_cases DIRECTLY, outside the
package's access gate (paper_report.assert_test_access) and outside the server
ledger, and it is a disclosed deviation from the Round 10 pre-registration. It
cannot change any Round 10 decision. Every call appends one line per run to
round10_posthoc/posthoc_test_access.log and writes POSTHOC_TEST_NOTE.md into
the run directory.

What it does per run directory (best checkpoint, constants from the sidecar,
never refit): rebuilds the model under the run's own knobs, scores all 70 test
files with the package's own test_model (same metrics, same plot style as the
validation plots), and writes
  variants/<v>/summary_errors_test_posthoc.csv
  variants/<v>/predictions_test_posthoc.npz
  variants/<v>/plots/plot_<case>.png          (70 test plots)
Physics arms get their test references from physics_reference.from_frame with
round10_resources/physics/params.json, exactly as F11 did.

Threshold-study runs (run name r10th_<level>_...) are scored against THEIR OWN
degraded reference, recomputed on the test frames with exactly the formulas of
build_threshold_caches.py (diffusivity: k(T) x value; blend: (1-a) nominal + a T0).
Any other physics run uses the nominal reference. --physics-level overrides.

Usage (from src/models/GRU; point it at COPIES if originals must stay untouched):
  python round10_posthoc/plot_test_posthoc.py --reason "..." RUN_DIR [RUN_DIR ...]
  python round10_posthoc/plot_test_posthoc.py --reason "..." --runs-root runs --glob "r10th_*_seed*" --workers 3
Resume-safe: a run that already has summary_errors_test_posthoc.csv and 70 test plots is skipped.
"""
import argparse
import datetime
import re
import subprocess
import time
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
LOG = os.path.join(HERE, "round10_posthoc", "posthoc_test_access.log")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def data_root():
    return os.path.abspath(os.environ.get("DATA_ROOT") or os.path.join(HERE, "..", "..", "..", "data"))


LEVEL_RE = re.compile(r"r10th_([sb][0-9]{3})_")


def level_of(run_dir, override="auto"):
    """'nominal' or a threshold level name (s080, b050, ...)."""
    if override != "auto":
        return override
    m = LEVEL_RE.match(os.path.basename(os.path.normpath(run_dir)))
    return m.group(1) if m else "nominal"


def physics_for(df, params, level):
    """Test-frame physics reference for `level`, same formulas as build_threshold_caches.py."""
    from physics_reference import from_frame
    nom = from_frame(df, params=params)
    if level == "nominal":
        return nom
    from build_threshold_caches import LEVELS
    spec = dict((name, (family, value)) for name, family, value in LEVELS)
    if level not in spec:
        raise RuntimeError("unknown physics level %r" % level)
    family, value = spec[level]
    if family == "diffusivity":
        return from_frame(df, k_scale=value, params=params)
    out = dict(nom)
    out["T_outer_phys"] = (1.0 - value) * nom["T_outer_phys"] + value * nom["T_outer_phys"][0]
    out["T_avg_line_phys"] = (1.0 - value) * nom["T_avg_line_phys"] + value * nom["T_avg_line_phys"][0]
    return out


def done_already(run_dir):
    v = os.path.join(run_dir, "variants", "forward_direct")
    return (os.path.isfile(os.path.join(v, "summary_errors_test_posthoc.csv")) and
            len(glob.glob(os.path.join(v, "plots", "*.png"))) >= 70)


def score_one(run_dir, reason, which="best", physics_level="auto"):
    import eval_checkpoint as ec
    c = json.load(open(os.path.join(run_dir, "run_config.json"), encoding="utf-8"))
    env = ec.env_from_config(c)
    for var in list(ec.KEYS.values()) + ec.ENV_VARS_EXTRA:
        os.environ.pop(var, None)
    os.environ.update(env)
    os.environ["SEAL_TEST"] = "1"          # never let the package itself load the test set
    for m in [m for m in list(sys.modules) if m.startswith("tes_gru")]:
        del sys.modules[m]
    import numpy as np
    import pandas as pd
    import torch
    from tes_gru import config
    from tes_gru.data import ThermalDataset
    from tes_gru.models import build_model
    from tes_gru.rollout import variant_to_encoder
    from tes_gru.evaluate import test_model
    import tes_gru.evaluate as _ev
    from tes_gru.config import DEVICE, LATEST_PARAMS, set_seed

    variant = c["variants"][0]
    set_seed(c["seed"])
    model = build_model(variant, LATEST_PARAMS, variant_to_encoder(variant)).to(DEVICE)
    scaler, const_src = ec.load_or_refit_constants(model, variant, run_dir, None, None)
    if const_src != "sidecar":
        raise RuntimeError("constants must come from the sidecar")
    ckpt = ec.checkpoint_path(run_dir, variant, which)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    root = data_root()
    test_dir = os.path.join(root, config.TEST_SUBDIR)
    paths = sorted(glob.glob(os.path.join(test_dir, "**", "*.csv"), recursive=True))
    if len(paths) != 70:
        raise RuntimeError("expected 70 test files under %s, found %d" % (test_dir, len(paths)))
    frames = [(pd.read_csv(p), os.path.relpath(p, root).replace(os.sep, "/")) for p in paths]
    physics = None
    level = level_of(run_dir, physics_level) if config.PHYS_MODE else "none"
    if config.PHYS_MODE:
        sys.path.insert(0, os.path.join(HERE, "round10_posthoc"))
        with open(os.path.join(HERE, "round10_resources", "physics", "params.json"), encoding="utf-8") as f:
            params = json.load(f)
        physics = dict((fn, physics_for(df, params, level)) for df, fn in frames)
    sets = [ThermalDataset(df, variant, scaler=scaler, file_name=fn,
                           physics_override=(physics[fn] if physics else None)) for df, fn in frames]

    dest = os.path.join(run_dir, "variants", variant)
    _ev.SAVE_PREDICTIONS = False
    config.RUN_NAME = "_posthoc_test_plot"
    summary_df, _ = test_model(variant, model, sets, output_dir=dest, set_name="posthoc_test",
                               suffix="_test_posthoc",
                               npz_path=os.path.join(dest, "predictions_test_posthoc.npz"),
                               plots_dir=os.path.join(dest, "plots"))
    rows = summary_df[summary_df["Case"] != "AVERAGE"]
    res = dict(run=os.path.basename(run_dir.rstrip("/\\")), seed=c["seed"],
               T_inner=float(rows["MAE_T_inner (C)"].mean()), T_outer=float(rows["MAE_T_outer (C)"].mean()),
               T_avg=float(rows["MAE_T_avg (C)"].mean()), n_files=int(len(rows)),
               n_plots=len(glob.glob(os.path.join(dest, "plots", "*.png"))), physics_level=level)
    with open(os.path.join(dest, "test_posthoc_aggregate.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG, "a", encoding="utf-8") as f:
        f.write("%s\t%s\tseed=%s\tckpt_sha256=%s\tn_test=%d\tphysics=%s\treason=%s\n"
                % (stamp, res["run"], c["seed"], sha256(ckpt), len(frames), level, reason))
    with open(os.path.join(run_dir, "POSTHOC_TEST_NOTE.md"), "w", encoding="utf-8") as f:
        f.write("# Post-hoc test scoring (not part of the Round 10 registered test reads)\n\n"
                "Scored %s on the 70 test files with `round10_posthoc/plot_test_posthoc.py`, after the Round 10 "
                "recipe was frozen and F11 was completed. Reason: %s. This run was not used for any decision; the "
                "numbers are descriptive. Test MAE T_inner / T_outer / T_avg = %.3f / %.3f / %.3f C. Plots: "
                "variants/%s/plots/ (%d files).\n" % (stamp, reason, res["T_inner"], res["T_outer"], res["T_avg"],
                                                      variant, res["n_plots"]))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--reason", required=True)
    ap.add_argument("--which", default="best", choices=("best", "final"))
    ap.add_argument("--runs-root", default=None)
    ap.add_argument("--glob", default=None, help="run-dir pattern under --runs-root, e.g. 'r10th_*_seed*'")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--physics-level", default="auto")
    ap.add_argument("--threads", type=int, default=1, help="CPU threads per scoring process (0 = PyTorch default)")
    args = ap.parse_args(argv)
    if not args.reason.strip():
        raise SystemExit("--reason must say why the test set is read")
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):   # inherited by the child processes
        if args.threads > 0:
            os.environ[var] = str(args.threads)
        else:
            os.environ.pop(var, None)
    os.environ.setdefault("PHYSICS_CACHE_DIR", os.path.join(HERE, "round10_resources", "physics", "nominal"))
    runs = [os.path.abspath(r) for r in args.runs]
    if args.glob:
        root = os.path.abspath(args.runs_root or os.path.join(HERE, "runs"))
        runs += sorted(d for d in glob.glob(os.path.join(root, args.glob))
                       if os.path.isfile(os.path.join(d, "variants", "forward_direct", "done.flag")))
    todo = [r for r in runs if not done_already(r)]
    print("POST-HOC TEST ACCESS: %d run(s) matched, %d already done, %d to score; logged to %s"
          % (len(runs), len(runs) - len(todo), len(todo), LOG), flush=True)
    if args.workers > 1 and len(todo) > 1:
        # one child process per run; tes_gru is re-imported per run, so runs never share a process
        procs, failed, queue = [], [], list(todo)
        while queue or procs:
            while queue and len(procs) < args.workers:
                r = queue.pop(0)
                cmd = [sys.executable, os.path.abspath(__file__), "--reason", args.reason, "--which", args.which,
                       "--physics-level", args.physics_level, r]
                procs.append((r, subprocess.Popen(cmd)))
            for item in list(procs):
                if item[1].poll() is not None:
                    procs.remove(item)
                    if item[1].returncode != 0:
                        failed.append(item[0])
            if procs:
                time.sleep(2)
        print("ALL DONE; failed: %d %s" % (len(failed), failed), flush=True)
        if failed:
            raise SystemExit(1)
        return
    for r in todo:
        res = score_one(r, args.reason, args.which, args.physics_level)
        print("%s  seed %s  physics %s  test MAE %.3f / %.3f / %.3f  plots %d" % (
            res["run"], res["seed"], res["physics_level"], res["T_inner"], res["T_outer"], res["T_avg"],
            res["n_plots"]), flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
