"""Training throughput benchmark: concurrent workers x CPU threads per worker (2026-10-22).

Why: every training process opens PyTorch/OpenMP thread pools as wide as the
machine. On the 48-thread lab server three training processes showed 83 threads
and about 15 busy cores each while the GPU sat at 37%. The model is small, so an
epoch is paced by the GRU's sequential GPU steps, not by CPU math; the pool
threads mostly spin while they wait. Limiting threads per process frees the CPU
for more concurrent runs.

For each setting WxT (W concurrent runs, T threads per run; T=0 = no limit, the
PyTorch default) this trains one registered Round 10 arm for a few epochs and
records seconds per epoch per run and total epochs per second. It also checks
that the thread limit leaves the training numbers unchanged: the first seed is
trained in every setting and its train/validation loss history is compared with
the first setting's.

Usage (from src/models/GRU; run while nothing else is training):
  python round10_posthoc/bench_threads.py --grid 3x0,3x1,6x1,9x1,12x1,6x2 --epochs 60
  python round10_posthoc/bench_threads.py --arm B2_mask --variant abs_sliding --grid 1x0,1x8,4x8,6x6,8x4 --epochs 3
    (the second line times the old autoregressive sliding-window model; a few epochs suffice)
Writes runs/_bench_threads/bench_threads.csv and deletes its own run directories
(prefix r10bench_) unless --keep. Python 3.8+.
"""
import argparse
import copy
import glob
import json
import os
import shutil
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.chdir(HERE)

import round10_pipeline as p  # noqa: E402

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def load_avg():
    try:
        return os.getloadavg()[0]
    except (AttributeError, OSError):      # Windows
        return float("nan")


def run_setting(arm, workers, threads, epochs, seeds, out):
    prefix = "r10bench_w%dt%d" % (workers, threads)
    procs = []
    t0 = time.time()
    for seed in seeds[:workers]:
        env = dict(arm["env"], SEEDS=str(seed), SEAL_TEST="1", ROUND10_NO_PLOTS="1",
                   MAX_EPOCHS=str(epochs), RUN_PREFIX=prefix)
        penv = p.isolated_env(env)
        for v in THREAD_VARS:
            if threads > 0:
                penv[v] = str(threads)
            else:
                penv.pop(v, None)
        log = open(os.path.join(out, "%s_seed%d.log" % (prefix, seed)), "w", encoding="utf-8")
        procs.append((seed, log, subprocess.Popen([sys.executable, os.path.join(HERE, "GRU_input_ablation.py")],
                                                  env=penv, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)))
    peak_load = 0.0
    while any(pr.poll() is None for _, _, pr in procs):
        peak_load = max(peak_load, load_avg())
        time.sleep(2)
    wall = time.time() - t0
    rows = []
    for seed, log, pr in procs:
        log.close()
        dirs = glob.glob(os.path.join(HERE, "runs", prefix + "*_seed%d" % seed))
        metas = glob.glob(os.path.join(dirs[0], "variants", "*", "meta.json")) if len(dirs) == 1 else []
        meta = metas[0] if len(metas) == 1 else ""
        if not os.path.isfile(meta):
            raise SystemExit("%s seed %d left no meta.json (exit %s); see %s" % (prefix, seed, pr.returncode, log.name))
        m = json.load(open(meta, encoding="utf-8"))
        h = json.load(open(os.path.join(os.path.dirname(meta), "train_history.json"), encoding="utf-8"))
        rows.append(dict(seed=seed, epochs=m["Total_Epoch"], s_per_epoch=m["Train_Time_s"] / m["Total_Epoch"],
                         hist=h["train_hist"] + h["val_hist"], run_dir=dirs[0]))
    return wall, peak_load, rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="C11", help="registered Round 10 arm to train (default C11, a physics arm)")
    ap.add_argument("--grid", default="3x0,3x1,6x1,9x1,12x1,6x2")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--variant", default=None,
                    help="override VARIANTS, e.g. abs_sliding for the old autoregressive model (use with --arm B2_mask)")
    args = ap.parse_args(argv)
    pr = p.protocol()
    table = dict((a["arm"], a) for a in p.full_arm_table(pr))
    arm = copy.deepcopy(table[args.arm])
    if args.variant:
        arm["env"]["VARIANTS"] = args.variant
    seeds = [int(s) for s in pr["seeds"]["S20"]]
    grid = [tuple(int(x) for x in g.split("x")) for g in args.grid.split(",")]
    if max(w for w, _ in grid) > len(seeds):
        raise SystemExit("at most %d workers" % len(seeds))
    out = os.path.join(HERE, "runs", "_bench_threads")
    os.makedirs(out, exist_ok=True)
    print("arm %s, %d epochs, %d logical CPUs, settings %s" % (args.arm, args.epochs, os.cpu_count(), args.grid), flush=True)

    ref, results = None, []
    diff_key = "seed%d_loss_history_max_abs_diff_vs_first" % seeds[0]
    for w, t in grid:
        wall, peak, rows = run_setting(arm, w, t, args.epochs, seeds, out)
        spe = statistics.median(r["s_per_epoch"] for r in rows)
        first = [r for r in rows if r["seed"] == seeds[0]][0]
        if ref is None:
            ref = first["hist"]
        diff = (max(abs(a - b) for a, b in zip(ref, first["hist"])) if len(ref) == len(first["hist"])
                else "length differs")
        res = dict(workers=w, threads=t if t else "no limit", epochs=args.epochs, wall_s=round(wall, 1),
                   s_per_epoch_median=round(spe, 4), epochs_per_s_total=round(w / spe, 2),
                   peak_load_1min=round(peak, 1))
        res[diff_key] = diff
        results.append(res)
        print("  %2d workers x %-8s threads: %.3f s/epoch per run, %.2f epochs/s total, wall %.0f s, peak load %.1f, "
              "loss history diff vs first setting %s" % (w, res["threads"], spe, w / spe, wall, peak, diff), flush=True)
        if not args.keep:
            for r in rows:
                if os.path.basename(r["run_dir"]).startswith("r10bench_"):
                    shutil.rmtree(r["run_dir"], ignore_errors=True)
    import csv
    with open(os.path.join(out, "bench_threads.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(results[0]))
        wr.writeheader()
        wr.writerows(results)
    best = max(results, key=lambda r: r["epochs_per_s_total"])
    print("best: %d workers x %s threads (%.2f epochs/s total); table in %s" % (
        best["workers"], best["threads"], best["epochs_per_s_total"], os.path.join(out, "bench_threads.csv")), flush=True)


if __name__ == "__main__":
    main()
