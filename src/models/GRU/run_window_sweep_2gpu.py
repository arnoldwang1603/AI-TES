"""One-command window-size sweep driver for a DUAL-GPU machine (2x4090).

Runs the W-sweep with one training process per GPU in parallel. The unit of
work is a single (W, seed) pair -- NOT a whole W -- so the two GPUs stay
balanced even though per-seed cost scales ~linearly with W (roughly
3/6/11/28 h for W=5/10/20/50 at the 1000-ep cap). A shared work queue hands
out jobs LONGEST-W-FIRST (LPT scheduling); with the default 2 seeds that
balances the two lanes to ~48 h each. Splitting by whole-W instead would pin
one indivisible ~56 h W=50 process to a single card while the other idles.

This round is a 2-SEED SCREENING pass (default SEEDS 7,42 -- the same pair
for every W, so per-seed differences cancel seed effects). The top W values
get the remaining seeds later as a confirmation pass.

Usage (on the 2x4090 machine):
    python run_window_sweep_2gpu.py            # run everything
    python run_window_sweep_2gpu.py --dry-run  # print the plan only

Env overrides:
    SEEDS="7,21,42,123"   full 4-seed protocol instead of the screening pair
    GPUS="0"              use one GPU only (jobs run sequentially, LPT order)

Per-(W, seed) console output goes to sweep_logs/W<W>_seed<seed>.log (this
console only shows start/finish lines). Fully resume-friendly: re-running
skips finished (W, seed) pairs and resumes interrupted training -- Ctrl+C
terminates children and is safe to relaunch.
"""
import os
import queue
import subprocess
import sys
import threading
import time

WINDOW_SIZES = [5, 10, 20, 50]         # scheduled longest-W-first (see below)
DEFAULT_SEEDS = "7,42"                 # screening pair; same for every W

HERE = os.path.dirname(os.path.abspath(__file__))
LAUNCHER = os.path.join(HERE, "GRU_input_ablation.py")
LOG_DIR = os.path.join(HERE, "sweep_logs")

_print_lock = threading.Lock()


def say(msg):
    with _print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def worker(gpu_id, work_q, results, dry_run):
    while True:
        try:
            W, seed = work_q.get_nowait()
        except queue.Empty:
            return
        tag = f"W{W}_seed{seed}"
        env = dict(os.environ, WINDOW_SIZE=str(W), SEEDS=str(seed),
                   CUDA_VISIBLE_DEVICES=str(gpu_id))
        if dry_run:
            say(f"GPU{gpu_id}: (dry run) would launch {tag}  "
                f"CUDA_VISIBLE_DEVICES={gpu_id}")
            work_q.task_done()
            continue
        log_path = os.path.join(LOG_DIR, f"{tag}.log")
        say(f"GPU{gpu_id}: {tag} STARTED  (log -> {log_path})")
        t0 = time.time()
        with open(log_path, "a", encoding="utf-8", errors="replace") as fh:
            fh.write(f"\n===== launched {time.strftime('%Y-%m-%d %H:%M:%S')} "
                     f"on GPU{gpu_id} =====\n")
            fh.flush()
            proc = subprocess.Popen([sys.executable, LAUNCHER],
                                    env=env, cwd=HERE,
                                    stdout=fh, stderr=subprocess.STDOUT)
            results[(W, seed)] = {"proc": proc, "gpu": gpu_id}
            rc = proc.wait()
        hours = (time.time() - t0) / 3600.0
        results[(W, seed)] = {"rc": rc, "hours": hours, "gpu": gpu_id}
        status = "OK" if rc == 0 else f"FAILED (exit {rc})"
        say(f"GPU{gpu_id}: {tag} {status} after {hours:.1f} h")
        work_q.task_done()


def main():
    dry_run = "--dry-run" in sys.argv
    gpus = [g.strip() for g in os.environ.get("GPUS", "0,1").split(",") if g.strip()]
    seeds = [s.strip() for s in os.environ.get("SEEDS", DEFAULT_SEEDS).split(",")
             if s.strip()]
    os.makedirs(LOG_DIR, exist_ok=True)

    # One job per (W, seed). LPT: largest W first so the long jobs start
    # immediately and the tail is filled with cheap ones -> balanced lanes.
    jobs = [(W, int(s)) for W in sorted(WINDOW_SIZES, reverse=True) for s in seeds]

    print("#" * 70)
    print(f"# DUAL-GPU WINDOW-SIZE SWEEP  W={sorted(WINDOW_SIZES)}  seeds={seeds}")
    print(f"# GPUs={gpus}  pad=variable (config default)  {len(jobs)} (W,seed) jobs")
    print(f"# per-(W,seed) logs -> {LOG_DIR}")
    print("#" * 70)

    work_q = queue.Queue()
    for job in jobs:
        work_q.put(job)

    results = {}
    threads = [threading.Thread(target=worker, args=(g, work_q, results, dry_run),
                                daemon=True) for g in gpus]
    for t in threads:
        t.start()
    try:
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(timeout=1.0)
    except KeyboardInterrupt:
        say("Interrupted -- terminating child processes...")
        for r in results.values():
            proc = r.get("proc")
            if proc is not None and proc.poll() is None:
                proc.terminate()
        say("Stopped. Re-running this script resumes where it left off "
            "(done variants skipped, partial training resumed).")
        sys.exit(130)

    if dry_run:
        print("\nDry run complete -- nothing launched.")
        return

    print("\n" + "#" * 70)
    print("# SWEEP SUMMARY")
    for (W, seed) in sorted(results):
        r = results[(W, seed)]
        print(f"#   W={W:<3} seed{seed:<4} GPU{r['gpu']}  "
              f"{'OK    ' if r.get('rc') == 0 else 'FAILED'}  {r.get('hours', 0):.1f} h")
    print("#" * 70)
    if any(r.get("rc") != 0 for r in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
