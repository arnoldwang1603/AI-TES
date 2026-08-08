"""Round 4 — comprehensive sweep (budget ~3 GPU-days; ~30 h planned here).

Budget note: roughly half the 70 h budget is DELIBERATELY not pre-spent.
Stages S1-S5 answer the open questions (anchor stability, Case-40 lookahead,
loss weights, capacity, interactions) with wide seed coverage; the champion
configuration, its 20-seed confirmation, and the targeted T_avg / Case-40
per-case work all depend on knowing the winners, so pre-spending that budget
on blind configs would buy noise, not information. Round 4.5 uses the
reserve after a first look at S1-S4 (available ~10 h in).

Stages (cheap first; S6 AR last):
  S1 stability    E,H x20 seeds; F,G x12. Real distributions for the anchor
                  bimodality question instead of 2-seed coin flips.
  S2 lookahead    INPUT_LOOKAHEAD in {1,3,5}: the model sees the inlet's next
                  k values (legal: given boundary condition) -- the targeted
                  Case-40 fix. la1 x20, la3/la5 x12.
  S3 weights      T_avg emphasis grid {1-6-5, 1-6-6, 1-6-8, 1-4-4, 1-3-3} +
                  interaction (la1 + 1-6-6), x12 seeds. (Arnold, twice.)
  S4 arch         hidden {64,128,256} x layers {2,3,5} x dropout {0.3,0.1},
                  x6 seeds (control h128x5dp0.3 = S1's E). The 5x128 recipe
                  was tuned for AR. (Arnold: layers + hidden first.)
  S5 arch-la      the two most promising small archs x la1, x8 seeds.
  S6 AR-baseline  arm B (abs_sliding) x seeds {21,123} -> 4-seed AR number.

All stages: TINNER anchor, lead 1, physics bound OFF (physically wrong per
Arnold). Progress lines print every PROGRESS_EVERY seconds (default 60).

Usage:
    python run_round4.py                 # everything
    python run_round4.py --stage S2      # one stage
    python run_round4.py --only E-la1    # one named config
    python run_round4.py --dry-run       # plan + ETA only
Env: SEEDS overrides all seed lists; PROGRESS_EVERY tunes the status cadence.
Resumable; Ctrl+C safe. Logs -> sweep_logs/r4-<config>.log
"""
import os
import re
import subprocess
import sys
import threading
import time

SEEDS_20 = "7,21,42,123,1,2,3,5,11,13,17,29,31,37,41,43,53,59,61,67"
SEEDS_12 = "7,21,42,123,1,2,3,5,11,13,17,29"
SEEDS_8 = "7,21,42,123,1,2,3,5"

BASE = dict(TINNER_MODE="anchor", ANCHOR_LEAD="1", PHYSICS_BOUND_WEIGHT="0",
            VARIANTS="forward_direct", LOSS_WEIGHTS="1,6,3",
            OTHER_CH_MODE="abs")

def cfg(**kw):
    d = dict(BASE)
    d.update({k: str(v) for k, v in kw.items()})
    return d

# (name, env, seeds, est_minutes_per_seed)
STAGES = {
    "S1": [
        ("E",      cfg(),                                SEEDS_20, 4),
        ("H",      cfg(OTHER_CH_MODE="anchor_avg"),      SEEDS_20, 4),
        ("F",      cfg(OTHER_CH_MODE="anchor"),          SEEDS_12, 4),
        ("G",      cfg(OTHER_CH_MODE="anchor_avg_grad"), SEEDS_12, 4),
    ],
    "S2": [
        ("E-la1",  cfg(INPUT_LOOKAHEAD=1),               SEEDS_20, 4),
        ("E-la3",  cfg(INPUT_LOOKAHEAD=3),               SEEDS_12, 4),
        ("E-la5",  cfg(INPUT_LOOKAHEAD=5),               SEEDS_12, 4),
    ],
    "S3": [
        ("w165",     cfg(LOSS_WEIGHTS="1,6,5"), SEEDS_12, 4),
        ("w166",     cfg(LOSS_WEIGHTS="1,6,6"), SEEDS_12, 4),
        ("w168",     cfg(LOSS_WEIGHTS="1,6,8"), SEEDS_12, 4),
        ("w144",     cfg(LOSS_WEIGHTS="1,4,4"), SEEDS_12, 4),
        ("w133",     cfg(LOSS_WEIGHTS="1,3,3"), SEEDS_12, 4),
        ("la1-w166", cfg(INPUT_LOOKAHEAD=1, LOSS_WEIGHTS="1,6,6"), SEEDS_12, 4),
    ],
    "S4": [
        (f"h{h}x{l}dp{d}".replace("0.", ""),
         cfg(HIDDEN_SIZE=h, NUM_LAYERS=l, DROPOUT=d), "SEEDS6",
         3 if h == 64 else (4 if h == 128 else 8))
        for h in (64, 128, 256) for l in (2, 3, 5) for d in (0.3, 0.1)
        if not (h == 128 and l == 5 and d == 0.3)     # control = S1's E
    ],
    "S5": [
        ("h64x3-la1",  cfg(HIDDEN_SIZE=64, NUM_LAYERS=3,
                           INPUT_LOOKAHEAD=1), SEEDS_8, 3),
        ("h128x3-la1", cfg(HIDDEN_SIZE=128, NUM_LAYERS=3,
                           INPUT_LOOKAHEAD=1), SEEDS_8, 4),
    ],
    "S6": [
        ("B-ar", cfg(VARIANTS="abs_sliding"), "21,123", 255),
    ],
}
SEEDS_6 = "7,21,42,123,1,2"
for _st in STAGES.values():                       # patch the S4 placeholder
    for i, (n, e, s, m) in enumerate(_st):
        if s == "SEEDS6":
            _st[i] = (n, e, SEEDS_6, m)

HERE = os.path.dirname(os.path.abspath(__file__))
LAUNCHER = os.path.join(HERE, "GRU_input_ablation.py")
LOG_DIR = os.path.join(HERE, "sweep_logs")


def tail_progress(log_path, label, n_seeds, done_runs, total_runs,
                  spent_h, total_h, stop):
    """Print a one-line status from the child's log every PROGRESS_EVERY s."""
    every = float(os.environ.get("PROGRESS_EVERY", "60"))
    while not stop.wait(every):
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                txt = f.read()
        except OSError:
            continue
        seeds = re.findall(r"# SEED (\d+)/(\d+)\s+seed=(\d+)", txt)
        eps = re.findall(r"epoch\s+(\d+)\s+\(", txt)
        fins = len(re.findall(r"DONE\s+overall_MAE", txt))
        cur = (f"seed {seeds[-1][0]}/{seeds[-1][1]} (={seeds[-1][2]})"
               if seeds else "starting")
        ep = f"epoch {eps[-1]}" if eps else "loading data"
        print(f"  [{time.strftime('%H:%M')}] {label}: {cur}, {ep}, "
              f"{fins}/{n_seeds} seeds finished | round: "
              f"{done_runs + fins}/{total_runs} runs, "
              f"~{spent_h:.1f}/{total_h:.1f} h", flush=True)


def main():
    dry = "--dry-run" in sys.argv
    stages = list(STAGES)
    if "--stage" in sys.argv:
        stages = [sys.argv[sys.argv.index("--stage") + 1].upper()]
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    os.makedirs(LOG_DIR, exist_ok=True)

    plan, total_min, total_runs = [], 0.0, 0
    for st in stages:
        for name, env, seeds, mins in STAGES[st]:
            if only and name != only:
                continue
            seeds = os.environ.get("SEEDS", seeds)
            n = len([s for s in seeds.split(",") if s.strip()])
            plan.append((st, name, env, seeds, n, n * mins))
            total_min += n * mins
            total_runs += n

    print("#" * 72)
    print(f"# ROUND 4  configs={len(plan)}  runs={total_runs}  "
          f"estimated ~{total_min/60:.1f} h")
    print("#" * 72)
    for st, name, env, seeds, n, mins in plan:
        knobs = {k: v for k, v in env.items() if BASE.get(k) != v}
        print(f"  [{st}] {name:<12} x{n:<3} ~{mins/60:4.1f} h  {knobs}")
    if dry:
        print("\nDry run complete -- nothing launched.")
        return

    results, done_runs, spent_min = {}, 0, 0.0
    try:
        for st, name, env_over, seeds, n, est in plan:
            print(f"\n{'=' * 72}\n=== [{st}] {name}  ({n} seeds, ~{est/60:.1f} h) ===")
            env = dict(os.environ, **env_over, SEEDS=seeds)
            log_path = os.path.join(LOG_DIR, f"r4-{name}.log")
            stop = threading.Event()
            mon = threading.Thread(target=tail_progress,
                                   args=(log_path, f"[{st}] {name}", n,
                                         done_runs, total_runs,
                                         spent_min / 60, total_min / 60, stop),
                                   daemon=True)
            t0 = time.time()
            with open(log_path, "a", encoding="utf-8", errors="replace") as fh:
                fh.write(f"\n===== {name} {env_over} seeds={seeds} launched "
                         f"{time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
                fh.flush()
                mon.start()
                rc = subprocess.call([sys.executable, LAUNCHER], env=env,
                                     cwd=HERE, stdout=fh,
                                     stderr=subprocess.STDOUT)
            stop.set()
            hrs = (time.time() - t0) / 3600.0
            spent_min += hrs * 60
            done_runs += n
            results[name] = (rc, hrs)
            note = "OK" if rc == 0 else (f"exit {rc} (check done.flags -- "
                                         "Windows CUDA teardown lies)")
            print(f"  {name} {note} after {hrs:.1f} h")
    except KeyboardInterrupt:
        print("\nInterrupted. Done so far:",
              {k: f"{v[1]:.1f}h" for k, v in results.items()})
        print("Re-running resumes where it left off.")
        sys.exit(130)

    print("\n" + "#" * 72)
    print("# ROUND 4 SUMMARY")
    for name, (rc, h) in results.items():
        print(f"#   {name:<14} {'OK    ' if rc == 0 else f'exit {rc}'}  {h:.1f} h")
    print("#" * 72)


if __name__ == "__main__":
    main()
