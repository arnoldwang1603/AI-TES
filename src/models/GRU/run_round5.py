"""Round 5 — confirmation + capacity refinement + the open questions.

What round 4 established (300 runs, 288 usable):
  * Anchors on T_outer/T_avg LOSE. E (no anchor) 1.284+-0.236 over 20 seeds
    vs H 1.867+-1.200 and F 1.930+-0.810 (permutation p=0.037). They wreck
    T_outer (0.83 -> 2.6) while barely moving T_avg (2.76 -> 2.71).
  * Inlet lookahead does NOT fix the Case-40 residual: 8.42 -> 8.38/8.46/8.49
    for k=1/3/5, i.e. untouched, and overall MAE got worse (p=0.003). The
    +8.4 C is a RANGE limit of the anchor head (8.5 C is ~13 sigma of the
    fitted delta), not an information limit -- hence ANCHOR_SCALE this round.
  * Loss weights: the LSTM-inherited 1/6/3 is already the best of six tried.
    Raising T_avg's weight makes T_avg *worse* (2.76 -> 3.18 at 1-6-8).
  * Architecture is the ONLY lever that helped: h128x2dp0.3 = 1.114+-0.181
    beats the 5-layer default 1.284 (p=0.123, suggestive, n=6), and it is
    also the only thing that improved T_avg (2.76 -> 2.42; h256x5dp0.1 2.32).

So round 5 spends everything on: confirming the architecture win at n=20,
refining around it, testing the range hypothesis for Case 40, and closing
the two ablation gaps (G control, AR baseline seeds).

  T1 confirm    the two round-4 leaders + default, 20 seeds each. Turns
                p=0.12 into a verdict.
  T2 refine     capacity grid around the h128x2 winner: h {96,128,160,192}
                x L {1,2,3} x dropout {0.2,0.3}, 10 seeds. Includes L=1,
                never tried, and the dropout the round-4 grid skipped.
  T3 range      ANCHOR_SCALE {3,6,12} x {la0, la1}, 10 seeds. Direct test of
                the Case-40 range hypothesis; la1 included because lookahead
                may only pay off once the head can actually express the
                correction it enables.
  T4 gaps       G (anchor_avg_grad) 12 seeds -- the detach control killed by
                the config bug; plus arm B (AR) seeds 1,2 for a 6-seed AR
                baseline.
  T5 champion   best-of-T1/T2 x ANCHOR_SCALE best-of-T3, 20 seeds. Only runs
                if you pass --champion H=..,L=..,DP=..,AS=.. after reading
                T1-T3 (it cannot be planned blind).

Usage:
    python run_round5.py                    # T1 -> T4  (~29 h)
    python run_round5.py --stage T2
    python run_round5.py --dry-run
    python run_round5.py --champion H=128,L=2,DP=0.3,AS=6   # T5, after analysis
Env: SEEDS overrides everything; PROGRESS_EVERY tunes the status cadence.
Resumable; Ctrl+C safe. Logs -> sweep_logs/r5-<config>.log
"""
import os
import re
import subprocess
import sys
import threading
import time

S20 = "7,21,42,123,1,2,3,5,11,13,17,29,31,37,41,43,53,59,61,67"
S12 = "7,21,42,123,1,2,3,5,11,13,17,29"
S10 = "7,21,42,123,1,2,3,5,11,13"

BASE = dict(TINNER_MODE="anchor", ANCHOR_LEAD="1", PHYSICS_BOUND_WEIGHT="0",
            VARIANTS="forward_direct", LOSS_WEIGHTS="1,6,3", OTHER_CH_MODE="abs")

def cfg(**kw):
    d = dict(BASE); d.update({k: str(v) for k, v in kw.items()}); return d

def mins(h, l):
    """Rough per-seed minutes for the direct model."""
    return max(2, round((h / 128.0) ** 1.3 * (1 + 0.45 * (l - 1))))

STAGES = {
    "T1": [
        ("c-h128x2dp3", cfg(HIDDEN_SIZE=128, NUM_LAYERS=2, DROPOUT=0.3), S20, mins(128, 2)),
        ("c-h256x5dp1", cfg(HIDDEN_SIZE=256, NUM_LAYERS=5, DROPOUT=0.1), S20, mins(256, 5)),
        ("c-h128x5dp3", cfg(HIDDEN_SIZE=128, NUM_LAYERS=5, DROPOUT=0.3), S20, mins(128, 5)),
    ],
    "T2": [
        (f"r-h{h}x{l}dp{str(dp).replace('0.','')}",
         cfg(HIDDEN_SIZE=h, NUM_LAYERS=l, DROPOUT=dp), S10, mins(h, l))
        for h in (96, 128, 160, 192) for l in (1, 2, 3) for dp in (0.2, 0.3)
        if not (h == 128 and l == 2 and dp == 0.3)          # in T1
    ],
    "T3": [
        (f"as{a}{'-la1' if la else ''}",
         cfg(ANCHOR_SCALE=a, INPUT_LOOKAHEAD=la, HIDDEN_SIZE=128, NUM_LAYERS=2,
             DROPOUT=0.3), S10, mins(128, 2))
        for a in (3, 6, 12) for la in (0, 1)
    ],
    "T6": [
        # T_avg position head: the diagnosis says our T_avg error is a POSITION
        # error x surface gap, and that we currently sit exactly at the
        # fixed-blend ceiling (2.42 vs oracle 2.41) while a per-case position
        # would reach 1.54. Paired with the best architecture so the comparison
        # is against the strongest baseline.
        ("pos-h128x5", cfg(OTHER_CH_MODE="pos_head"), S10, mins(128, 5)),
        ("pos-h128x2", cfg(OTHER_CH_MODE="pos_head", HIDDEN_SIZE=128,
                           NUM_LAYERS=2, DROPOUT=0.3), S10, mins(128, 2)),
    ],
    "T4": [
        ("g-gradctl", cfg(OTHER_CH_MODE="anchor_avg_grad"), S12, mins(128, 5)),
        ("b-ar", cfg(VARIANTS="abs_sliding"), "1,2", 255),
    ],
}

HERE = os.path.dirname(os.path.abspath(__file__))
LAUNCHER = os.path.join(HERE, "GRU_input_ablation.py")
LOG_DIR = os.path.join(HERE, "sweep_logs")


def champion_stage(spec):
    """--champion H=128,L=2,DP=0.3,AS=6  -> the T5 config."""
    kv = dict(p.split("=") for p in spec.split(","))
    h, l = int(kv.get("H", 128)), int(kv.get("L", 2))
    dp, a = float(kv.get("DP", 0.3)), float(kv.get("AS", 1))
    la = int(kv.get("LA", 0))
    name = f"champ-h{h}x{l}dp{str(dp).replace('0.','')}as{a:g}" + (f"la{la}" if la else "")
    return [(name, cfg(HIDDEN_SIZE=h, NUM_LAYERS=l, DROPOUT=dp,
                       ANCHOR_SCALE=a, INPUT_LOOKAHEAD=la), S20, mins(h, l))]


def tail_progress(log_path, label, n_seeds, done, total, spent_h, tot_h, stop):
    every = float(os.environ.get("PROGRESS_EVERY", "60"))
    while not stop.wait(every):
        try:
            txt = open(log_path, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        sd = re.findall(r"# SEED (\d+)/(\d+)\s+seed=(\d+)", txt)
        ep = re.findall(r"epoch\s+(\d+)\s+\(", txt)
        fin = len(re.findall(r"DONE\s+overall_MAE", txt))
        cur = f"seed {sd[-1][0]}/{sd[-1][1]} (={sd[-1][2]})" if sd else "starting"
        print(f"  [{time.strftime('%H:%M')}] {label}: {cur}, "
              f"{'epoch ' + ep[-1] if ep else 'loading'}, {fin}/{n_seeds} done "
              f"| round: {done + fin}/{total} runs, ~{spent_h:.1f}/{tot_h:.1f} h",
              flush=True)


def main():
    dry = "--dry-run" in sys.argv
    if "--champion" in sys.argv:
        plan_src = [("T5", c) for c in
                    champion_stage(sys.argv[sys.argv.index("--champion") + 1])]
    else:
        stages = ([sys.argv[sys.argv.index("--stage") + 1].upper()]
                  if "--stage" in sys.argv else list(STAGES))
        plan_src = [(st, c) for st in stages for c in STAGES[st]]
    os.makedirs(LOG_DIR, exist_ok=True)

    plan, tot_min, tot_runs = [], 0.0, 0
    for st, (name, env, seeds, per) in plan_src:
        seeds = os.environ.get("SEEDS", seeds)
        n = len([s for s in seeds.split(",") if s.strip()])
        plan.append((st, name, env, seeds, n, n * per))
        tot_min += n * per; tot_runs += n

    print("#" * 74)
    print(f"# ROUND 5  configs={len(plan)}  runs={tot_runs}  ~{tot_min/60:.1f} h")
    print("#" * 74)
    for st, name, env, seeds, n, m in plan:
        knobs = {k: v for k, v in env.items() if BASE.get(k) != v}
        print(f"  [{st}] {name:<16} x{n:<3} ~{m/60:4.1f} h  {knobs}")
    if dry:
        print("\nDry run complete -- nothing launched."); return

    res, done, spent = {}, 0, 0.0
    try:
        for st, name, env_over, seeds, n, est in plan:
            print(f"\n{'='*74}\n=== [{st}] {name}  ({n} seeds, ~{est/60:.1f} h) ===")
            env = dict(os.environ, **env_over, SEEDS=seeds)
            lp = os.path.join(LOG_DIR, f"r5-{name}.log")
            print(f"  console -> {lp}")
            stop = threading.Event()
            threading.Thread(target=tail_progress,
                             args=(lp, f"[{st}] {name}", n, done, tot_runs,
                                   spent / 60, tot_min / 60, stop),
                             daemon=True).start()
            t0 = time.time()
            with open(lp, "a", encoding="utf-8", errors="replace") as fh:
                fh.write(f"\n===== {name} {env_over} seeds={seeds} launched "
                         f"{time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
                fh.flush()
                rc = subprocess.call([sys.executable, LAUNCHER], env=env,
                                     cwd=HERE, stdout=fh, stderr=subprocess.STDOUT)
            stop.set()
            hrs = (time.time() - t0) / 3600.0
            spent += hrs * 60; done += n; res[name] = (rc, hrs)
            print(f"  {name} {'OK' if rc == 0 else f'exit {rc} (check done.flags)'}"
                  f" after {hrs:.1f} h")
    except KeyboardInterrupt:
        print("\nInterrupted:", {k: f"{v[1]:.1f}h" for k, v in res.items()})
        print("Re-running resumes where it left off."); sys.exit(130)

    print("\n" + "#" * 74 + "\n# ROUND 5 SUMMARY")
    for nm, (rc, h) in res.items():
        print(f"#   {nm:<18} {'OK' if rc == 0 else f'exit {rc}':<8} {h:.1f} h")
    print("#" * 74)


if __name__ == "__main__":
    main()
