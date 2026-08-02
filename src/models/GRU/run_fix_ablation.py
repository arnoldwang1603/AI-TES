"""Incremental ablation of the 2026-07-23 fixes (single GPU).

Three problems were found in the 2026-07-21 anchor run:
  * bad first steps on cases with a jumping inlet (Case 40: -37 C) -- the
    anchor used Input_T(t) to predict T_inner(t+1), one step stale;
  * T_avg carries a permanent ~+2.5 C level offset (70% of its error);
  * T_outer / T_avg are generally weak while T_inner is now near-perfect --
    the uniform [1,1,1] loss gives the hard channels no extra pull.

Three fixes, run cumulatively so each one's effect is attributable:

  A  lead        ANCHOR_LEAD=1 only (weights uniform, no physics bound)
                 -> isolates the anchor-lag fix. Compare vs the 2026-07-21
                    anchor run.
  B  lead+wts    + LOSS_WEIGHTS=1,6,3   (Ti x1, To x6, Ta x3, as used by the
                    LSTM line) -> isolates the loss-shaping effect vs A.
  C  full        + PHYSICS_BOUND_WEIGHT=1  (hinge keeping T_avg inside
                    [T_inner, T_outer]; holds on 98.76% of our data)
                 -> isolates the physics constraint vs B.
  E  direct      forward_direct variant: exogenous-only [Time, Input_T]
                 seq2seq, one forward pass, zero AR feedback (the LSTM
                 line's formulation inside our GRU harness). Same loss
                 shaping as C -> formulation is the only variable vs C.
                 Cheapest arm by far (no rollout loop).

Usage:
    python run_fix_ablation.py            # A, B, C in order (~18 h)
    python run_fix_ablation.py --only C   # just the full stack (~6 h)
    python run_fix_ablation.py --dry-run

Env overrides: SEEDS (default 7,42), ARMS (e.g. "A,C").
Logs -> sweep_logs/fix-<arm>.log. Resumable; Ctrl+C safe.
"""
import os
import subprocess
import sys
import time

ARMS = {
    "A": dict(ANCHOR_LEAD="1", LOSS_WEIGHTS="1,1,1", PHYSICS_BOUND_WEIGHT="0",
              OTHER_CH_MODE="abs"),
    "B": dict(ANCHOR_LEAD="1", LOSS_WEIGHTS="1,6,3", PHYSICS_BOUND_WEIGHT="0",
              OTHER_CH_MODE="abs"),
    "C": dict(ANCHOR_LEAD="1", LOSS_WEIGHTS="1,6,3", PHYSICS_BOUND_WEIGHT="1",
              OTHER_CH_MODE="abs"),
    # D = Arnold's request from the 2026-07-23 meeting: "just do the same
    # thing [delta] for outer and average". Same as C but T_outer / T_avg are
    # predicted as per-step changes off their own previous value.
    "D": dict(ANCHOR_LEAD="1", LOSS_WEIGHTS="1,6,3", PHYSICS_BOUND_WEIGHT="1",
              OTHER_CH_MODE="persistence"),
    # E = formulation test: forward_direct variant -- exogenous-only
    # [Time, Input_T] inputs, ALL state channels predicted in one forward
    # pass, no AR feedback anywhere (the LSTM line's problem setup, minus
    # bidirectionality, inside our GRU + multi-seed harness). Loss shaping
    # kept identical to C so the FORMULATION is the only variable vs C.
    # Anchor/persistence knobs are abs_sliding-only and inert here. Much
    # cheaper than the other arms: single-pass training, no rollout loop.
    "E": dict(VARIANTS="forward_direct",
              LOSS_WEIGHTS="1,6,3", PHYSICS_BOUND_WEIGHT="1"),
}
DEFAULT_SEEDS = "7,42"

HERE = os.path.dirname(os.path.abspath(__file__))
LAUNCHER = os.path.join(HERE, "GRU_input_ablation.py")
LOG_DIR = os.path.join(HERE, "sweep_logs")


def main():
    dry = "--dry-run" in sys.argv
    if "--only" in sys.argv:
        arms = [sys.argv[sys.argv.index("--only") + 1].upper()]
    else:
        arms = [a.strip().upper() for a in
                os.environ.get("ARMS", "A,B,C,D,E").split(",") if a.strip()]
    for a in arms:
        if a not in ARMS:
            sys.exit(f"unknown arm {a!r}; pick from {list(ARMS)}")
    seeds = os.environ.get("SEEDS", DEFAULT_SEEDS)
    os.makedirs(LOG_DIR, exist_ok=True)

    print("#" * 70)
    print(f"# FIX ABLATION  arms={arms}  seeds={seeds}  (TINNER_MODE=anchor)")
    print("#" * 70)

    results = {}
    try:
        for arm in arms:
            cfg = ARMS[arm]
            print(f"\n{'=' * 70}\n=== ARM {arm}: {cfg} ===\n{'=' * 70}")
            env = dict(os.environ, TINNER_MODE="anchor", **cfg)
            env.setdefault("SEEDS", DEFAULT_SEEDS)
            if dry:
                print(f"  (dry run) would launch arm {arm} with {cfg}")
                continue
            log_path = os.path.join(LOG_DIR, f"fix-{arm}.log")
            print(f"  console -> {log_path}")
            t0 = time.time()
            with open(log_path, "a", encoding="utf-8", errors="replace") as fh:
                fh.write(f"\n===== arm {arm} {cfg} launched "
                         f"{time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
                fh.flush()
                rc = subprocess.call([sys.executable, LAUNCHER], env=env,
                                     cwd=HERE, stdout=fh,
                                     stderr=subprocess.STDOUT)
            hrs = (time.time() - t0) / 3600.0
            results[arm] = (rc, hrs)
            print(f"  arm {arm} {'OK' if rc == 0 else f'FAILED ({rc})'} "
                  f"after {hrs:.1f} h")
    except KeyboardInterrupt:
        print("\nInterrupted. Done so far:", results)
        print("Re-running resumes where it left off.")
        sys.exit(130)

    if dry:
        print("\nDry run complete -- nothing launched.")
        return
    print("\n" + "#" * 70)
    print("# SUMMARY")
    for a, (rc, h) in results.items():
        print(f"#   arm {a}  {'OK    ' if rc == 0 else 'FAILED'}  {h:.1f} h")
    print("#" * 70)
    if any(rc != 0 for rc, _ in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
