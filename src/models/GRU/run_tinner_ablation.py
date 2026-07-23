"""One-command T_inner ablation driver (single GPU).

Runs the two T_inner fixes for the forward `abs_sliding` variant, PRIORITY
ORDER FIRST, one fresh subprocess per mode (TINNER_MODE is baked into input
dims and dataset columns at import time, so it must never change inside a
running process):

    1. anchor       -- T_inner head predicts a z-scored residual off the GT
                       exogenous Input_T (delta = T_inner - Input_T). The
                       priority arm: it hard-wires the T_inner~Input_T
                       tracking Arnold pointed out.
    2. output_only  -- v22-style A/B: T_inner predicted but removed from the
                       input (4-d), so its errors cannot feed back.

The control arm ("arfed", the current 5-input recipe) is NOT re-run: the
2026-07-16 variable-pad sweep already provides it (800-ep cap caveat noted
in the log).

Usage:
    python run_tinner_ablation.py            # run both modes sequentially
    python run_tinner_ablation.py --dry-run  # print the plan only

Env overrides:
    SEEDS="7,21,42,123"   full 4-seed protocol (default: screening pair 7,42)
    TINNER_MODES="anchor" run a subset / different order

Run dirs: runs/2026-07-21_abs_sliding_W10_1000ep_ES150_P0_variable_Tin-<mode>_seed<N>/
Per-mode console output -> sweep_logs/Tin-<mode>.log. Fully resume-friendly:
re-running skips finished (mode, seed) pairs and resumes interrupted
training -- Ctrl+C is safe.
"""
import os
import subprocess
import sys
import time

DEFAULT_MODES = ["anchor", "output_only"]   # priority order
DEFAULT_SEEDS = "7,42"                      # screening pair; same for both arms

HERE = os.path.dirname(os.path.abspath(__file__))
LAUNCHER = os.path.join(HERE, "GRU_input_ablation.py")
LOG_DIR = os.path.join(HERE, "sweep_logs")


def main():
    dry_run = "--dry-run" in sys.argv
    modes = [m.strip() for m in
             os.environ.get("TINNER_MODES", ",".join(DEFAULT_MODES)).split(",")
             if m.strip()]
    seeds = os.environ.get("SEEDS", DEFAULT_SEEDS)
    os.makedirs(LOG_DIR, exist_ok=True)

    print("#" * 70)
    print(f"# T_INNER ABLATION  modes={modes} (priority order)  seeds={seeds}")
    print(f"# W=10, pad=variable (config defaults); single GPU, sequential")
    print("#" * 70)

    results = {}
    try:
        for mode in modes:
            print(f"\n{'=' * 70}\n=== TINNER_MODE = {mode} ===\n{'=' * 70}")
            env = dict(os.environ, TINNER_MODE=mode)
            env.setdefault("SEEDS", DEFAULT_SEEDS)
            if dry_run:
                print(f"  (dry run) would launch: TINNER_MODE={mode} "
                      f"SEEDS={env['SEEDS']} {os.path.basename(LAUNCHER)}")
                continue
            log_path = os.path.join(LOG_DIR, f"Tin-{mode}.log")
            print(f"  console output -> {log_path}")
            t0 = time.time()
            with open(log_path, "a", encoding="utf-8", errors="replace") as fh:
                fh.write(f"\n===== launched {time.strftime('%Y-%m-%d %H:%M:%S')} "
                         f"seeds {env['SEEDS']} =====\n")
                fh.flush()
                rc = subprocess.call([sys.executable, LAUNCHER], env=env,
                                     cwd=HERE, stdout=fh,
                                     stderr=subprocess.STDOUT)
            hours = (time.time() - t0) / 3600.0
            results[mode] = (rc, hours)
            status = "OK" if rc == 0 else f"FAILED (exit {rc})"
            print(f"  TINNER_MODE={mode} {status} after {hours:.1f} h")
    except KeyboardInterrupt:
        print("\nInterrupted. Completed so far:", results)
        print("Re-running this script resumes where it left off.")
        sys.exit(130)

    if dry_run:
        print("\nDry run complete -- nothing launched.")
        return

    print("\n" + "#" * 70)
    print("# ABLATION SUMMARY")
    for mode, (rc, hours) in results.items():
        print(f"#   Tin-{mode:<12} {'OK    ' if rc == 0 else 'FAILED'}  {hours:.1f} h")
    print("#" * 70)
    if any(rc != 0 for rc, _ in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
