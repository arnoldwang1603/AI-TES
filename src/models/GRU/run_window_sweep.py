"""One-command window-size sweep driver.

Runs the full W-sweep (W in WINDOW_SIZES below) by launching
`GRU_input_ablation.py` once per window size in a FRESH subprocess with the
WINDOW_SIZE env var set. One W per process is mandatory: W is baked into
default arguments and star-import snapshots at import time, so it cannot be
changed inside a running process.

Usage:
    python run_window_sweep.py            # run the whole sweep sequentially
    python run_window_sweep.py --dry-run  # print the plan, launch nothing

Notes:
  * Pad mode is inherited from config (default: "variable" -- the production
    choice from the 2026-07 t=0 study). Override with the SLIDING_PAD_MODE
    env var if ever needed.
  * Each W writes its own run dirs:
        runs/<date>_abs_sliding_W<W>_1000ep_ES150_P0_variable_seed<N>/
  * Fully resume-friendly: re-running this driver skips every (W, seed,
    variant) that already has a done.flag, and resumes any interrupted
    training from its last checkpoint -- safe to Ctrl+C and relaunch.
  * Sequential on purpose (single shared GPU). Smallest W first so quick
    results land early; W=50 rollouts cost ~5x W=10 per epoch, so expect the
    later stages to be much slower.
"""
import os
import subprocess
import sys
import time

WINDOW_SIZES = [5, 10, 20, 50]

HERE = os.path.dirname(os.path.abspath(__file__))
LAUNCHER = os.path.join(HERE, "GRU_input_ablation.py")


def main():
    dry_run = "--dry-run" in sys.argv
    pad_mode = os.environ.get("SLIDING_PAD_MODE", "variable (config default)")
    print("#" * 70)
    print(f"# WINDOW-SIZE SWEEP  W={WINDOW_SIZES}  pad_mode={pad_mode}")
    print(f"# launcher: {LAUNCHER}")
    print("#" * 70)

    results = {}
    try:
        for W in WINDOW_SIZES:
            print(f"\n{'=' * 70}\n=== W = {W} ===\n{'=' * 70}")
            if dry_run:
                print(f"  (dry run) would launch: WINDOW_SIZE={W} "
                      f"SEEDS={os.environ.get('SEEDS', '7,42')} "
                      f"{sys.executable} {os.path.basename(LAUNCHER)}")
                continue
            env = dict(os.environ, WINDOW_SIZE=str(W))
            # 2-seed screening protocol (same pair for every W -- paired
            # design). Export SEEDS yourself to override, e.g. the full
            # "7,21,42,123" for a confirmation pass.
            env.setdefault("SEEDS", "7,42")
            t0 = time.time()
            rc = subprocess.call([sys.executable, LAUNCHER], env=env, cwd=HERE)
            hours = (time.time() - t0) / 3600.0
            results[W] = (rc, hours)
            status = "OK" if rc == 0 else f"FAILED (exit {rc})"
            print(f"\n=== W={W} finished: {status} after {hours:.1f} h ===")
    except KeyboardInterrupt:
        print("\nSweep interrupted by user. Completed so far:", results)
        print("Re-running this script resumes where it left off "
              "(done variants are skipped, partial training resumes).")
        sys.exit(130)

    if dry_run:
        print("\nDry run complete -- nothing launched.")
        return

    print("\n" + "#" * 70)
    print("# SWEEP SUMMARY")
    for W, (rc, hours) in results.items():
        print(f"#   W={W:<3} {'OK    ' if rc == 0 else 'FAILED'}  {hours:.1f} h")
    print("#" * 70)
    if any(rc != 0 for rc, _ in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
