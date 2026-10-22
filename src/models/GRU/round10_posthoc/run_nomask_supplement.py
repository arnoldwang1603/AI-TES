"""Post-hoc supplement to Round 10 v4 (2026-10-21): unmasked versions of C12, R3 and C11.

Same environment as the registered arms except TAVG_MASK="" (training labels and
validation scalar unmasked). Validation only (launch_arm forces SEAL_TEST=1).
Seeds: C12 uses the registered S20 (paper-relevant: mask x physics-residual 2x2);
R3 (Yimin's whole-output ratio, PI interest) and C11 use S10. Resume-safe.

Usage (from src/models/GRU):
  python round10_posthoc/run_nomask_supplement.py --dry-run
  python round10_posthoc/run_nomask_supplement.py [--workers 8] [--only C12_nomask,R3_nomask]
"""
import argparse
import copy
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.chdir(HERE)

import round10_pipeline as p  # noqa: E402


def build_arms():
    pr = p.protocol()
    s20, s10 = pr["seeds"]["S20"], pr["seeds"]["S10"]
    table = {a["arm"]: a for a in p.full_arm_table(pr)}
    arms = []
    for src, name, seeds in (("C12", "C12_nomask", s20), ("R3", "R3_nomask", s10), ("C11", "C11_nomask", s10)):
        a = copy.deepcopy(table[src])
        if a["env"].get("TAVG_MASK") != "split":
            raise RuntimeError("registered %s is not masked: %r" % (src, a["env"].get("TAVG_MASK")))
        a["env"]["TAVG_MASK"] = ""
        a["arm"], a["stage"], a["seeds"], a["condition"] = name, "POSTHOC", list(map(int, seeds)), None
        arms.append((a, table[src]))
    p.resolve_names([a for a, _ in arms])
    return arms


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--threads", type=int, default=1, help="CPU threads per training run (0 = PyTorch default)")
    args = ap.parse_args(argv)
    p.set_cpu_threads(args.threads)
    arms = build_arms()
    for a, src in arms:
        if "_mk" in a["run_name"]:
            raise RuntimeError("unmasked arm resolved to a masked run name: " + a["run_name"])
        diff = sorted(k for k in set(a["env"]) | set(src["env"]) if a["env"].get(k) != src["env"].get(k))
        print("%-11s seeds=%-2d %s  differs from registered arm in %s" % (a["arm"], len(a["seeds"]), a["run_name"], diff))
    if args.dry_run:
        return
    for a, _ in arms:
        if args.only and a["arm"] not in args.only.split(","):
            continue
        p.launch_arm(a, workers=args.workers, result_dir=None)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
