"""Launch the reference-threshold study (PREREG_reference_threshold.md).

Every level is the registered C11 arm (PHYS_MODE=residual, h128, mask, vs3) with
only PHYSICS_CACHE_DIR swapped to that level's degraded cache and a level-specific
RUN_PREFIX (r10th_<level>) so run names never collide. Validation only
(launch_arm forces SEAL_TEST=1). Resume-safe: finished seeds are skipped.

Pass 1 runs the S10 seeds of every level, pass 2 the other ten S20 seeds, so a
partial run already covers all levels.

Usage (from src/models/GRU):
  python round10_posthoc/run_threshold.py --dry-run
  python round10_posthoc/run_threshold.py --workers 3 [--levels s080,b010] [--seeds S10|S20]
  [--cache-root runs/_threshold_caches]
"""
import argparse
import copy
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.chdir(HERE)

import round10_pipeline as p  # noqa: E402

LEVELS = ["s080", "s060", "s040", "s025", "b005", "b010", "b020", "b035", "b050"]


def build_arms(levels, seeds, cache_root):
    pr = p.protocol()
    base = {a["arm"]: a for a in p.full_arm_table(pr)}["C11"]
    if base["env"].get("PHYS_MODE") != "residual" or base["env"].get("TAVG_MASK") != "split":
        raise RuntimeError("registered C11 env changed: %r" % base["env"])
    arms = []
    for level in levels:
        cache = os.path.abspath(os.path.join(cache_root, level))
        n = len([f for f in os.listdir(cache) if f.endswith(".npz")]) if os.path.isdir(cache) else 0
        if n != 304:
            raise RuntimeError("cache %s has %d npz files, expected 304; run build_threshold_caches.py first" % (cache, n))
        a = copy.deepcopy(base)
        a["env"]["PHYSICS_CACHE_DIR"] = cache
        a["env"]["RUN_PREFIX"] = "r10th_" + level
        a["arm"], a["stage"], a["condition"] = "TH_" + level, "POSTHOC", None
        a["seeds"] = list(map(int, seeds))
        arms.append(a)
    p.resolve_names(arms)
    return arms, base


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default=",".join(LEVELS))
    ap.add_argument("--seeds", default="S20", choices=("S10", "S20"))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--cache-root", default=os.path.join(HERE, "runs", "_threshold_caches"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    pr = p.protocol()
    s10, s20 = list(pr["seeds"]["S10"]), list(pr["seeds"]["S20"])
    levels = [l for l in args.levels.split(",") if l]
    unknown = [l for l in levels if l not in LEVELS]
    if unknown:
        raise SystemExit("unknown levels: %s" % unknown)
    passes = [s10] if args.seeds == "S10" else [s10, [s for s in s20 if s not in s10]]

    arms, base = build_arms(levels, s20, args.cache_root)
    for a in arms:
        diff = sorted(k for k in set(a["env"]) | set(base["env"]) if a["env"].get(k) != base["env"].get(k))
        done = [s for s in a["seeds"] if os.path.isfile(os.path.join(HERE, "runs", "%s_seed%d" % (a["run_name"], s),
                                                                        "variants", "forward_direct", "done.flag"))]
        print("%-8s %s  differs from C11 in %s  done %d/%d" % (a["arm"], a["run_name"], diff, len(done), len(a["seeds"])))
    total = len(arms) * sum(len(x) for x in passes)
    print("planned runs: %d (%d levels x %s)" % (total, len(arms), args.seeds))
    if args.dry_run:
        return
    for i, seeds in enumerate(passes, 1):
        print("=== pass %d: seeds %s" % (i, seeds), flush=True)
        for a in arms:
            b = dict(a, seeds=list(seeds))
            p.launch_arm(b, workers=args.workers, result_dir=None)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
