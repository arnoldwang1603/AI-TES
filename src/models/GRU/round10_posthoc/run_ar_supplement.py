"""Post-hoc supplement (2026-10-25): the old autoregressive GRU under the current protocol.

The arm is the registered direct arm B2_mask (h128x1, T_inner inlet anchor, loss 1-6-3,
T_avg mask split, vs3 split, 2000-epoch cap, TF32 0) with exactly one change:
VARIANTS=abs_sliding, the autoregressive sliding-window model that Round 8 replaced
with forward_direct. RUN_PREFIX r10ar keeps the names apart from every other run.
Validation only during training (launch_arm forces SEAL_TEST=1); the test set is read
afterwards with plot_test_posthoc.py. Resume-safe: finished seeds are skipped.

One difference in meaning remains: TAVG_MASK=split removes the masked files' T_avg labels
from the loss in both arms, but during abs_sliding's teacher-forcing epochs (0 to 99,
probability 0.5 falling to 0) those labels are still fed back as window inputs;
forward_direct never sees them. Left as is, so the AR recipe stays the one Round 8 compared.

Pass 1 runs the S10 seeds, pass 2 the other ten S20 seeds; stop after pass 1 for a
10-seed answer. Lab-server benchmark (bench_threads.py, 2026-10-22): one AR run takes
about 17.5 s per epoch; total throughput is about 0.18 to 0.19 epochs/s from 4 to 8
workers. The default 5 workers (not benchmarked directly) runs each 10-seed pass as two
full waves. At about 700 epochs per run: S10 about 11 h, S20 about 22 h.

Usage (from src/models/GRU):
  python round10_posthoc/run_ar_supplement.py --dry-run
  nohup python round10_posthoc/run_ar_supplement.py > ar_run.log 2>&1 &
  python round10_posthoc/run_ar_supplement.py --smoke      # 2 epochs, seed 7, prefix r10arsmoke
"""
import argparse
import copy
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.chdir(HERE)

import round10_pipeline as p  # noqa: E402

AR_VARIANT = "abs_sliding"


def build_arm(smoke=False):
    pr = p.protocol()
    base = {a["arm"]: a for a in p.full_arm_table(pr)}["B2_mask"]
    if base["env"].get("TAVG_MASK") != "split" or base["env"].get("PHYS_MODE"):
        raise RuntimeError("registered B2_mask env changed: %r" % base["env"])
    a = copy.deepcopy(base)
    a["env"]["VARIANTS"] = AR_VARIANT
    a["env"]["RUN_PREFIX"] = "r10arsmoke" if smoke else "r10ar"
    if smoke:
        a["env"]["MAX_EPOCHS"] = "2"
    a["arm"], a["stage"], a["condition"] = ("AR_mask_smoke" if smoke else "AR_mask"), "POSTHOC", None
    a["seeds"] = [7] if smoke else list(map(int, pr["seeds"]["S20"]))
    p.resolve_names([a])
    if not a["run_name"].startswith(a["env"]["RUN_PREFIX"] + "_" + AR_VARIANT) or "_mk" not in a["run_name"]:
        raise RuntimeError("unexpected AR run name: " + a["run_name"])
    return a, base


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="S20", choices=("S10", "S20"))
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--threads", type=int, default=8, help="CPU threads per training run (0 = PyTorch default)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="2-epoch end-to-end check on one seed")
    args = ap.parse_args(argv)
    p.set_cpu_threads(args.threads)
    # launch_arm's completion check looks for variants/<p.VARIANT>/; the AR run writes variants/abs_sliding/
    p.VARIANT = AR_VARIANT

    pr = p.protocol()
    s10, s20 = list(map(int, pr["seeds"]["S10"])), list(map(int, pr["seeds"]["S20"]))
    a, base = build_arm(args.smoke)
    passes = [a["seeds"]] if args.smoke else ([s10] if args.seeds == "S10" else [s10, [s for s in s20 if s not in s10]])
    diff = sorted(k for k in set(a["env"]) | set(base["env"]) if a["env"].get(k) != base["env"].get(k))
    done = [s for s in a["seeds"] if os.path.isfile(os.path.join(HERE, "runs", "%s_seed%d" % (a["run_name"], s),
                                                                    "variants", AR_VARIANT, "done.flag"))]
    print("%s  %s\n  differs from B2_mask in %s\n  done %d/%d; planned %d runs; %d workers x %d threads"
          % (a["arm"], a["run_name"], diff, len(done), len(a["seeds"]), sum(len(x) for x in passes),
             args.workers, args.threads), flush=True)
    if args.dry_run:
        return
    for i, seeds in enumerate(passes, 1):
        print("=== pass %d: seeds %s" % (i, seeds), flush=True)
        p.launch_arm(dict(a, seeds=list(seeds)), workers=args.workers, result_dir=None)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
