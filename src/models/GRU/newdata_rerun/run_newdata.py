"""New-data rerun (2026-11-18): the Round 10 h128 arms on the 2026-11-15 temperature database.

Arms (registered Round 10 envs, h128): direct = B2_mask, C12 = physics residual + physics
input (the frozen recipe); C11 on request (--arms direct,C12,C11). Changes vs the
registered env, nothing else:
  DATA_ROOT          AI-TES/data_2026-11-15 (checked against the split's md5 digest)
  SPLIT_NAME         newdata_1115_s42.json (tag nd1115; see make_split.py)
  TAVG_MASK          "" (no inner-weighted T_avg left in this database)
  TINNER_MASK        split (16 Case 10 files with an HX-face T_inner, tag mi)
  PHYSICS_CACHE_DIR  round10_resources/physics/nominal_nd1115 (built here if missing)
  RUN_PREFIX         r11nd
Same S20 seeds as Round 10, so each run pairs with its Round 10 twin on the unchanged
70-file test set. Sealed like Round 10 (validation only, 20 validation plots per run);
score the test set afterwards with round10_posthoc/plot_test_posthoc.py (70 plots per run).
Pass 1 = S10 seeds of every arm, pass 2 = the other ten. Resume-safe: rerun the same command.

Usage (from src/models/GRU):
  python newdata_rerun/run_newdata.py --dry-run
  nohup python newdata_rerun/run_newdata.py > newdata_run.log 2>&1 &
  python newdata_rerun/run_newdata.py --smoke      # 2 epochs, seed 7, prefix r11ndsmoke
"""
import argparse
import concurrent.futures
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "newdata_rerun"))
os.chdir(HERE)

import round10_pipeline as p  # noqa: E402
import make_split  # noqa: E402
import build_physics_cache as bpc  # noqa: E402

PREFIX = "r11nd"
REGISTERED = {"direct": "B2_mask", "C12": "C12", "C11": "C11"}


def check_data(data_root):
    spec = json.load(open(os.path.join(HERE, "splits", make_split.OUT_NAME), encoding="utf-8"))
    train_dir = os.path.join(data_root, spec["train_subdir"])
    if not os.path.isdir(train_dir):
        raise SystemExit("no training folder under --data-root %s" % data_root)
    keys = make_split.train_keys(train_dir)
    if make_split.manifest_sha256(train_dir, keys) != spec["data_manifest_sha256"]:
        raise SystemExit("training files under %s differ from the ones %s was made from"
                         % (train_dir, make_split.OUT_NAME))
    test_dir = os.path.join(data_root, "Test_70_cases")
    n_test = sum(1 for dp, _, fs in os.walk(test_dir) for f in fs if f.lower().endswith(".csv"))
    if n_test != 70:
        raise SystemExit("expected 70 test files under %s, found %d" % (test_dir, n_test))
    return spec


def build_arms(names, data_root, smoke=False):
    pr = p.protocol()
    table = {a["arm"]: a for a in p.full_arm_table(pr)}
    arms = []
    for name in names:
        base = table[REGISTERED[name]]
        if base["env"].get("TAVG_MASK") != "split" or base["env"].get("HIDDEN_SIZE") != "128":
            raise RuntimeError("registered %s env changed: %r" % (REGISTERED[name], base["env"]))
        a = copy.deepcopy(base)
        a["env"].update(DATA_ROOT=data_root, SPLIT_NAME=make_split.OUT_NAME, TAVG_MASK="", TINNER_MASK="split",
                        PHYSICS_CACHE_DIR=bpc.OUT, RUN_PREFIX=PREFIX + ("smoke" if smoke else ""))
        if smoke:
            a["env"]["MAX_EPOCHS"] = "2"
        a["arm"] = "ND_%s%s" % (name, "_smoke" if smoke else "")
        a["stage"], a["condition"] = "NEWDATA", None
        a["seeds"] = [7] if smoke else list(map(int, pr["seeds"]["S20"]))
        arms.append((a, base))
    p.resolve_names([a for a, _ in arms])
    for a, _ in arms:
        if not a["run_name"].startswith(a["env"]["RUN_PREFIX"] + "_") or "_%s_mi" % make_split.TAG not in a["run_name"]:
            raise RuntimeError("unexpected run name: " + a["run_name"])
    return arms


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="direct,C12", help="comma list from %s" % sorted(REGISTERED))
    ap.add_argument("--seeds", default="S20", choices=("S10", "S20"))
    ap.add_argument("--workers", type=int, default=8, help="training runs at once, split evenly over the arms")
    ap.add_argument("--threads", type=int, default=1, help="CPU threads per training run (0 = PyTorch default)")
    ap.add_argument("--data-root", default=make_split.default_data_root())
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="2-epoch end-to-end check on seed 7")
    args = ap.parse_args(argv)
    names = [n.strip() for n in args.arms.split(",") if n.strip()]
    if not names or any(n not in REGISTERED for n in names):
        raise SystemExit("--arms must be a comma list from %s" % sorted(REGISTERED))
    data_root = os.path.abspath(args.data_root)
    p.set_cpu_threads(args.threads)

    spec = check_data(data_root)
    if not bpc.is_complete():
        bpc.build(data_root)
    arms = build_arms(names, data_root, args.smoke)
    pr = p.protocol()
    s10, s20 = list(map(int, pr["seeds"]["S10"])), list(map(int, pr["seeds"]["S20"]))
    passes = [[7]] if args.smoke else ([s10] if args.seeds == "S10" else [s10, [s for s in s20 if s not in s10]])
    per_arm = max(1, args.workers // len(arms))
    print("data %s: train %d, val %d, drop %d, T_inner masked %d; physics cache %s"
          % (data_root, spec["n_train"], spec["n_val"], spec["n_drop"], len(spec["tinner_mask"]), bpc.OUT), flush=True)
    for a, base in arms:
        diff = sorted(k for k in set(a["env"]) | set(base["env"]) if a["env"].get(k) != base["env"].get(k))
        done = [s for s in a["seeds"] if os.path.isfile(os.path.join(HERE, "runs", "%s_seed%d" % (a["run_name"], s),
                                                                       "variants", p.VARIANT, "done.flag"))]
        print("%s  %s\n  differs from %s in %s\n  done %d/%d"
              % (a["arm"], a["run_name"], base["arm"], diff, len(done), len(a["seeds"])), flush=True)
    print("planned %d runs; %d arm(s) x %d workers x %d threads"
          % (sum(len(x) for x in passes) * len(arms), len(arms), per_arm, args.threads), flush=True)
    if args.dry_run:
        return
    for i, seeds in enumerate(passes, 1):
        print("=== pass %d: seeds %s" % (i, seeds), flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(arms)) as pool:
            jobs = [pool.submit(p.launch_arm, dict(a, seeds=list(seeds)), per_arm, None) for a, _ in arms]
            errors = []
            for job in jobs:
                try:
                    job.result()
                except BaseException as e:      # let the other arm finish its wave, then stop
                    errors.append(e)
        if errors:
            raise errors[0]
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
