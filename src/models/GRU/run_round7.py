"""Round 7 -- combine the weight lever with the robust gates, and tell the GRU
which regime it is in.

Where round 6 left us (916 runs, 55 configs, 2026-08-26 log entry):
  * The lever was the LOSS WEIGHT, not the gate: (T_inner, T_outer, T_avg) =
    1-6-1 gives 0.774 +- 0.02 over 12 seeds against 0.973 for the inherited
    1-6-3, monotone in T_avg's weight (10 > 6 > 3 > 1; 0.5 is slightly worse
    again). T_outer is what improves (0.70 -> 0.44) and T_avg follows, because
    T_avg is computed from T_outer.
  * Gates that fix the 12 excursion cases WITHOUT damaging the other 58:
    tgate10-soft40 (0.822, 0/20 blow-ups), cgate-afb (0.841, 0/20),
    cgate-ah (0.948, 2/20). Floors and learned gates are dead. All of these
    were measured under 1-6-3.
  * Capacity: h256/L3 is the best point of the grid (0.845 under 1-6-3);
    L5 hurts everywhere.
  * The shared-channel case gate (cgate, 3 channels) ruined the 49 cases it
    never switched (T_avg 7.90 vs 1.90) because the GRU is never told which
    regime a step belongs to and one channel had to mean two things.
  * 2026-09-05 meeting: Arnold asked for a sanity protocol (drop every
    both-phase run from train AND test, rerun, see whether the rest recovers),
    and the discussion pinned down that the network never sees the gate.

Methods referred to below (plain descriptions are in the 2026-09-06 email to
Arnold and the merged summary doc): method 2 = case gate handing T_avg to a
raw absolute 4th channel (cgate-ah); method 3 = case gate handing over to a
4th channel anchored on the surface midpoint (cgate-afb); method 4 = the
per-timestep soft gate on the predicted surface gap (tgateN-softM).

Stages:
  S0 sanity   EXCLUDE_BOTH_PHASE on the round-6 configuration (1-6-3), 10 seeds
              = the round-5 seed list. With the both-phase runs gone no gate
              can fire, so the gated arms are CODE checks, each against its
              own twin with the same number of output channels:
                xb-cgate      (3 ch) must equal xb-base (3 ch) seed by seed
                xb-cgate-afb  (4 ch) must equal xb-ah   (4 ch) seed by seed
              (the unselected head gets no gradient; bit-identical up to CUDA
              nondeterminism). A difference inside a pair is a shared-code
              bug. Note the pairing: a 4th output channel shifts the init RNG
              stream, so xb-cgate-afb can NOT be expected to match xb-base.
              The scientific reading of S0 is a different comparison, made by
              sanity_compare.py after the run: xb-base vs the round-6 base, and
              xb-w161 (12 seeds, pairs with w161's seeds) vs w161, on the 49
              retained test cases -- does removing the both-phase runs from
              TRAINING help or hurt the normal regime? These arms are scored
              on 49 cases; export_results keeps them out of the ranking.
  S1 main     9 arms x 20 seeds (the round-6 seed list, so cross-round paired
              comparisons stay valid). Everything on 1-6-1:
                w161            the new base, confirmed at 20 (round 6 had 12)
                w161-ah         4 channels, nothing gating to them: the null
                                for every 4-channel arm (RNG shift at init)
                w161-tg10s40    method 4, round-6 best gate
                w161-tg20       timestep hard gate: the 10/20/30 mini-sweep is
                w161-tg30         repeated because 1-6-1 makes the predicted
                w161-tg30s30      gap (the gate's input) more accurate, which
                                  can move the best threshold
                w161-cgate-afb  method 3
                w161-cgate-ah   method 2 -- do the 2/20 blow-ups survive the
                                weight change?
                w161-tg10s40-cgate-afb  both gate kinds, anchored head
  S2 weights  T_avg pinned at 1, T_outer pushed: 1-8-1, 1-10-1, 1-12-1, and
              1-8-2 as a cross-check. 12 seeds, 3 channels, paired vs w161.
  S3 flag     CASE_FLAG_INPUT on w161: the detector's verdict as an INPUT
              column so the GRU can adapt to the regime itself instead of
              being switched from outside. "case" (per run), "phase" (per
              timestep: 1 from the first step after the input-temperature
              peak plateau, i.e. once the discharge has begun) and "zero" --
              the null, because an extra input column shifts the init RNG
              exactly like the extra output channel did. Plus phase x
              cgate-afb with ITS null, zero x cgate-afb. 20 seeds.
  S4 arch     h256/L3 and h192/L2 under 1-6-1; h256/L3 with each of the two
              robust gates and with the 4-channel null (h256x3-ah). 12 seeds.
              Not gated on S1's outcome, so the round runs unattended.
  S5 AR       the autoregressive formulation with the position head under
              1-6-1, alone and with cgate-afb. 4 seeds, ~5 h each -- last, so
              every direct-model answer is in hand before it starts. With 4
              seeds this pair is descriptive; it is not entered in the rule.

Decision rule, fixed BEFORE the run:
  pairing    every arm is paired per seed against ITS OWN NULL (same seed
             list, same number of input and output channels):
               3-channel weight arms (S2)            vs w161
               4-channel gated arms (S1)             vs w161-ah
               flag-input arms fic / fip (S3)        vs w161-fiz
               w161-fip-cgate-afb                    vs w161-fiz-cgate-afb
               h256x3 gated arms (S4)                vs w161-h256x3-ah
             w161-ah - w161 and w161-fiz - w161 are reported as the measured
             null shifts (expected ~0 in the mean).
  primary    overall MAE: paired mean difference vs the null, exact sign-flip
             permutation p and a 95% CI. Arms are RANKED, not each tested vs
             base at 0.05 (winner's curse with ~20 arms). "Within noise" =
             the CI includes 0.
  secondary  T_avg MAE on the 12 flagged cases (32-40, 51-53); also reported
             on the 21 switched cases and the 49 untouched ones, with "no
             paired degradation on the 49" as an explicit requirement.
  veto       blow-ups: any seed above 1.5 C on the common seed set. Jumps: a
             paired increase in JumpSteps_gt2C vs the null with p < 0.05. An
             arm that wins the mean but adds either is not a winner.
  deployment seed-ensemble MAE (ensemble_eval.py --seeds <S12> --subset-size
             10), i.e. at a common n across stages with a subset sd; the
             shipped recipe is the ensemble, so it is reported for every arm.
  simplicity a winner within noise of a simpler arm loses to it; the order is
             3-ch < 4-ch, no flag < flag, h128x2 < larger, one gate < two.
  S0         identity within each pair (above); the training-set question is
             answered by sanity_compare.py, descriptively.

Resume note: w161 shares its RUN_NAME with round 6's S4 w1-6-1 arm, and the
code path with every round-7 knob off is numerically unchanged, so on a
machine that ran round 6 those 12 seeds resume via done.flag and only 8 new
seeds train. Every other arm carries a new tag and starts clean. (Dev box,
2026-09-06: one seed-7 smoke run of xb-cgate-afb exists locally and would
resume as that seed; it was made with the current code.)

Usage:
    python run_round7.py                 # everything, ~27 h direct + ~40 h AR
    python run_round7.py --stage S1      # or --stage S0,S1
    python run_round7.py --only w161-tg10s40
    python run_round7.py --dry-run
Env: SEEDS overrides everything; PROGRESS_EVERY tunes the status cadence.
Resumable; Ctrl+C safe. Logs -> sweep_logs/r7-<config>.log
"""
import glob
import os
import re
import subprocess
import sys
import threading
import time

S20 = "7,21,42,123,1,2,3,5,11,13,17,29,31,37,41,43,53,59,61,67"
S12 = "7,21,42,123,1,2,3,5,11,13,17,29"
S10 = "7,21,42,123,1,2,3,5,11,13"          # = the round-5 champion's seeds
S4 = "7,21,42,123"

# The round-6 champion configuration. EVERY knob the child reads is pinned
# here (2026-08-22 review finding: a stray env var in the launching shell must
# not be able to turn every arm into a gated one).
BASE = dict(TINNER_MODE="anchor", ANCHOR_LEAD="1", PHYSICS_BOUND_WEIGHT="0",
            VARIANTS="forward_direct", LOSS_WEIGHTS="1,6,3",
            OTHER_CH_MODE="pos_head", HIDDEN_SIZE="128", NUM_LAYERS="2",
            DROPOUT="0.3", WINDOW_SIZE="10", SLIDING_PAD_MODE="variable",
            ANCHOR_SCALE="1.0", INPUT_LOOKAHEAD="0",
            POS_GAP_FLOOR="0", POS_FLOOR_SOFT="0",
            POS_TEMP_GATE="0", POS_TEMP_SOFT="0", POS_CASE_GATE="0",
            POS_LEARNED_GATE="0", POS_GATE_BIAS="2.0", POS_LEARNED_POOL="0",
            POS_ABS_HEAD="0", POS_ANCHORED_FALLBACK="0", POS_FIT_CLEAN="0",
            CASE_FLAG_INPUT="", EXCLUDE_BOTH_PHASE="0")
W = "1,6,1"                                  # the round-6 lever


def cfg(**kw):
    d = dict(BASE)
    d.update({k: str(v) for k, v in kw.items()})
    return d


def w161(**kw):
    """Everything in S1-S5 sits on the 1-6-1 weights."""
    kw.setdefault("LOSS_WEIGHTS", W)
    return cfg(**kw)


def mins(h=128, l=2):
    """Rough per-seed minutes for the direct model: ~3 at h128x2 (the dev
    box measured 2-2.6 min; the lab server is a little slower), ~9 at
    h256x3."""
    return max(2, round(2.0 * (h / 128.0) ** 1.3 * (1 + 0.45 * (l - 1))))


STAGES = {
    "S0": [
        ("xb-base", cfg(EXCLUDE_BOTH_PHASE=1), S10, mins()),
        ("xb-ah", cfg(EXCLUDE_BOTH_PHASE=1, POS_ABS_HEAD=1), S10, mins()),
        ("xb-cgate", cfg(EXCLUDE_BOTH_PHASE=1, POS_CASE_GATE=1), S10, mins()),
        ("xb-cgate-afb", cfg(EXCLUDE_BOTH_PHASE=1, POS_CASE_GATE=1,
                             POS_ANCHORED_FALLBACK=1), S10, mins()),
        # pairs with w161's first 12 seeds for the training-set question
        ("xb-w161", w161(EXCLUDE_BOTH_PHASE=1), S12, mins()),
    ],
    "S1": [
        ("w161", w161(), S20, mins()),
        ("w161-ah", w161(POS_ABS_HEAD=1), S20, mins()),
        ("w161-tg10s40", w161(POS_TEMP_GATE=10, POS_TEMP_SOFT=40), S20, mins()),
        ("w161-tg20", w161(POS_TEMP_GATE=20), S20, mins()),
        ("w161-tg30", w161(POS_TEMP_GATE=30), S20, mins()),
        ("w161-tg30s30", w161(POS_TEMP_GATE=30, POS_TEMP_SOFT=30), S20, mins()),
        ("w161-cgate-afb", w161(POS_CASE_GATE=1, POS_ANCHORED_FALLBACK=1),
         S20, mins()),
        ("w161-cgate-ah", w161(POS_CASE_GATE=1, POS_ABS_HEAD=1), S20, mins()),
        ("w161-tg10s40-cgate-afb", w161(POS_TEMP_GATE=10, POS_TEMP_SOFT=40,
                                        POS_CASE_GATE=1,
                                        POS_ANCHORED_FALLBACK=1), S20, mins()),
    ],
    "S2": [
        ("w" + w.replace(",", ""), cfg(LOSS_WEIGHTS=w), S12, mins())
        for w in ("1,8,1", "1,10,1", "1,12,1", "1,8,2")
    ],
    "S3": [
        ("w161-fic", w161(CASE_FLAG_INPUT="case"), S20, mins()),
        ("w161-fip", w161(CASE_FLAG_INPUT="phase"), S20, mins()),
        ("w161-fiz", w161(CASE_FLAG_INPUT="zero"), S20, mins()),
        ("w161-fip-cgate-afb", w161(CASE_FLAG_INPUT="phase", POS_CASE_GATE=1,
                                    POS_ANCHORED_FALLBACK=1), S20, mins()),
        ("w161-fiz-cgate-afb", w161(CASE_FLAG_INPUT="zero", POS_CASE_GATE=1,
                                    POS_ANCHORED_FALLBACK=1), S20, mins()),
    ],
    "S4": [
        ("w161-h256x3", w161(HIDDEN_SIZE=256, NUM_LAYERS=3), S12, mins(256, 3)),
        ("w161-h192x2", w161(HIDDEN_SIZE=192, NUM_LAYERS=2), S12, mins(192, 2)),
        ("w161-h256x3-ah", w161(HIDDEN_SIZE=256, NUM_LAYERS=3, POS_ABS_HEAD=1),
         S12, mins(256, 3)),
        ("w161-h256x3-tg10s40", w161(HIDDEN_SIZE=256, NUM_LAYERS=3,
                                     POS_TEMP_GATE=10, POS_TEMP_SOFT=40),
         S12, mins(256, 3)),
        ("w161-h256x3-cgate-afb", w161(HIDDEN_SIZE=256, NUM_LAYERS=3,
                                       POS_CASE_GATE=1,
                                       POS_ANCHORED_FALLBACK=1),
         S12, mins(256, 3)),
    ],
    "S5": [
        # 300 min/seed from measurement (round 4: ~4.25 h/seed; round 6 S5
        # booked 4.5-5.9 h/seed). 4 seeds per arm.
        ("AR-w161", w161(VARIANTS="abs_sliding", NUM_LAYERS=5), S4, 300),
        ("AR-w161-cgate-afb", w161(VARIANTS="abs_sliding", NUM_LAYERS=5,
                                   POS_CASE_GATE=1, POS_ANCHORED_FALLBACK=1),
         S4, 300),
    ],
}
# Cheapest and most informative first; the AR pair last so killing the round
# after S4 costs nothing.
DEFAULT_STAGES = ["S0", "S1", "S2", "S3", "S4", "S5"]

HERE = os.path.dirname(os.path.abspath(__file__))
LAUNCHER = os.path.join(HERE, "GRU_input_ablation.py")
LOG_DIR = os.path.join(HERE, "sweep_logs")


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
        cur = "seed {}/{} (={})".format(*sd[-1]) if sd else "starting"
        print("  [{}] {}: {}, {}, {}/{} done | round: {}/{} runs, ~{:.1f}/{:.1f} h"
              .format(time.strftime("%H:%M"), label, cur,
                      "epoch " + ep[-1] if ep else "loading", fin, n_seeds,
                      done + fin, total, spent_h, tot_h), flush=True)


def preflight_compile():
    """Compile the package under the interpreter that will run it. The dev
    box runs a newer Python than the lab server; that mismatch once crashed
    every arm at import."""
    import compileall
    if not compileall.compile_dir(os.path.join(HERE, "tes_gru"),
                                  quiet=1, force=True):
        sys.exit("tes_gru does not compile under " + sys.version.split()[0]
                 + " -- fix the syntax error above before launching")


def preflight_stale_dirs():
    """A done.flag makes the sweep silently reuse a directory. Any dir
    carrying a gating / round-7 tag must have been written by code that
    records the current provenance keys; older dirs are refused."""
    import json as _json
    tags = ("_tg", "_fl", "_cgate", "_lg", "_ah", "_afb", "_cf", "_fi", "_xb")
    stale = []
    runs_root = os.path.join(HERE, "runs")
    if os.path.isdir(runs_root):
        for d in os.listdir(runs_root):
            if not any(t in d for t in tags):
                continue
            rdir = os.path.join(runs_root, d)
            rc = os.path.join(rdir, "run_config.json")
            flags = glob.glob(os.path.join(rdir, "variants", "*", "done.flag"))
            if not flags or not os.path.isfile(rc):
                continue
            try:
                keys = _json.load(open(rc))
            except Exception:
                stale.append(d + "  (unreadable run_config)")
                continue
            if "pos_fit_clean" not in keys:
                stale.append(d + "  (pre-round-6 code)")
            elif ("_fi" in d or "_xb" in d) and "case_flag_input" not in keys:
                stale.append(d + "  (pre-round-7 code)")
    if stale:
        print("\nREFUSING TO LAUNCH -- stale done-flagged run dirs would be")
        print("silently reused by these arms. Delete or rename them first:")
        for d in stale:
            print("   runs/" + d)
        sys.exit(2)


# Runs in a CHILD with the BASE env pinned and CUDA hidden, so the parent
# never holds a GPU context for the whole round and a stray shell variable
# cannot change what is being checked. Compares the numpy detector (what
# EXCLUDE_BOTH_PHASE and the flag column use, on the raw column) with the
# torch one (what every gate uses at runtime, on the scaled, last-row-dropped
# X) for EVERY train / val / test run.
_TWIN_CHECK = r'''
import sys
from tes_gru.config import EXCLUDE_BOTH_PHASE
from tes_gru.data import load_all_data, ThermalDataset, input_case_flag_np
from tes_gru.rollout import _input_case_gate
tr, va, te, sc = load_all_data()
bad, counts, fired_test = [], [], []
for name, dfs in (("train", tr), ("val", va), ("test", te)):
    n = 0
    for df, cid in dfs:
        col = "Input Temperature (C)" if "Input Temperature (C)" in df.columns else "T_inner (C)"
        a = input_case_flag_np(df[col].values)
        ds = ThermalDataset(df, "forward_direct", scaler=sc, file_name=cid)
        b = bool(_input_case_gate(ds.X[:, :, 1])[0].item())
        if a != b:
            bad.append((name, cid, a, b))
        n += int(b)
        if name == "test" and b:
            fired_test.append(cid)
    counts.append((name, n, len(dfs)))
print("  detector twins: " + ", ".join("%s %d/%d fire" % c for c in counts))
if bad:
    print("  numpy / torch detectors DISAGREE:")
    for x in bad[:10]:
        print("    ", x)
    sys.exit(3)
if EXCLUDE_BOTH_PHASE:
    if any(c[1] for c in counts):
        print("  EXCLUDE_BOTH_PHASE left runs the gate still fires on")
        sys.exit(4)
    print("  EXCLUDE_BOTH_PHASE: retained train/val/test = %d/%d/%d, no gate can fire"
          % tuple(c[2] for c in counts))
else:
    print("  test cases the gate switches (%d): %s" % (
        len(fired_test), ", ".join(c.split("/")[-1] for c in fired_test)))
'''


def preflight_detector_twins(plan):
    envs = [("full data", {})]
    if any(env.get("EXCLUDE_BOTH_PHASE") == "1" for _, _, env, _, _, _ in plan):
        envs.append(("EXCLUDE_BOTH_PHASE=1", {"EXCLUDE_BOTH_PHASE": "1"}))
    for label, extra in envs:
        env = dict(os.environ, **BASE)
        env.update(extra)
        env.update(PYTHONUNBUFFERED="1")   # (config.py probes the GPU at import;
                                            # the child exits, so no context lingers)
        print("  [{}]".format(label))
        rc = subprocess.call([sys.executable, "-c", _TWIN_CHECK], env=env, cwd=HERE)
        if rc != 0:
            sys.exit("detector twin check failed (exit {}) -- not launching".format(rc))


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass
    argv = sys.argv[1:]
    dry = "--dry-run" in argv

    def after(flag):
        if flag not in argv:
            return None
        i = argv.index(flag)
        if i + 1 >= len(argv) or argv[i + 1].startswith("--"):
            sys.exit("{} needs a value".format(flag))
        return argv[i + 1]

    stages = after("--stage")
    stages = ([s.strip().upper() for s in stages.split(",") if s.strip()]
              if stages else DEFAULT_STAGES)
    for st in stages:
        if st not in STAGES:
            sys.exit("unknown stage " + st + "; valid: " + ", ".join(STAGES))
    only = after("--only")
    only = {s.strip() for s in only.split(",") if s.strip()} if only else None
    os.makedirs(LOG_DIR, exist_ok=True)

    plan, tot_min, tot_runs = [], 0.0, 0
    for st in stages:
        for name, env, seeds, per in STAGES[st]:
            if only and name not in only:
                continue
            seeds = os.environ.get("SEEDS", seeds)
            n = len([s for s in seeds.split(",") if s.strip()])
            plan.append((st, name, env, seeds, n, n * per))
            tot_min += n * per
            tot_runs += n
    if not plan:
        sys.exit("nothing to run; arms in {}: {}".format(
            ",".join(stages), ", ".join(nm for st in stages for nm, _, _, _ in STAGES[st])))

    print("#" * 74)
    print("# ROUND 7  configs={}  runs={}  ~{:.1f} h".format(
        len(plan), tot_runs, tot_min / 60))
    print("#" * 74)
    for st, name, env, seeds, n, m in plan:
        knobs = {k: v for k, v in env.items() if BASE.get(k) != v}
        print("  [{}] {:<24} x{:<3} ~{:4.1f} h  {}".format(st, name, n, m / 60, knobs))
    if dry:
        print("\nDry run complete -- nothing launched.")
        return

    print("\nPre-flight:")
    preflight_compile()
    preflight_stale_dirs()
    preflight_detector_twins(plan)

    res, done, spent = {}, 0, 0.0
    try:
        for st, name, env_over, seeds, n, est in plan:
            print("\n{}\n=== [{}] {}  ({} seeds, ~{:.1f} h) ===".format(
                "=" * 74, st, name, n, est / 60))
            # PYTHONUNBUFFERED: the child's stdout is a FILE here, so Python
            # would block-buffer it and the progress reader would see nothing
            # for minutes at a time.
            env = dict(os.environ, **env_over, SEEDS=seeds,
                       PYTHONUNBUFFERED="1")
            lp = os.path.join(LOG_DIR, "r7-{}.log".format(name))
            print("  console -> " + lp)
            stop = threading.Event()
            threading.Thread(target=tail_progress,
                             args=(lp, "[{}] {}".format(st, name), n, done,
                                   tot_runs, spent / 60, tot_min / 60, stop),
                             daemon=True).start()
            t0 = time.time()
            with open(lp, "a", encoding="utf-8", errors="replace") as fh:
                fh.write("\n===== {} {} seeds={} launched {} =====\n".format(
                    name, env_over, seeds, time.strftime("%Y-%m-%d %H:%M:%S")))
                fh.flush()
                rc = subprocess.call([sys.executable, LAUNCHER], env=env,
                                     cwd=HERE, stdout=fh,
                                     stderr=subprocess.STDOUT)
            stop.set()
            hrs = (time.time() - t0) / 3600.0
            spent += hrs * 60
            done += n
            res[name] = (rc, hrs)
            print("  {} {} after {:.1f} h".format(
                name, "OK" if rc == 0 else
                "exit {} (check done.flags)".format(rc), hrs))
    except KeyboardInterrupt:
        print("\nInterrupted:", {k: "{:.1f}h".format(v[1]) for k, v in res.items()})
        print("Re-running resumes where it left off.")
        sys.exit(130)

    print("\n" + "#" * 74 + "\n# ROUND 7 SUMMARY")
    for nm, (rc, h) in res.items():
        print("#   {:<24} {:<8} {:.1f} h".format(
            nm, "OK" if rc == 0 else "exit {}".format(rc), h))
    print("#" * 74)
    print("# Ensemble scores:  python ensemble_eval.py runs --seeds " + S12
          + " --subset-size 10")
    print("# S0 reading:       python sanity_compare.py --a runs "
          "\"sanity_xb_pos_head/h128/L2_dp0.3_xb_w1-6-1\" --b runs "
          "\"pos_head/h128/L2_dp0.3_w1-6-1\"")
    print("# Export with:      python export_results.py runs Round7_upload "
          "--with-plots 2")


if __name__ == "__main__":
    main()
