"""Round 8 -- close the base-data line: combine the two round-7 survivors,
give T_outer a reference built from the inlet temperature (Arnold), and give
the autoregressive branch a fair final comparison.

Where round 7 left us (448 runs, 2026-09-12 log entry):
  * Winner: the case gate with a plain absolute 4th channel (`cgate-ah`) under
    1-6-1: 0.718 +- 0.058, 0/20 blow-ups, the 12 hard cases halved, ensemble
    0.676. Round 6's best gate (tgate10-soft40) blew up 2/20 under the new
    weights; the gate ranking does not transfer across loss weights.
  * The phase flag as an input (`fip`) leaves the 12 hard cases alone but
    improves the other 49 by 0.25 and removes a third of the spurious jumps.
    fip and cgate-ah work on DIFFERENT cases and were never combined.
  * Weights and capacity are closed (1-6-1, h128x2). Hard timestep gates add
    jumps; the constant case flag (fic) hurts.
  * S0: with both-phase runs removed the gated arms reproduce their nulls bit
    for bit; removing those runs from TRAINING costs the other 49 cases
    0.55 C. The both-phase runs stay.
  * AR: every AR arm so far ran at 5 layers (the 2026-07 protocol) while the
    direct model moved to 2; under pos_head, L5 is the worst depth of the
    grid. AR's best (1.06) has never been measured at L2.
  * 2026-09-10 meeting: Arnold asked for T_outer anchored on the live input
    temperature, exactly as T_inner is ("time-dependent, not a constant").

The T_outer reference, measured before launch (training set, residual std
against T_outer's own 115.8 C): raw Input_T(t) 97.2; Input_T lagged 720
steps 39.8; an exponential moving average of Input_T with time constant tau
= 480 / 720 / 1000 / 1200 / 1500 steps -> 38.3 / 21.3 / 9.4 / 7.3 / 12.0.
The outer wall is a first-order low-pass of the inlet (fitted slope 1.001),
and EMA(1200) removes 94% of its variance; the round-4 initial-value anchor
removed 50% and lost, the T_inner anchor removes 99% and won. Both
references run: the literal request (To-ai) and the physical one (To-em1200).

Stages:
  S1 combine  20 seeds, h128x2, 1-6-1:
                w161                    the round-7 base, S2's null (resumes)
                w161-cgate-ah           the round-7 winner (resumes from the
                                        round-7 dir on the lab server)
                w161-fip-cgate-ah       the two survivors combined
                w161-fiz-cgate-ah       its null (constant-zero input column)
  S2 anchor   20 seeds, TOUTER_MODE. The anchor adds no channel and no input
              column, so its null is the same arm without it:
                w161-To-ai              T_outer = Input_T(t+1) + delta   vs w161
                w161-To-em900 / -em1200 / -em1500
                                        T_outer = EMA_tau(Input_T)(t+1) + delta
                                        vs w161; the tau bracket is there
                                        because the reference's own best
                                        smoothing (1200) need not be the
                                        model's best
                w161-To-em1200-cgate-ah on the round-7 winner   vs w161-cgate-ah
                w161-fip-To-em1200-cgate-ah
                                        on the full stack        vs w161-fip-cgate-ah
                w161-To-em1200-os2      output range x2             vs w161-To-em1200
                w163-To-em1200, w131-To-em1200
                                        loss balance re-checked under the
                                        anchor                    vs w161-To-em1200
              (delta z-scored on the train set; the T_inner recipe verbatim,
              with its own range multiplier TOUTER_SCALE; the EMA is seeded
              at T_outer(0); the position formula consumes the anchored
              T_outer, so a T_outer gain should show up in T_avg too -- that
              is the mechanism that made 1-6-1 win.)
              The formula ALONE (reference + train-set mean offset, no
              model) is scored on the 70 test cases before launch and logged
              as the floor: an arm whose T_outer is not below its own floor
              has learned nothing on top of the reference.
  S3 AR       4 seeds, abs_sliding at L2 (the direct model's depth), 1-6-1:
                AR-w161-L2              plain position head -- isolates what
                                        L5 cost the round-6/7 AR arms
                AR-w161-cgate-ah-L2     the winner, autoregressive
                AR-w161-fip-cgate-ah-L2 the combination, autoregressive (the
                                        flag column now rides in the window)
                AR-w161-To-em1200-L2, AR-w161-To-em1200-cgate-ah-L2,
                AR-w161-fip-To-em1200-cgate-ah-L2
                                        the S2 main line mirrored: the EMA
                                        anchor alone, on the winner, on the
                                        full stack (applied per step, before
                                        the position formula and before the
                                        value feeds back into the window)
              Not in DEFAULT_STAGES (2026-09-13): the direct results come
              first; the AR stage then runs on request with the base arm and
              the direct winner only, so the round never blocks for a week.
              Descriptive: n=4 enters no rule, and at n=4 no null could
              resolve the flag's effect (its direct-model size is 0.04 C
              against an AR seed spread of 0.3), so the fiz null is not run.
              Booked at the measured L5 cost (300 min/seed; L2 is unmeasured
              and should be faster).

Decision rule (the round-7 rule, plus the round-8 specifics):
  pairing    fip-cgate-ah vs fiz-cgate-ah; To-ai and To-em{900,1200,1500}
             vs w161; To-em1200-cgate-ah vs cgate-ah; fip-To-em1200-cgate-ah
             vs fip-cgate-ah; AR arms descriptive only.
  primary    paired overall-MAE difference vs the null, sign-flip p, 95% CI,
             arms ranked; "within noise" = CI includes 0.
  secondary  T_avg on the 12 flagged / 21 switched / 49 untouched cases, no
             paired degradation on any subset vs the arm's null.
  veto       any seed > 1.5 C on the common seeds; a paired increase in
             JumpSteps_gt2C with p < 0.05.
  deployment seed-ensemble MAE on the common 20 seeds (ensemble_eval.py
             --seeds S20); ties on the primary go to the ensemble.
  simplicity 3-ch < 4-ch, no flag < flag, no T_outer anchor < anchor, one
             gate < two; a winner within noise of a simpler arm loses to it.
  expected   the results are expected to step up along the stack:
  progression  cgate-ah (0.718 measured)
               -> fip-cgate-ah        ~0.70  (fip's -0.16 on the 49 untouched
                                              cases added to the gate's -0.84
                                              on the 21 switched ones)
               -> To-em1200-cgate-ah  T_outer below its 0.45 and T_avg
                                       following it through the position
                                       formula, the mechanism that made
                                       1-6-1 win; overall ~0.68 if so
               -> fip-To-em1200-cgate-ah  ~0.65, both effects together
             Each step is CHECKED against its parent (the same arm without
             the added piece): paired 95% CI below zero, no veto, and for
             fip-cgate-ah also against fiz-cgate-ah (the flag's null; the
             null shift (fiz-cgate-ah - cgate-ah) reported, and the
             comparison repeated without any blow-up seed of the null). A
             step that does not materialise is dropped from the stack and
             the next one is tested on the previous survivor. The last
             survivor ships as its 20-seed ensemble.
             The reference itself: To-ai is expected within noise of w161
             (its floor is 73 C); the tau bracket is reported as a curve,
             with 1200 expected best; os2 is expected to matter only on the
             tail cases (the 12 flagged, where the residual is largest);
             the weight arms are expected to confirm 1-6-1, with 1-3-1 the
             candidate if the anchor has made T_outer easy enough that its
             weight was starving T_avg.

Resume note: w161-cgate-ah and every round-7 arm share RUN_NAMEs with the
round-7 dirs, and the code path with every round-8 knob off is numerically
unchanged (TOUTER_MODE=abs is a no-op; the AR flag column only exists when
CASE_FLAG_INPUT is set; the case gate is now computed once per rollout
instead of once per step, which is the same boolean), so on the lab server
those resume via done.flag. A MAX_EPOCHS smoke run gets its own `ep<N>`
directory and can never collide with a real seed; the stale-dir guard below
additionally refuses any finished dir that neither hit the 1000-epoch cap
nor early-stopped (Total_Epoch < 151 is impossible for a real run).

Usage:
    python run_round8.py                 # S1 + S2 (the direct arms), ~12 h
    python run_round8.py --stage S3 --only AR-w161-L2,AR-w161-<winner>-L2
                                         # afterwards: the AR branch, base +
                                         # the direct winner only (~40 h);
                                         # all six S3 arms would be ~120 h
    python run_round8.py --only w161-To-em1200
    python run_round8.py --dry-run
Env: SEEDS overrides everything; PROGRESS_EVERY tunes the status cadence.
Resumable; Ctrl+C safe. Logs -> sweep_logs/r8-<config>.log
"""
import glob
import json
import os
import re
import subprocess
import sys
import threading
import time

S20 = "7,21,42,123,1,2,3,5,11,13,17,29,31,37,41,43,53,59,61,67"
S4 = "7,21,42,123"

# The round-7 champion configuration. EVERY knob the child reads is pinned.
BASE = dict(TINNER_MODE="anchor", ANCHOR_LEAD="1", PHYSICS_BOUND_WEIGHT="0",
            VARIANTS="forward_direct", LOSS_WEIGHTS="1,6,1",
            OTHER_CH_MODE="pos_head", HIDDEN_SIZE="128", NUM_LAYERS="2",
            DROPOUT="0.3", WINDOW_SIZE="10", SLIDING_PAD_MODE="variable",
            ANCHOR_SCALE="1.0", INPUT_LOOKAHEAD="0",
            POS_GAP_FLOOR="0", POS_FLOOR_SOFT="0",
            POS_TEMP_GATE="0", POS_TEMP_SOFT="0", POS_CASE_GATE="0",
            POS_LEARNED_GATE="0", POS_GATE_BIAS="2.0", POS_LEARNED_POOL="0",
            POS_ABS_HEAD="0", POS_ANCHORED_FALLBACK="0", POS_FIT_CLEAN="0",
            CASE_FLAG_INPUT="", EXCLUDE_BOTH_PHASE="0",
            TOUTER_MODE="abs", TOUTER_TAU="1200", TOUTER_SCALE="1.0",
            MAX_EPOCHS="1000")
RUN_PREFIX = "2026-08-06_"          # every arm's RUN_NAME starts with this


def cfg(**kw):
    d = dict(BASE)
    d.update({k: str(v) for k, v in kw.items()})
    return d


def cah(**kw):
    """cgate-ah: case gate handing T_avg to a plain absolute 4th channel."""
    kw.setdefault("POS_CASE_GATE", 1)
    kw.setdefault("POS_ABS_HEAD", 1)
    return cfg(**kw)


def ar(**kw):
    kw.setdefault("VARIANTS", "abs_sliding")
    return cfg(**kw)


def mins(h=128, l=2):
    return max(2, round(2.0 * (h / 128.0) ** 1.3 * (1 + 0.45 * (l - 1))))


STAGES = {
    "S1": [
        # both references resume from the round-7 dirs when present, and
        # train only if the runs/ folder was wiped -- the S2 arms are paired
        # against w161, so it must exist next to them
        ("w161", cfg(), S20, mins()),
        ("w161-cgate-ah", cah(), S20, mins()),
        ("w161-fip-cgate-ah", cah(CASE_FLAG_INPUT="phase"), S20, mins()),
        ("w161-fiz-cgate-ah", cah(CASE_FLAG_INPUT="zero"), S20, mins()),
    ],
    "S2": [
        ("w161-To-ai", cfg(TOUTER_MODE="anchor_input"), S20, mins()),
        # tau bracket around the train-set optimum (1200): the model's best
        # smoothing need not be the reference's own best
        ("w161-To-em900", cfg(TOUTER_MODE="anchor_ema", TOUTER_TAU=900), S20, mins()),
        ("w161-To-em1200", cfg(TOUTER_MODE="anchor_ema", TOUTER_TAU=1200), S20, mins()),
        ("w161-To-em1500", cfg(TOUTER_MODE="anchor_ema", TOUTER_TAU=1500), S20, mins()),
        # the anchor on the main line: on the winner, and on the full stack
        ("w161-To-em1200-cgate-ah", cah(TOUTER_MODE="anchor_ema", TOUTER_TAU=1200),
         S20, mins()),
        ("w161-fip-To-em1200-cgate-ah", cah(TOUTER_MODE="anchor_ema", TOUTER_TAU=1200,
                                            CASE_FLAG_INPUT="phase"), S20, mins()),
        # the same companions the position head got in rounds 5-6, 20 seeds:
        # output range (the residual's tail is 6 sigma) and the loss balance
        # (1-6-1 was tuned with T_outer as a direct head)
        ("w161-To-em1200-os2", cfg(TOUTER_MODE="anchor_ema", TOUTER_TAU=1200,
                                   TOUTER_SCALE=2.0), S20, mins()),
        ("w163-To-em1200", cfg(TOUTER_MODE="anchor_ema", TOUTER_TAU=1200,
                               LOSS_WEIGHTS="1,6,3"), S20, mins()),
        ("w131-To-em1200", cfg(TOUTER_MODE="anchor_ema", TOUTER_TAU=1200,
                               LOSS_WEIGHTS="1,3,1"), S20, mins()),
    ],
    "S3": [
        ("AR-w161-L2", ar(), S4, 300),
        ("AR-w161-cgate-ah-L2", ar(POS_CASE_GATE=1, POS_ABS_HEAD=1), S4, 300),
        ("AR-w161-fip-cgate-ah-L2", ar(POS_CASE_GATE=1, POS_ABS_HEAD=1,
                                       CASE_FLAG_INPUT="phase"), S4, 300),
        # the S2 main line mirrored: the EMA anchor alone, on the winner, and
        # on the full stack (the AR path applies it per step, before the
        # position formula and before the value feeds back)
        ("AR-w161-To-em1200-L2", ar(TOUTER_MODE="anchor_ema", TOUTER_TAU=1200),
         S4, 300),
        ("AR-w161-To-em1200-cgate-ah-L2", ar(POS_CASE_GATE=1, POS_ABS_HEAD=1,
                                             TOUTER_MODE="anchor_ema",
                                             TOUTER_TAU=1200), S4, 300),
        ("AR-w161-fip-To-em1200-cgate-ah-L2", ar(POS_CASE_GATE=1, POS_ABS_HEAD=1,
                                                 CASE_FLAG_INPUT="phase",
                                                 TOUTER_MODE="anchor_ema",
                                                 TOUTER_TAU=1200), S4, 300),
    ],
}
# S3 (AR, ~120 h for all six arms) is OFF by default: run the direct stages
# first, then give the AR branch only the direct winner, e.g.
#     python run_round8.py --stage S3 --only AR-w161-L2,AR-w161-fip-To-em1200-cgate-ah-L2
DEFAULT_STAGES = ["S1", "S2"]

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
    import compileall
    if not compileall.compile_dir(os.path.join(HERE, "tes_gru"),
                                  quiet=1, force=True):
        sys.exit("tes_gru does not compile under " + sys.version.split()[0]
                 + " -- fix the syntax error above before launching")


def preflight_stale_dirs():
    """A done.flag makes the sweep silently reuse a directory. Only dirs an
    arm of this round could resume (current name scheme) are examined.
    Refused: dirs written by older code (missing provenance keys), and any
    finished dir that neither hit the 1000-epoch cap nor early-stopped --
    a real run cannot finish below 151 epochs (patience 150), so a leftover
    smoke test is unforgeable here even if its run_config was rewritten."""
    tags = ("_tg", "_fl", "_cgate", "_lg", "_ah", "_afb", "_cf", "_fi", "_xb",
            "_To-", "abs_sliding")
    stale = []
    runs_root = os.path.join(HERE, "runs")
    if os.path.isdir(runs_root):
        for d in os.listdir(runs_root):
            if not d.startswith(RUN_PREFIX):
                continue
            rdir = os.path.join(runs_root, d)
            rc = os.path.join(rdir, "run_config.json")
            flags = glob.glob(os.path.join(rdir, "variants", "*", "done.flag"))
            if not flags or not os.path.isfile(rc):
                continue
            if "_ep" in d and re.search(r"_ep\d+(_|$)", d):
                stale.append(d + "  (MAX_EPOCHS smoke-test dir; delete it)")
                continue
            short = []
            for f in flags:
                mp = os.path.join(os.path.dirname(f), "meta.json")
                try:
                    tot = int(json.load(open(mp)).get("Total_Epoch", 1000))
                except Exception:
                    tot = 1000
                if tot < 151:
                    short.append(tot)
            if short:
                stale.append(d + "  (finished after only {} epochs -- smoke test?)".format(short[0]))
                continue
            if not any(t in d for t in tags):
                continue
            try:
                keys = json.load(open(rc))
            except Exception:
                stale.append(d + "  (unreadable run_config)")
                continue
            if "pos_fit_clean" not in keys:
                stale.append(d + "  (pre-round-6 code)")
            elif ("_fi" in d or "_xb" in d) and "case_flag_input" not in keys:
                stale.append(d + "  (pre-round-7 code)")
            elif "_To-" in d and "touter_mode" not in keys:
                stale.append(d + "  (pre-round-8 code)")
    if stale:
        print("\nREFUSING TO LAUNCH -- stale done-flagged run dirs would be")
        print("silently reused by these arms. Delete or rename them first:")
        for d in stale:
            print("   runs/" + d)
        sys.exit(2)


_TWIN_CHECK = r'''
import sys
from tes_gru.config import EXCLUDE_BOTH_PHASE
from tes_gru.data import load_all_data, ThermalDataset, input_case_flag_np
from tes_gru.rollout import _input_case_gate
tr, va, te, sc = load_all_data()
bad, counts = [], []
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
    counts.append((name, n, len(dfs)))
print("  detector twins: " + ", ".join("%s %d/%d fire" % c for c in counts))
if bad:
    print("  numpy / torch detectors DISAGREE:")
    for x in bad[:10]:
        print("    ", x)
    sys.exit(3)
'''

# The AR window must carry the flag column: build one abs_sliding dataset with
# CASE_FLAG_INPUT=phase and check the tensor width against INPUT_DIMS, the
# last column against the numpy phase flag, and the Input_T column against
# the dataset's own scaled inlet series (so idx 4 really is Input_T).
_AR_FLAG_CHECK = r'''
import sys, numpy as np
from tes_gru.config import INPUT_DIMS
from tes_gru.data import load_all_data, ThermalDataset, input_phase_flag_np, input_case_flag_np
tr, va, te, sc = load_all_data()
col = lambda df: "Input Temperature (C)" if "Input Temperature (C)" in df.columns else "T_inner (C)"
fire = [(df, cid) for df, cid in te if input_case_flag_np(df[col(df)].values)]
df, cid = fire[0]
ds = ThermalDataset(df, "abs_sliding", scaler=sc, file_name=cid)
X = ds.X[0].numpy()
want_flag = input_phase_flag_np(df[col(df)].values)[:-1]
want_inp = np.asarray(ds.full_input_temp[0], dtype=float)[:-1]
w_ok = X.shape[1] == INPUT_DIMS["abs_sliding"] == 6
f_ok = np.array_equal(X[:, -1], want_flag) and set(np.unique(X[:, -1])) <= {0.0, 1.0}
i_ok = np.allclose(X[:, 4], want_inp, atol=1e-6)
print("  AR flag column: width %d (INPUT_DIMS %d) ok=%s | last column == numpy phase flag: %s | idx 4 == scaled Input_T: %s"
      % (X.shape[1], INPUT_DIMS["abs_sliding"], w_ok, f_ok, i_ok))
sys.exit(0 if (w_ok and f_ok and i_ok) else 5)
'''


def preflight_children(plan):
    checks = [("detector twins", _TWIN_CHECK, {})]
    if any(env.get("VARIANTS") == "abs_sliding" and env.get("CASE_FLAG_INPUT")
           for _, _, env, _, _, _ in plan):
        checks.append(("AR flag column", _AR_FLAG_CHECK,
                       {"VARIANTS": "abs_sliding", "CASE_FLAG_INPUT": "phase"}))
    for label, code, extra in checks:
        env = dict(os.environ, **BASE)
        env.update(extra)
        env.update(PYTHONUNBUFFERED="1")
        print("  [{}]".format(label))
        rc = subprocess.call([sys.executable, "-c", code], env=env, cwd=HERE)
        if rc != 0:
            sys.exit("{} check failed (exit {}) -- not launching".format(label, rc))


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
    print("# ROUND 8  configs={}  runs={}  ~{:.1f} h  (resumed seeds finish instantly)".format(
        len(plan), tot_runs, tot_min / 60))
    print("#" * 74)
    for st, name, env, seeds, n, m in plan:
        knobs = {k: v for k, v in env.items() if BASE.get(k) != v}
        print("  [{}] {:<26} x{:<3} ~{:4.1f} h  {}".format(st, name, n, m / 60, knobs))
    if dry:
        print("\nDry run complete -- nothing launched.")
        return

    print("\nPre-flight:")
    preflight_compile()
    preflight_stale_dirs()
    preflight_children(plan)

    res, done, spent = {}, 0, 0.0
    try:
        for st, name, env_over, seeds, n, est in plan:
            print("\n{}\n=== [{}] {}  ({} seeds, ~{:.1f} h) ===".format(
                "=" * 74, st, name, n, est / 60))
            env = dict(os.environ, **env_over, SEEDS=seeds,
                       PYTHONUNBUFFERED="1")
            lp = os.path.join(LOG_DIR, "r8-{}.log".format(name))
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

    print("\n" + "#" * 74 + "\n# ROUND 8 SUMMARY")
    for nm, (rc, h) in res.items():
        print("#   {:<26} {:<8} {:.1f} h".format(
            nm, "OK" if rc == 0 else "exit {}".format(rc), h))
    print("#" * 74)
    print("# Ensemble (deployment metric, common 20 seeds):")
    print("#     python ensemble_eval.py runs --seeds " + S20 + " --subset-size 12")
    print("# AR vs direct at a common n=4:")
    print("#     python ensemble_eval.py runs --seeds " + S4 + " --min-seeds 4")
    print("# Export with:  python export_results.py runs Round8_upload --with-plots 2")


if __name__ == "__main__":
    main()
