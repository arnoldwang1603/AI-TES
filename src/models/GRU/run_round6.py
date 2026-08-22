"""Round 6 — settle how the position parametrization should be gated.

Where round 5 left us (384 runs, 36 configs):
  * pos_head h128x2 = 0.890 +- 0.077 over 10 seeds, the project's best and by
    far its tightest (first config with zero runs above 1.5 C). Reparametrizing
    T_avg as a POSITION between the two surfaces is what did it; the flagged
    late-drift cases dropped 72%.
  * Architecture barely matters: the whole h{64..256} x L{1,2,3,5} x
    dropout{0.1,0.2,0.3} grid spans 1.14-1.48 with overlapping error bars, and
    the round-4 "2 layers wins" result did NOT replicate at n=20 (p=0.35).
  * Cases 32-40 / 51-53 stay weak. T_avg rises ABOVE both surfaces for 3-6% of
    their timesteps, so it is no longer between anything and the position stops
    meaning anything. Ranking all 70 cases by how far the position falls
    outside the head's reachable band puts exactly those 12 on top.

The whole question this round is WHERE to stop trusting the position, and
four mechanisms answer it differently. S1 sweeps them properly rather than
picking one:

  recondition   POS_GAP_FLOOR: never switch away, just bound the denominator so
                pos = excess/gap cannot blow up (measured max 66 against a
                typical 0.27). POS_FLOOR_SOFT is the same asymptote without the
                kink at the threshold.
  timestep gate POS_TEMP_GATE: hand T_avg to a dedicated absolute head while
                the surfaces are within N degrees. POS_TEMP_SOFT ramps the
                handover, which also keeps that head fed with gradient on both
                sides of the threshold.
  case gate     POS_CASE_GATE (Arnold, 2026-08-20): hand over whole runs that
                contain both a charging and a discharging phase -- 19 of the 20
                excursion cases qualify. Read off the input temperature profile
                alone: 20/20 hits, 1 false alarm, 69/70 correct.
  learned gate  POS_LEARNED_GATE: the model predicts the handover weight
                itself. Worth having because every threshold above is derived
                from where the excursions sit in THIS dataset, which may not be
                the boundary that actually matters. POS_LEARNED_POOL makes it
                a per-CASE router (pooled logit, jump-free); POS_GATE_BIAS=10
                pins it inert as the 5-channel null.
  fallback      POS_ANCHORED_FALLBACK: what the gates hand over TO. Default is
                a raw absolute head -- the family this project refuted three
                times; the anchored version predicts the deviation from the
                model's own surface midpoint instead, inheriting the surfaces'
                accuracy. POS_FIT_CLEAN separately drops excursion steps from
                the position fit (they inflate sd ~34%).

  S1 gating   34 arms x 20 seeds. Every mechanism swept alone before any
              combination, plus two controls: plain pos_head, and the extra
              absolute head with nothing gating to it. That second control
              matters -- the extra output channel shifts the RNG stream, so a
              4-channel arm and a 3-channel arm do not share an initialization
              even at the same seed, and without it a gating "win" could just
              be that shift.
  S3 arch     capacity re-scanned under pos_head, 12 seeds. The round-5 grid
              was measured under the OLD parametrization. L=1 is excluded --
              PyTorch applies dropout only BETWEEN layers, so at L=1 the
              dropout axis is a no-op and round 5 wasted 40 runs on exact
              duplicates.
  S4 weights  loss weights re-scanned under pos_head, 12 seeds. 1/6/3 was
              inherited from the LSTM line and confirmed best under the old
              parametrization; T_avg's weight now steers a position head.
  S5 AR       does the position idea transfer to the autoregressive
              formulation? Runs last (~5 h/seed measured in round 4 vs ~3 min
              for a direct run; 8 runs ~ 40 h, more than S1-S4 combined), so
              every direct-model conclusion is in hand before it starts and
              killing the round after S4 costs nothing. It uses the plain
              position head -- once S1 names a gate, this pair is worth
              re-running with it.

Decision rule, fixed BEFORE the run (31 selection arms invite a winner's
curse -- the best of 32 at n=20 overstates its edge, so the analysis must
rank, not test-each-vs-base at 0.05):
  primary    overall MAE, paired per seed against `base` (same seed lists)
  secondary  T_avg MAE on the 20 excursion cases (32-40, 50-53, 59-65)
  tertiary   spurious-jump count (max over channels of |step change| beyond
             truth, per case) -- hard per-step gates can CREATE exactly the
             discontinuities Arnold flagged in earlier reviews, so an arm
             that wins MAE but jumps is not a winner
  a winner within noise of a simpler arm loses to the simpler arm.

Usage:
    python run_round6.py                 # everything, ~73 h (S5 is ~40 of it)
    python run_round6.py --stage S1      # just the gating sweep, ~23 h
    python run_round6.py --stage S5      # just the AR comparison, ~40 h
    python run_round6.py --only tgate30
    python run_round6.py --dry-run
Env: SEEDS overrides everything; PROGRESS_EVERY tunes the status cadence.
Resumable; Ctrl+C safe. Logs -> sweep_logs/r6-<config>.log
"""
import os
import re
import subprocess
import sys
import threading
import time

S20 = "7,21,42,123,1,2,3,5,11,13,17,29,31,37,41,43,53,59,61,67"
S16 = "7,21,42,123,1,2,3,5,11,13,17,29,31,37,41,43"
S12 = "7,21,42,123,1,2,3,5,11,13,17,29"

# Everything inherits the round-5 champion; stages vary one axis at a time.
# EVERY knob the child process reads is pinned here, so a stray
# `set POS_CASE_GATE=1` left in the launching shell cannot silently turn
# every arm into a gated one (2026-08-22 review finding).
BASE = dict(TINNER_MODE="anchor", ANCHOR_LEAD="1", PHYSICS_BOUND_WEIGHT="0",
            VARIANTS="forward_direct", LOSS_WEIGHTS="1,6,3",
            OTHER_CH_MODE="pos_head", HIDDEN_SIZE="128", NUM_LAYERS="2",
            DROPOUT="0.3", WINDOW_SIZE="10", SLIDING_PAD_MODE="variable",
            ANCHOR_SCALE="1.0", INPUT_LOOKAHEAD="0",
            POS_GAP_FLOOR="0", POS_FLOOR_SOFT="0",
            POS_TEMP_GATE="0", POS_TEMP_SOFT="0", POS_CASE_GATE="0",
            POS_LEARNED_GATE="0", POS_GATE_BIAS="2.0", POS_LEARNED_POOL="0",
            POS_ABS_HEAD="0", POS_ANCHORED_FALLBACK="0", POS_FIT_CLEAN="0")


def cfg(**kw):
    d = dict(BASE)
    d.update({k: str(v) for k, v in kw.items()})
    return d


def mins(h=128, l=2):
    """Rough per-seed minutes for the direct model."""
    return max(2, round((h / 128.0) ** 1.3 * (1 + 0.45 * (l - 1))))


STAGES = {
    "S1": (
        # -- controls ------------------------------------------------------
        # base: RUN_NAME matches the round-5 champion dirs, so on a machine
        # that ran round 5 the first 10 seeds resume via done.flag. The base
        # code path is numerically unchanged (gates off, floor off), so
        # mixing is safe -- delete those dirs to force a clean re-run.
        [("base", cfg(), S20, mins()),
         # 4 channels, nothing gating to them: isolates the RNG shift and the
         # extra parameters from any actual gating effect.
         ("abshead", cfg(POS_ABS_HEAD=1), S20, mins())]
        # -- (a) recondition only, never switch ----------------------------
        + [("floor{}".format(f), cfg(POS_GAP_FLOOR=f), S20, mins())
           for f in (10, 20, 40, 80)]
        + [("floor{}-soft".format(f), cfg(POS_GAP_FLOOR=f, POS_FLOOR_SOFT=1),
            S20, mins()) for f in (20, 40, 80)]
        # -- (b) timestep gate, hard switch --------------------------------
        + [("tgate{}".format(t), cfg(POS_TEMP_GATE=t), S20, mins())
           for t in (10, 20, 30, 50)]
        # -- (c) timestep gate, ramped handover ----------------------------
        #    threshold and ramp width swept semi-independently: a wide ramp
        #    from a low threshold is a different shape from a narrow ramp off
        #    a high one, and only the ramp keeps the absolute head trained.
        + [("tgate{}-soft{}".format(t, w), cfg(POS_TEMP_GATE=t, POS_TEMP_SOFT=w),
            S20, mins())
           for t, w in ((20, 20), (30, 15), (30, 30), (30, 60), (50, 50),
                        (10, 40))]
        # -- (d) case gate --------------------------------------------------
        #    cgate alone keeps 3 channels (Arnold's original: gated cases
        #    simply predict T_avg absolutely) -- but that makes channel 2
        #    serve two roles, z on ungated cases and a scaled temperature on
        #    gated ones. cgate-ah gives the fallback its own 4th channel, so
        #    the pair separates "case gate" from "dual-use channel".
        + [("cgate", cfg(POS_CASE_GATE=1), S20, mins()),
           ("cgate-ah", cfg(POS_CASE_GATE=1, POS_ABS_HEAD=1), S20, mins())]
        # -- (e) learned gate ----------------------------------------------
        #    bias sets where training starts: +2 trusts the position formula
        #    (w=0.88), 0 is uncommitted, +4 nearly always trusts it.
        + [("learned", cfg(POS_LEARNED_GATE=1), S20, mins()),
           ("learned-b0", cfg(POS_LEARNED_GATE=1, POS_GATE_BIAS=0), S20, mins()),
           ("learned-b4", cfg(POS_LEARNED_GATE=1, POS_GATE_BIAS=4), S20, mins()),
           # per-CASE learned router: pooled logit, jump-free by construction
           ("learned-case", cfg(POS_LEARNED_GATE=1, POS_LEARNED_POOL=1),
            S20, mins()),
           # 5-channel null: bias +10 pins sigmoid at ~1-5e-5, so the gate is
           # inert -- the 5-channel analogue of abshead. Without it a
           # "learned" win is confounded with the OUTPUT_SIZE=5 RNG shift.
           ("learned-b10", cfg(POS_LEARNED_GATE=1, POS_GATE_BIAS=10),
            S20, mins())]
        # -- (e2) anchored fallback: the gate hands over to a fallback that is
        #    itself anchored on the model's own surfaces (midpoint + z*sd)
        #    instead of a raw absolute head -- the raw-absolute family is the
        #    one this project has refuted three times.
        + [("tgate30s30-afb", cfg(POS_TEMP_GATE=30, POS_TEMP_SOFT=30,
                                  POS_ANCHORED_FALLBACK=1), S20, mins()),
           ("cgate-afb", cfg(POS_CASE_GATE=1, POS_ANCHORED_FALLBACK=1),
            S20, mins())]
        # -- (e3) clean fit: excursion steps excluded from the mu/sd fit
        #    (they inflate sd ~34% and cost normal-regime resolution)
        + [("base-cleanfit", cfg(POS_FIT_CLEAN=1), S20, mins())]
        # -- (f) combinations ----------------------------------------------
        + [("floor40-cgate-ah", cfg(POS_GAP_FLOOR=40, POS_CASE_GATE=1,
                                    POS_ABS_HEAD=1), S20, mins()),
           ("floor40-tgate30s30", cfg(POS_GAP_FLOOR=40, POS_TEMP_GATE=30,
                                      POS_TEMP_SOFT=30), S20, mins()),
           ("floor40-learned", cfg(POS_GAP_FLOOR=40, POS_LEARNED_GATE=1),
            S20, mins()),
           ("tgate30s30-cgate", cfg(POS_TEMP_GATE=30, POS_TEMP_SOFT=30,
                                    POS_CASE_GATE=1), S20, mins()),
           ("learned-cgate", cfg(POS_LEARNED_GATE=1, POS_CASE_GATE=1),
            S20, mins())]
    ),
    "S3": [
        ("arch-h{}x{}".format(h, l), cfg(HIDDEN_SIZE=h, NUM_LAYERS=l), S12,
         mins(h, l))
        for h in (96, 128, 192, 256) for l in (2, 3, 5)
        if not (h == 128 and l == 2)          # that is S1's base
    ],
    "S4": [
        ("w" + w.replace(",", ""), cfg(LOSS_WEIGHTS=w), S12, mins())
        for w in ("1,6,6", "1,6,1", "1,6,10", "1,4,3", "1,3,3", "1,8,3",
                  "1,6,0.5", "2,6,3")
    ],
    "S5": [
        # 300 min/seed from measurement, not folklore: round 4 ran the
        # identical abs_sliding config at ~4.25 h/seed (B-ar stage, 2 seeds in
        # ~8.5 h) and the 2026-07-16 log books 4.5-5.9 h/seed. 4 seeds per
        # arm -- n=2 sits below the project's own evidence bar.
        ("AR-pos_head", cfg(VARIANTS="abs_sliding", NUM_LAYERS=5),
         "7,21,42,123", 300),
        ("AR-base", cfg(VARIANTS="abs_sliding", NUM_LAYERS=5,
                        OTHER_CH_MODE="abs"), "7,21,42,123", 300),
    ],
}
# Everything, cheapest first: S1-S4 land in ~31 h, then the AR
# comparison adds ~72 h on its own (~18 h per seed against ~3 min
# for a direct run). Ordered last so every direct-model result is
# already in hand before that starts, and so killing the round
# after S4 costs nothing.
DEFAULT_STAGES = ["S1", "S3", "S4", "S5"]

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


def main():
    dry = "--dry-run" in sys.argv
    stages = ([sys.argv[sys.argv.index("--stage") + 1].upper()]
              if "--stage" in sys.argv else DEFAULT_STAGES)
    for st in stages:
        if st not in STAGES:
            sys.exit("unknown stage " + st + "; valid: " + ", ".join(STAGES)
                     + "  (S2 is intentionally absent -- stage numbering"
                       " carries over from round 5)")
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    os.makedirs(LOG_DIR, exist_ok=True)

    plan, tot_min, tot_runs = [], 0.0, 0
    for st in stages:
        for name, env, seeds, per in STAGES[st]:
            if only and name != only:
                continue
            seeds = os.environ.get("SEEDS", seeds)
            n = len([s for s in seeds.split(",") if s.strip()])
            plan.append((st, name, env, seeds, n, n * per))
            tot_min += n * per
            tot_runs += n

    print("#" * 74)
    print("# ROUND 6  configs={}  runs={}  ~{:.1f} h".format(
        len(plan), tot_runs, tot_min / 60))
    print("#" * 74)
    for st, name, env, seeds, n, m in plan:
        knobs = {k: v for k, v in env.items() if BASE.get(k) != v}
        print("  [{}] {:<16} x{:<3} ~{:4.1f} h  {}".format(st, name, n, m / 60, knobs))
    if dry:
        print("\nDry run complete -- nothing launched.")
        return

    # Pre-flight (2026-08-22 review): a done.flag makes the sweep silently
    # reuse a directory, and one stale smoke-test dir already got mixed into
    # an arm once. Round-6 gating dirs must carry the full provenance the
    # current runio.py writes; round-5 reuse (base / arch arms) is exempt
    # because those names carry no round-6 tag.
    import json as _json
    _tags = ("_tg", "_fl", "_cgate", "_lg", "_ah", "_afb", "_cf")
    stale = []
    runs_root = os.path.join(HERE, "runs")
    if os.path.isdir(runs_root):
        for d in os.listdir(runs_root):
            if not any(t in d for t in _tags):
                continue
            rdir = os.path.join(runs_root, d)
            rc = os.path.join(rdir, "run_config.json")
            flags = [f for dp, _, fs in os.walk(rdir) for f in fs
                     if f == "done.flag"]
            if not flags or not os.path.isfile(rc):
                continue
            try:
                keys = _json.load(open(rc))
            except Exception:
                stale.append(d + "  (unreadable run_config)")
                continue
            if "pos_fit_clean" not in keys:
                stale.append(d + "  (pre-round-6 code)")
    if stale:
        print("\nREFUSING TO LAUNCH -- stale done-flagged run dirs would be")
        print("silently reused by these arms. Delete or rename them first:")
        for d in stale:
            print("   runs/" + d)
        sys.exit(2)

    res, done, spent = {}, 0, 0.0
    try:
        for st, name, env_over, seeds, n, est in plan:
            print("\n{}\n=== [{}] {}  ({} seeds, ~{:.1f} h) ===".format(
                "=" * 74, st, name, n, est / 60))
            # PYTHONUNBUFFERED: the child's stdout is a FILE here, so Python would
            # block-buffer it and the progress reader below would see
            # nothing for minutes at a time.
            env = dict(os.environ, **env_over, SEEDS=seeds,
                       PYTHONUNBUFFERED="1")
            lp = os.path.join(LOG_DIR, "r6-{}.log".format(name))
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

    print("\n" + "#" * 74 + "\n# ROUND 6 SUMMARY")
    for nm, (rc, h) in res.items():
        print("#   {:<18} {:<8} {:.1f} h".format(
            nm, "OK" if rc == 0 else "exit {}".format(rc), h))
    print("#" * 74)
    print("# Export with:  python export_results.py runs Round6_upload "
          "--with-plots 2 --title 'Round 6'")


if __name__ == "__main__":
    main()
