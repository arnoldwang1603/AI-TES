"""Round 9 -- re-test the round 4-8 conclusions under a validation set that
covers every case type (branch V, stage S1: the configurations, then every
loss-weight vector that was ever scanned), and one network per temperature
instead of one network for all three (branch I, stage S2). Agreed at the
2026-09-17 meeting: "do that [the validation re-run] first for sure"; S2 is
the three-model split presented there. The user's standing rule (2026-09-19):
whatever was tested under the old validation set -- weights included -- is
re-run under the new one; nothing carries over untested.

Why (2026-09-16 audit; calit2-thermal-wiki/HANDOVER.md section 2):
  * Every round through 8 picked its best epoch and stopped early on a
    validation set of 15 runs that, sorted by path, all come from the
    "Case 1 (linear charging)" folder. Early stopping never saw a
    charge-then-discharge run.
  * Round 8's T_outer anchor (To-em1200) improved T_outer and wrecked T_avg,
    worst on the 21 gated cases (0.86 -> 4.26). Its validation loss bottomed
    out ~300 epochs earlier than the un-anchored arm (best epoch 324 vs 619),
    so training stopped while the 4th channel -- trained only on the 107
    gated runs, none of which are in validation -- was still converging.
    Hypothesis: the validation set, not the anchor, is what failed.
  * VAL_SPLIT=stratified (2026-09-17): 2 runs from each of the 10 case-type
    folders -> 20 validation / 291 training, drawn once (SPLIT_SEED=42) and
    frozen in splits/stratified_v2_s42.json, which ships with the code. The
    test set is untouched (70). Tag `vs2` in every run name; export_results
    files these under `vs2_*` families.

Stage S1 (branch V) -- the decision-relevant configuration of each round,
re-trained under the stratified split, 20 seeds each (the seed list every
round has used, so each arm pairs per seed with its legacy-split twin: the
round-6/7/8 twins sit in runs/ on the lab server, the round-4/5 twins in the
Round_4/5_Results folders of the wiki; r5-pos-h128x5 pairs on its 10 seeds):

    arm                    round   what it settled under the old split
    r4-abs-h128x5          4       direct T_avg, 5 layers: 1.28 (the pre-pos_head base)
    r4-anchor_avg-h128x5   4       the fixed-weight T_avg anchor: lost, 1.87
    r5-abs-h128x2          5       direct T_avg at 2 layers: 1.37 (20 seeds; the 1.11 in
                                   the round-4 log was an n=6 false positive) -- Arnold's
                                   2026-09-17 "use the absolute value for T_avg as well",
                                   at the current capacity (pairs with round-5 c-h128x2dp3)
    r5-pos-h128x5          5       position head at 5 layers: 1.07
    r5-pos-h128x2          5/6     position head at 2 layers: 0.890 / 0.973 -- the base
    r6-tg10s40             6       timestep soft gate: 0.822, 0/20 blow-ups
    r6-cgate-afb           6       case gate + midpoint head: 0.841, 0/20
    r6-cgate-ah            6       case gate + absolute head: 0.948, 2/20
    r6-ah                  6       the 4-channel null of the round-6 gates (1-6-3, ah)
    r7-w161                6/7     loss weight 1-6-1: 0.774 (12 seeds) / 0.787 -- the lever
    r7-w161-ah             7       the 4-channel null under 1-6-1 (round 7 paired the
                                   gates against it: +0.031 [-0.01, +0.08] vs w161)
    r7-w161-cgate-ah       7       the winner: 0.718, 0/20
    r7-w161-fip            7       phase flag as input: 0.733 (helps the 49)
    r7-w161-fiz            7       its null: 0.775
    r7-w161-tg30s30        7       soft timestep gate under 1-6-1: 0.752
    r8-w161-fip-cgate-ah   8       the two survivors combined: 0.707, within noise
    r8-w161-fiz-cgate-ah   8       its input-column null (round 8 paired fip-cgate-ah
                                   against it: -0.031 [-0.069, +0.006])
    r8-w161-To-em1200      8       T_outer anchored on the smoothed inlet: 0.935
    r8-w161-To-em1200-cgate-ah  8  the anchor on the winner: 1.147, T_avg wrecked

Decision rule, fixed BEFORE the run (the pilot below -- 3 seeds of two S1
arms and one S2 arm -- was in hand when the numeric thresholds were set; the
sweep's 20 seeds include those three, and every comparison is also reported
on the 17 non-pilot seeds):
  statistic  paired difference per seed, sign-flip p, 95% bootstrap CI;
             blow-up seeds (> 1.5 C) on either side counted and the
             comparison repeated without them (the round-7/8 practice).
  protocol   every vs2 arm is paired per seed against its legacy-split twin
             (same seeds, same config, tag differs). What changes between
             the twins is the protocol as a whole: the validation set (15
             linear-charging runs -> 2 of each type), the training set (296
             -> 291; 13 of the 15 old validation files move into training,
             so the scaler and every fitted constant move a little) and the
             epoch cap (1000 -> 2000, needed because vs2 does not go flat,
             see "Epoch cap"). Reported for every arm; the sign is not
             assumed; the rest-49 shift is the sensitivity indicator for the
             training-set part.
  primary    under vs2, each round's conclusion is re-asked with its own
             pairing (own null = same channel and input count where one
             exists, as rounds 6-8 did):
               r5-pos-h128x2 vs r5-abs-h128x2        position head still wins? (same capacity)
               r5-pos-h128x2 vs r4-abs-h128x5        ... and against the round-4 base
               r4-anchor_avg-h128x5 vs r4-abs-h128x5  the fixed anchor still loses?
               r5-pos-h128x2 vs r5-pos-h128x5        2 layers still beat 5?
               r7-w161 vs r5-pos-h128x2              the weight lever still holds?
               r6-tg10s40 / r6-cgate-afb / r6-cgate-ah vs r6-ah   round 6's gates vs their 4-channel null
               r7-w161-cgate-ah vs r7-w161-ah         the gate still wins (4-channel null)?
               r7-w161-cgate-ah vs r7-w161            ... and vs the 3-channel base (simplicity clause)
               r7-w161-tg30s30 vs r7-w161-ah          the soft timestep gate, as in round 7
               r7-w161-fip vs r7-w161-fiz             the flag still helps the 49?
               r8-w161-fip-cgate-ah vs r8-w161-fiz-cgate-ah   the round-8 pairing; and vs r7-w161-cgate-ah
               r8-w161-To-em1200 vs r7-w161           the anchor, retested
               r8-w161-To-em1200-cgate-ah vs r7-w161-cgate-ah  the anchor on the winner
  hypothesis numeric, on the paired (To-em1200-cgate-ah minus cgate-ah)
             switched-21 T_avg under vs2, legacy value +3.40 [+1.76, +5.57]:
               (a) "the validation set caused the round-8 collapse" if the
                   vs2 CI's upper bound is below +1.0 C (i.e. most of the
                   +3.4 is gone; pilot: +0.69, seeds +1.04/+0.64/+0.38);
               (b) "the reference itself still costs T_avg" if the vs2 CI
                   excludes 0 -- (a) and (b) can both be true, and the pilot
                   says they will be; then branch I asks whether the cost is
                   the coupling.
             Also reported, descriptive: the difference-in-differences
             (anchored - null)_vs2 - (anchored - null)_legacy per seed, and
             the best epochs of both arms with the count of seeds at the
             cap (Total_Epoch == 2000); a seed at the cap is censored, its
             best epoch a lower bound.
  secondary  T_avg on the 12 flagged / 21 switched / 49 untouched cases;
             per-channel validation curves (train_history.json val_hist_ch)
             for which channel drove each arm's stop.
  veto       any seed > 1.5 C; paired increase in JumpSteps_gt2C, p < 0.05.
  deployment seed-ensemble MAE on the common 20 seeds (ensemble_eval.py
             --seeds S20). The recipe changes only if a vs2 arm beats the
             legacy cgate-ah ensemble (0.676) AND its vs2 null.
  no AR      the autoregressive branch is not run this round (user decision
             2026-09-17): the direct model is the deployment candidate and
             AR has tracked it in every round so far.

Stage S2 (branch I) -- MODEL_SPLIT=independent: three complete networks
(init encoder + GRU + head), one each for T_inner, T_outer and T_avg, trained
one after another, each with its own optimiser, LR schedule, early stopping
and best epoch, the finished ones frozen (eval mode) while the next trains.
The T_avg network is fitted against the walls it will see at test time, and
nothing one temperature does can stop or reshape another's training -- the
coupling suspected in round 8 (the T_outer reference made the shared network
early-stop before the gated T_avg channel had converged). Position formula,
gates and anchors unchanged; the loss-weight tag (w161) is inherited so the
arm pairs by name with its shared twin -- with one channel per network the
relative weights act only through the gradient clip (measured: never engages
on the anchored arms, rarely on the others).
"Independent" here means the whole bundle -- separate weights, separate
stopping decision, separate optimiser, sequential training on frozen walls
-- and S2 tests the bundle against the shared model; it does not attribute
a gain to one ingredient. MODEL_SPLIT=avg_only (tag inda) is the two-network
form: the walls keep their shared trunk, only T_avg is separate. The pilot
motivated it: the lone T_outer net was worse than the shared trunk's T_outer
(0.42 vs 0.33, 3/3 seeds) while the lone T_avg net was better on the 49
normal cases (-0.38, 3/3 seeds).
All under vs2, 20 seeds, so every arm pairs per seed with its S1 twin (the
RNG stream differs between a split and a shared model, so the pairing is by
seed label, as for the input-column nulls):

    arm                             pairs with (S1)            question
    ind-w161                        r7-w161                    independence alone, no gate
    ind-w161-cgate-ah               r7-w161-cgate-ah           on the winner
    ind-w161-To-em1200              r8-w161-To-em1200          the reference where it cannot
    ind-w161-To-em1200-cgate-ah     r8-w161-To-em1200-cgate-ah   interfere with T_avg's training
    ind-h64-w161-cgate-ah           ind-w161-cgate-ah          capacity control: three h64 nets
                                                               (GRU 114k params) vs three h128
                                                               (449k); the shared h128 net has
                                                               150k -- GRU capacity only; the
                                                               init-encoder MLPs are fixed-width
    inda-w161-cgate-ah              r7-w161-cgate-ah, ind-w161-cgate-ah   walls shared, T_avg separate
    inda-w161-To-em1200-cgate-ah    r8-w161-To-em1200-cgate-ah, ind-...   same, with the reference

  rule     primary: each split arm vs its S1 twin (the statistic above).
           Then ind-To-em1200-cgate-ah vs ind-cgate-ah, and inda-To-em1200-
           cgate-ah vs inda-cgate-ah: the round-8 question re-asked where the
           reference cannot stop T_avg's training. inda vs ind on the same
           knobs: does the wall pair want its shared trunk. Capacity: if
           ind-h64-cgate-ah is within noise of ind-cgate-ah, an ind gain is
           not a GRU-capacity gain (round 7: more capacity did not help the
           shared net). Simplicity order: shared < inda < ind.
  expected from the pilot (3 seeds): ind beats shared on the 49 normal
           cases and loses on the 21 switched and on T_outer; inda is
           expected to keep the shared T_outer (0.33) and the ind rest-49
           gain, which would make it the arm to beat. If every split arm
           matches its shared twin, one network was never the problem and
           the split is dropped.
  cost     each stage forwards the frozen nets too; pilot on the dev GPU:
           ind 7-9 min/seed at the 1000 cap vs 3 for shared, so budget ~3x
           (inda ~2x).
  history  train_history.json carries `stages` (per-network best/total epoch
           and best val; also flat Best_Epoch_<net> / Total_Epoch_<net> in
           meta.json) and val_hist_ch; the top-level best_epoch indexes the
           concatenated curve. In-run resume is off for split seeds (a killed
           seed restarts; done.flag resume across seeds unchanged).

Stage S1, part 2 -- the loss-weight scans of rounds 6-8 re-run under vs2, on
the base they were run on (the plain position head, h128x2) and with their
seed counts, so each pairs with its legacy twin. Round 6 moved the T_avg
weight (1-6-0.5 / 1 / 3 / 6 / 10, plus 1-3-3, 1-4-3, 1-8-3, 2-6-3) and found
3 -> 1 the biggest single gain of the project; round 7 moved the T_outer
weight (1-8-1, 1-8-2, 1-10-1, 1-12-1) and nothing beat 1-6-1; round 8
re-asked 1-6-3 and 1-3-1 under the EMA reference. All of that was decided by
the linear-charging validation set. 1-6-1 and 1-6-3 are already in part 1.

    arm                 seeds  pairs with           question under vs2
    r6-w1-6-0.5          12    r6 twin              T_avg weight below 1?
    r6-w1-6-6, r6-w1-6-10 12   r6 twins             ... above 3? (Arnold 2026-09-17:
                                                    "explore increasing the weight average")
    r6-w1-3-3, r6-w1-4-3, r6-w1-8-3, r6-w2-6-3  12   T_outer / T_inner weight at T_avg 3
    r7-w1-8-1, r7-w1-8-2, r7-w1-10-1, r7-w1-12-1  12  T_outer weight above 6 under T_avg 1
    r8-w163-To-em1200, r8-w131-To-em1200  20        the weight under the reference
    w166-cgate-ah        12    r7-w161-cgate-ah     Arnold's ask on the winner (new arm,
                                                    no legacy twin)

  rule     each weight arm vs r7-w161 (part 1) on the common seeds, the
           statistic above; the two EMA arms vs r8-w161-To-em1200;
           w166-cgate-ah vs r7-w161-cgate-ah and r6-cgate-ah (1-6-3). The
           question is whether 1-6-1 is still the vector to use: it stays
           unless a vector beats it with the CI clear of zero and no veto.

Stages S1b / S1c -- EVERYTHING ELSE that was ever trained (the user's rule,
restated 2026-09-20: after the validation fix, every configuration tested
before is re-run under the new set). splits/round9_legacy_configs.json is
built from the run_config.json of every finished forward_direct run in the
round 4-8 result folders (146 distinct configurations on the 70-case test
set; the 5 exclude-both-phase sanity arms and the 4 AR arms are not in it);
the runner loads it and schedules every configuration that S1 does not
already contain, with its legacy seed list, so each pairs per seed with its
twin: 114 configurations / 1570 runs / ~150 h. Names are "L-" + the legacy
tag. The manifest tiers them by a MEASUREMENT, not by argument (the first
tiering, by which arms "could plausibly flip", was backwards on the data --
the user's objection 2026-09-20): a validation set decides WHEN training
stops, so the question is how early the old set stopped each configuration.
For every legacy run, legacy_post_checkpoint_train_drop = how much the
training loss still fell in the 150 epochs after the checkpoint the old set
picked (median over seeds). Calibration on S1's twins: cgate-ah 12.8% and
w161 11.4% (vs2 moved cgate-ah by +0.03 -- nothing); To-em1200-cgate-ah
25.5% (vs2 took it from 1.53 to 0.89); abs-h128x5 31.4%.
  S1b  drop > 12.8% (above the calibrated no-change point), 90 configs /
       1170 runs / ~105 h, default ON: the configurations the old set
       demonstrably cut short -- the entire abs-model capacity, lookahead,
       weight and anchor-scale scans of rounds 4-5 (drops 32-56%, best
       epochs 290-410), the learned gates (33-48%), the round-6 soft gates
       and floors at 1-6-3 (13-30%), To-ai, os2, To-em900, the full stack.
  S1c  drop <= 12.8%, 24 configs / 400 runs / ~44 h, default OFF: trained
       at least as long as the winner did, and vs2 did not move the winner
       -- the 1-6-1 gate/flag family (cgate-afb, fip/fiz-cgate-afb, tg10s40,
       tg20/30, fic), the h192/h256 arms at 1-6-1 and 1-6-3 (best epochs
       680-830, many at the 1000 cap), fl20/fl40/fl40s, the shared-channel
       cgate, To-em1500. --stage S1c completes the record.
Neither stage has a pre-registered comparison of its own: these arms feed
the protocol clause (vs2 minus legacy, per arm) and the family tables of
export_results; any of them that beats the vs2 winner goes through the
deployment clause (must beat the legacy cgate-ah ensemble AND its vs2 null,
no veto). S1b runs after S1 and before S2, so the decision arms finish first.

Stage S1, part 3 -- the early-window fix (new; user 2026-09-19: "the core
problem since round 5 is still there: T_avg in the first ~50 steps of cases
15-24 / 58-65, even in the best arm"). Diagnosis on the 20-seed cgate-ah
ensemble: the head's position output is flat at ~0.24 from t=0 while the
true position of a charging run starts near 0.05 and climbs to ~0.25 over
the first ~200 steps (heat has not penetrated; the bulk average sits at the
cold outer wall); error = delta-pos x wall gap, worst where the gap opens
fast (case 24: 157 C at step 50 -> 10.6 C; cases 24/29/54-57 carry 3-11 C
there; 58-65 are fine, < 0.8 C). The climb is the same in every run: the
per-step median position over the 311 training runs is 0.04 / 0.16 / 0.25 /
0.28 at steps 1 / 50 / 200 / 500, spread 0.037 in the first 200 steps
against 0.08 for the flat mean. POS_TIME_PRIOR=median (tag tp): the head
predicts the residual from that curve. Offline, with the TRUE walls and the
curve alone (network output 0), the first-200-step T_avg error on the 49
non-gated test cases is 0.51 C against 2.64 for the flat mean and 1.39 for
the trained ensemble, and the full-run error is not worse (1.55 vs 1.59).

    arm                 seeds  pairs with           question
    tp-w161              20    r7-w161              the prior on the plain base
    tp-w161-cgate-ah     20    r7-w161-cgate-ah     the prior on the winner

  rule     primary: T_avg |error| over the first 200 steps on the 49
           non-gated cases, paired per seed vs the twin (the statistic
           above); the arm passes if that CI is below zero AND the overall
           paired difference is not above zero (CI) AND no veto. Secondary:
           the first 50 steps; cases 24/29/54-57 individually; the switched
           21 unchanged (the gate bypasses the position formula there).
  expected the first-200-step error at least halved (offline the curve alone
           takes it from 2.6 to 0.5 with true walls; the network still has
           to get the walls); overall within noise or slightly better (the
           window is 14 % of the steps and 11 % of the T_avg error).

Epoch cap: every arm runs with MAX_EPOCHS=2000 (tag "ep2000" in the run
name). Under the stratified validation set the 1000 cap binds -- in the
dev-box pilot all three cgate-ah seeds were still improving on validation at
epoch 1000 (best epochs 950-958; the old validation set went flat by 600-800)
-- and a binding cap turns "stop when validation stops improving" into
"train to the cap". The legacy twins hit their cap in 4% of runs (63 of the
1556 direct runs of rounds 6-8) against 4 of 6 pilot seeds under vs2; the
twins stay as they are, the vs2 arms get the headroom. Expect best epochs
of 1000-1500 under vs2 and ~1.5x the round-8 time per seed.

Usage:
    python run_round9.py                 # everything: S1, S2, S1b, S1c = 218 GPU-h,
                                         #   ~108 h wall at --parallel 3 (user 2026-09-20: run it all)
    python run_round9.py --stage S1      # the decision arms only, 44 GPU-h
    python run_round9.py --stage S1,S2   # decision arms + the split models, 68 GPU-h
    python run_round9.py --parallel 4    # 4 seeds at once (measured 2.27x on a 4090; 3 -> 2.03x)
    python run_round9.py --parallel 1    # one seed at a time, as rounds 4-8 ran

Stage S4 (off by default; --stage S4, ~11 GPU-h) -- leave-one-type-out. The
recipe (w161-cgate-ah) trained ten times, each with one of the ten case-type
folders removed from training AND validation (TRAIN_EXCLUDE_TYPE; the frozen
split is filtered, so the other nine types' validation runs are identical
across folds) and that folder's runs appended to the test set. Scored per
fold on (a) the 70 standard cases -- how much the missing type costs
in-distribution -- and (b) the held-out type's own runs -- what the model
does on an operating regime it never saw. 12 seeds. The folds that remove
Case 4 / 6 / 7 also test the case gate on both-phase runs it never trained
on. Tag xt<N>; export_results files them under loto_xt<N>_ families
because their Test_MAE_Overall mixes the two sets. No pre-registered
threshold: this is a generalisation report for the paper, paired per seed
against r7-w161-cgate-ah on the 70 standard cases.

Parallel seeds (--parallel N, default 3; env PARALLEL_SEEDS): each arm's
seed list is dealt round-robin to N launcher processes that train at the
same time on the one GPU, each writing its own run dirs and its own log
(sweep_logs/r9-<arm>.w<k>.log). A run's result does not depend on what
else is on the GPU (fixed seeds, separate processes), and done.flag resume
works per seed as before. Measured on the dev 4090 (60-epoch runs, the GPU
shared with another job): 3 concurrent = 2.03x the throughput of one, 4 =
2.27x -- the models are small and each process is Python/launch-bound, so
the gain flattens. GPU memory per process: 1.1-1.4 GB for the shared
models (h128x2 .. h256x3, 5-layer), 3.8 GB for the three-network arms; the
preflight reads the free GPU memory and lowers N if N x 4.5 GB does not fit.
    python run_round9.py --only r7-w161,r6-w1-6-6,r7-w1-8-1     # a subset by name
    python run_round9.py --only tp-w161-cgate-ah,r7-w161-cgate-ah  # the early-window pair
    python run_round9.py --only r8-w161-To-em1200-cgate-ah,ind-w161-To-em1200-cgate-ah
    python run_round9.py --dry-run
Env: SEEDS overrides everything; PROGRESS_EVERY tunes the status cadence.
Resumable; Ctrl+C safe. Logs -> sweep_logs/r9-<config>.log
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
S12 = "7,21,42,123,1,2,3,5,11,13,17,29"

# Every knob the child reads is pinned; the ONLY thing every arm in S1 shares
# and no earlier round had is VAL_SPLIT=stratified.
BASE = dict(TINNER_MODE="anchor", ANCHOR_LEAD="1", PHYSICS_BOUND_WEIGHT="0",
            VARIANTS="forward_direct", LOSS_WEIGHTS="1,6,3",
            OTHER_CH_MODE="pos_head", HIDDEN_SIZE="128", NUM_LAYERS="2",
            DROPOUT="0.3", WINDOW_SIZE="10", SLIDING_PAD_MODE="variable",
            ANCHOR_SCALE="1.0", INPUT_LOOKAHEAD="0",
            POS_GAP_FLOOR="0", POS_FLOOR_SOFT="0",
            POS_TEMP_GATE="0", POS_TEMP_SOFT="0", POS_CASE_GATE="0",
            POS_LEARNED_GATE="0", POS_GATE_BIAS="2.0", POS_LEARNED_POOL="0",
            POS_ABS_HEAD="0", POS_ANCHORED_FALLBACK="0", POS_FIT_CLEAN="0",
            POS_TIME_PRIOR="0",
            CASE_FLAG_INPUT="", EXCLUDE_BOTH_PHASE="0",
            TOUTER_MODE="abs", TOUTER_TAU="1200", TOUTER_SCALE="1.0", TOUTER_EMA_FORM="euler",
            MAX_EPOCHS="2000", MODEL_SPLIT="shared",
            VAL_SPLIT="stratified", VAL_PER_TYPE="2", TRAIN_EXCLUDE_TYPE="")
RUN_PREFIX = "2026-08-06_"
SPLIT_FILE = os.path.join("splits", "stratified_v2_s42.json")
W = "1,6,1"


def cfg(**kw):
    d = dict(BASE)
    d.update({k: str(v) for k, v in kw.items()})
    return d


def w161(**kw):
    kw.setdefault("LOSS_WEIGHTS", W)
    return cfg(**kw)


def cah(**kw):
    kw.setdefault("POS_CASE_GATE", 1)
    kw.setdefault("POS_ABS_HEAD", 1)
    return w161(**kw)


def ind(**kw):
    kw.setdefault("MODEL_SPLIT", "independent")
    return w161(**kw)


def mins(h=128, l=2, nets=1):
    # minutes per seed on the lab GPU; x1.5 for the 2000-epoch cap (vs2 runs
    # train longer -- pilot: shared arms 950-1000+ epochs, vs ~600-800 before)
    return nets * max(3, round(3.0 * (h / 128.0) ** 1.3 * (1 + 0.45 * (l - 1))))


LEGACY_MANIFEST = os.path.join("splits", "round9_legacy_configs.json")


def legacy_arms(tier):
    """S1b ("keep") / S1c ("archive"): the manifest's configurations of that
    tier, as (name, env, seeds, minutes). The manifest stores knobs that
    differ from BASE; BASE supplies the rest (incl. the round-9 protocol
    knobs)."""
    here = os.path.dirname(os.path.abspath(__file__))
    spec = json.load(open(os.path.join(here, LEGACY_MANIFEST), encoding="utf-8"))
    arms = []
    for d in spec["configs"]:
        if d.get("tier", "keep") != tier:
            continue
        env = cfg(**d["env"])
        h, l = int(env["HIDDEN_SIZE"]), int(env["NUM_LAYERS"])
        arms.append((d["name"], env, ",".join(str(s) for s in d["seeds"]), mins(h, l)))
    return arms


STAGES = {
    "S1": [
        ("r4-abs-h128x5", cfg(OTHER_CH_MODE="abs", NUM_LAYERS=5), S20, mins(128, 5)),
        ("r4-anchor_avg-h128x5", cfg(OTHER_CH_MODE="anchor_avg", NUM_LAYERS=5), S20, mins(128, 5)),
        ("r5-abs-h128x2", cfg(OTHER_CH_MODE="abs"), S20, mins()),
        ("r5-pos-h128x5", cfg(NUM_LAYERS=5), S20, mins(128, 5)),
        ("r5-pos-h128x2", cfg(), S20, mins()),
        ("r6-tg10s40", cfg(POS_TEMP_GATE=10, POS_TEMP_SOFT=40), S20, mins()),
        ("r6-cgate-afb", cfg(POS_CASE_GATE=1, POS_ANCHORED_FALLBACK=1), S20, mins()),
        ("r6-cgate-ah", cfg(POS_CASE_GATE=1, POS_ABS_HEAD=1), S20, mins()),
        ("r6-ah", cfg(POS_ABS_HEAD=1), S20, mins()),
        ("r7-w161", w161(), S20, mins()),
        ("r7-w161-ah", w161(POS_ABS_HEAD=1), S20, mins()),
        ("r7-w161-cgate-ah", cah(), S20, mins()),
        ("r7-w161-fip", w161(CASE_FLAG_INPUT="phase"), S20, mins()),
        ("r7-w161-fiz", w161(CASE_FLAG_INPUT="zero"), S20, mins()),
        ("r7-w161-tg30s30", w161(POS_TEMP_GATE=30, POS_TEMP_SOFT=30), S20, mins()),
        ("r8-w161-fip-cgate-ah", cah(CASE_FLAG_INPUT="phase"), S20, mins()),
        ("r8-w161-fiz-cgate-ah", cah(CASE_FLAG_INPUT="zero"), S20, mins()),
        ("r8-w161-To-em1200", w161(TOUTER_MODE="anchor_ema", TOUTER_TAU=1200), S20, mins()),
        ("r8-w161-To-em1200-cgate-ah", cah(TOUTER_MODE="anchor_ema", TOUTER_TAU=1200), S20, mins()),
        # ---- part 2: the loss-weight scans of rounds 6-8, on their own base ----
        ("r6-w1-6-0.5", cfg(LOSS_WEIGHTS="1,6,0.5"), S12, mins()),
        ("r6-w1-6-6", cfg(LOSS_WEIGHTS="1,6,6"), S12, mins()),
        ("r6-w1-6-10", cfg(LOSS_WEIGHTS="1,6,10"), S12, mins()),
        ("r6-w1-3-3", cfg(LOSS_WEIGHTS="1,3,3"), S12, mins()),
        ("r6-w1-4-3", cfg(LOSS_WEIGHTS="1,4,3"), S12, mins()),
        ("r6-w1-8-3", cfg(LOSS_WEIGHTS="1,8,3"), S12, mins()),
        ("r6-w2-6-3", cfg(LOSS_WEIGHTS="2,6,3"), S12, mins()),
        ("r7-w1-8-1", cfg(LOSS_WEIGHTS="1,8,1"), S12, mins()),
        ("r7-w1-8-2", cfg(LOSS_WEIGHTS="1,8,2"), S12, mins()),
        ("r7-w1-10-1", cfg(LOSS_WEIGHTS="1,10,1"), S12, mins()),
        ("r7-w1-12-1", cfg(LOSS_WEIGHTS="1,12,1"), S12, mins()),
        ("r8-w163-To-em1200", cfg(LOSS_WEIGHTS="1,6,3", TOUTER_MODE="anchor_ema", TOUTER_TAU=1200), S20, mins()),
        ("r8-w131-To-em1200", cfg(LOSS_WEIGHTS="1,3,1", TOUTER_MODE="anchor_ema", TOUTER_TAU=1200), S20, mins()),
        ("w166-cgate-ah", cah(LOSS_WEIGHTS="1,6,6"), S12, mins()),
        # ---- part 3: the early-window fix (position time prior) ----
        ("tp-w161", w161(POS_TIME_PRIOR="median"), S20, mins()),
        ("tp-w161-cgate-ah", cah(POS_TIME_PRIOR="median"), S20, mins()),
    ],
    "S2": [
        ("ind-w161", ind(), S20, mins(nets=3)),
        ("ind-w161-cgate-ah", ind(POS_CASE_GATE=1, POS_ABS_HEAD=1), S20, mins(nets=3)),
        ("ind-w161-To-em1200", ind(TOUTER_MODE="anchor_ema", TOUTER_TAU=1200), S20, mins(nets=3)),
        ("ind-w161-To-em1200-cgate-ah", ind(POS_CASE_GATE=1, POS_ABS_HEAD=1, TOUTER_MODE="anchor_ema", TOUTER_TAU=1200), S20, mins(nets=3)),
        ("ind-h64-w161-cgate-ah", ind(POS_CASE_GATE=1, POS_ABS_HEAD=1, HIDDEN_SIZE=64), S20, mins(64, 2, nets=3)),
        ("inda-w161-cgate-ah", ind(MODEL_SPLIT="avg_only", POS_CASE_GATE=1, POS_ABS_HEAD=1), S20, mins(nets=2)),
        ("inda-w161-To-em1200-cgate-ah", ind(MODEL_SPLIT="avg_only", POS_CASE_GATE=1, POS_ABS_HEAD=1, TOUTER_MODE="anchor_ema", TOUTER_TAU=1200), S20, mins(nets=2)),
    ],
}
STAGES["S1b"] = legacy_arms("keep")
STAGES["S1c"] = legacy_arms("archive")
# S4 -- leave-one-type-out (2026-09-23, for the paper's generalisation
# section; off by default, run after the round with --stage S4): the current
# recipe re-trained ten times, each time with one case-type folder removed
# from training and validation and scored on it. 12 seeds per fold.
STAGES["S4"] = [("loto-xt%d-w161-cgate-ah" % t, cah(TRAIN_EXCLUDE_TYPE=t), S12, mins())
                for t in range(1, 11)]
# Everything, in decision order: the S1 decision arms, the split models,
# then the legacy inventory the old validation set cut short, then the rest.
DEFAULT_STAGES = ["S1", "S2", "S1b", "S1c"]
PARALLEL_DEFAULT = int(os.environ.get("PARALLEL_SEEDS", "3"))
GB_PER_WORKER = 4.5          # measured peak 3.8 GB (three-network arms) + margin

HERE = os.path.dirname(os.path.abspath(__file__))
LAUNCHER = os.path.join(HERE, "GRU_input_ablation.py")
LOG_DIR = os.path.join(HERE, "sweep_logs")


def tail_progress(log_paths, label, n_seeds, done, total, spent_h, tot_h, stop):
    """One status line per PROGRESS_EVERY seconds, summed over the arm's
    worker logs (one launcher process per log)."""
    every = float(os.environ.get("PROGRESS_EVERY", "60"))
    while not stop.wait(every):
        fin, cur = 0, []
        for lp in log_paths:
            try:
                txt = open(lp, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            txt = txt.rsplit("===== ", 1)[-1]    # this launch only (the log is appended to)
            sd = re.findall(r"# SEED (\d+)/(\d+)\s+seed=(\d+)", txt)
            ep = re.findall(r"epoch\s+(\d+)\s+\(", txt)
            fin += len(re.findall(r"DONE\s+overall_MAE|DONE flag found", txt))
            cur.append("s{}@ep{}".format(sd[-1][2], ep[-1] if ep else 0) if sd else "starting")
        print("  [{}] {}: {} | {}/{} done | round: {}/{} runs, ~{:.1f}/{:.1f} GPU-h"
              .format(time.strftime("%H:%M"), label, " ".join(cur), fin, n_seeds,
                      done + fin, total, spent_h, tot_h), flush=True)


def gpu_free_gb():
    """Free GPU memory in GB (0.0 if no GPU / no torch), read in a child so
    the parent never holds a CUDA context."""
    code = "import torch; f, t = torch.cuda.mem_get_info(); print('FREE %.2f' % (f / 2**30))"
    try:
        out = subprocess.run([sys.executable, "-c", code], cwd=HERE, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, universal_newlines=True, timeout=120).stdout
        m = re.search(r"FREE ([\d.]+)", out)
        return float(m.group(1)) if m else 0.0
    except Exception:
        return 0.0


def deal_seeds(seeds, n_workers):
    """Round-robin the seed list to n_workers non-empty chunks."""
    lst = [s.strip() for s in seeds.split(",") if s.strip()]
    chunks = [lst[i::n_workers] for i in range(n_workers)]
    return [",".join(c) for c in chunks if c]


def run_arm(name, env_over, seeds, n_workers, log_paths):
    """Launch the arm's seeds on n_workers concurrent launcher processes;
    returns the worst exit code. Ctrl+C terminates every child."""
    procs = []
    try:
        for k, chunk in enumerate(deal_seeds(seeds, n_workers)):
            env = dict(os.environ, **env_over, SEEDS=chunk, PYTHONUNBUFFERED="1")
            fh = open(log_paths[k], "a", encoding="utf-8", errors="replace")
            fh.write("\n===== {} {} seeds={} worker {}/{} launched {} =====\n".format(
                name, env_over, chunk, k + 1, n_workers, time.strftime("%Y-%m-%d %H:%M:%S")))
            fh.flush()
            procs.append((subprocess.Popen([sys.executable, LAUNCHER], env=env, cwd=HERE,
                                           stdout=fh, stderr=subprocess.STDOUT), fh))
        rcs = []
        for p, fh in procs:
            rcs.append(p.wait())
            fh.close()
        return max(rcs) if rcs else 0
    except KeyboardInterrupt:
        for p, fh in procs:
            try:
                p.terminate()
            except Exception:
                pass
        for p, fh in procs:
            try:
                p.wait(timeout=30)
            except Exception:
                p.kill()
            fh.close()
        raise


def preflight_compile():
    import compileall
    if not compileall.compile_dir(os.path.join(HERE, "tes_gru"),
                                  quiet=1, force=True):
        sys.exit("tes_gru does not compile under " + sys.version.split()[0]
                 + " -- fix the syntax error above before launching")


RUN_NAMES = {}      # arm name -> RUN_NAME_BASE, filled by preflight_arm_configs


def preflight_arm_configs(plan):
    """Import tes_gru.config under EVERY arm's env in a child process: an
    arm whose knob combination config.py refuses (an assert) must fail here,
    not hours into the sweep; and no two arms may resolve to the same
    RUN_NAME_BASE (done.flag resume would silently merge them)."""
    seen = {}
    code = "from tes_gru import config; print(config.RUN_NAME_BASE)"
    for i, (st, name, env, _, _, _) in enumerate(plan):
        e = dict(os.environ, **env)
        e.update(PYTHONUNBUFFERED="1")
        p = subprocess.run([sys.executable, "-c", code], env=e, cwd=HERE,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True)
        lines = [l for l in p.stdout.splitlines() if l.startswith(RUN_PREFIX)]
        if not lines:
            print(p.stdout[-1500:])
            sys.exit("arm {} [{}]: tes_gru.config refuses this knob combination".format(name, st))
        rn = lines[-1].strip()
        if rn in seen:
            sys.exit("arms {} and {} resolve to the same run name {}".format(seen[rn], name, rn))
        seen[rn] = name
        RUN_NAMES[name] = rn
        if (i + 1) % 25 == 0 or i + 1 == len(plan):
            print("  config import ok for {}/{} arms".format(i + 1, len(plan)), flush=True)


def preflight_split_file():
    """The frozen validation list must be the committed one: present, 10
    types x VAL_PER_TYPE, disjoint from its own train list, and every named
    file present on this machine (checked again inside load_all_data)."""
    p = os.path.join(HERE, SPLIT_FILE)
    if not os.path.isfile(p):
        sys.exit("missing {} -- it ships with the code; do not let the server draw its own".format(SPLIT_FILE))
    spec = json.load(open(p, encoding="utf-8"))
    per = int(BASE["VAL_PER_TYPE"])
    n_types = len(spec["types"])
    if spec.get("val_per_type") != per or len(spec["val"]) != per * n_types:
        sys.exit("{}: expected {} x {} validation files, found {}".format(
            SPLIT_FILE, n_types, per, len(spec["val"])))
    if set(spec["val"]) & set(spec["train"]):
        sys.exit("{}: validation and training lists overlap".format(SPLIT_FILE))
    by_type = {}
    for f in spec["val"]:
        by_type[f.split("/")[0]] = by_type.get(f.split("/")[0], 0) + 1
    if set(by_type.values()) != {per}:
        sys.exit("{}: validation count per type is {}".format(SPLIT_FILE, by_type))
    print("  split file: {} types x {} = {} validation, {} training (seed {})".format(
        n_types, per, len(spec["val"]), len(spec["train"]), spec["split_seed"]))


def preflight_stale_dirs():
    """Refuse leftover smoke-test dirs (every 2026-08-06_ dir is policed for
    an ep<151 tag or Total_Epoch < 151) and vs/ind-tagged dirs whose
    run_config lacks the matching provenance (written by code older than
    2026-09-17). Note: a vs2/ind dir trained by an EARLIER round-9 build is
    indistinguishable here (no code stamp in run_config.json); on the dev
    box, delete pilot dirs before a real run. The ep2000 tag keeps this
    sweep's dirs apart from the 1000-cap pilot dirs anyway."""
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
            m = re.search(r"_ep(\d+)(_|$)", d)
            if m and int(m.group(1)) < 151:
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
            if "_vs" not in d and "_ind_" not in d:
                continue
            try:
                keys = json.load(open(rc))
            except Exception:
                stale.append(d + "  (unreadable run_config)")
                continue
            if "_vs" in d and (keys.get("data") or {}).get("val_split") != "stratified":
                stale.append(d + "  (vs tag but no stratified provenance)")
            if re.search(r"_xt\d+(_|$)", d) and not (keys.get("data") or {}).get("train_exclude_type"):
                stale.append(d + "  (xt tag but no train_exclude_type provenance)")
            if "_ind_" in d and keys.get("model_split") != "independent":
                stale.append(d + "  (ind tag but no model_split provenance)")
    if stale:
        print("\nREFUSING TO LAUNCH -- stale done-flagged run dirs would be")
        print("silently reused by these arms. Delete or rename them first:")
        for d in stale:
            print("   runs/" + d)
        sys.exit(2)


# Child checks, run with the BASE env pinned (VAL_SPLIT=stratified): the
# split resolves to the committed file and the numpy/torch case detectors
# agree on every train / val / test run under it.
_CHILD_CHECK = r'''
import sys
from tes_gru.config import VAL_SPLIT
from tes_gru.data import load_all_data, ThermalDataset, input_case_flag_np
from tes_gru.rollout import _input_case_gate
assert VAL_SPLIT == "stratified", VAL_SPLIT
tr, va, te, sc = load_all_data()
if not (len(tr) == 291 and len(va) == 20 and len(te) == 70):
    print("  unexpected split sizes: train %d val %d test %d" % (len(tr), len(va), len(te))); sys.exit(4)
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
print("  split %d/%d/%d; detector twins: " % (len(tr), len(va), len(te)) + ", ".join("%s %d/%d fire" % c for c in counts))
if counts[1][1] == 0:
    print("  WARNING: no both-phase run in the validation set -- the point of vs2 is lost"); sys.exit(6)
if bad:
    print("  numpy / torch detectors DISAGREE:"); [print("    ", x) for x in bad[:10]]; sys.exit(3)
print("CHECK_OK")
'''

# Round-9 code paths, run with the knobs they need: (1) the closed-form EMA
# reference (rollout.touter_reference, 2026-09-19) against the per-step
# recurrence it replaced; (2) the three-network wrapper: same output layout
# as the shared model, no parameter shared between the three, the pos-head
# constants on the wrapper, and the staged loss masks summing to the full
# weight vector.
_CHILD_CHECK_2 = r'''
import sys, torch
import tes_gru.rollout as R
from tes_gru.config import MODEL_SPLIT, MODEL_GROUPS, OUTPUT_SIZE, LATEST_PARAMS, DEVICE, LOSS_WEIGHTS
import tes_gru.config as C
from tes_gru.rollout import touter_reference, variant_to_encoder
from tes_gru.models import build_model, model_parts, IndependentThermalGRU, stage_loss_weights
assert MODEL_SPLIT == "independent" and OUTPUT_SIZE == 4, (MODEL_SPLIT, OUTPUT_SIZE)
class M: touter_ref_affine = ((2.0, 0.5), (1.0, 0.0))
x = torch.rand(8, 1440, device=DEVICE); e0 = torch.rand(8, device=DEVICE)
seed = (e0 - 0.5) / 2.0
worst = 0.0
import math
for form in ("euler", "exact"):
    C.TOUTER_EMA_FORM = form                      # touter_ema_r reads the config module globals
    for tau in (100, 480, 900, 1200, 1500):       # the taus round 8 scanned
        C.TOUTER_TAU = tau
        r = C.touter_ema_r()
        assert abs(r - (math.exp(-1.0 / tau) if form == "exact" else 1.0 - 1.0 / tau)) < 1e-15
        loop = torch.empty_like(x); loop[:, 0] = seed
        for t in range(1, x.shape[1]): loop[:, t] = r * loop[:, t-1] + (1.0 - r) * x[:, t]
        ref = touter_reference(x, e0=e0, model=M())
        d = (ref - loop).abs().max().item()
        if not (torch.isfinite(ref).all().item() and d <= 1e-5):   # NaN fails this too
            print("  EMA closed form vs recurrence FAILS at tau=%d (%s): max diff %r" % (tau, form, d)); sys.exit(7)
        worst = max(worst, d)
C.TOUTER_EMA_FORM = "euler"; C.TOUTER_TAU = 1200
print("  EMA closed form vs recurrence: max diff %.1e over tau in 100..1500, euler and exact forms (scaled units)" % worst)
m = build_model("forward_direct", LATEST_PARAMS, variant_to_encoder("forward_direct")).to(DEVICE)
assert isinstance(m, IndependentThermalGRU) and len(model_parts(m)) == 3
ids = [set(id(p) for p in part.parameters()) for part in m.parts]
assert not (ids[0] & ids[1]) and not (ids[1] & ids[2]) and not (ids[0] & ids[2]), "parts share parameters"
h = m.init_hidden(8, x0=torch.rand(8, 4, device=DEVICE))
out, h2 = m(torch.rand(8, 1440, 2, device=DEVICE), h)
assert out.shape == (8, 1440, OUTPUT_SIZE) and len(h2) == 3, out.shape
n = sum(p.numel() for p in m.parameters())
# the position time prior's step-index recovery from the scaled Time input
from tes_gru.rollout import pos_time_curve
class TP: pass
tpm = TP(); tpm.pos_time_prior = torch.arange(1440, dtype=torch.float32); tpm.pos_time_affine = (1.0 / 14390.0, 0.0, 10.0)
ts = (torch.arange(1440, dtype=torch.float32) * 10.0) * (1.0 / 14390.0)
rec = pos_time_curve(tpm, ts.view(1, -1).to(DEVICE))
assert torch.equal(rec.cpu().view(-1), torch.arange(1440, dtype=torch.float32)), "time prior index recovery"
print("  position time prior: step index recovered exactly from scaled time for 1440 steps")
# the stage loss vectors train.py will use (same helper), summing to the full vector
masks = [stage_loss_weights(g, LOSS_WEIGHTS) for g in MODEL_GROUPS]
assert torch.equal(sum(masks), torch.tensor(LOSS_WEIGHTS, dtype=torch.float32)), masks
print("  independent wrapper: 3 nets, %s params, output %s, hidden x%d, stage masks %s" % (
    format(n, ","), tuple(out.shape), len(h2), [mk.tolist() for mk in masks]))
print("CHECK_OK")
'''


def _run_check(code, env, label):
    """A check passes when its last line is CHECK_OK -- every assert and
    sys.exit in the check fires before that line. The exit code is reported
    but not trusted on its own: on the Windows dev box a process that has
    built a CUDA model exits 0xC0000409 at interpreter teardown after a
    complete, correct run (also seen on the launcher; not on Linux)."""
    print("  [{}]".format(label))
    p = subprocess.run([sys.executable, "-c", code], env=env, cwd=HERE,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True)
    lines = [l for l in p.stdout.splitlines() if l.strip()]
    for l in lines:
        if l != "CHECK_OK":
            print("   " + l if not l.startswith("  ") else l)
    ok = bool(lines) and lines[-1] == "CHECK_OK"
    if not ok:
        sys.exit("{} failed (exit {}) -- not launching".format(label, p.returncode))
    if p.returncode != 0:
        print("   (passed; exit code {} at teardown ignored)".format(p.returncode))


_CHILD_CHECK_LOTO = r'''
import sys
from tes_gru.config import TRAIN_EXCLUDE_TYPE
from tes_gru.data import load_all_data
assert TRAIN_EXCLUDE_TYPE == "4", TRAIN_EXCLUDE_TYPE
tr, va, te, sc = load_all_data()
pfx = "Case 4 ("
in_tr = sum(1 for _, c in tr if pfx in c); in_va = sum(1 for _, c in va if pfx in c); in_te = sum(1 for _, c in te if pfx in c)
print("  LOTO fold xt4: train %d (%d of type 4), val %d (%d of type 4), test %d (%d of type 4)" % (len(tr), in_tr, len(va), in_va, len(te), in_te))
if in_tr or in_va or in_te == 0 or len(va) != 18 or len(te) != 70 + in_te:
    print("  LOTO split is wrong"); sys.exit(8)
print("CHECK_OK")
'''


def preflight_child(plan=None):
    env = dict(os.environ, **BASE)
    env.update(PYTHONUNBUFFERED="1")
    _run_check(_CHILD_CHECK, env, "split + detector twins")
    if plan and any(e.get("TRAIN_EXCLUDE_TYPE") for _, _, e, _, _, _ in plan):
        _run_check(_CHILD_CHECK_LOTO, dict(env, TRAIN_EXCLUDE_TYPE="4"), "leave-one-type-out split")
    env2 = dict(env, MODEL_SPLIT="independent", POS_CASE_GATE="1", POS_ABS_HEAD="1",
                TOUTER_MODE="anchor_ema", TOUTER_TAU="1200", LOSS_WEIGHTS="1,6,1")
    _run_check(_CHILD_CHECK_2, env2, "EMA closed form + independent wrapper")


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
    by_upper = {k.upper(): k for k in STAGES}          # --stage is case-insensitive (S1b == s1B)
    stages = ([by_upper.get(s.strip().upper(), s.strip()) for s in stages.split(",") if s.strip()]
              if stages else DEFAULT_STAGES)
    for st in stages:
        if st not in STAGES:
            sys.exit("unknown stage " + st + "; valid: " + ", ".join(STAGES))
    only = after("--only")
    only = {s.strip() for s in only.split(",") if s.strip()} if only else None
    par = after("--parallel")
    n_workers = max(1, int(par)) if par else PARALLEL_DEFAULT
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
    # No two arms may resolve to the same run directory (with done.flag
    # resume, the second would silently reuse the first's seeds).
    names = {}
    for st, name, env, _, _, _ in plan:
        key = tuple(sorted((k, v) for k, v in env.items() if k != "SEEDS"))
        if key in names:
            sys.exit("arms {} and {} have identical knobs (same RUN_NAME)".format(names[key], name))
        names[key] = name

    speed = {1: 1.0, 2: 1.6, 3: 2.03, 4: 2.27}.get(n_workers, 2.3)   # measured on a 4090
    print("#" * 74)
    print("# ROUND 9  configs={}  runs={}  {:.0f} GPU-h  ~{:.0f} h wall at --parallel {}  (finished seeds are skipped)".format(
        len(plan), tot_runs, tot_min / 60, tot_min / 60 / speed, n_workers))
    print("#" * 74)
    for st, name, env, seeds, n, m in plan:
        knobs = {k: v for k, v in env.items() if BASE.get(k) != v}
        print("  [{}] {:<28} x{:<3} ~{:4.1f} h  {}".format(st, name, n, m / 60, knobs))
    if dry:
        print("\nDry run complete -- nothing launched.")
        return

    print("\nPre-flight:")
    preflight_compile()
    preflight_split_file()
    preflight_stale_dirs()
    preflight_child(plan)
    print("  [config import under every arm's knobs]")
    preflight_arm_configs(plan)
    if n_workers > 1:
        free = gpu_free_gb()
        fit = max(1, int(free // GB_PER_WORKER)) if free > 0 else n_workers
        if fit < n_workers:
            print("  GPU has {:.1f} GB free: {} workers at {} GB each do not fit -> --parallel {}".format(
                free, n_workers, GB_PER_WORKER, fit))
            n_workers = fit
        else:
            print("  GPU has {:.1f} GB free: {} workers x {} GB fit".format(free, n_workers, GB_PER_WORKER))

    res, done, spent = {}, 0, 0.0
    try:
        for st, name, env_over, seeds, n, est in plan:
            nw = min(n_workers, n)
            print("\n{}\n=== [{}] {}  ({} seeds on {} worker{}, ~{:.1f} GPU-h) ===".format(
                "=" * 74, st, name, n, nw, "s" if nw > 1 else "", est / 60))
            lps = [os.path.join(LOG_DIR, "r9-{}.w{}.log".format(name, k + 1)) for k in range(nw)]
            print("  console -> " + (lps[0] if nw == 1 else lps[0].replace(".w1.log", ".w<k>.log")))
            stop = threading.Event()
            threading.Thread(target=tail_progress,
                             args=(lps, "[{}] {}".format(st, name), n, done,
                                   tot_runs, spent / 60, tot_min / 60, stop),
                             daemon=True).start()
            t0 = time.time()
            rc = run_arm(name, env_over, seeds, nw, lps)
            stop.set()
            hrs = (time.time() - t0) / 3600.0
            spent += hrs * 60
            done += n
            # The verdict is the done.flags, not the exit code (the Windows
            # dev box exits 0xC0000409 at teardown after a complete run).
            rn = RUN_NAMES.get(name)
            missing = [s for s in seeds.split(",") if s.strip() and rn and not glob.glob(
                os.path.join(HERE, "runs", "{}_seed{}".format(rn, s.strip()), "variants", "*", "done.flag"))]
            ok = rn is not None and not missing
            res[name] = (ok, rc, hrs, missing)
            print("  {} {} after {:.1f} h".format(
                name, "OK ({} seeds done)".format(n) if ok
                else "INCOMPLETE: seeds {} have no done.flag (exit {})".format(",".join(missing) or "?", rc), hrs))
    except KeyboardInterrupt:
        print("\nInterrupted:", {k: "{:.1f}h".format(v[2]) for k, v in res.items()})
        print("Re-running resumes where it left off.")
        sys.exit(130)

    print("\n" + "#" * 74 + "\n# ROUND 9 SUMMARY")
    for nm, (ok, rc, h, missing) in res.items():
        print("#   {:<40} {:<12} {:.1f} h".format(nm, "OK" if ok else "INCOMPLETE", h))
    bad = [nm for nm, v in res.items() if not v[0]]
    if bad:
        print("# {} arm(s) incomplete -- re-run the same command; finished seeds are skipped".format(len(bad)))
    print("#" * 74)
    print("# Ensemble (common 20 seeds):  python ensemble_eval.py runs --seeds " + S20 + " --subset-size 12")
    print("# Export with:                 python export_results.py runs Round9_upload --with-plots 2")


if __name__ == "__main__":
    main()
