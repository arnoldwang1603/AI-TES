"""Package a sweep's runs/ directory for sharing.

A finished round is ~7 GB, almost all of it per-case plots, laid out as one
flat directory per (config, seed) -- 384 look-alike names that hide the actual
comparison. This writes a ~160 MB tree instead, organised the way the results
are actually read:

    configs/01_pos_head/            family, ranked by its best configuration
        _family.csv                 every configuration in this family, ranked
        h128/                       hidden size
            L2_dp0.3/               layers (+ whatever else differs)
                _summary.csv        per-seed table
                run_config.json     the exact settings
                seed007/            metrics (plus plots for the top configs)

Metrics are kept for every run, so any comparison in the round stays
reproducible. Plots and raw predictions are kept only for the top
--with-plots configurations; checkpoints are always dropped -- regenerable
output, not results.

    python export_results.py runs Round6_upload
    python export_results.py runs out --with-plots 2
"""
import argparse
import csv
import json
import os
import re
import shutil
import statistics as st

METRICS = ("meta.json", "summary_errors.csv", "summary_errors_val.csv",     # _val: sealed runs (H3)
           "train_history.json", "done.flag")
CHANNELS = ("T_inner", "T_outer", "T_avg")
DEFAULT_W = (1.0, 6.0, 3.0)


def _pfx(meta):
    """Round 10 / H3: a sealed run (test set never read) carries Val_<metric>
    keys in meta.json instead of Test_<metric>."""
    return "Val_" if meta.get("sealed") else "Test_"


def _m(meta, key):
    """meta[<Test_|Val_><key>], e.g. _m(meta, "MAE_Overall")."""
    return meta[_pfx(meta) + key]


def _mask_tag(c):
    """Round 10 / H2 run-name token for the T_avg mask knob, or ""."""
    m = str(c.get("tavg_mask") or "")
    if not m:
        return ""
    if m == "split":
        return "mk"
    if m == "valonly":
        return "mkv"
    if m.startswith("random:"):
        return "mkr" + m.split(":", 1)[1]
    return "mk_" + m


def family(c):
    """Coarse bucket: which experiment arm is this, ignoring capacity."""
    fam = _family_core(c)
    tags = []
    if c.get('init_burnin_steps'):
        tags.append('bi%d' % c['init_burnin_steps'])
    if c.get('synthetic_rest_files'):
        tags.append('rest%d' % c['synthetic_rest_files'])
    if c.get('constants_from_sha256'):
        tags.append('cf' + c['constants_from_sha256'][:12])
    if c.get('phys_mode'):
        tags.append('ph' + str(c['phys_mode']))
    if c.get('other_ch_mode') == 'ratio_input':
        tags.append('rt' + str(c.get('ratio_channels', 'R3')) + ('k' if c.get('ratio_ref', 'kelvin') == 'kelvin' else 'c'))
    if c.get('other_ch_mode') == 'anchor_initial':
        tags.append('init')
    # Keep combinations distinct when an adopted encoder is paired with a
    # ratio/physics challenger in the confirmation and final-seed stages.
    if c.get('enc_widths', '256,256') != '256,256':
        tags.append('enc' + (str(c['enc_widths']).replace(',', '-') or 'l'))
    if c.get('h0_mode', 'linear') != 'linear':
        tags.append(str(c['h0_mode']))
    if c.get('loss_w0_mode'):
        tags.append(str(c['loss_w0_mode']))
    if tags:
        fam += '_' + '_'.join(tags)
    # Round 10 / H3: a sealed run is scored on its 20 validation files, never
    # on the test set -- its own family, never ranked next to a test-scored one.
    if c.get("sealed"):
        return "sealed_" + _family_split(c, fam)
    return _family_split(c, fam)


def _family_split(c, fam):
    # Round-7 sanity arms are scored on a REDUCED test set (both-phase runs
    # removed), so their numbers must never sit next to full-set ones.
    if c.get("exclude_both_phase"):
        return "sanity_xb_" + fam
    # Round-9 stratified-validation runs are a different training PROTOCOL
    # (291/20 split instead of 296/15); same test set, so they are
    # comparable, but they are filed as their own family so the protocol
    # change is visible in every table.
    # Leave-one-type-out folds are scored on 70 + the held-out type's runs,
    # so their overall numbers are their own family, never ranked with the
    # full-set ones (same reason as sanity_xb_).
    xt = (c.get("data") or {}).get("train_exclude_type")
    if xt:
        return "loto_xt{}_".format(xt) + fam
    if (c.get("data") or {}).get("val_split") == "stratified":
        # Round 10 / H1: the json's own tag (vs3) when the run recorded one.
        d = c.get("data") or {}
        tag = str(d.get("split_tag") or "vs{}".format(d.get("val_per_type", 2)))
        if _mask_tag(c):
            tag += "_" + _mask_tag(c)              # H2: masked-label runs are their own family
        if c.get("relabel_csv"):
            tag += "_rl"                           # A4: algebraically relabelled T_avg
        return tag + "_" + fam
    return ("rl_" if c.get("relabel_csv") else "") + fam


def _family_core(c):
    if c.get("early_step_weight", 1) != 1:
        return "early_step_weight"
    if c.get("feat_cols", "Time (s),Input Temperature (C)").lower() in ("input temperature (c)", "input_t", "input", "no_time"):
        return "no_time"
    if c.get("enc_widths", "256,256") != "256,256" or c.get("h0_mode", "linear") != "linear" or c.get("init_encoder", "point") != "point":
        return "encoder_" + str(c.get("enc_widths", "256,256")).replace(",", "-") + "_" + str(c.get("h0_mode", "linear")) + "_" + str(c.get("init_encoder", "point"))
    if c.get("loss_w0_mode") == "inv_kc":
        return "anchor_inverse_kc"
    v = (c.get("variants") or ["?"])[0]
    if v == "abs_sliding":
        # The AR arms differ in other_ch_mode too -- without this the
        # AR-pos_head / AR-base pair collapses into one group and the
        # per-seed files overwrite each other.
        m = c.get("other_ch_mode", "abs")
        return "AR" if m == "abs" else "AR_" + m
    if v != "forward_direct":
        return v
    mode = c.get("other_ch_mode", "abs")
    if mode != "abs":
        base = mode
        if c.get("model_split") == "independent":
            base += "_independent"
        elif c.get("model_split") == "avg_only":
            base += "_avg_split"
        if c.get("case_flag_input"):
            base += "_flag_input"
        if c.get("touter_mode", "abs") != "abs":
            base += "_touter_anchor"
        if (c.get("pos_gap_floor", 0) or c.get("pos_case_gate")
                or c.get("pos_temp_gate", 0) or c.get("pos_abs_head")
                or c.get("pos_learned_gate")):
            base += "_excursion_fix"
        return base
    if c.get("anchor_scale", 1.0) != 1.0 or c.get("input_lookahead", 0):
        return "anchor_range"
    if tuple(c.get("loss_weights") or ()) not in ((), DEFAULT_W):
        return "loss_weights"
    return "baseline"


def leaf_name(c, lp):
    """Layer level: layers, dropout, and anything else that still differs."""
    p = ["L{}".format(lp["num_layers"]), "dp{:g}".format(lp["dropout"])]
    if c.get('init_burnin_steps'):
        p.append('bi%d' % c['init_burnin_steps'])
    if c.get('synthetic_rest_files'):
        p.append('rest%d' % c['synthetic_rest_files'])
    if c.get('constants_from_sha256'):
        p.append('cf' + c['constants_from_sha256'][:12])
    if c.get('phys_mode'):
        p.append('ph' + str(c['phys_mode']))
    if c.get('other_ch_mode') == 'ratio_input':
        p.append('rt' + str(c.get('ratio_channels', 'R3')) + ('k' if c.get('ratio_ref', 'kelvin') == 'kelvin' else 'c'))
    if c.get('other_ch_mode') == 'anchor_initial':
        p.append('init')
    if c.get("early_step_weight", 1) != 1:
        p.append("esw%g" % c["early_step_weight"])
    if c.get("feat_cols", "Time (s),Input Temperature (C)").lower() in ("input temperature (c)", "input_t", "input", "no_time"):
        p.append("nt")
    if c.get("loss_w0_mode") == "inv_kc":
        p.append("w0ikc")
    if c.get("enc_widths", "256,256") != "256,256":
        p.append("enc" + (str(c.get("enc_widths", "")).replace(",", "-") or "l"))
    if c.get("h0_mode", "linear") != "linear":
        p.append("h0t" if c["h0_mode"] == "tanh" else "h0z")
    if c.get("init_encoder", "point") != "point":
        p.append("ew%d" % c.get("window_size", 10))
    if c.get("pos_gap_floor", 0):
        p.append("floor{:g}".format(c["pos_gap_floor"])
                 + ("s" if c.get("pos_floor_soft") else ""))
    if c.get("pos_temp_gate", 0):
        p.append("tg{:g}".format(c["pos_temp_gate"])
                 + ("s{:g}".format(c["pos_temp_soft"])
                    if c.get("pos_temp_soft", 0) else ""))
    if c.get("pos_case_gate"):
        p.append("cgate")
    if c.get("pos_learned_gate"):
        p.append("lg" + ("p" if c.get("pos_learned_pool") else "")
                 + ("b{:g}".format(c["pos_gate_bias"])
                    if c.get("pos_gate_bias", 2.0) != 2.0 else ""))
    if c.get("pos_anchored_fallback"):
        p.append("afb")
    if c.get("pos_fit_clean"):
        p.append("cf")
    if str(c.get("pos_time_prior", "0")) != "0":
        p.append("tp")
    if (c.get("pos_abs_head") and not c.get("pos_temp_gate", 0)
            and not c.get("pos_learned_gate")
            and not c.get("pos_anchored_fallback")):
        p.append("ah")
    if c.get("input_lookahead", 0):
        p.append("la{}".format(c["input_lookahead"]))
    tf = c.get("tf32")
    if tf not in (None, False, "0"):
        p.append("tf32" if tf in (True, "1") else "tf32c")   # H9 (Round 10)
    if c.get("case_flag_input"):
        p.append("fi" + str(c["case_flag_input"])[0])
    if c.get("exclude_both_phase"):
        p.append("xb")
    if (c.get("data") or {}).get("train_exclude_type"):
        p.append("xt{}".format(c["data"]["train_exclude_type"]))
    if c.get("model_split") == "independent":
        p.append("ind")
    elif c.get("model_split") == "avg_only":
        p.append("inda")
    if c.get("anchor_scale", 1.0) != 1.0:
        p.append("as{:g}".format(c["anchor_scale"]))
    w = tuple(c.get("loss_weights") or ())
    if w and w != DEFAULT_W:
        p.append("w" + "-".join("{:g}".format(x) for x in w))
    if c.get("tinner_mode") != "anchor":
        p.append("Tin-{}".format(c.get("tinner_mode")))
    if c.get("touter_mode", "abs") != "abs":
        p.append("To-ai" if c["touter_mode"] == "anchor_input"
                 else "To-em{}{}".format(c.get("touter_tau", ""), "x" if c.get("touter_ema_form") == "exact" else "")
                 if c["touter_mode"] == "anchor_ema"
                 else "To-{}".format(c["touter_mode"]))
        if c.get("touter_scale", 1.0) != 1.0:
            p.append("os{:g}".format(c["touter_scale"]))
    return "_".join(p)


def collect(src):
    runs = []
    for d in sorted(os.listdir(src)):
        cfg = os.path.join(src, d, "run_config.json")
        if not os.path.isfile(cfg):
            continue
        c = json.load(open(cfg))
        vroot = os.path.join(src, d, "variants")
        if not os.path.isdir(vroot):
            continue
        vds = [v for v in sorted(os.listdir(vroot))
               if os.path.isfile(os.path.join(vroot, v, "meta.json"))]
        if not vds:
            continue
        vd = os.path.join(vroot, vds[0])
        lp = c["latest_params"]
        runs.append(dict(dir=d, cfg=c, lp=lp, vd=vd, seed=c.get("seed"),
                         meta=json.load(open(os.path.join(vd, "meta.json"))),
                         fam=family(c), h=lp["hidden_size"],
                         leaf=leaf_name(c, lp)))
    return runs


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def summary_csv_name(r):
    return "summary_errors_val.csv" if r["meta"].get("sealed") else "summary_errors.csv"


def n_test_cases(r):
    """Rows of summary_errors.csv minus the AVERAGE line (70 on the full
    test set, 49 for the round-7 sanity arms, 20 validation files for a
    sealed run)."""
    p = os.path.join(r["vd"], summary_csv_name(r))
    if not os.path.isfile(p):
        return ""
    return sum(1 for row in csv.DictReader(open(p, newline=""))
               if row.get("Case") != "AVERAGE")


def stats_row(rs, **extra):
    o = [_m(x["meta"], "MAE_Overall") for x in rs]
    row = dict(extra)
    row.update(seeds=len(o), scored_on=("validation" if rs[0]["meta"].get("sealed") else "test"),
               test_cases=n_test_cases(rs[0]),
               overall_mean=round(st.mean(o), 4),
               overall_sd=round(st.stdev(o) if len(o) > 1 else 0.0, 4),
               overall_best=round(min(o), 4), overall_worst=round(max(o), 4),
               runs_above_1_5C=sum(1 for x in o if x > 1.5))
    for c in CHANNELS:
        row[c + "_mean"] = round(
            st.mean([_m(x["meta"], "MAE_" + c) for x in rs]), 4)
    row["R2_mean"] = round(
        st.mean([_m(x["meta"], "R2_Overall") for x in rs]), 6)
    return row



# ---------------------------------------------------------------------------
# Plain-language documentation. Only the entries that actually occur in a given
# export get written, so the README never explains something that is not there.
# ---------------------------------------------------------------------------
FAMILY_DOC = [
    ("pos_head_touter_anchor", """The outer-surface temperature is predicted
the way the inner one always has been: as an offset from a reference built
from the inlet temperature, instead of as an absolute value (Arnold's
request, 2026-09-10). Two references: the inlet temperature at that moment
("To-ai" -- the offset is almost as wide as the temperature itself), and a
slow running average of it ("To-emN" -- the outer wall follows the inlet
like a low-pass filter, and this offset is only a few degrees wide)."""),
    ("pos_head_touter_anchor_excursion_fix", """The outer-surface anchor above
combined with a gate."""),
    ("pos_head_independent", """Three separate networks, one per temperature,
instead of one network with three outputs (round 9, branch I). They are
trained one after another -- inner surface, outer surface, then the average,
each against the finished predictions of the others -- so no temperature's
training can stop or reshape another's. The suspicion behind this: in round
8 the outer-surface reference made the shared network stop training early,
before the average-temperature part had converged."""),
    ("pos_head_independent_excursion_fix", """The three-network setup above
with a gate; the gate and its extra channel belong to the average-temperature
network."""),
    ("pos_head_independent_touter_anchor", """The three-network setup with
the outer-surface reference, no gate."""),
    ("pos_head_independent_touter_anchor_excursion_fix", """The three-network
setup with the outer-surface reference AND a gate -- the round-8 combination
that failed, retried where the reference cannot interfere with the
average-temperature network's training."""),
    ("pos_head_avg_split_excursion_fix", """Two networks instead of one: the two
surfaces keep sharing one network (as in every earlier round), only the
average temperature gets its own, trained afterwards against the finished
surfaces. The three-network pilot showed the lone outer-surface network doing
worse than the shared one, so this keeps the surfaces together and decouples
only the average temperature's training and stopping."""),
    ("pos_head_avg_split_touter_anchor_excursion_fix", """The two-network setup
above with the outer-surface reference."""),
    ("pos_head_flag_input", """The network is TOLD which kind of run it is
looking at: one extra input column, read off the input temperature curve,
says whether the run both charges and discharges (or, in the per-timestep
form, whether the discharge has begun). In round 6 the gates made that
decision outside the network and the network never saw it; here it can adapt
its own predictions to the regime instead of being switched from outside."""),
    ("pos_head_flag_input_excursion_fix", """The regime flag above combined
with one of the round-6 gates."""),
    ("sanity_xb_pos_head", """A check requested at the 2026-09-05 meeting:
every run containing both a charging and a discharging phase is REMOVED from
training and from the test set, and the model is retrained. Scored on the 49
remaining test cases only -- these numbers are not comparable with any other
folder here."""),
    ("sanity_xb_pos_head_excursion_fix", """The same check with a gate switched
on. With the both-phase runs gone the gate can never fire, so each gated
configuration must reproduce its ungated twin with the same number of output
channels: cgate (3 channels) reproduces the plain check, cgate_afb (4
channels) reproduces the 4-channel "ah" control. A difference within such a
pair would point at the shared code. Scored on 49 test cases."""),
    ("pos_head_excursion_fix", """Fixes for where the position idea breaks
down. In some runs the average temperature climbs ABOVE both surfaces -- the
material is still giving back stored heat while both surfaces have already
cooled -- so it is no longer "between" anything and a position stops meaning
anything. These configurations try different ways of noticing that and handling
it; see the floor / cgate / tg entries below."""),
    ("pos_head", """The average temperature is not predicted directly. Instead
the model predicts WHERE it sits between the inner and the outer surface -- a
number around 0.27 that barely moves within a run -- and the average is then
computed from that position plus the two surface temperatures. The large swings
come from the surfaces automatically, so the model only has to supply one
small, well-behaved number."""),
    ("baseline", """The plain version: the model predicts all three
temperatures directly, as absolute values. Everything else is compared against
this."""),
    ("anchor_range", """Experiments on how large a correction the inner-surface
head is allowed to produce, aimed at a leftover error in the first timestep of
a few runs."""),
    ("anchor_avg_grad", """An earlier attempt: the average temperature was
computed from a FIXED recipe -- a fixed 29/71 mix of the two surfaces -- rather
than predicted. Kept for comparison. It did not work, because the right mix
differs from run to run and one fixed number cannot follow it."""),
    ("anchor_avg", """The same fixed-recipe attempt, but with the average's
training signal blocked from reaching the two surface predictions."""),
    ("anchor", """The fixed-recipe attempt applied to both the outer and the
average temperature."""),
    ("loss_weights", """Runs that change how much each of the three
temperatures counts in the training objective."""),
    ("AR", """The original autoregressive setup: the model feeds its own
previous prediction back in and walks forward one timestep at a time. It trains
roughly 300x slower than the direct version and scores worse."""),
]

KNOB_DOC = [
    ("h", "hidden size -- how wide each layer of the network is."),
    ("L", "how many GRU layers are stacked."),
    ("dp", """dropout -- the fraction of the network randomly switched off
during training, to stop it memorising. It does nothing at L1, because PyTorch
only applies dropout BETWEEN layers."""),
    ("floor", """The position is worked out by dividing by the gap between the
two surfaces, so when they nearly meet that division explodes. This sets a
lower bound, in degrees, on the divisor. An "s" suffix is the smooth version,
which has no kink at the threshold."""),
    ("cgate", """Whole runs containing both a charging and a discharging phase
are handed over to a separate output that predicts the average temperature
directly, since the position idea is unreliable for them. Which runs those are
is read off the input temperature curve alone."""),
    ("tg", """The same handover, decided timestep by timestep instead of per
run: it applies only while the two surfaces are within N degrees of each other.
An "s" suffix ramps the handover smoothly over that many degrees instead of
switching abruptly."""),
    ("ah", """An extra fourth output predicting the average temperature
directly, present but with nothing gating to it -- a control, so a gain from
the gates above cannot be mistaken for the effect of simply having one more
output."""),
    ("lg", """Instead of us picking the temperature threshold, the model
predicts the handover weight itself and learns where the position idea can be
trusted. It starts out biased towards trusting it."""),
    ("afb", """The handover target is anchored on the model's own two
surface predictions (their midpoint) instead of being a raw absolute
temperature, so the fallback inherits the surfaces' accuracy."""),
    ("cf", """The position statistics are fitted with the excursion steps
excluded, which sharpens the head's resolution in the normal regime."""),
    ("tp", """the position gets a time prior: early in a run the true position
is far below its later value (the heat has not penetrated yet) and climbs
over the first ~200 steps in almost the same way in every run; that curve
(the per-step median over the training runs) is fixed and the network
predicts only the deviation from it. Aimed at the first ~200 steps, where
the plain head overestimates T_avg."""),
    ("la", "the model additionally sees the input temperature N steps ahead."),
    ("fi", """the regime flag fed to the network as an extra input: "fic" is
one value per run (both charging and discharging, or not), "fip" is per
timestep (1 from the first step after the input temperature's peak plateau,
i.e. once the discharge has begun, in such a run), "fiz" is a column of
zeros -- the control, because adding any input column changes the random
starting point of training."""),
    ("xb", """runs with both a charging and a discharging phase were removed
from training AND from the test set before this run (the 2026-09-05 sanity
check). Scored on 49 test cases; not comparable with the others."""),
    ("ind", """three separate networks, one per temperature, trained one
after another (inner, outer, then average, each against the finished
others), instead of one network with three outputs. The capacity tags (h, L)
then describe EACH of the three."""),
    ("inda", """two networks: the two surfaces share one (as before), the
average temperature has its own, trained afterwards against the finished
surfaces."""),
    ("xt", """leave-one-type-out: every run of case type N was removed from
training and validation and added to the test set, so the numbers include a
regime the model never saw. Not comparable with the full-set families."""),
    ("vs", """the validation set used to pick the best epoch: "vs2" means 2
runs from each of the 10 case-type folders (20 in all, fixed once and
recorded in splits/); absent means the original protocol, where the 15
validation runs were all linear-charging cases."""),
    ("To-", """how the outer surface temperature is parametrised: "To-ai"
means it is predicted as an offset from the inlet temperature at that moment
(the inner surface's recipe); "To-emN" as an offset from a slow running
average of the inlet temperature with time constant N steps (N=1200 is 3.3
h; the outer wall follows the inlet like a low-pass filter, and this
reference is within 7 degrees of it on average, against 97 for the raw
inlet); "To-emNx" is the same with the exact one-step solution of the wall's
energy balance instead of its first-order approximation (identical to 0.04 %
at N = 1200; kept for media with a short time constant); absent means it is
predicted directly."""),
    ("as", "the inner-surface correction head's output range, multiplied by N."),
    ("os", "the outer-surface correction head's output range, multiplied by N."),
    ("w", """the relative weight of the three temperatures in the training
objective, in the order inner / outer / average. The default is 1-6-3."""),
    ("Tin-", "how the inner surface temperature is parametrised."),
]

NUMERIC_KNOBS = ("h", "L", "la", "as", "tg", "floor")


def _wrap(text, indent=""):
    words = " ".join(text.split()).split(" ")
    lines, cur = [], indent
    for w in words:
        if len(cur) + len(w) + 1 > 78 and cur.strip():
            lines.append(cur.rstrip())
            cur = indent + w + " "
        else:
            cur += w + " "
    if cur.strip():
        lines.append(cur.rstrip())
    return "\n".join(lines)


def write_readme(dst, runs, fams_present, index, with_plots):
    """A legend for the folder: what each configuration actually means."""
    leaves = " ".join(r["leaf"] for r in runs)
    # "vs" is carried as the family prefix (vs2_...), never as a leaf token
    has_vs = any(re.match(r"(sealed_)?vs\d+_", f) for f in fams_present)
    used = [(k, t) for k, t in KNOB_DOC
            if k == "h"
            or (k == "L" and re.search(r"\bL\d", leaves))
            or (k == "vs" and has_vs)
            or (k not in ("h", "L", "vs") and k in leaves)]
    best = index[0]
    n_cases = best.get("test_cases") or 70
    out = ["# What is in this folder", ""]
    out.append(_wrap(
        "{} training runs, grouped into {} configurations. Each configuration "
        "was trained several times from different random starting points -- one "
        "seed per run -- because a single run can come out good or bad by luck "
        "alone. The spread across seeds is what tells you whether a difference "
        "is real.".format(len(runs), len(index))))
    out += ["", "## Layout", "",
            "    summary/configs_ranked.csv        every configuration, best first",
            "    summary/best_config_per_case.csv  the winner's error on each of the {} test cases".format(n_cases),
            "    summary/sanity_xb_reduced_testset.csv  (if present) the 2026-09-05 sanity check, scored on 49 cases -- not comparable",
            "    configs/01_.../                   one folder per FAMILY, best family first",
            "        _family.csv                   that family's configurations, best first",
            "        h128/L2_dp0.3/                hidden size, then layers (+ anything else that differs);",
            "                                      the round's best configuration is suffixed ' (cur best)'",
            "            _summary.csv              one row per seed",
            "            run_config.json           the exact settings used",
            "            seed007/                  that run's metrics",
            "",
            _wrap("Folders are ordered by size so they are easy to browse; the "
                  "_family.csv and configs_ranked.csv files are ordered by "
                  "score instead. Plots and raw predictions are kept only for "
                  "the top {} configuration(s) -- ask if you want them for "
                  "another one. Checkpoints are not included, since they can "
                  "be regenerated.".format(with_plots)),
            "", "## The numbers", "",
            _wrap("overall MAE is the average error in degrees Celsius across "
                  "the three temperatures and all {} test cases (the sanity_xb "
                  "families are the exception: 49 cases); lower is "
                  "better. T_inner, T_outer and T_avg are those three on their "
                  "own -- the inner surface, the outer surface, and the average "
                  "temperature of the material. sd is how much the result moved "
                  "between seeds, so a small sd means the configuration trains "
                  "reliably rather than getting lucky.".format(n_cases)),
            "",
            _wrap("Best in this folder: {} / h{} / {} at {} +/- {} C over {} "
                  "seeds.".format(best["family"], best["hidden_size"],
                                  best["config"], best["overall_mean"],
                                  best["overall_sd"], best["seeds"])),
            "", "## The families", ""]
    # A "vsN_" prefix marks the validation protocol, not the arm: the arm's
    # text applies to the prefixed family too (the sanity_xb_ families have
    # their own entries).
    documented = {n for n, _ in FAMILY_DOC}
    for name, text in FAMILY_DOC:
        hits = sorted(f for f in fams_present
                      if f == name or (re.match(r"(sealed_)?vs\d+_", f)
                                       and re.sub(r"^(sealed_)?vs\d+_", "", f) == name
                                       and f not in documented))
        for f in hits:
            out += ["**{}**".format(f), "", _wrap(text), ""]
    out += ["## Reading a configuration name", "",
            _wrap("A name lists only what differs from the defaults, which are: "
                  "the direct (non-autoregressive) rollout, the inner surface "
                  "anchored on the input temperature, loss weights 1-6-3, and no "
                  "physics constraint."), ""]
    for key, text in used:
        label = key + ("N" if key in NUMERIC_KNOBS else "")
        out += ["`{}`".format(label), "", _wrap(text, "  "), ""]
    with open(os.path.join(dst, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out).rstrip() + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="the runs/ directory of a finished round")
    ap.add_argument("dst", help="output directory (must not exist)")
    ap.add_argument("--with-plots", type=int, default=1,
                    help="how many top configurations keep plots + predictions")
    args = ap.parse_args()
    if os.path.exists(args.dst):
        raise SystemExit("{} exists -- refusing to overwrite".format(args.dst))

    runs = collect(args.src)
    if not runs:
        raise SystemExit("no runs found under {}".format(args.src))

    # group into family -> hidden -> leaf
    cfgs = {}
    for r in runs:
        cfgs.setdefault((r["fam"], r["h"], r["leaf"]), []).append(r)
    # Round-7 sanity arms (EXCLUDE_BOTH_PHASE) are scored on 49 test cases,
    # so they never compete for the ranking, the plots or "(cur best)"; they
    # are listed after every full-set family instead. Round 10 / H3: sealed
    # runs are scored on their validation files; they rank among themselves
    # only (a Round 10 export is all sealed, a mixed one ranks the
    # test-scored runs and lists the sealed families after them).
    sanity = [k for k in cfgs if cfgs[k][0]["cfg"].get("exclude_both_phase")]
    sealed = [k for k in cfgs if k not in sanity and cfgs[k][0]["meta"].get("sealed")]
    full = [k for k in cfgs if k not in sanity and k not in sealed]
    if not full and sealed:
        full, sealed = sealed, []
        print("all runs are sealed (validation-scored): ranking them among themselves")
    if not full:
        raise SystemExit("only sanity (reduced test set) runs found -- "
                         "nothing to rank")
    def _score(k):
        return st.mean([_m(x["meta"], "MAE_Overall") for x in cfgs[k]])
    ranked = sorted(full, key=_score)
    heavy = set(ranked[:args.with_plots])

    fams = {}
    for k in cfgs:
        fams.setdefault(k[0], []).append(k)
    def _side(f):
        return f.startswith("sanity_xb_") or (sealed and f.startswith("sealed_"))
    fam_order = sorted([f for f in fams if not _side(f)],
                       key=lambda f: min(_score(k) for k in fams[f]))
    fam_order += sorted(f for f in fams if _side(f))

    print("{} runs | {} configurations | {} families".format(
        len(runs), len(cfgs), len(fam_order)))

    index = []
    for frank, fam in enumerate(fam_order, 1):
        fdir = os.path.join(args.dst, "configs", "{:02d}_{}".format(frank, fam))
        # Browse order, so this listing IS a map of the directory tree:
        # hidden size, then layer count, then the rest. _family.csv ranks them.
        keys = sorted(fams[fam],
                      key=lambda k: (k[1], cfgs[k][0]["lp"]["num_layers"], k[2]))
        best = _score(keys[0])
        print("\n  {:02d} {:<26} {} configs, best {:.3f}".format(
            frank, fam, len(keys), best))
        frows = []
        for k in keys:
            rs = sorted(cfgs[k], key=lambda x: _m(x["meta"], "MAE_Overall"))
            _, h, leaf = k
            # The single best configuration of the round gets a visible tag on
            # its folder, so it can be found by browsing without opening a CSV.
            ldir = os.path.join(fdir, "h{}".format(h),
                                leaf + (" (cur best)" if k == ranked[0] else ""))
            os.makedirs(ldir, exist_ok=True)
            shutil.copy2(os.path.join(args.src, rs[0]["dir"], "run_config.json"),
                         os.path.join(ldir, "run_config.json"))
            per = []
            for r in rs:
                sd = os.path.join(ldir, "seed{:03d}".format(r["seed"]))
                os.makedirs(sd, exist_ok=True)
                want = METRICS + (("predictions.npz", "predictions_val.npz") if k in heavy else ())
                for f in want:
                    s = os.path.join(r["vd"], f)
                    if os.path.isfile(s):
                        shutil.copy2(s, os.path.join(sd, f))
                if k in heavy and os.path.isdir(os.path.join(r["vd"], "plots")):
                    shutil.copytree(os.path.join(r["vd"], "plots"),
                                    os.path.join(sd, "plots"))
                th = os.path.join(r["vd"], "train_history.json")
                row = dict(seed=r["seed"],
                           overall_MAE_C=round(_m(r["meta"], "MAE_Overall"), 4))
                for c in CHANNELS:
                    row[c + "_MAE_C"] = round(_m(r["meta"], "MAE_" + c), 4)
                row["R2_overall"] = round(_m(r["meta"], "R2_Overall"), 6)
                row["best_epoch"] = (json.load(open(th)).get("best_epoch", "")
                                     if os.path.isfile(th) else "")
                row["run_dir"] = r["dir"]
                per.append(row)
            write_csv(os.path.join(ldir, "_summary.csv"), per)
            e = stats_row(rs, hidden_size=h, config=leaf)
            frows.append(e)
            g = dict(e)
            g["family"] = fam
            index.append(g)
            print("     h{:<5} {:<28} n={:<3} {:.3f} +- {:.3f}{}{}".format(
                h, leaf, e["seeds"], e["overall_mean"], e["overall_sd"],
                "   [+plots]" if k in heavy else "",
                "   <-- (cur best)" if k == ranked[0] else ""))
        frows.sort(key=lambda r: r["overall_mean"])
        write_csv(os.path.join(fdir, "_family.csv"), frows)

    os.makedirs(os.path.join(args.dst, "summary"), exist_ok=True)
    # Full-set configurations ranked; sanity (49-case) configurations go to
    # their own file so a reader cannot line the two up by accident (sealed
    # families likewise, when test-scored runs are present).
    sanity_rows = [r for r in index if r["family"].startswith("sanity_xb_")]
    sealed_rows = [r for r in index if _side(r["family"]) and not r["family"].startswith("sanity_xb_")]
    index = [r for r in index if not _side(r["family"])]
    index.sort(key=lambda r: r["overall_mean"])
    for i, r in enumerate(index, 1):
        r_ = dict(rank=i)
        r_.update(r)
        index[i - 1] = r_
    write_csv(os.path.join(args.dst, "summary", "configs_ranked.csv"), index)
    if sanity_rows:
        sanity_rows.sort(key=lambda r: r["overall_mean"])
        write_csv(os.path.join(args.dst, "summary", "sanity_xb_reduced_testset.csv"),
                  sanity_rows)
    if sealed_rows:
        # H3: validation-scored (sealed) configurations next to test-scored
        # ones -- their own file, never lined up with the test numbers.
        sealed_rows.sort(key=lambda r: r["overall_mean"])
        write_csv(os.path.join(args.dst, "summary", "sealed_validation_scored.csv"),
                  sealed_rows)

    # per-case table for the winner (validation files, keyed by name, for a
    # sealed winner; NaN T_avg on masked files is skipped)
    acc = {}
    for r in cfgs[ranked[0]]:
        path = os.path.join(r["vd"], summary_csv_name(r))
        if not os.path.isfile(path):
            continue
        for row in csv.DictReader(open(path, newline="")):
            if row["Case"] == "AVERAGE":
                continue
            m = re.search(r"\((\d+)\)", row["CaseFile"])
            cid = int(m.group(1)) if m else row["Case"]
            a = acc.setdefault(cid, dict(file=row["CaseFile"], max_in=[],
                                         **{c: [] for c in CHANNELS}))
            for c in CHANNELS:
                v = float(row["MAE_{} (C)".format(c)] or "nan")
                if v == v:
                    a[c].append(v)
            a["max_in"].append(float(row["MaxErr_T_inner (C)"]))
    rows = []
    for cid in sorted(acc, key=lambda k: (1, 0, k) if isinstance(k, str) else (0, k, "")):
        a = acc[cid]
        means = [st.mean(a[c]) if a[c] else float("nan") for c in CHANNELS]
        row = dict(case=cid, file=a["file"])
        for c, v in zip(CHANNELS, means):
            row[c + "_MAE_C"] = round(v, 4)
        row["overall_MAE_C"] = round(st.mean([v for v in means if v == v]), 4)
        row["MaxErr_T_inner_C"] = round(st.mean(a["max_in"]), 3)
        row["seeds"] = len(a["T_inner"])
        rows.append(row)
    write_csv(os.path.join(args.dst, "summary", "best_config_per_case.csv"), rows)

    write_readme(args.dst, runs, set(r['fam'] for r in runs),
                 index, args.with_plots)

    dsize = sum(os.path.getsize(os.path.join(dp, f))
                for dp, _, fs in os.walk(args.dst) for f in fs)
    ssize = sum(os.path.getsize(os.path.join(dp, f))
                for dp, _, fs in os.walk(args.src) for f in fs)
    print("\n{}: {:.0f} MB (from {:.0f} MB)".format(
        args.dst, dsize / 1e6, ssize / 1e6))


if __name__ == "__main__":
    main()
