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

METRICS = ("meta.json", "summary_errors.csv", "train_history.json", "done.flag")
CHANNELS = ("T_inner", "T_outer", "T_avg")
DEFAULT_W = (1.0, 6.0, 3.0)


def family(c):
    """Coarse bucket: which experiment arm is this, ignoring capacity."""
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
    if (c.get("pos_abs_head") and not c.get("pos_temp_gate", 0)
            and not c.get("pos_learned_gate")
            and not c.get("pos_anchored_fallback")):
        p.append("ah")
    if c.get("input_lookahead", 0):
        p.append("la{}".format(c["input_lookahead"]))
    if c.get("anchor_scale", 1.0) != 1.0:
        p.append("as{:g}".format(c["anchor_scale"]))
    w = tuple(c.get("loss_weights") or ())
    if w and w != DEFAULT_W:
        p.append("w" + "-".join("{:g}".format(x) for x in w))
    if c.get("tinner_mode") != "anchor":
        p.append("Tin-{}".format(c.get("tinner_mode")))
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


def stats_row(rs, **extra):
    o = [x["meta"]["Test_MAE_Overall"] for x in rs]
    row = dict(extra)
    row.update(seeds=len(o), overall_mean=round(st.mean(o), 4),
               overall_sd=round(st.stdev(o) if len(o) > 1 else 0.0, 4),
               overall_best=round(min(o), 4), overall_worst=round(max(o), 4),
               runs_above_1_5C=sum(1 for x in o if x > 1.5))
    for c in CHANNELS:
        row[c + "_mean"] = round(
            st.mean([x["meta"]["Test_MAE_" + c] for x in rs]), 4)
    row["R2_mean"] = round(
        st.mean([x["meta"]["Test_R2_Overall"] for x in rs]), 6)
    return row



# ---------------------------------------------------------------------------
# Plain-language documentation. Only the entries that actually occur in a given
# export get written, so the README never explains something that is not there.
# ---------------------------------------------------------------------------
FAMILY_DOC = [
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
    ("la", "the model additionally sees the input temperature N steps ahead."),
    ("as", "the inner-surface correction head's output range, multiplied by N."),
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
    used = [(k, t) for k, t in KNOB_DOC
            if k == "h"
            or (k == "L" and re.search(r"\bL\d", leaves))
            or (k not in ("h", "L") and k in leaves)]
    best = index[0]
    out = ["# What is in this folder", ""]
    out.append(_wrap(
        "{} training runs, grouped into {} configurations. Each configuration "
        "was trained several times from different random starting points -- one "
        "seed per run -- because a single run can come out good or bad by luck "
        "alone. The spread across seeds is what tells you whether a difference "
        "is real.".format(len(runs), len(index))))
    out += ["", "## Layout", "",
            "    summary/configs_ranked.csv        every configuration, best first",
            "    summary/best_config_per_case.csv  the winner's error on each of the 70 test cases",
            "    configs/01_.../                   one folder per FAMILY, best family first",
            "        _family.csv                   that family's configurations, best first",
            "        h128/L2_dp0.3/                hidden size, then layers (+ anything else that differs)",
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
                  "the three temperatures and all 70 test cases; lower is "
                  "better. T_inner, T_outer and T_avg are those three on their "
                  "own -- the inner surface, the outer surface, and the average "
                  "temperature of the material. sd is how much the result moved "
                  "between seeds, so a small sd means the configuration trains "
                  "reliably rather than getting lucky."),
            "",
            _wrap("Best in this folder: {} / h{} / {} at {} +/- {} C over {} "
                  "seeds.".format(best["family"], best["hidden_size"],
                                  best["config"], best["overall_mean"],
                                  best["overall_sd"], best["seeds"])),
            "", "## The families", ""]
    for name, text in FAMILY_DOC:
        if name in fams_present:
            out += ["**{}**".format(name), "", _wrap(text), ""]
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
    ranked = sorted(cfgs, key=lambda k: st.mean(
        [x["meta"]["Test_MAE_Overall"] for x in cfgs[k]]))
    heavy = set(ranked[:args.with_plots])

    fams = {}
    for k in cfgs:
        fams.setdefault(k[0], []).append(k)
    fam_order = sorted(fams, key=lambda f: min(
        st.mean([x["meta"]["Test_MAE_Overall"] for x in cfgs[k]]) for k in fams[f]))

    print("{} runs | {} configurations | {} families".format(
        len(runs), len(cfgs), len(fam_order)))

    index = []
    for frank, fam in enumerate(fam_order, 1):
        fdir = os.path.join(args.dst, "configs", "{:02d}_{}".format(frank, fam))
        # Browse order, so this listing IS a map of the directory tree:
        # hidden size, then layer count, then the rest. _family.csv ranks them.
        keys = sorted(fams[fam],
                      key=lambda k: (k[1], cfgs[k][0]["lp"]["num_layers"], k[2]))
        best = st.mean([x["meta"]["Test_MAE_Overall"] for x in cfgs[keys[0]]])
        print("\n  {:02d} {:<26} {} configs, best {:.3f}".format(
            frank, fam, len(keys), best))
        frows = []
        for k in keys:
            rs = sorted(cfgs[k], key=lambda x: x["meta"]["Test_MAE_Overall"])
            _, h, leaf = k
            ldir = os.path.join(fdir, "h{}".format(h), leaf)
            os.makedirs(ldir, exist_ok=True)
            shutil.copy2(os.path.join(args.src, rs[0]["dir"], "run_config.json"),
                         os.path.join(ldir, "run_config.json"))
            per = []
            for r in rs:
                sd = os.path.join(ldir, "seed{:03d}".format(r["seed"]))
                os.makedirs(sd, exist_ok=True)
                want = METRICS + (("predictions.npz",) if k in heavy else ())
                for f in want:
                    s = os.path.join(r["vd"], f)
                    if os.path.isfile(s):
                        shutil.copy2(s, os.path.join(sd, f))
                if k in heavy and os.path.isdir(os.path.join(r["vd"], "plots")):
                    shutil.copytree(os.path.join(r["vd"], "plots"),
                                    os.path.join(sd, "plots"))
                th = os.path.join(r["vd"], "train_history.json")
                row = dict(seed=r["seed"],
                           overall_MAE_C=round(r["meta"]["Test_MAE_Overall"], 4))
                for c in CHANNELS:
                    row[c + "_MAE_C"] = round(r["meta"]["Test_MAE_" + c], 4)
                row["R2_overall"] = round(r["meta"]["Test_R2_Overall"], 6)
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
            print("     h{:<5} {:<28} n={:<3} {:.3f} +- {:.3f}{}".format(
                h, leaf, e["seeds"], e["overall_mean"], e["overall_sd"],
                "   [+plots]" if k in heavy else ""))
        frows.sort(key=lambda r: r["overall_mean"])
        write_csv(os.path.join(fdir, "_family.csv"), frows)

    os.makedirs(os.path.join(args.dst, "summary"), exist_ok=True)
    index.sort(key=lambda r: r["overall_mean"])
    for i, r in enumerate(index, 1):
        r_ = dict(rank=i)
        r_.update(r)
        index[i - 1] = r_
    write_csv(os.path.join(args.dst, "summary", "configs_ranked.csv"), index)

    # per-case table for the winner
    acc = {}
    for r in cfgs[ranked[0]]:
        path = os.path.join(r["vd"], "summary_errors.csv")
        if not os.path.isfile(path):
            continue
        for row in csv.DictReader(open(path, newline="")):
            if row["Case"] == "AVERAGE":
                continue
            m = re.search(r"\((\d+)\)", row["CaseFile"])
            if not m:
                continue
            a = acc.setdefault(int(m.group(1)),
                               dict(file=row["CaseFile"], max_in=[],
                                    **{c: [] for c in CHANNELS}))
            for c in CHANNELS:
                a[c].append(float(row["MAE_{} (C)".format(c)]))
            a["max_in"].append(float(row["MaxErr_T_inner (C)"]))
    rows = []
    for cid in sorted(acc):
        a = acc[cid]
        means = [st.mean(a[c]) for c in CHANNELS]
        row = dict(case=cid, file=a["file"])
        for c, v in zip(CHANNELS, means):
            row[c + "_MAE_C"] = round(v, 4)
        row["overall_MAE_C"] = round(st.mean(means), 4)
        row["MaxErr_T_inner_C"] = round(st.mean(a["max_in"]), 3)
        row["seeds"] = len(a["T_avg"])
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
