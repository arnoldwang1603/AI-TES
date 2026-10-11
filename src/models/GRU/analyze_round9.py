"""Round 9 analysis under the rule pre-registered in run_round9.py's docstring.

    python analyze_round9.py <round-9 runs dir> [--legacy DIR ...] [--out DIR]

Every arm of run_round9.STAGES (S1, S2, S1b, S1c) is resolved to its run name
(tes_gru.config imported under the arm's knobs -- the runner's own
preflight), every seed's meta.json / summary_errors.csv / train_history.json
is read, and each arm is paired per seed with
  * its legacy-protocol twin (same knobs, run name without "_ep2000"/"_vs2"),
    searched in the --legacy folders (rounds 4-8), and
  * the comparison partners the rule names.
Statistic (the round-7 rule): paired difference per seed, 95 % bootstrap CI
(10 000 resamples, fixed RNG), two-sided sign-flip p (exact up to 16 pairs,
else 50 000 random flips, fixed RNG); blow-ups (> 1.5 C) counted on both
sides and the comparison repeated without them; the primary comparisons
and the hypothesis also on the 17 seeds other than the pilot seeds 7/21/42.
Deployment metric: seed-ensemble MAE (predictions averaged over the common
seeds, then scored) -- checked against the legacy cgate-ah value 0.676.

Writes <out>/r9_runs.json (per-run metrics cache), r9_arms.csv (one row per
arm), r9_report.md (every table) and prints the report.
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics as st
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FLAGGED = set(range(32, 41)) | {51, 52, 53}
SWITCHED = FLAGGED | {50, 58} | set(range(59, 66))
PILOT = {7, 21, 42}
BLOW = 1.5


# ----------------------------------------------------------------- loading
def run_metrics(d):
    vd = os.path.join(d, "variants", "forward_direct")
    if not os.path.isfile(os.path.join(vd, "done.flag")):
        return None
    meta = json.load(open(os.path.join(vd, "meta.json")))
    th = json.load(open(os.path.join(vd, "train_history.json")))
    per, jumps = {}, 0.0
    for row in csv.DictReader(open(os.path.join(vd, "summary_errors.csv"))):
        m = re.search(r"\((\d+)\)", row.get("CaseFile") or "")
        if not m:
            continue
        c = int(m.group(1))
        per[c] = float(row["MAE_T_avg (C)"])
        try:
            jumps += float(row.get("JumpSteps_gt2C") or 0.0)
        except ValueError:
            pass
    c = json.load(open(os.path.join(d, "run_config.json")))
    cap = int(c["latest_params"].get("max_epochs", 1000))
    return dict(overall=meta["Test_MAE_Overall"], T_inner=meta["Test_MAE_T_inner"],
                T_outer=meta["Test_MAE_T_outer"], T_avg=meta["Test_MAE_T_avg"],
                f12=st.mean(per[i] for i in per if i in FLAGGED),
                sw21=st.mean(per[i] for i in per if i in SWITCHED),
                r49=st.mean(per[i] for i in per if i not in SWITCHED),
                jumps=jumps, best_epoch=int(th["best_epoch"]), total_epoch=int(th["total_epoch"]),
                # split models: each stage has its own cap; the top-level total is their sum
                capped=int(any(s_["total_epoch"] >= cap for s_ in th["stages"]) if th.get("stages")
                           else th["total_epoch"] >= cap),
                stages=th.get("stages"))


def load_npz(d):
    z = np.load(os.path.join(d, "variants", "forward_direct", "predictions.npz"), allow_pickle=True)
    ids = [int(re.search(r"\((\d+)\)", c).group(1)) for c in z["case_ids"]]
    P = np.stack([np.asarray(p, dtype=float) for p in z["inv_preds"]])
    A = np.stack([np.asarray(a, dtype=float) for a in z["inv_actuals"]])
    return ids, P, A


# columns of predictions.npz: Time, T_outer, T_inner, T_avg, Input_T
def ensemble(dirs):
    ids0, Ps, A0 = None, [], None
    for d in dirs:
        ids, P, A = load_npz(d)
        if ids0 is None:
            ids0, A0 = ids, A
        assert ids == ids0
        Ps.append(P)
    E = np.mean(Ps, axis=0)
    ch = [np.abs(E[:, :, j] - A0[:, :, j]).mean(axis=1).mean() for j in (2, 1, 3)]
    return float(np.mean(ch)), [float(x) for x in ch]


def early_window(d, k):
    ids, P, A = load_npz(d)
    non = [i for i, c in enumerate(ids) if c not in SWITCHED]
    return float(np.abs(P[non, :k, 3] - A[non, :k, 3]).mean())


# ----------------------------------------------------------------- statistics
RNG = np.random.default_rng(0)


def paired(a, b, key, seeds=None):
    """a, b: {seed: metrics}. Returns stats of a[key] - b[key] on common seeds."""
    common = sorted(set(a) & set(b) & (set(seeds) if seeds else set(a)))
    if len(common) < 3:
        return None
    d = np.array([a[s][key] - b[s][key] for s in common])
    n = len(d)
    boot = RNG.choice(d, size=(10000, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    obs = abs(d.mean())
    if n <= 16:
        signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
        p = float((np.abs((signs * d).mean(axis=1)) >= obs - 1e-12).mean())
    else:
        signs = RNG.choice([-1.0, 1.0], size=(50000, n))
        p = float((np.abs((signs * d).mean(axis=1)) >= obs - 1e-12).mean())
    return dict(n=n, mean=float(d.mean()), lo=float(lo), hi=float(hi), p=p, seeds=common)


def fmt(r, digits=3):
    if r is None:
        return "n/a"
    f = "%+." + str(digits) + "f [%+." + str(digits) + "f, %+." + str(digits) + "f] p=%.3f n=%d"
    return f % (r["mean"], r["lo"], r["hi"], r["p"], r["n"])


def verdict(r):
    if r is None:
        return "n/a"
    if r["hi"] < 0:
        return "better"
    if r["lo"] > 0:
        return "worse"
    return "noise"


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--legacy", nargs="*", default=[])
    ap.add_argument("--out", default="r9_analysis")
    ap.add_argument("--arms", default=None, help="cached arm list (json) to skip resolving run names")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    os.makedirs(a.out, exist_ok=True)
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    # --- arms and run names ---
    if a.arms and os.path.isfile(a.arms):
        arms = json.load(open(a.arms))
    else:
        import run_round9 as r
        plan = [(st_, n, e, s, len(s.split(",")), 0) for st_ in ("S1", "S2", "S1b", "S1c") for n, e, s, _ in r.STAGES[st_]]
        r.preflight_arm_configs(plan)
        arms = [dict(stage=st_, name=n, run_name=r.RUN_NAMES[n], seeds=[int(x) for x in s.split(",")], env=e)
                for st_, n, e, s, _, _ in plan]
        json.dump(arms, open(os.path.join(a.out, "r9_arms.json"), "w"), indent=1)
    A = {x["name"]: x for x in arms}

    # --- per-run metrics (cached) ---
    cache_p = os.path.join(a.out, "r9_runs.json")
    cache = json.load(open(cache_p)) if os.path.isfile(cache_p) else {}
    roots = [a.runs] + list(a.legacy)

    def find(name, s):
        for root in roots:
            d = os.path.join(root, "%s_seed%d" % (name, s))
            if os.path.isdir(d):
                return d
        return None

    def metrics_of(name, seeds):
        res = {}
        for s in seeds:
            key = "%s_seed%d" % (name, s)
            if key not in cache:
                d = find(name, s)
                cache[key] = run_metrics(d) if d else None
            if cache[key] is not None:
                res[s] = cache[key]
        return res

    V, L = {}, {}
    for x in arms:
        V[x["name"]] = metrics_of(x["run_name"], x["seeds"])
        leg = x["run_name"].replace("_ep2000", "").replace("_vs2", "")
        L[x["name"]] = metrics_of(leg, x["seeds"]) if leg != x["run_name"] and "_ind" not in leg and "_tp" not in leg and "_inda" not in leg else {}
    json.dump(cache, open(cache_p, "w"))

    def mean(d, k):
        return st.mean(v[k] for v in d.values()) if d else float("nan")

    def blows(d):
        return sum(v["overall"] > BLOW for v in d.values())

    # --- arm table ---
    rows = []
    for x in arms:
        v, l = V[x["name"]], L[x["name"]]
        prot = paired(v, l, "overall") if l else None
        rows.append(dict(stage=x["stage"], arm=x["name"], n=len(v), overall=mean(v, "overall"),
                         sd=st.stdev([q["overall"] for q in v.values()]) if len(v) > 1 else 0.0,
                         T_inner=mean(v, "T_inner"), T_outer=mean(v, "T_outer"), T_avg=mean(v, "T_avg"),
                         f12=mean(v, "f12"), sw21=mean(v, "sw21"), r49=mean(v, "r49"), blowups=blows(v),
                         best_epoch_med=st.median(q["best_epoch"] for q in v.values()) if v else float("nan"),
                         capped=sum(q["capped"] for q in v.values()),
                         legacy_n=len(l), legacy_overall=mean(l, "overall"), legacy_blowups=blows(l),
                         legacy_best_epoch_med=st.median(q["best_epoch"] for q in l.values()) if l else float("nan"),
                         protocol_diff=prot["mean"] if prot else float("nan"),
                         protocol_lo=prot["lo"] if prot else float("nan"), protocol_hi=prot["hi"] if prot else float("nan")))
    with open(os.path.join(a.out, "r9_arms.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    R = {x["arm"]: x for x in rows}

    def table(stage_names, title):
        out("\n## " + title + "\n")
        out("| arm | n | overall ± sd | T_outer | T_avg | 12 hard | 21 sw | 49 rest | >1.5 | best ep (med) | at cap | legacy | vs2 − legacy [95% CI] | legacy best ep |")
        out("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for x in rows:
            if x["arm"] not in stage_names:
                continue
            leg = ("%.3f (%d)" % (x["legacy_overall"], x["legacy_blowups"])) if x["legacy_n"] else "—"
            pd_ = ("%+.3f [%+.3f, %+.3f]" % (x["protocol_diff"], x["protocol_lo"], x["protocol_hi"])) if x["legacy_n"] >= 3 else "—"
            out("| %s | %d | %.3f ± %.3f | %.3f | %.3f | %.2f | %.2f | %.2f | %d | %.0f | %d | %s | %s | %s |" % (
                x["arm"], x["n"], x["overall"], x["sd"], x["T_outer"], x["T_avg"], x["f12"], x["sw21"], x["r49"],
                x["blowups"], x["best_epoch_med"], x["capped"], leg, pd_,
                ("%.0f" % x["legacy_best_epoch_med"]) if x["legacy_n"] else "—"))

    S1 = [x["name"] for x in arms if x["stage"] == "S1"]
    S2 = [x["name"] for x in arms if x["stage"] == "S2"]
    out("# Round 9 results -- %d arms, %d runs" % (len(arms), sum(len(v) for v in V.values())))
    table(S1, "S1 -- decision arms (vs2, cap 2000)")
    table(S2, "S2 -- split models (vs2, cap 2000)")

    # --- comparisons ---
    def cmp(x, y, keys=("overall", "T_outer", "T_avg", "sw21", "r49", "f12"), seeds=None, label=None):
        r_ = {k: paired(V[x], V[y], k, seeds) for k in keys}
        j = paired(V[x], V[y], "jumps", seeds)
        ex = {s for s in set(V[x]) | set(V[y]) if (s in V[x] and V[x][s]["overall"] > BLOW) or (s in V[y] and V[y][s]["overall"] > BLOW)}
        rob = paired(V[x], V[y], "overall", sorted((set(V[x]) & set(V[y])) - ex)) if ex else None
        return r_, j, ex, rob

    def cmp_block(pairs, title, seeds=None):
        out("\n## " + title + "\n")
        out("| comparison | overall | verdict | T_outer | T_avg | 21 sw | 49 rest | 12 hard | jumps | blow-ups a/b | w/o blow-up seeds |")
        out("|---|---|---|---|---|---|---|---|---|---|---|")
        res = {}
        for x, y in pairs:
            if x not in V or y not in V:
                continue
            r_, j, ex, rob = cmp(x, y, seeds=seeds)
            res[(x, y)] = (r_, j, ex, rob)
            jv = ("%+.1f p=%.3f" % (j["mean"], j["p"])) if j else "n/a"
            if j and j["mean"] > 0 and j["p"] < 0.05:
                jv += " **VETO**"
            out("| %s − %s | %s | %s | %s | %s | %s | %s | %s | %s | %d/%d | %s |" % (
                x, y, fmt(r_["overall"]), verdict(r_["overall"]), fmt(r_["T_outer"], 2).split(" p=")[0],
                fmt(r_["T_avg"], 2).split(" p=")[0], fmt(r_["sw21"], 2).split(" p=")[0],
                fmt(r_["r49"], 2).split(" p=")[0], fmt(r_["f12"], 2).split(" p=")[0], jv,
                blows(V[x]), blows(V[y]), fmt(rob) if rob else "—"))
        return res

    primary = [("r5-pos-h128x2", "r5-abs-h128x2"), ("r5-pos-h128x2", "r4-abs-h128x5"),
               ("r4-anchor_avg-h128x5", "r4-abs-h128x5"), ("r5-pos-h128x2", "r5-pos-h128x5"),
               ("r7-w161", "r5-pos-h128x2"),
               ("r6-tg10s40", "r6-ah"), ("r6-cgate-afb", "r6-ah"), ("r6-cgate-ah", "r6-ah"),
               ("r7-w161-cgate-ah", "r7-w161-ah"), ("r7-w161-cgate-ah", "r7-w161"),
               ("r7-w161-tg30s30", "r7-w161-ah"), ("r7-w161-fip", "r7-w161-fiz"),
               ("r8-w161-fip-cgate-ah", "r8-w161-fiz-cgate-ah"), ("r8-w161-fip-cgate-ah", "r7-w161-cgate-ah"),
               ("r8-w161-To-em1200", "r7-w161"), ("r8-w161-To-em1200-cgate-ah", "r7-w161-cgate-ah")]
    P1 = cmp_block(primary, "Primary comparisons (vs2, all common seeds)")
    cmp_block(primary, "Primary comparisons on the 17 non-pilot seeds",
              seeds=[s for s in range(0, 200) if s not in PILOT])

    # --- hypothesis ---
    out("\n## Hypothesis: (To-em1200-cgate-ah − cgate-ah) switched-21 T_avg under vs2 (legacy +3.40 [+1.76, +5.57])\n")
    h_all = paired(V["r8-w161-To-em1200-cgate-ah"], V["r7-w161-cgate-ah"], "sw21")
    h_np = paired(V["r8-w161-To-em1200-cgate-ah"], V["r7-w161-cgate-ah"], "sw21", [s for s in range(200) if s not in PILOT])
    h_leg = paired(L["r8-w161-To-em1200-cgate-ah"], L["r7-w161-cgate-ah"], "sw21")
    out("- vs2, all seeds:   " + fmt(h_all, 2))
    out("- vs2, 17 non-pilot: " + fmt(h_np, 2))
    out("- legacy (recomputed): " + fmt(h_leg, 2))
    if h_all:
        out("- (a) validation set caused the collapse (CI upper < +1.0): **%s**" % ("YES" if h_all["hi"] < 1.0 else "NO"))
        out("- (b) the reference still costs T_avg (CI excludes 0): **%s**" % ("YES" if h_all["lo"] > 0 else "NO"))
    # difference in differences
    common = sorted(set(V["r8-w161-To-em1200-cgate-ah"]) & set(V["r7-w161-cgate-ah"]) & set(L["r8-w161-To-em1200-cgate-ah"]) & set(L["r7-w161-cgate-ah"]))
    if common:
        did = [(V["r8-w161-To-em1200-cgate-ah"][s]["sw21"] - V["r7-w161-cgate-ah"][s]["sw21"]) -
               (L["r8-w161-To-em1200-cgate-ah"][s]["sw21"] - L["r7-w161-cgate-ah"][s]["sw21"]) for s in common]
        out("- difference-in-differences (vs2 gap − legacy gap), per seed mean %+.2f, median %+.2f, n=%d" % (st.mean(did), st.median(did), len(did)))
    for arm in ("r8-w161-To-em1200-cgate-ah", "r7-w161-cgate-ah", "r8-w161-To-em1200", "r7-w161"):
        out("- best epoch %-28s vs2 median %.0f (at cap %d/%d) | legacy median %s" % (
            arm, R[arm]["best_epoch_med"], R[arm]["capped"], R[arm]["n"],
            ("%.0f" % R[arm]["legacy_best_epoch_med"]) if R[arm]["legacy_n"] else "—"))

    # --- weights ---
    wt = [n for n in S1 if n.startswith(("r6-w", "r7-w1-", "r8-w1"))]
    cmp_block([(n, "r7-w161") for n in wt if not n.endswith("To-em1200")] +
              [("r8-w163-To-em1200", "r8-w161-To-em1200"), ("r8-w131-To-em1200", "r8-w161-To-em1200"),
               ("w166-cgate-ah", "r7-w161-cgate-ah"), ("w166-cgate-ah", "r6-cgate-ah"),
               ("r6-cgate-ah", "r7-w161-cgate-ah")],
              "Loss-weight vectors under vs2 (vs 1-6-1)")

    # --- early window (time prior) ---
    out("\n## Early-window fix (time prior): T_avg |error| on the 49 non-switched cases\n")
    runs_root = a.runs
    for tp, base in (("tp-w161-cgate-ah", "r7-w161-cgate-ah"), ("tp-w161", "r7-w161")):
        ew = {}
        for arm in (tp, base):
            ew[arm] = {}
            for s in V[arm]:
                d = find(A[arm]["run_name"], s)
                ew[arm][s] = dict(t50=early_window(d, 50), t200=early_window(d, 200))
        r50 = paired(ew[tp], ew[base], "t50")
        r200 = paired(ew[tp], ew[base], "t200")
        rov = paired(V[tp], V[base], "overall")
        j = paired(V[tp], V[base], "jumps")
        passed = r200 and r200["hi"] < 0 and rov and rov["lo"] <= 0 and blows(V[tp]) == 0 and not (j and j["mean"] > 0 and j["p"] < 0.05)
        out("- %s vs %s: first 200 steps %s (%.2f -> %.2f); first 50 %s (%.2f -> %.2f); overall %s; jumps %s; blow-ups %d -> **%s**" % (
            tp, base, fmt(r200, 2), st.mean(v["t200"] for v in ew[base].values()), st.mean(v["t200"] for v in ew[tp].values()),
            fmt(r50, 2), st.mean(v["t50"] for v in ew[base].values()), st.mean(v["t50"] for v in ew[tp].values()),
            fmt(rov), fmt(j, 1), blows(V[tp]), "PASS" if passed else "FAIL"))

    # --- split models ---
    s2pairs = [("ind-w161", "r7-w161"), ("ind-w161-cgate-ah", "r7-w161-cgate-ah"),
               ("ind-w161-To-em1200", "r8-w161-To-em1200"), ("ind-w161-To-em1200-cgate-ah", "r8-w161-To-em1200-cgate-ah"),
               ("ind-w161-To-em1200-cgate-ah", "ind-w161-cgate-ah"), ("ind-h64-w161-cgate-ah", "ind-w161-cgate-ah"),
               ("inda-w161-cgate-ah", "r7-w161-cgate-ah"), ("inda-w161-cgate-ah", "ind-w161-cgate-ah"),
               ("inda-w161-To-em1200-cgate-ah", "r8-w161-To-em1200-cgate-ah"),
               ("inda-w161-To-em1200-cgate-ah", "inda-w161-cgate-ah")]
    cmp_block(s2pairs, "S2 -- split models vs their twins")
    out("\nper-network best / total epoch (medians):")
    for n in S2:
        st_ = [q["stages"] for q in V[n].values() if q.get("stages")]
        if st_:
            parts = []
            for i, s0 in enumerate(st_[0]):
                parts.append("%s %.0f/%.0f" % (s0["channel"], st.median(x[i]["best_epoch"] for x in st_), st.median(x[i]["total_epoch"] for x in st_)))
            out("- %-30s %s" % (n, "; ".join(parts)))

    # --- legacy inventory: protocol effect distribution and ranking ---
    out("\n## Protocol effect over all arms with a legacy twin (vs2 − legacy, overall)\n")
    pe = [x for x in rows if x["legacy_n"] >= 3]
    out("- %d arms: better under vs2 (CI < 0) %d, worse (CI > 0) %d, noise %d; median shift %+.3f" % (
        len(pe), sum(x["protocol_hi"] < 0 for x in pe), sum(x["protocol_lo"] > 0 for x in pe),
        sum(x["protocol_lo"] <= 0 <= x["protocol_hi"] for x in pe), st.median(x["protocol_diff"] for x in pe)))
    for stage in ("S1", "S1b", "S1c"):
        xs = [x for x in pe if x["stage"] == stage]
        if xs:
            out("  - %s: %d arms, better %d, worse %d, median %+.3f" % (stage, len(xs), sum(x["protocol_hi"] < 0 for x in xs),
                                                                    sum(x["protocol_lo"] > 0 for x in xs), st.median(x["protocol_diff"] for x in xs)))
    out("\n## Top 15 arms under vs2 by mean overall (all stages)\n")
    out("| rank | arm | stage | n | overall | >1.5 | sw21 | r49 | T_outer |")
    out("|---|---|---|---|---|---|---|---|---|")
    for i, x in enumerate(sorted(rows, key=lambda x: x["overall"])[:15]):
        out("| %d | %s | %s | %d | %.3f ± %.3f | %d | %.2f | %.2f | %.3f |" % (i + 1, x["arm"], x["stage"], x["n"], x["overall"], x["sd"], x["blowups"], x["sw21"], x["r49"], x["T_outer"]))

    # --- deployment: ensembles ---
    out("\n## Deployment metric: seed-ensemble MAE\n")
    leg_cah = [find(A["r7-w161-cgate-ah"]["run_name"].replace("_ep2000", "").replace("_vs2", ""), s) for s in A["r7-w161-cgate-ah"]["seeds"]]
    leg_cah = [d for d in leg_cah if d]
    if leg_cah:
        e, ch = ensemble(leg_cah)
        out("- legacy cgate-ah, %d seeds: %.3f (the recorded 0.676; T_inner %.3f T_outer %.3f T_avg %.3f)" % (len(leg_cah), e, *ch))
    top = [x["arm"] for x in sorted(rows, key=lambda x: x["overall"])[:8]]
    for arm in dict.fromkeys(top + ["r5-abs-h128x2", "r4-abs-h128x5", "r7-w161-cgate-ah", "tp-w161-cgate-ah",
                                    "r8-w161-To-em1200-cgate-ah", "ind-w161-To-em1200"]):
        ds = [find(A[arm]["run_name"], s) for s in sorted(V[arm])]
        e, ch = ensemble([d for d in ds if d])
        out("- %-34s %2d seeds: ensemble %.3f (single-model mean %.3f; T_inner %.3f T_outer %.3f T_avg %.3f)" % (arm, len(ds), e, R[arm]["overall"], *ch))

    open(os.path.join(a.out, "r9_report.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\nwrote %s/r9_report.md, r9_arms.csv, r9_runs.json" % a.out)


if __name__ == "__main__":
    main()
