"""Analyse the reference-threshold study (PREREG_reference_threshold.md).

Reads validation results (summary_errors_val.csv) of every level, of the Round 10
baselines B2_mask (C1, direct), C11 (nominal reference) and C2 (initial value),
pairs them by seed with C1, bootstraps 95% CIs over seeds, joins the training
residual shares from level_stats.csv, and writes:
  threshold_table.csv / .md, threshold_curve.png, threshold_boundary.csv, threshold_collapse.csv

Usage (from src/models/GRU):
  python round10_posthoc/analyze_threshold.py --runs-root runs --baseline-root <dir with r10v4_ runs>
       [--cache-root runs/_threshold_caches] [--out runs/_threshold_results]
Python 3.8+; numpy, pandas, matplotlib.
"""
import argparse
import glob
import math
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BODY = "_forward_direct_Tin-anchor_l1_w1-6-3_oth-abs_h128x1dp0.0_"
TAGS = {"C1": "r10v4" + BODY + "ep2000_vs3_mk", "C11": "r10v4" + BODY + "phr_ep2000_vs3_mk",
        "C2": "r10v4_forward_direct_Tin-anchor_l1_w1-6-3_oth-anchor_initial_h128x1dp0.0_init_ep2000_vs3_mk"}
LEVELS = ["s080", "s060", "s040", "s025", "b005", "b010", "b020", "b035", "b050",
          "s015", "s010", "s006", "b075", "b085", "b100"]   # last six = 2026-10-21 extension
BANDS = {"T_inner": 0.02, "T_outer": 0.05, "T_avg": 0.08, "ToTa": 0.05}
CH = ("T_inner", "T_outer", "T_avg")
# Training residual shares of the Round 10 end points (review 2026-10-17, C16 table)
ENDPOINTS = {"C11": (0.62, 0.44), "C2": (49.0, 66.0)}


def seed_table(root, base):
    """seed -> dict(channel -> file-mean MAE); T_avg excludes NaN (masked) files."""
    out = {}
    for d in glob.glob(os.path.join(root, base + "_seed*")):
        m = re.search(r"_seed(\d+)$", d.rstrip("/\\"))
        f = os.path.join(d, "variants", "forward_direct", "summary_errors_val.csv")
        if not m or not os.path.isfile(os.path.join(d, "variants", "forward_direct", "done.flag")) or not os.path.isfile(f):
            continue
        t = pd.read_csv(f)
        t = t[~t.iloc[:, 0].astype(str).str.upper().str.contains("AVERAGE")]
        out[int(m.group(1))] = {"T_inner": t["MAE_T_inner (C)"].mean(), "T_outer": t["MAE_T_outer (C)"].mean(),
                                "T_avg": t["MAE_T_avg (C)"].dropna().mean(), "n_avg": int(t["MAE_T_avg (C)"].notna().sum())}
    return out


def paired(a, b, rng, n_boot=10000):
    seeds = sorted(set(a) & set(b))
    res = {"n_seeds": len(seeds)}
    if len(seeds) < 3:
        return res
    for ch in CH + ("ToTa",):
        get = (lambda r: (r["T_outer"] + r["T_avg"]) / 2.0) if ch == "ToTa" else (lambda r, c=ch: r[c])
        d = np.array([get(a[s]) - get(b[s]) for s in seeds])
        boots = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
        res[ch + "_A"] = float(np.mean([get(a[s]) for s in seeds]))
        res[ch + "_diff"] = float(d.mean())
        res[ch + "_lo"], res[ch + "_hi"] = (float(x) for x in np.percentile(boots, [2.5, 97.5]))
    return res


def boot_crossing(points, c1, ch, rng, n_boot=2000):
    """Seed-bootstrap distribution of the share where the gain curve first reaches 0.

    points: list of (share, seed_table) sorted by share. Seeds are resampled jointly
    (the same labels for every level and for C1), so the paired structure is kept.
    Returns (median, lo, hi, fraction of draws with no crossing) or None.
    """
    seeds = sorted(set(c1).intersection(*[set(t) for _, t in points]))
    if len(points) < 2 or len(seeds) < 5:
        return None
    get = (lambda r: (r["T_outer"] + r["T_avg"]) / 2.0) if ch == "ToTa" else (lambda r: r[ch])
    d = np.array([[get(t[s]) - get(c1[s]) for s in seeds] for _, t in points])   # levels x seeds
    x = [sh for sh, _ in points]
    found = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(seeds), len(seeds))
        c = crossing(x, d[:, idx].mean(axis=1).tolist())
        if c is not None:
            found.append(c)
    if not found:
        return (None, None, None, 1.0)
    lo, med, hi = np.percentile(found, [2.5, 50, 97.5])
    return (float(med), float(lo), float(hi), 1.0 - len(found) / float(n_boot))


def crossing(x, y):
    """first log-x where y crosses 0 going up; None if never."""
    for i in range(1, len(x)):
        if y[i - 1] < 0 <= y[i]:
            lx0, lx1 = math.log(x[i - 1]), math.log(x[i])
            return math.exp(lx0 + (0 - y[i - 1]) * (lx1 - lx0) / (y[i] - y[i - 1]))
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default=os.path.join(HERE, "runs"))
    ap.add_argument("--baseline-root", default=os.path.join(HERE, "runs"))
    ap.add_argument("--cache-root", default=os.path.join(HERE, "runs", "_threshold_caches"))
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "_threshold_results"))
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(20261021)

    c1 = seed_table(args.baseline_root, TAGS["C1"])
    if len(c1) < 10:
        raise SystemExit("baseline C1 (B2_mask) not found under %s (found %d seeds)" % (args.baseline_root, len(c1)))
    stats = pd.read_csv(os.path.join(args.cache_root, "level_stats.csv")).set_index("level")

    rows = [dict(level="C1 direct", family="none", share_To=100.0, share_Ta=100.0,
                 **{k: v for k, v in paired(c1, c1, rng).items() if k == "n_seeds"},
                 **{ch + "_A": float(np.mean([r[ch] for r in c1.values()])) for ch in CH})]
    tables = {}
    for name in ("C11", "C2"):
        t = seed_table(args.baseline_root, TAGS[name])
        tables[name] = t
        if t:
            rows.append(dict(level=name, family="nominal" if name == "C11" else "initial value",
                             share_To=ENDPOINTS[name][0], share_Ta=ENDPOINTS[name][1], **paired(t, c1, rng)))
    for lev in LEVELS:
        t = seed_table(args.runs_root, "r10th_" + lev + BODY + "phr_ep2000_vs3_mk")
        if not t:
            continue
        tables[lev] = t
        s = stats.loc[lev]
        rows.append(dict(level=lev, family=s["family"], value=s["value"], share_To=s["train_share_T_outer_pct"],
                         share_Ta=s["train_share_T_avg_pct"], ref_val_mae_To=s["val_mae_T_outer"],
                         ref_val_mae_Ta=s["val_mae_T_avg"], **paired(t, c1, rng)))
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(args.out, "threshold_table.csv"), index=False)

    def fmt(r, ch):
        if pd.isna(r.get(ch + "_diff", np.nan)):
            return ""
        return "%+.3f [%+.3f, %+.3f]" % (r[ch + "_diff"], r[ch + "_lo"], r[ch + "_hi"])
    lines = ["| level | family | share T_outer % | share T_avg % | seeds | dT_inner | dT_outer | dT_avg | dToTa | ref own val MAE To/Ta |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in tab.iterrows():
        own = "" if pd.isna(r.get("ref_val_mae_To", np.nan)) else "%.2f / %.2f" % (r["ref_val_mae_To"], r["ref_val_mae_Ta"])
        lines.append("| %s | %s | %.2f | %.2f | %s | %s | %s | %s | %s | %s |" % (
            r["level"], r["family"], r["share_To"], r["share_Ta"], int(r.get("n_seeds", 0) or 0),
            fmt(r, "T_inner"), fmt(r, "T_outer"), fmt(r, "T_avg"), fmt(r, "ToTa"), own))
    lines.append("")
    lines.append("Differences are level minus C1 (B2_mask, direct), seed-paired, validation MAE in C; negative = the reference helps. "
                 "Bands for wording: T_inner 0.02, T_outer 0.05, T_avg 0.08 C. Direct model's own validation MAE: "
                 "T_outer %.3f, T_avg %.3f." % (np.mean([r["T_outer"] for r in c1.values()]), np.mean([r["T_avg"] for r in c1.values()])))

    # boundary per family and channel. C11 is the left end of both families. The blend family's right
    # end is b100 (initial value through the same residual pipeline) when it exists, else C2.
    bnd = []
    have_b100 = "b100" in set(tab.level)
    for fam in ("diffusivity", "blend"):
        keep = [fam, "nominal"] + (["initial value"] if fam == "blend" and not have_b100 else [])
        sub = tab[tab.family.isin(keep)]
        for ch, sc in (("T_outer", "share_To"), ("T_avg", "share_Ta"), ("ToTa", "share_To")):
            s = sub.dropna(subset=[ch + "_diff"]).sort_values(sc)
            if len(s) < 2:
                continue
            mean_cross = crossing(s[sc].tolist(), s[ch + "_diff"].tolist())
            first_ci = s[s[ch + "_hi"] >= 0][sc].min() if (s[ch + "_hi"] >= 0).any() else None
            bc = boot_crossing([(r[sc], tables[r["level"]]) for _, r in s.iterrows() if tables.get(r["level"])],
                               c1, ch, rng)
            bnd.append(dict(family=fam, channel=ch, share_where_mean_gain_reaches_0=mean_cross,
                            boot_median=bc[0] if bc else None, boot_ci_lo=bc[1] if bc else None,
                            boot_ci_hi=bc[2] if bc else None, boot_frac_no_crossing=bc[3] if bc else None,
                            first_share_with_CI_upper_ge_0=first_ci, levels_used=",".join(s.level)))
    pd.DataFrame(bnd).to_csv(os.path.join(args.out, "threshold_boundary.csv"), index=False)
    lines.append("")
    lines.append("Boundary (share of target spread, %; seed-bootstrap 95% CI of the crossing, and the share of draws with no crossing):")
    for b in bnd:
        lines.append("- %s %s: mean crosses 0 at %s; bootstrap %s; no crossing in %s of draws; first level with CI upper >= 0: %s" % (
            b["family"], b["channel"],
            "%.1f" % b["share_where_mean_gain_reaches_0"] if b["share_where_mean_gain_reaches_0"] else "not reached",
            ("%.1f [%.1f, %.1f]" % (b["boot_median"], b["boot_ci_lo"], b["boot_ci_hi"])) if b["boot_median"] else "n/a",
            ("%.0f%%" % (100 * b["boot_frac_no_crossing"])) if b["boot_frac_no_crossing"] is not None else "n/a",
            "%.1f" % b["first_share_with_CI_upper_ge_0"] if b["first_share_with_CI_upper_ge_0"] else "none"))

    # collapse: interpolate the blend curve at the diffusivity shares (log axis), compare with the band
    col = []
    d_s, b_s = tab[tab.family == "diffusivity"], tab[tab.family.isin(["blend", "nominal"])]
    for ch, sc in (("T_outer", "share_To"), ("T_avg", "share_Ta")):
        bb = b_s.dropna(subset=[ch + "_diff"]).sort_values(sc)
        if len(bb) < 2:
            continue
        for _, r in d_s.dropna(subset=[ch + "_diff"]).iterrows():
            x = r[sc]
            if not (bb[sc].min() <= x <= bb[sc].max()):
                continue
            yb = float(np.interp(math.log(x), np.log(bb[sc].values), bb[ch + "_diff"].values))
            col.append(dict(channel=ch, level=r["level"], share=x, diffusivity_diff=r[ch + "_diff"], blend_interp=yb,
                            gap=r[ch + "_diff"] - yb, band=BANDS[ch], within_band=abs(r[ch + "_diff"] - yb) <= BANDS[ch]))
    pd.DataFrame(col).to_csv(os.path.join(args.out, "threshold_collapse.csv"), index=False)
    if col:
        lines.append("")
        lines.append("Collapse test (diffusivity level vs blend curve at the same share): " + "; ".join(
            "%s %s gap %+.3f (%s)" % (c["channel"], c["level"], c["gap"], "within band" if c["within_band"] else "OUTSIDE band") for c in col))
    with open(os.path.join(args.out, "threshold_table.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for ax, (ch, sc) in zip(axes, (("T_outer", "share_To"), ("T_avg", "share_Ta"), ("ToTa", "share_To"))):
            for fam, mk in (("diffusivity", "o"), ("blend", "s"), ("nominal", "*"), ("initial value", "D")):
                s = tab[tab.family == fam].dropna(subset=[ch + "_diff"]).sort_values(sc)
                if len(s):
                    x, y = s[sc].to_numpy(float), s[ch + "_diff"].to_numpy(float)
                    yerr = [y - s[ch + "_lo"].to_numpy(float), s[ch + "_hi"].to_numpy(float) - y]
                    ax.errorbar(x, y, yerr=yerr,
                                marker=mk, capsize=3, linestyle="-" if fam in ("diffusivity", "blend") else "none", label=fam)
            ax.axhline(0, color="k", lw=0.8)
            ax.axhspan(-BANDS[ch], BANDS[ch], color="0.9", zorder=0)
            ax.set_xscale("log")
            ax.set_xlabel("training residual share of target spread (%)" + (" (T_outer axis)" if ch == "ToTa" else ""))
            ax.set_ylabel("validation MAE minus direct (C)")
            ax.set_title(ch)
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "threshold_curve.png"), dpi=150)
        print("wrote", os.path.join(args.out, "threshold_curve.png"))
    except Exception as e:  # plotting must never block the tables
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
