"""paper_stats.py -- Round 10 / A5: the pre-registered statistics layer.

numpy + pandas only; importable and a CLI. Every decision of Round 10 is a
seed-paired comparison of two arms on the validation files (prereg
`prereg/round10_v3.json`, sections statistics / bands / vetoes_validation_side
/ verdict_classes). This module turns long tables into those verdicts.

Input: a LONG table with one row per (arm, seed, file, channel):
    arm, seed, file, channel, MAE
  * `file` may also be called case_file (or case_id); `MAE` may be called
    value.
  * channel in T_inner / T_outer / T_avg (deg C, per-file MAE). A masked
    validation file (H2) carries NaN or no row for T_avg: it is skipped, so
    T_avg is scored on the files finite in every seed of both arms (vs3:
    17), T_inner / T_outer on 20.
  * optional rows with channel JumpSteps_gt2C_ToTa / JumpSteps_gt1C_ToTa
    (per-file counts; summed per seed) feed veto V2.
  `long_table_from_runs` builds this table from run directories
  (summary_errors_val.csv of a sealed run, or eval_checkpoint --set val).

Statistic (analyze_round9.paired, calibrate_band.paired_stats verbatim):
  per-seed difference d_s = mean over files of (A - B); mean; 95 % CI from
  10,000 bootstrap resamples over seeds; two-sided sign-flip p (exact up to
  16 seeds, else 50,000 random flips). One numpy Generator is threaded
  through every call, so a run of this module is reproducible from its
  --seed; within one comparison the draws go T_inner, T_outer, T_avg, ToTa,
  JumpSteps_gt2C_ToTa, JumpSteps_gt1C_ToTa.

Band per comparison and channel (prereg bands.per_comparison_rule):
  band_c = max(fixed_c, sqrt((1.96 * sd_case / sqrt(n))^2 + seed_halfwidth^2))
  fixed: T_inner 0.02, T_outer 0.05, T_avg 0.08 deg C; sd_case = sd over the
  n files of the seed-mean per-file difference; seed_halfwidth = half the
  bootstrap CI. A CI inside +/- band is worded "indistinguishable at n=<n>".

Vetoes (validation side):
  V1  any seed of the challenger (arm A) whose mean of the three channel
      MAEs exceeds 1.5 deg C;
  V2  paired JumpSteps_gt2C_ToTa or _gt1C_ToTa increase (A - B > 0) with
      sign-flip p < 0.05.

Verdict classes (prereg verdict_classes, tested in this order):
  BETTER      CI of (T_outer+T_avg)/2 entirely < 0 AND T_avg CI upper <= 0
              AND T_outer CI upper <= +band AND T_inner CI upper <= +band
              AND no veto
  EQUIVALENT  every channel's CI inside +/- band_c
  NO_WORSE    every channel's CI upper <= +band_c and no veto
  WORSE       a veto fired, or some channel's CI lies entirely above 0
  UNDECIDED   otherwise
  (EQUIVALENT is tested before NO_WORSE because it implies it; the prereg
  attaches the veto to BETTER and NO_WORSE only, so a vetoed arm with every
  CI inside the bands is still EQUIVALENT -- the veto columns say so.)

Holm: within a family, the sign-flip p-values of one channel across the
family's comparisons are Holm-adjusted (`p_holm`, `holm_reject`).

CLI (from src/models/GRU):
    python paper_stats.py --long long.csv --pair B2_mask:B2_orig[:label] [--pair ...]
                          [--family P1] [--out paper_stats_out] [--seed 0]
    python paper_stats.py --from-runs runs --arm B2_mask=<RUN_NAME_BASE> --arm B2_orig=<RUN_NAME_BASE>
                          --seeds 1,2,3,5,7,11,13,21,42,123 --long-out long.csv [--pair ...]
Writes <out>/<family>_tidy.csv (one row per comparison x channel),
<out>/<family>_verdicts.csv and <out>/<family>.md.
"""
import argparse
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd

CHANNELS = ("T_inner", "T_outer", "T_avg")
JUMP_COLS = ("JumpSteps_gt2C_ToTa", "JumpSteps_gt1C_ToTa")
# These two values implement the registered analyze_round9.paired algorithm;
# decision thresholds and bootstrap coverage/count come from the prereg file.
N_FLIP = 50000
EXACT_FLIP_MAX_N = 16
VERDICTS = ("BETTER", "EQUIVALENT", "NO_WORSE", "WORSE", "UNDECIDED")
TIDY_COLS = ["family", "label", "armA", "armB", "channel", "n_seeds", "n_files", "meanA", "meanB",
             "diff_mean", "ci_lo", "ci_hi", "ci_halfwidth", "p_signflip", "p_holm", "holm_reject",
             "sd_seed_diff", "se_seed_diff", "sd_case_diff", "case_halfwidth", "band_fixed", "band",
             "inside_band", "upper_within_band", "frac_files_A_better", "wording"]
VERDICT_COLS = ["family", "label", "armA", "armB", "verdict", "vetoed", "veto_V1", "veto_V2",
                "blowup_seeds_A", "blowup_seeds_B", "n_seeds", "n_files_T_inner", "n_files_T_outer",
                "n_files_T_avg", "ToTa_diff", "ToTa_ci_lo", "ToTa_ci_hi", "ToTa_p",
                "J2_diff", "J2_p", "J1_diff", "J1_p", "reason"]


def load_registered_policy(prereg=None):
    """Parse the active registration, failing closed on an unsupported rule.

    The JSON currently expresses some rules as prose. Their numeric literals
    are read from that prose instead of maintaining another threshold table.
    A future change to the rule's meaning requires an explicit implementation,
    rather than silently retaining a stale interpretation.
    """
    if prereg is None or isinstance(prereg, (str, os.PathLike)):
        path = prereg or os.environ.get("ROUND10_PREREG") or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "prereg", "round10_v3.json")
        with open(path, encoding="utf-8-sig") as f:
            prereg = json.load(f)
    text = prereg["statistics"]["paired_test"]
    count = re.search(r"([0-9,]+)\s+bootstrap resamples", text)
    coverage = re.search(r"([0-9.]+)%\s*CI", text)
    blowup = re.search(r">\s*([0-9.]+)\s*degC", prereg["vetoes_validation_side"]["V1_blowup"])
    jump_rule = prereg["vetoes_validation_side"]["V2_jumps"]
    alpha = re.search(r"p\s*<\s*([0-9.]+)", jump_rule)
    jump_thresholds = re.findall(r"([0-9]+)\s*degC", jump_rule)
    z = re.search(r"([0-9.]+)\s*\*\s*sd_case", prereg["bands"]["per_comparison_rule"])
    if not all((count, coverage, blowup, alpha, z)) or len(jump_thresholds) != 2:
        raise ValueError("Cannot interpret registered bootstrap/band/veto numeric rules")
    expected = {
        "BETTER": ("(T_outer+T_avg)/2", "entirely < 0", "T_avg CI upper <= 0",
                   "T_outer CI upper <= +band_T_outer", "T_inner CI upper <= +band_T_inner", "no veto"),
        "NO_WORSE": ("every channel", "CI upper <= +band_c", "no veto"),
        "EQUIVALENT": ("every channel", "CI inside +/- band_c"),
    }
    for name, fragments in expected.items():
        rule = prereg["verdict_classes"][name]
        if not all(fragment in rule for fragment in fragments):
            raise ValueError("Unsupported registered %s rule: %s" % (name, rule))
    if ("mean of the three validation channel MAEs" not in prereg["vetoes_validation_side"]["V1_blowup"]
            or "T_outer+T_avg only" not in jump_rule or "increase across seeds" not in jump_rule):
        raise ValueError("Unsupported registered validation veto semantics")
    bands = {ch: float(prereg["bands"]["fixed_bands_degC"][ch]) for ch in CHANNELS}
    confidence = float(coverage.group(1)) / 100.0
    policy = dict(fixed_bands=bands, n_boot=int(count.group(1).replace(",", "")),
                  confidence=confidence, blowup=float(blowup.group(1)), alpha=float(alpha.group(1)),
                  z=float(z.group(1)), jump_cols=tuple("JumpSteps_gt%sC_ToTa" % t for t in jump_thresholds))
    if (policy["n_boot"] <= 0 or not 0 < confidence < 1 or not 0 < policy["alpha"] < 1
            or policy["blowup"] <= 0 or policy["z"] <= 0 or any(v < 0 for v in bands.values())):
        raise ValueError("Invalid registered statistics policy")
    return policy


def configure_policy(prereg=None):
    """Select the registration used by this process (pipeline passes its JSON)."""
    global FIXED_BANDS, BLOWUP_C, N_BOOT, ALPHA, Z95, CI_PERCENTILES, JUMP_COLS
    policy = load_registered_policy(prereg)
    FIXED_BANDS = policy["fixed_bands"]
    BLOWUP_C, N_BOOT, ALPHA, Z95 = policy["blowup"], policy["n_boot"], policy["alpha"], policy["z"]
    JUMP_COLS = policy["jump_cols"]
    tail = (1.0 - policy["confidence"]) * 50.0
    # Rounding avoids changing the archived 95% algorithm's exact 2.5/97.5
    # percentile calls because binary subtraction produces 2.500000000000002.
    CI_PERCENTILES = [round(tail, 12), round(100.0 - tail, 12)]
    return policy


configure_policy()


# ----------------------------------------------------------------- statistic
def paired_stats(d, rng, n_boot=None, n_flip=N_FLIP):
    """d: per-seed paired differences. Mean, 95 % bootstrap CI over seeds
    (n_boot resamples), two-sided sign-flip p (exact up to 16 pairs, else
    n_flip random flips) -- analyze_round9.paired, draw for draw."""
    d = np.asarray(d, dtype=float)
    n_boot = N_BOOT if n_boot is None else int(n_boot)
    n = len(d)
    if n < 3:
        raise ValueError("paired_stats needs at least 3 pairs, got %d" % n)
    boot = rng.choice(d, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, CI_PERCENTILES)
    obs = abs(d.mean())
    if n <= EXACT_FLIP_MAX_N:
        signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
    else:
        signs = rng.choice([-1.0, 1.0], size=(n_flip, n))
    p = float((np.abs((signs * d).mean(axis=1)) >= obs - 1e-12).mean())
    return dict(n=n, mean=float(d.mean()), lo=float(lo), hi=float(hi), p=p)


def band_for(sd_case, n_files, seed_halfwidth, fixed):
    """prereg bands.per_comparison_rule."""
    case_hw = Z95 * sd_case / math.sqrt(n_files) if n_files > 0 and np.isfinite(sd_case) else float("nan")
    comb = math.sqrt(case_hw ** 2 + seed_halfwidth ** 2) if np.isfinite(case_hw) else seed_halfwidth
    return float(max(fixed, comb)), float(case_hw)


def holm(pvals, alpha=None):
    """Holm step-down. Returns (adjusted p-values, reject flags); NaN p-values
    stay NaN and do not count towards m."""
    p = np.asarray(pvals, dtype=float)
    alpha = ALPHA if alpha is None else float(alpha)
    adj = np.full(p.shape, np.nan)
    idx = np.where(np.isfinite(p))[0]
    m = len(idx)
    if m == 0:
        return adj, np.zeros(p.shape, dtype=bool)
    order = idx[np.argsort(p[idx], kind="stable")]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    reject = np.zeros(p.shape, dtype=bool)
    reject[idx] = adj[idx] <= alpha
    return adj, reject


# ----------------------------------------------------------------- long table
def normalise_long(df):
    """Canonical columns arm, seed (int), file (str), channel, value (float)."""
    d = df.copy()
    cols = {c.lower(): c for c in d.columns}
    fcol = cols.get("file") or cols.get("case_file") or cols.get("case_id")
    vcol = cols.get("mae") or cols.get("value")
    if fcol is None or vcol is None or "arm" not in cols or "seed" not in cols or "channel" not in cols:
        raise ValueError("long table needs arm, seed, file|case_file|case_id, channel, MAE|value; got %s" % list(df.columns))
    out = pd.DataFrame({"arm": d[cols["arm"]].astype(str), "seed": d[cols["seed"]].astype(int),
                        "file": d[fcol].astype(str), "channel": d[cols["channel"]].astype(str),
                        "value": pd.to_numeric(d[vcol], errors="coerce").astype(float)})
    return out


def seed_file_matrix(long, arm, channel):
    """DataFrame seeds x files of `channel` for `arm`, keeping the files that
    are finite in EVERY seed of the arm (masked T_avg files drop out)."""
    sub = long[(long["arm"] == arm) & (long["channel"] == channel)]
    if sub.empty:
        return pd.DataFrame()
    piv = sub.pivot(index="seed", columns="file", values="value")      # raises on duplicate rows
    return piv.dropna(axis=1, how="any").sort_index()


def paired_matrices(long, armA, armB, channel):
    """(seeds, files, XA, XB): common seeds and common files, arrays (S, F)."""
    pa, pb = seed_file_matrix(long, armA, channel), seed_file_matrix(long, armB, channel)
    seeds = sorted(set(pa.index) & set(pb.index))
    files = sorted(set(pa.columns) & set(pb.columns))
    if len(seeds) < 3 or not files:
        raise ValueError("%s vs %s on %s: %d common seeds, %d common files" % (armA, armB, channel, len(seeds), len(files)))
    return seeds, files, pa.loc[seeds, files].to_numpy(dtype=float), pb.loc[seeds, files].to_numpy(dtype=float)


def paired_difference(XA, XB, rng, fixed_band=None):
    """Seed-paired difference A - B from per-seed x per-file matrices (S, F):
    the statistic on the per-seed means plus the file-side spread and the
    per-comparison band (NaN when fixed_band is None)."""
    XA, XB = np.asarray(XA, dtype=float), np.asarray(XB, dtype=float)
    D = XA - XB
    d_seed = D.mean(axis=1)
    st = paired_stats(d_seed, rng)
    dbar_case = D.mean(axis=0)
    n_files = D.shape[1]
    sd_case = float(dbar_case.std(ddof=1)) if n_files > 1 else float("nan")
    seed_hw = (st["hi"] - st["lo"]) / 2.0
    if fixed_band is None:
        band, case_hw = float("nan"), float("nan")
    else:
        band, case_hw = band_for(sd_case, n_files, seed_hw, fixed_band)
    inside = bool(np.isfinite(band) and st["lo"] >= -band and st["hi"] <= band)
    return dict(n_seeds=int(D.shape[0]), n_files=int(n_files),
                meanA=float(XA.mean(axis=1).mean()), meanB=float(XB.mean(axis=1).mean()),
                diff_mean=st["mean"], ci_lo=st["lo"], ci_hi=st["hi"], ci_halfwidth=float(seed_hw),
                p_signflip=st["p"], sd_seed_diff=float(d_seed.std(ddof=1)),
                se_seed_diff=float(d_seed.std(ddof=1) / math.sqrt(len(d_seed))),
                sd_case_diff=sd_case, case_halfwidth=case_hw,
                band_fixed=(float("nan") if fixed_band is None else float(fixed_band)), band=band,
                inside_band=inside,
                upper_within_band=bool(np.isfinite(band) and st["hi"] <= band),
                frac_files_A_better=float((dbar_case < 0).mean()),
                wording=("indistinguishable at n=%d" % n_files) if inside else "")


def per_seed_means(long, arm):
    """{channel: Series seed -> mean MAE over the arm's files finite in every seed}."""
    return {ch: seed_file_matrix(long, arm, ch).mean(axis=1) for ch in CHANNELS}


def blowup_seeds(long, arm):
    """Seeds whose mean of the three channel MAEs exceeds BLOWUP_C (veto V1)."""
    ms = per_seed_means(long, arm)
    if any(s.empty for s in ms.values()):
        return []
    tbl = pd.concat([ms[ch].rename(ch) for ch in CHANNELS], axis=1, join="inner")
    return [int(s) for s in tbl.index[tbl.mean(axis=1) > BLOWUP_C]]


def seed_totals(long, arm, channel):
    """Series seed -> total over files (JumpSteps); empty when absent."""
    piv = seed_file_matrix(long, arm, channel)
    return piv.sum(axis=1) if not piv.empty else pd.Series(dtype=float)


# ----------------------------------------------------------------- comparison
def classify(ch_rows, tota, vetoed):
    """Verdict class from the per-channel rows, the ToTa row and the veto flag."""
    lo = {c: ch_rows[c]["ci_lo"] for c in CHANNELS}
    hi = {c: ch_rows[c]["ci_hi"] for c in CHANNELS}
    band = {c: ch_rows[c]["band"] for c in CHANNELS}
    inside = all(lo[c] >= -band[c] and hi[c] <= band[c] for c in CHANNELS)
    if (not vetoed and tota["ci_hi"] < 0 and hi["T_avg"] <= 0
            and hi["T_outer"] <= band["T_outer"] and hi["T_inner"] <= band["T_inner"]):
        return "BETTER"
    if inside:
        return "EQUIVALENT"
    if not vetoed and all(hi[c] <= band[c] for c in CHANNELS):
        return "NO_WORSE"
    if vetoed or any(lo[c] > 0 for c in CHANNELS):
        return "WORSE"
    return "UNDECIDED"


def compare(long, armA, armB, rng, bands=None, label=None, family=""):
    """One pre-registered comparison, A (challenger) minus B (reference).
    Returns (tidy rows: list of dicts, one per channel + ToTa + jumps;
    verdict: dict). RNG draw order: T_inner, T_outer, T_avg, ToTa, J2, J1."""
    bands = dict(FIXED_BANDS, **(bands or {}))
    label = label or ("%s-vs-%s" % (armA, armB))
    base = dict(family=family, label=label, armA=armA, armB=armB)
    rows, ch_rows, means = [], {}, {}
    for ch in CHANNELS:
        seeds, files, XA, XB = paired_matrices(long, armA, armB, ch)
        r = paired_difference(XA, XB, rng, bands[ch])
        r.update(base, channel=ch)
        rows.append(r)
        ch_rows[ch] = r
        means[ch] = (pd.Series(XA.mean(axis=1), index=seeds), pd.Series(XB.mean(axis=1), index=seeds))
    # (T_outer + T_avg) / 2 per seed, on the seeds common to both channels
    common = sorted(set(means["T_outer"][0].index) & set(means["T_avg"][0].index))
    tA = 0.5 * (means["T_outer"][0].loc[common] + means["T_avg"][0].loc[common])
    tB = 0.5 * (means["T_outer"][1].loc[common] + means["T_avg"][1].loc[common])
    st = paired_stats((tA - tB).to_numpy(), rng)
    tota = dict(base, channel="ToTa", n_seeds=st["n"], n_files=float("nan"), meanA=float(tA.mean()),
                meanB=float(tB.mean()), diff_mean=st["mean"], ci_lo=st["lo"], ci_hi=st["hi"],
                ci_halfwidth=(st["hi"] - st["lo"]) / 2.0, p_signflip=st["p"],
                sd_seed_diff=float((tA - tB).std(ddof=1)),
                se_seed_diff=float((tA - tB).std(ddof=1) / math.sqrt(st["n"])),
                sd_case_diff=float("nan"), case_halfwidth=float("nan"), band_fixed=float("nan"),
                band=float("nan"), inside_band=False, upper_within_band=False,
                frac_files_A_better=float("nan"), wording="")
    rows.append(tota)
    # vetoes
    bu_A, bu_B = blowup_seeds(long, armA), blowup_seeds(long, armB)
    veto1 = bool(bu_A)
    jumps, veto2 = {}, False
    for jc in JUMP_COLS:
        sA, sB = seed_totals(long, armA, jc), seed_totals(long, armB, jc)
        cs = sorted(set(sA.index) & set(sB.index))
        if len(cs) < 3:
            jumps[jc] = dict(diff=float("nan"), p=float("nan"), lo=float("nan"), hi=float("nan"), n=len(cs))
            continue
        d = (sA.loc[cs] - sB.loc[cs]).to_numpy(dtype=float)
        sj = paired_stats(d, rng)
        jumps[jc] = dict(diff=sj["mean"], p=sj["p"], lo=sj["lo"], hi=sj["hi"], n=sj["n"])
        if sj["mean"] > 0 and sj["p"] < ALPHA:
            veto2 = True
        rows.append(dict(base, channel=jc, n_seeds=sj["n"], n_files=float("nan"),
                         meanA=float(sA.loc[cs].mean()), meanB=float(sB.loc[cs].mean()),
                         diff_mean=sj["mean"], ci_lo=sj["lo"], ci_hi=sj["hi"],
                         ci_halfwidth=(sj["hi"] - sj["lo"]) / 2.0, p_signflip=sj["p"],
                         sd_seed_diff=float(d.std(ddof=1)), se_seed_diff=float(d.std(ddof=1) / math.sqrt(len(d))),
                         sd_case_diff=float("nan"), case_halfwidth=float("nan"), band_fixed=float("nan"),
                         band=float("nan"), inside_band=False, upper_within_band=False,
                         frac_files_A_better=float("nan"), wording=""))
    vetoed = veto1 or veto2
    verdict = classify(ch_rows, tota, vetoed)
    reason = []
    if veto1:
        reason.append("V1 blow-up seeds of %s: %s" % (armA, bu_A))
    if veto2:
        reason.append("V2 jump increase: " + ", ".join(
            "%s %+.1f (p=%.3f)" % (jc, jumps[jc]["diff"], jumps[jc]["p"]) for jc in JUMP_COLS
            if np.isfinite(jumps[jc]["p"]) and jumps[jc]["diff"] > 0 and jumps[jc]["p"] < ALPHA))
    inside = [c for c in CHANNELS if ch_rows[c]["inside_band"]]
    if inside:
        reason.append("inside band: " + ", ".join("%s (n=%d)" % (c, ch_rows[c]["n_files"]) for c in inside))
    v = dict(base, verdict=verdict, vetoed=vetoed, veto_V1=veto1, veto_V2=veto2,
             blowup_seeds_A=" ".join(map(str, bu_A)), blowup_seeds_B=" ".join(map(str, bu_B)),
             n_seeds=ch_rows["T_avg"]["n_seeds"],
             n_files_T_inner=ch_rows["T_inner"]["n_files"], n_files_T_outer=ch_rows["T_outer"]["n_files"],
             n_files_T_avg=ch_rows["T_avg"]["n_files"],
             ToTa_diff=tota["diff_mean"], ToTa_ci_lo=tota["ci_lo"], ToTa_ci_hi=tota["ci_hi"], ToTa_p=tota["p_signflip"],
             J2_diff=jumps[JUMP_COLS[0]]["diff"], J2_p=jumps[JUMP_COLS[0]]["p"],
             J1_diff=jumps[JUMP_COLS[1]]["diff"], J1_p=jumps[JUMP_COLS[1]]["p"],
             reason="; ".join(reason))
    return rows, v


def compare_family(long, pairs, rng=None, bands=None, family="family", seed=0):
    """pairs: iterable of (armA, armB) or (armA, armB, label). Runs `compare`
    for each in order (one RNG), Holm-adjusts the sign-flip p per channel
    across the family. Returns (tidy DataFrame, verdict DataFrame)."""
    rng = rng if rng is not None else np.random.default_rng(seed)
    long = normalise_long(long)
    rows, verdicts = [], []
    for p in pairs:
        armA, armB = p[0], p[1]
        label = p[2] if len(p) > 2 else None
        r, v = compare(long, armA, armB, rng, bands=bands, label=label, family=family)
        rows.extend(r)
        verdicts.append(v)
    tidy = pd.DataFrame(rows)
    tidy["p_holm"] = np.nan
    tidy["holm_reject"] = False
    for ch in list(CHANNELS) + ["ToTa"] + list(JUMP_COLS):
        m = tidy["channel"] == ch
        if m.any():
            adj, rej = holm(tidy.loc[m, "p_signflip"].to_numpy())
            tidy.loc[m, "p_holm"] = adj
            tidy.loc[m, "holm_reject"] = rej
    tidy = tidy[[c for c in TIDY_COLS if c in tidy.columns]]
    vdf = pd.DataFrame(verdicts)[VERDICT_COLS]
    return tidy, vdf


# ----------------------------------------------------------------- reporting
def _fmt(r):
    return "%+.3f [%+.3f, %+.3f]" % (r["diff_mean"], r["ci_lo"], r["ci_hi"])


def to_markdown(tidy, verdicts, family="family"):
    lines = ["# %s -- seed-paired comparisons (A - B, deg C, validation files)" % family, "",
             "Statistic: per-seed mean difference, 95 %% CI from %d seed bootstraps, exact sign-flip p; "
             "band_c = max(fixed, sqrt((1.96 sd_case/sqrt(n))^2 + seed_halfwidth^2)); "
             "a CI inside +/- band reads \"indistinguishable at n\"." % N_BOOT, ""]
    for _, v in verdicts.iterrows():
        lines.append("## %s: %s vs %s -> **%s**%s" % (v["label"], v["armA"], v["armB"], v["verdict"],
                                                    "  (VETO)" if v["vetoed"] else ""))
        if v["reason"]:
            lines.append("")
            lines.append("_%s_" % v["reason"])
        lines.append("")
        lines.append("| channel | n seeds | n files | mean A | mean B | diff [95% CI] | band | p sign-flip | p Holm | reading |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        sub = tidy[tidy["label"] == v["label"]]
        for _, r in sub.iterrows():
            band = ("%.3f" % r["band"]) if np.isfinite(r["band"]) else "-"
            nf = ("%d" % r["n_files"]) if np.isfinite(r["n_files"]) else "-"
            ph = ("%.3f" % r["p_holm"]) if np.isfinite(r["p_holm"]) else "-"
            if r["channel"] in CHANNELS:
                reading = r["wording"] or ("upper within band" if r["upper_within_band"] else
                                           "A better" if r["ci_hi"] < 0 else "A worse" if r["ci_lo"] > 0 else "undecided")
            else:
                reading = "A better" if r["ci_hi"] < 0 else "A worse" if r["ci_lo"] > 0 else "noise"
            lines.append("| %s | %d | %s | %.3f | %.3f | %s | %s | %.3f | %s | %s |" % (
                r["channel"], r["n_seeds"], nf, r["meanA"], r["meanB"], _fmt(r), band, r["p_signflip"], ph, reading))
        lines.append("")
    return "\n".join(lines) + "\n"


def write_outputs(tidy, verdicts, out_dir, family="family"):
    os.makedirs(out_dir, exist_ok=True)
    tidy.to_csv(os.path.join(out_dir, "%s_tidy.csv" % family), index=False)
    verdicts.to_csv(os.path.join(out_dir, "%s_verdicts.csv" % family), index=False)
    md = to_markdown(tidy, verdicts, family)
    with open(os.path.join(out_dir, "%s.md" % family), "w", encoding="utf-8") as fh:
        fh.write(md)
    return md


# ----------------------------------------------------------------- from run dirs
def long_table_from_runs(arm_dirs, runs_root, seeds, variant="forward_direct",
                         csv_name="summary_errors_val.csv"):
    """arm_dirs: {arm: RUN_NAME_BASE}. Reads runs/<RUN_NAME_BASE>_seed<s>/
    variants/<variant>/<csv_name> (a sealed run's own validation scoring or
    eval_checkpoint --set val) and returns the long table (MAE per channel
    per file; JumpSteps columns per file for V2). Missing runs are reported
    and skipped."""
    rows, missing = [], []
    for arm, base in arm_dirs.items():
        for s in seeds:
            p = os.path.join(runs_root, "%s_seed%d" % (base, int(s)), "variants", variant, csv_name)
            if not os.path.isfile(p):
                missing.append(p)
                continue
            df = pd.read_csv(p)
            if "Case" in df.columns:
                df = df[df["Case"] != "AVERAGE"]
            for _, r in df.iterrows():
                f = str(r["CaseFile"])
                for ch in CHANNELS:
                    col = "MAE_%s (C)" % ch
                    if col in df.columns:
                        rows.append(dict(arm=arm, seed=int(s), file=f, channel=ch,
                                         MAE=pd.to_numeric(r[col], errors="coerce")))
                for jc in JUMP_COLS:
                    if jc in df.columns:
                        rows.append(dict(arm=arm, seed=int(s), file=f, channel=jc,
                                         MAE=pd.to_numeric(r[jc], errors="coerce")))
    if missing:
        print("long_table_from_runs: %d run csv(s) missing, e.g. %s" % (len(missing), missing[0]), file=sys.stderr)
    return pd.DataFrame(rows, columns=["arm", "seed", "file", "channel", "MAE"])


# ----------------------------------------------------------------- CLI
def _parse_pairs(items):
    pairs = []
    for it in items or []:
        parts = it.split(":")
        if len(parts) not in (2, 3):
            sys.exit("--pair takes A:B or A:B:label, got %r" % it)
        pairs.append(tuple(parts))
    return pairs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--long", default=None, help="long table csv (arm, seed, file, channel, MAE)")
    ap.add_argument("--from-runs", default=None, help="runs root; build the long table from run dirs")
    ap.add_argument("--arm", action="append", default=[], help="arm=RUN_NAME_BASE (with --from-runs)")
    ap.add_argument("--seeds", default="1,2,3,5,7,11,13,21,42,123")
    ap.add_argument("--csv-name", default="summary_errors_val.csv")
    ap.add_argument("--long-out", default=None, help="write the built long table here")
    ap.add_argument("--pair", action="append", default=[], help="A:B[:label]; repeatable")
    ap.add_argument("--family", default="family")
    ap.add_argument("--out", default="paper_stats_out")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the bootstraps")
    ap.add_argument("--bands", default=None, help='json, e.g. {"T_avg": 0.1}, overriding the fixed bands')
    a = ap.parse_args(argv)
    if a.from_runs:
        arm_dirs = dict(x.split("=", 1) for x in a.arm)
        if not arm_dirs:
            sys.exit("--from-runs needs at least one --arm name=RUN_NAME_BASE")
        seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
        long = long_table_from_runs(arm_dirs, a.from_runs, seeds, csv_name=a.csv_name)
        if a.long_out:
            long.to_csv(a.long_out, index=False)
            print("long table -> %s (%d rows)" % (a.long_out, len(long)))
    elif a.long:
        long = pd.read_csv(a.long)
    else:
        sys.exit("give --long or --from-runs")
    pairs = _parse_pairs(a.pair)
    if not pairs:
        if a.from_runs and a.long_out:
            return 0
        sys.exit("no --pair given")
    bands = json.loads(a.bands) if a.bands else None
    tidy, verdicts = compare_family(long, pairs, bands=bands, family=a.family, seed=a.seed)
    md = write_outputs(tidy, verdicts, a.out, a.family)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
