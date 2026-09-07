"""Compare two configurations on the test cases they share, paired by seed.

Written for the round-7 sanity stage (Arnold, 2026-09-05): a model trained
WITHOUT the both-charging-and-discharging runs (EXCLUDE_BOTH_PHASE, 49 test
cases) against the full-trained model restricted to those same 49 cases. The
identity checks inside S0 (xb-cgate == xb-base, xb-cgate-afb == xb-ah) are
code checks; this comparison is the one that says whether removing those runs
from TRAINING helps or hurts the normal regime.

Usage:
    python sanity_compare.py --a runs "sanity_xb_pos_head/h128/L2_dp0.3_xb" \\
                             --b runs "pos_head/h128/L2_dp0.3"
    python sanity_compare.py --a runs "sanity_xb_pos_head/h128/L2_dp0.3_xb_w1-6-1" \\
                             --b ../round6/runs "pos_head/h128/L2_dp0.3_w1-6-1"

Each side is <runs dir> <family/hNNN/leaf> exactly as export_results names
them (see configs_ranked.csv). Per seed present on both sides, the mean MAE per
channel over the common cases is computed from summary_errors.csv, then the
paired difference (a - b), its sd, and an exact sign-flip permutation p.
"""
import csv
import os
import random
import re
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_results import collect                      # noqa: E402

CH = ("T_inner", "T_outer", "T_avg")


def side(argname):
    i = sys.argv.index(argname)
    return sys.argv[i + 1], sys.argv[i + 2]


def per_case(vd):
    out = {}
    p = os.path.join(vd, "summary_errors.csv")
    if not os.path.isfile(p):
        return out
    for row in csv.DictReader(open(p, newline="")):
        if row["Case"] == "AVERAGE":
            continue
        out[row["CaseFile"]] = {c: float(row["MAE_{} (C)".format(c)]) for c in CH}
    return out


def pick(src, key):
    fam, h, leaf = key.split("/", 2)
    h = int(h.lstrip("h"))
    rs = [r for r in collect(src) if (r["fam"], r["h"], r["leaf"]) == (fam, h, leaf)]
    if not rs:
        names = sorted({"{}/h{}/{}".format(r["fam"], r["h"], r["leaf"]) for r in collect(src)})
        sys.exit("{} not found under {}; available:\n  ".format(key, src) + "\n  ".join(names))
    return {r["seed"]: per_case(r["vd"]) for r in rs}


def signflip_p(d, n=20000):
    if not d:
        return float("nan")
    obs = abs(st.mean(d))
    rng = random.Random(0)
    hits = 0
    for _ in range(n):
        m = st.mean(x if rng.random() < 0.5 else -x for x in d)
        if abs(m) >= obs - 1e-12:
            hits += 1
    return hits / n


def main():
    if "--a" not in sys.argv or "--b" not in sys.argv:
        sys.exit(__doc__)
    (sa, ka), (sb, kb) = side("--a"), side("--b")
    A, B = pick(sa, ka), pick(sb, kb)
    seeds = sorted(set(A) & set(B))
    if not seeds:
        sys.exit("no seed present on both sides (a: {}, b: {})".format(sorted(A), sorted(B)))
    print("a = {}  ({} seeds)\nb = {}  ({} seeds)\npaired seeds: {}".format(
        ka, len(A), kb, len(B), seeds))
    rows, diffs = [], {c: [] for c in CH + ("overall",)}
    for s in seeds:
        common = sorted(set(A[s]) & set(B[s]))
        if not common:
            continue
        ma = {c: st.mean(A[s][k][c] for k in common) for c in CH}
        mb = {c: st.mean(B[s][k][c] for k in common) for c in CH}
        ma["overall"] = st.mean(ma[c] for c in CH)
        mb["overall"] = st.mean(mb[c] for c in CH)
        for c in diffs:
            diffs[c].append(ma[c] - mb[c])
        rows.append((s, len(common), ma, mb))
    print("\n{:>5} {:>5}  {:>22}  {:>22}  {:>8}".format("seed", "cases", "a: overall / T_avg",
                                                     "b: overall / T_avg", "a-b"))
    for s, n, ma, mb in rows:
        print("{:>5} {:>5}  {:>10.3f} / {:>9.3f}  {:>10.3f} / {:>9.3f}  {:>+8.3f}".format(
            s, n, ma["overall"], ma["T_avg"], mb["overall"], mb["T_avg"],
            ma["overall"] - mb["overall"]))
    print("\npaired mean difference a - b over {} seeds (negative = a better):".format(len(rows)))
    for c in ("overall",) + CH:
        d = diffs[c]
        print("  {:<8} {:>+7.3f}  sd {:.3f}  sign-flip p = {:.3f}".format(
            c, st.mean(d), st.stdev(d) if len(d) > 1 else 0.0, signflip_p(d)))


if __name__ == "__main__":
    main()
