"""Seed-ensemble evaluation.

For every configuration in a runs folder, average the per-seed predictions
case by case and score the average. This is the recipe that gets deployed
(round 6: w1-6-1 single-seed mean 0.774, 12-seed ensemble 0.742), so every
round reports it next to the single-seed numbers.

Usage:
    python ensemble_eval.py runs                     # table on stdout + ensemble.csv
    python ensemble_eval.py runs --out my.csv
    python ensemble_eval.py runs --min-seeds 4       # skip thinner configs
    python ensemble_eval.py runs --seeds 7,21,42,... # restrict every config to
                                                     # these seeds, so configs run
                                                     # at 12 and at 20 seeds are
                                                     # ensembled at the same n
    python ensemble_eval.py runs --subset-size 12 --subset-reps 20
                                                     # also report mean +- sd of the
                                                     # ensemble over 20 random
                                                     # 12-seed subsets (uncertainty)

Reads predictions.npz (written by evaluate.py for every run) and meta.json.
Self-check: each run's single-seed per-channel MAE recomputed from its npz
must match its meta.json to 1e-3 on every channel, which guards the column
layout assumed below (a permutation of the three temperature columns would
fail on at least two channels).

Case subsets reported for T_avg: the 12 flagged cases (32-40, 51-53), the 21
cases the case gate switches (the 20 excursion cases + case 58), and the 49
it never touches. Configurations scored on a reduced test set (the round-7
sanity arms, 49 cases) are printed in their own block.
"""
import csv
import os
import re
import statistics as st
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_results import collect                      # noqa: E402

# inv_pred / inv_actual rows follow the scaler's column order
# (Time, T_outer, T_inner, T_avg, Input_T) -- see evaluate.SCALER_COL.
COL = {"T_inner": 2, "T_outer": 1, "T_avg": 3}
CH = ("T_inner", "T_outer", "T_avg")
FLAGGED = set(list(range(32, 41)) + [51, 52, 53])                 # 12
EXCURSION = set(list(range(32, 41)) + [50, 51, 52, 53] + list(range(59, 66)))  # 20
SWITCHED = EXCURSION | {58}                                      # 21 (case 58: false alarm)


def arg(name, default=None, cast=str):
    if name in sys.argv:
        return cast(sys.argv[sys.argv.index(name) + 1])
    return default


def case_no(cid):
    m = re.search(r"\((\d+)\)", str(cid))
    return int(m.group(1)) if m else None


def load_npz(vd):
    p = os.path.join(vd, "predictions.npz")
    if not os.path.isfile(p):
        return None
    z = np.load(p, allow_pickle=True)
    return {str(c): (np.asarray(a, dtype=float), np.asarray(b, dtype=float))
            for c, a, b in zip(z["case_ids"], z["inv_actuals"], z["inv_preds"])}


def score(preds_by_case):
    """preds_by_case: {case_id: (actual, pred)} -> dict of MAEs."""
    per_ch = {c: [] for c in CH}
    sub = {"flagged12": [], "switched21": [], "rest49": []}
    for cid, (act, pred) in preds_by_case.items():
        for c in CH:
            per_ch[c].append(float(np.mean(np.abs(pred[:, COL[c]] - act[:, COL[c]]))))
        n = case_no(cid)
        ta = per_ch["T_avg"][-1]
        if n in FLAGGED:
            sub["flagged12"].append(ta)
        if n in SWITCHED:
            sub["switched21"].append(ta)
        else:
            sub["rest49"].append(ta)
    out = {c: st.mean(v) for c, v in per_ch.items()}
    out["overall"] = st.mean(out[c] for c in CH)
    for k, v in sub.items():
        out[k + "_T_avg"] = st.mean(v) if v else float("nan")
    return out


def ensemble(loaded, cases):
    ens = {}
    for cid in cases:
        act = loaded[0][1][cid][0]
        preds = [d[cid][1] for _, d, _ in loaded]
        n = min(p.shape[0] for p in preds)
        ens[cid] = (act[:n], np.mean([p[:n] for p in preds], axis=0))
    return ens


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src = sys.argv[1]
    out_path = arg("--out", os.path.join(src, "..", "ensemble.csv"))
    min_seeds = arg("--min-seeds", 2, int)
    only_seeds = arg("--seeds")
    only_seeds = ({int(s) for s in only_seeds.split(",") if s.strip()}
                  if only_seeds else None)
    sub_size = arg("--subset-size", 0, int)
    sub_reps = arg("--subset-reps", 20, int)
    rng = np.random.default_rng(0)

    groups = {}
    for r in collect(src):
        if only_seeds is not None and r["seed"] not in only_seeds:
            continue
        groups.setdefault((r["fam"], r["h"], r["leaf"]), []).append(r)

    rows, warned = [], 0
    for (fam, h, leaf), rs in sorted(groups.items()):
        loaded = []
        for r in rs:
            d = load_npz(r["vd"])
            if d is None:
                continue
            s = score(d)
            for c in CH:
                ref = r["meta"].get("Test_MAE_" + c)
                if ref is not None and abs(s[c] - ref) > 1e-3:
                    warned += 1
                    print("WARNING {} seed {}: npz {} MAE {:.4f} != meta {:.4f} "
                          "(column layout?)".format(r["dir"], r["seed"], c, s[c], ref))
            loaded.append((r["seed"], d, s["overall"]))
        if len(loaded) < min_seeds:
            continue
        cases = set.intersection(*[set(d) for _, d, _ in loaded])
        e = score(ensemble(loaded, cases))
        singles = [s for _, _, s in loaded]
        row = dict(
            family=fam, hidden=h, config=leaf, seeds=len(loaded),
            cases=len(cases),
            single_mean=round(st.mean(singles), 4),
            single_best=round(min(singles), 4),
            ensemble_overall=round(e["overall"], 4),
            ensemble_T_inner=round(e["T_inner"], 4),
            ensemble_T_outer=round(e["T_outer"], 4),
            ensemble_T_avg=round(e["T_avg"], 4),
            ensemble_flagged12_T_avg=round(e["flagged12_T_avg"], 4),
            ensemble_switched21_T_avg=round(e["switched21_T_avg"], 4),
            ensemble_rest49_T_avg=round(e["rest49_T_avg"], 4),
            gain_vs_single_mean=round(st.mean(singles) - e["overall"], 4))
        if sub_size and len(loaded) > sub_size:
            vals = []
            for _ in range(sub_reps):
                pick = rng.choice(len(loaded), size=sub_size, replace=False)
                vals.append(score(ensemble([loaded[i] for i in pick], cases))["overall"])
            row["subset{}_ens_mean".format(sub_size)] = round(st.mean(vals), 4)
            row["subset{}_ens_sd".format(sub_size)] = round(
                st.stdev(vals) if len(vals) > 1 else 0.0, 4)
        rows.append(row)

    if not rows:
        sys.exit("no configuration with >= {} seeds and predictions.npz".format(min_seeds))
    full_n = max(r["cases"] for r in rows)
    rows.sort(key=lambda r: (r["cases"] != full_n, r["ensemble_overall"]))
    with open(out_path, "w", newline="") as f:
        fields = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k)
                        if k in rows[0] else 99)
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    hdr = "{:<38} {:>3} {:>7} {:>7} {:>8} {:>7} {:>7} {:>7}".format(
        "config", "n", "single", "best", "ensembl", "flag12", "swit21", "rest49")
    print(hdr)
    block = full_n
    for r in rows:
        if r["cases"] != block:
            block = r["cases"]
            print("--- scored on {} test cases (sanity arms; not comparable) ---".format(block))
        print("{:<38} {:>3} {:>7.3f} {:>7.3f} {:>8.3f} {:>7.2f} {:>7.2f} {:>7.2f}{}".format(
            (r["family"] + "/h{}/".format(r["hidden"]) + r["config"])[-38:],
            r["seeds"], r["single_mean"], r["single_best"], r["ensemble_overall"],
            r["ensemble_flagged12_T_avg"], r["ensemble_switched21_T_avg"],
            r["ensemble_rest49_T_avg"],
            ("   sub{}: {:.3f} +- {:.3f}".format(sub_size, r["subset{}_ens_mean".format(sub_size)],
                                                 r["subset{}_ens_sd".format(sub_size)])
             if "subset{}_ens_mean".format(sub_size) in r else "")))
    print("\nwritten -> {}{}".format(
        os.path.abspath(out_path),
        "   ({} layout warnings above)".format(warned) if warned else ""))


if __name__ == "__main__":
    main()
