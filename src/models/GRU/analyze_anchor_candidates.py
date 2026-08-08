"""Data-level evaluation of anchor candidates for each output channel.

No training involved. For every candidate reference signal this computes the
RESIDUAL the model would still have to learn, and compares it against the
channel's absolute spread. That ratio is what decides whether an anchor is
worth using:

    residual_std / absolute_std   ->  the fraction of the work left over

Findings this script backs up (train set, 311 cases):
  * T_input is an excellent anchor for T_inner (1% left) and a poor one for
    T_avg (64%) and T_outer (83%) -- it is thermally adjacent to T_inner
    (0.40 C apart on average) but far from the others (53.8 / 75.2 C).
  * T_avg's best anchor is a weighted average of T_inner and T_outer
    (fitted w ~ 0.29), leaving only 4%.
  * T_outer has no strong candidate; its own initial value leaves 49%.

An anchor needs BOTH properties to be useful:
  (a) the reference is externally known, so errors do not accumulate
      (this is what rules out anchoring on the model's own previous
      prediction -- see arm D, which turns the rollout into an integrator);
  (b) the reference tracks the target closely, so the residual is small
      (this is what rules out T_input for T_avg / T_outer).

Usage:  python analyze_anchor_candidates.py
"""
import csv
import glob
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))          # src/models/GRU
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))   # repo root
TRAIN_DIR = os.path.join(BASE_DIR, "data",
                         "Latest Database (Use this for training)")

COLS = ("T_outer (C)", "T_inner (C)", "T_avg (C)", "Input Temperature (C)")


def load_cases(d):
    out = []
    for f in glob.glob(os.path.join(d, "**", "*.csv"), recursive=True):
        rows = list(csv.DictReader(open(f, newline="")))
        if not rows or "T_inner (C)" not in rows[0]:
            continue
        out.append({c: np.array([float(r[c]) for r in rows]) for c in COLS})
    return out


def fit_blend_weight(cases):
    """Least-squares w for  T_avg ~ w*T_inner + (1-w)*T_outer."""
    num = den = 0.0
    for g in cases:
        x = g["T_inner (C)"] - g["T_outer (C)"]
        num += float((x * (g["T_avg (C)"] - g["T_outer (C)"])).sum())
        den += float((x * x).sum())
    return num / den if den > 1e-9 else 0.5


def main():
    cases = load_cases(TRAIN_DIR)
    if not cases:
        raise SystemExit(f"no training CSVs found under {TRAIN_DIR}")
    print(f"train cases: {len(cases)}   (source: {TRAIN_DIR})\n")
    w = fit_blend_weight(cases)

    def report(target, name, fn):
        base = np.concatenate([g[target] for g in cases]).std()
        r = np.concatenate([fn(g) for g in cases])
        print(f"  {name:<34} std {r.std():7.2f}   mean {r.mean():+8.2f}   "
              f"|max| {np.abs(r).max():6.1f}   leaves "
              f"{100 * r.std() / base:3.0f}% of the work")

    for target, label, candidates in (
        ("T_inner (C)", "T_inner", [
            ("vs Input_T", lambda g: g["T_inner (C)"] - g["Input Temperature (C)"]),
        ]),
        ("T_avg (C)", "T_avg", [
            ("vs Input_T", lambda g: g["T_avg (C)"] - g["Input Temperature (C)"]),
            ("vs T_inner", lambda g: g["T_avg (C)"] - g["T_inner (C)"]),
            ("vs T_avg(0) [initial value]", lambda g: g["T_avg (C)"] - g["T_avg (C)"][0]),
            (f"vs {w:.2f}*T_inner+{1-w:.2f}*T_outer",
             lambda g: g["T_avg (C)"] - (w * g["T_inner (C)"] + (1 - w) * g["T_outer (C)"])),
        ]),
        ("T_outer (C)", "T_outer", [
            ("vs Input_T", lambda g: g["T_outer (C)"] - g["Input Temperature (C)"]),
            ("vs T_inner", lambda g: g["T_outer (C)"] - g["T_inner (C)"]),
            ("vs T_outer(0) [initial value]", lambda g: g["T_outer (C)"] - g["T_outer (C)"][0]),
        ]),
    ):
        base = np.concatenate([g[target] for g in cases]).std()
        print(f"=== {label} ===")
        print(f"  {'absolute (no anchor)':<34} std {base:7.2f}"
              f"{'':38}leaves 100% of the work")
        for name, fn in candidates:
            report(target, name, fn)
        print()

    print("=== how far each channel sits from Input_T ===")
    for label, col in (("T_inner", "T_inner (C)"),
                       ("T_avg", "T_avg (C)"),
                       ("T_outer", "T_outer (C)")):
        d = np.concatenate([np.abs(g[col] - g["Input Temperature (C)"])
                            for g in cases])
        print(f"  |{label:<8} - Input_T|   mean {d.mean():7.2f} C   "
              f"max {d.max():7.1f} C")
    print(f"\nfitted blend weight w = {w:.4f}  "
          f"(T_avg sits ~{100*w:.0f}% of the way from T_outer toward T_inner)")


if __name__ == "__main__":
    main()
