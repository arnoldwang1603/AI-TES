"""Gate check for Arms Z2-Z5: does the lumped heat-flow rule fit the data?

    python check_energy_fit.py

Fits  T_avg(t+1) - T_avg(t) = a*(T_inner - T_avg) + b*(T_outer - T_avg) + c
on the training set (the exact fit train.py will use) and reports how well it
explains the step-to-step change in T_avg, on train and on the 70-case test
set. No GPU, no training -- takes seconds.

Verdict thresholds (R^2 of the dT_avg prediction on the TEST set):
    >= 0.8   GO           -- run Z2-Z5 as planned
    0.5-0.8  MARGINAL     -- run Z2 only first and check its T_avg before Z3-Z5
    <  0.5   SWITCH       -- the rule is too crude; change Z2-Z5 to the
                             curvature penalty (see note printed below)
"""
import numpy as np

from tes_gru.data import load_all_data, fit_energy_balance, energy_residual_stats


def main():
    train_dfs, val_dfs, test_dfs, _ = load_all_data()
    if train_dfs is None:
        raise SystemExit("Could not load data.")

    fit = fit_energy_balance(train_dfs)
    te = energy_residual_stats(test_dfs, fit)
    va = energy_residual_stats(val_dfs, fit) if val_dfs else None

    print("\n" + "=" * 64)
    print("LUMPED ENERGY-BALANCE FIT  (dTa = a(Ti-Ta) + b(To-Ta) + c)")
    print("=" * 64)
    print(f"  a = {fit['a']:.6f}   b = {fit['b']:.6f}   c = {fit['c']:+.6f} C/step")
    print(f"  train: R2 = {fit['r2']:.4f}   resid std = {fit['sigma']:.4f} C   n = {fit['n']}")
    if va is not None:
        print(f"  val  : R2 = {va['r2']:.4f}   resid std = {va['sigma']:.4f} C   n = {va['n']}")
    print(f"  test : R2 = {te['r2']:.4f}   resid std = {te['sigma']:.4f} C   n = {te['n']}")

    sane = fit['a'] > 0 and fit['b'] > 0
    print(f"  signs: a, b > 0 (T_avg relaxes toward both walls)? {'yes' if sane else 'NO'}")

    r2 = te['r2']
    if not sane:
        verdict = "SWITCH (a or b <= 0: the fit is not physical)"
    elif r2 >= 0.8:
        verdict = "GO"
    elif r2 >= 0.5:
        verdict = "MARGINAL -- run Z2 alone first"
    else:
        verdict = "SWITCH"
    print(f"\n  VERDICT: {verdict}")
    if verdict.startswith("SWITCH"):
        print("  -> In run_fix_ablation.py, change _Z_ENERGY_BASE: remove\n"
              "     PINN_ENERGY_WEIGHT and add EARLY_SMOOTH_TO_WEIGHT='0.01',\n"
              "     EARLY_SMOOTH_TA_WEIGHT='0.02', EARLY_SMOOTH_TAU='6.0',\n"
              "     PINN_NORM='weighted' (curvature penalty at the same ratios).")
    print("=" * 64)


if __name__ == "__main__":
    main()
