"""Energy-domain metrics and seed-ensemble uncertainty for a finished configuration.

Reads every seed's predictions.npz of one configuration and reports what an
energy-systems reader asks for, on top of the per-channel MAE the runs
already record (2026-09-23, for the ECM / Applied Energy write-up):

  stored energy   E(t) = m cp (T_avg(t) - T_avg(0)). Without m cp the error
                  is reported in K (multiply by m cp yourself, or pass
                  --mcp-kj-per-k to get kWh).
  state of charge SoC(t) = (T_avg(t) - T_lo) / (T_hi - T_lo), T_lo / T_hi the
                  run's own true extremes; error in percentage points, so it
                  is comparable across runs of different swing.
  bound violation share of steps where the predicted T_avg lies outside the
                  predicted walls' interval, on the steps where the TRUE
                  T_avg is inside it (the position formula's premise); the
                  switched cases, where the truth itself leaves the interval,
                  are reported separately.
  wall balance    residual of the lumped outer-wall balance
                  tau dT_out/dt + T_out - T_in (tau in steps) for the
                  prediction vs for the truth. The truth's residual is not
                  zero (the one-node model is approximate); a prediction with
                  the same residual RMS is physically as consistent as the
                  data.
  uncertainty     the 20-seed spread (std across seeds, mean over time) vs
                  the ensemble mean's error, per case and channel:
                  correlation across cases, and a calibration table by spread
                  quintile. A usable UQ signal has error rising with spread.
  cost            training time per seed and inference time per case from
                  meta.json, next to the parameter count.

Usage:
    python energy_uq_report.py <runs root> "<glob of one config's seed dirs>" [--out prefix] [--mcp-kj-per-k X]
    python energy_uq_report.py ../../../../calit2-thermal-wiki/round8 "*_w1-6-1_oth-pos_head_h128x2dp0.3_cgate_ah_seed*"
Writes <prefix>_cases.csv (per case), <prefix>_summary.json and <prefix>_uq.png.
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

FLAGGED = set(range(32, 41)) | {51, 52, 53}
SWITCHED = FLAGGED | {50, 58} | set(range(59, 66))
# predictions.npz column layout (ThermalDataset.BASE_COLS order)
COL = {"time": 0, "T_outer": 1, "T_inner": 2, "T_avg": 3, "Input_T": 4}
TAU_STEPS = 1200.0      # the round-8 outer-wall time constant, in 10 s steps


def load_config(root, pattern):
    dirs = sorted(glob.glob(os.path.join(root, pattern)))
    seeds, preds, metas, actual, ids = [], [], [], None, None
    for d in dirs:
        vd = os.path.join(d, "variants", "forward_direct")
        f = os.path.join(vd, "predictions.npz")
        if not os.path.isfile(f) or not os.path.isfile(os.path.join(vd, "done.flag")):
            continue
        z = np.load(f, allow_pickle=True)
        cid = [int(re.search(r"\((\d+)\)", c).group(1)) for c in z["case_ids"]]
        P = np.stack([np.asarray(p, dtype=float) for p in z["inv_preds"]])
        A = np.stack([np.asarray(a, dtype=float) for a in z["inv_actuals"]])
        if ids is None:
            ids, actual = cid, A
        else:
            assert cid == ids, "case order differs between seeds"
        preds.append(P)
        seeds.append(int(d.rsplit("seed", 1)[1]))
        metas.append(json.load(open(os.path.join(vd, "meta.json"))))
    if not preds:
        sys.exit("no finished seeds match {} under {}".format(pattern, root))
    return np.array(ids), actual, np.stack(preds), seeds, metas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("pattern")
    ap.add_argument("--out", default=None, help="output prefix (default: energy_uq_<config>)")
    ap.add_argument("--mcp-kj-per-k", type=float, default=None,
                    help="m*cp of the storage medium in kJ/K; converts the energy error to kWh")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ids, A, P, seeds, metas = load_config(a.root, a.pattern)      # P: (S, N, T, 5)
    S, N, T, _ = P.shape
    ens = P.mean(axis=0)                                            # (N, T, 5)
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", a.pattern.strip("*").replace("_seed", ""))
    prefix = a.out or ("energy_uq_" + tag)
    print("config {}  seeds {}  cases {}  steps {}".format(tag, S, N, T))

    Ta, To, Ti, Tin = (COL[k] for k in ("T_avg", "T_outer", "T_inner", "Input_T"))
    rows = []
    for i, c in enumerate(ids):
        tru, prd = A[i], ens[i]
        # --- stored energy per unit m cp (K) and SoC (pp) ---
        e_tru = tru[:, Ta] - tru[0, Ta]
        e_prd = prd[:, Ta] - prd[0, Ta]
        e_mae_K = float(np.abs(e_prd - e_tru).mean())
        lo, hi = tru[:, Ta].min(), tru[:, Ta].max()
        swing = max(hi - lo, 1e-6)
        soc_mae_pp = float(100.0 * np.abs(prd[:, Ta] - tru[:, Ta]).mean() / swing)
        soc_end_pp = float(100.0 * (prd[-1, Ta] - tru[-1, Ta]) / swing)
        # --- bound violation on the steps where the truth is inside its walls ---
        t_in = (tru[:, Ta] >= np.minimum(tru[:, Ti], tru[:, To]) - 0.5) & (tru[:, Ta] <= np.maximum(tru[:, Ti], tru[:, To]) + 0.5)
        p_out = (prd[:, Ta] < np.minimum(prd[:, Ti], prd[:, To]) - 0.5) | (prd[:, Ta] > np.maximum(prd[:, Ti], prd[:, To]) + 0.5)
        viol = float((p_out & t_in).sum() / max(1, t_in.sum()))
        # --- lumped outer-wall balance residual, prediction vs truth ---
        # dT_out/dt from a +-30-step centred difference: the raw one-step
        # difference times tau (1200 steps) turns 0.01 C of step-to-step
        # noise into 12 C of residual and swamps the physics.
        def resid(x, w=30):
            d = (x[2 * w:, To] - x[:-2 * w, To]) / (2.0 * w)
            return TAU_STEPS * d + x[w:-w, To] - x[w:-w, Tin]
        r_tru, r_prd = resid(tru), resid(prd)
        # --- uncertainty: spread across seeds vs error of the ensemble mean ---
        spread = {k: float(P[:, i, :, COL[k]].std(axis=0).mean()) for k in ("T_inner", "T_outer", "T_avg")}
        err = {k: float(np.abs(ens[i, :, COL[k]] - tru[:, COL[k]]).mean()) for k in ("T_inner", "T_outer", "T_avg")}
        rows.append(dict(case=int(c), switched=int(c in SWITCHED), flagged=int(c in FLAGGED),
                         mae_T_inner=err["T_inner"], mae_T_outer=err["T_outer"], mae_T_avg=err["T_avg"],
                         energy_mae_K=e_mae_K, soc_mae_pp=soc_mae_pp, soc_end_err_pp=soc_end_pp,
                         swing_K=float(swing), bound_violation=viol,
                         wall_resid_rms_true=float(np.sqrt((r_tru ** 2).mean())),
                         wall_resid_rms_pred=float(np.sqrt((r_prd ** 2).mean())),
                         spread_T_inner=spread["T_inner"], spread_T_outer=spread["T_outer"], spread_T_avg=spread["T_avg"]))

    with open(prefix + "_cases.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    def mean(key, sel=lambda r: True):
        v = [r[key] for r in rows if sel(r)]
        return float(st.mean(v)) if v else float("nan")
    non = lambda r: not r["switched"]
    sw = lambda r: bool(r["switched"])
    summary = dict(config=tag, seeds=S, cases=N,
                   mae=dict(T_inner=mean("mae_T_inner"), T_outer=mean("mae_T_outer"), T_avg=mean("mae_T_avg")),
                   energy_mae_K=dict(all=mean("energy_mae_K"), non_switched=mean("energy_mae_K", non), switched=mean("energy_mae_K", sw)),
                   soc_mae_pp=dict(all=mean("soc_mae_pp"), non_switched=mean("soc_mae_pp", non), switched=mean("soc_mae_pp", sw)),
                   soc_end_abs_pp=float(st.mean(abs(r["soc_end_err_pp"]) for r in rows)),
                   bound_violation=dict(non_switched=mean("bound_violation", non), switched=mean("bound_violation", sw)),
                   wall_resid_rms=dict(truth=mean("wall_resid_rms_true"), prediction=mean("wall_resid_rms_pred")),
                   cost=dict(train_time_s_per_seed=float(st.mean(m.get("Train_Time_s", float("nan")) for m in metas)),
                             infer_ms_per_case=float(st.mean(m.get("Infer_ms_per_case", float("nan")) for m in metas)),
                             params=int(metas[0].get("Total_Params", 0))))
    if a.mcp_kj_per_k:
        summary["energy_mae_kWh"] = {k: v * a.mcp_kj_per_k / 3600.0 for k, v in summary["energy_mae_K"].items()}

    # --- UQ: correlation and calibration by spread quintile, per channel ---
    uq = {}
    for k in ("T_inner", "T_outer", "T_avg"):
        s = np.array([r["spread_" + k] for r in rows]); e = np.array([r["mae_" + k] for r in rows])
        rank = lambda x: np.argsort(np.argsort(x))
        pear = float(np.corrcoef(s, e)[0, 1]); spear = float(np.corrcoef(rank(s), rank(e))[0, 1])
        q = np.quantile(s, [0.2, 0.4, 0.6, 0.8])
        bins = np.digitize(s, q)
        calib = [dict(quintile=b + 1, n=int((bins == b).sum()), spread=float(s[bins == b].mean()), error=float(e[bins == b].mean()))
                 for b in range(5) if (bins == b).any()]
        uq[k] = dict(pearson=pear, spearman=spear, calibration=calib)
    summary["uq"] = uq
    json.dump(summary, open(prefix + "_summary.json", "w"), indent=1)

    # --- plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, k in zip(axes, ("T_inner", "T_outer", "T_avg")):
            s = [r["spread_" + k] for r in rows]; e = [r["mae_" + k] for r in rows]
            col = ["tab:red" if r["switched"] else "tab:blue" for r in rows]
            ax.scatter(s, e, c=col, s=18)
            for r in rows:
                if r["mae_" + k] > 2.0 or r["spread_" + k] > 1.0:
                    ax.annotate(str(r["case"]), (r["spread_" + k], r["mae_" + k]), fontsize=7)
            ax.set_xlabel("seed spread, std across {} seeds (C)".format(S)); ax.set_ylabel("ensemble-mean MAE (C)")
            ax.set_title("{}: Spearman {:.2f}".format(k, uq[k]["spearman"]))
            ax.grid(alpha=0.3)
        axes[0].scatter([], [], c="tab:blue", label="non-switched"); axes[0].scatter([], [], c="tab:red", label="switched (gated)"); axes[0].legend(fontsize=8)
        fig.suptitle("Seed spread vs error per test case -- {}".format(tag), fontsize=10)
        fig.tight_layout(); fig.savefig(prefix + "_uq.png", dpi=130); plt.close(fig)
    except Exception as ex:
        print("plot skipped:", ex)

    # --- report ---
    m = summary
    print("\nchannel MAE (ensemble of {} seeds): T_inner {:.3f}  T_outer {:.3f}  T_avg {:.3f}".format(S, m["mae"]["T_inner"], m["mae"]["T_outer"], m["mae"]["T_avg"]))
    print("stored-energy MAE per unit m*cp:  all {:.2f} K   non-switched {:.2f} K   switched {:.2f} K".format(*[m["energy_mae_K"][k] for k in ("all", "non_switched", "switched")]))
    if "energy_mae_kWh" in m:
        print("stored-energy MAE:                all {:.2f} kWh  non-switched {:.2f}  switched {:.2f}".format(*[m["energy_mae_kWh"][k] for k in ("all", "non_switched", "switched")]))
    print("SoC MAE (pp of each run's swing): all {:.1f}   non-switched {:.1f}   switched {:.1f};  |end-of-run SoC error| {:.1f} pp".format(
        m["soc_mae_pp"]["all"], m["soc_mae_pp"]["non_switched"], m["soc_mae_pp"]["switched"], m["soc_end_abs_pp"]))
    print("T_avg outside predicted walls where the truth is inside: non-switched {:.2%} of steps, switched {:.2%}".format(
        m["bound_violation"]["non_switched"], m["bound_violation"]["switched"]))
    print("outer-wall balance residual RMS (tau={:.0f} steps): truth {:.2f} C, prediction {:.2f} C".format(TAU_STEPS, m["wall_resid_rms"]["truth"], m["wall_resid_rms"]["prediction"]))
    print("cost: {:.0f} s training per seed, {:.1f} ms inference per case, {:,} parameters".format(
        m["cost"]["train_time_s_per_seed"], m["cost"]["infer_ms_per_case"], m["cost"]["params"]))
    print("\nseed spread vs ensemble error across the {} cases:".format(N))
    for k in ("T_inner", "T_outer", "T_avg"):
        u = uq[k]
        print("  {:<8} Pearson {:+.2f}  Spearman {:+.2f} | by spread quintile (spread -> error): ".format(k, u["pearson"], u["spearman"])
              + "  ".join("{:.2f}->{:.2f}".format(cb["spread"], cb["error"]) for cb in u["calibration"]))
    print("\nwrote {}_cases.csv, {}_summary.json, {}_uq.png".format(prefix, prefix, prefix))


if __name__ == "__main__":
    main()
