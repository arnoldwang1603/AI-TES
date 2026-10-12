"""Round 10 / H3: pure aggregate of a per-file score table.

numpy + pandas only (no torch, no config import), so main.py (meta.json),
eval_checkpoint.py, the runner's smoke table and the statistics layer (A5)
all read the same numbers from the same function.

Input: the summary_errors DataFrame that evaluate.test_model returns (its
AVERAGE row is ignored), or a list of row dicts with the same column names.
NaN entries -- T_avg on a masked validation file (H2), channels a variant
does not predict -- are skipped: every per-channel mean is over the files
that carry a finite value, and n_files_<ch> says how many that was (vs3:
T_avg on 17 of 20 validation files, T_inner / T_outer on 20).

Keys returned (AGG_KEYS):
  n_files, n_files_<ch>                 files in the table / scored per channel
  MAE_<ch>, RMSE_<ch>, MaxErr_<ch>      file means, deg C
  Err_step1_<ch>, MaxErr_1_5_<ch>       file means of the G7 start-up metrics
  MAE_Overall                           mean of the three channel MAEs
  MAE_ToTa                              (MAE_T_outer + MAE_T_avg) / 2
  blowup                                MAE_Overall > BLOWUP_C (veto V1)
  JumpSteps_gt2C_ToTa, JumpSteps_gt1C_ToTa, JumpSteps_gt2C
                                        totals over files (veto V2 reads the
                                        first two)
"""
import numpy as np
import pandas as pd

CHANNELS = ("T_inner", "T_outer", "T_avg")
BLOWUP_C = 1.5
AGG_KEYS = (
    "n_files", "n_files_T_inner", "n_files_T_outer", "n_files_T_avg",
    "MAE_T_inner", "MAE_T_outer", "MAE_T_avg", "MAE_Overall", "MAE_ToTa",
    "RMSE_T_inner", "RMSE_T_outer", "RMSE_T_avg",
    "MaxErr_T_inner", "MaxErr_T_outer", "MaxErr_T_avg",
    "Err_step1_T_inner", "Err_step1_T_outer", "Err_step1_T_avg",
    "MaxErr_1_5_T_inner", "MaxErr_1_5_T_outer", "MaxErr_1_5_T_avg",
    "blowup", "JumpSteps_gt2C_ToTa", "JumpSteps_gt1C_ToTa", "JumpSteps_gt2C",
)


def _rows(table):
    df = table if isinstance(table, pd.DataFrame) else pd.DataFrame(list(table))
    if "Case" in df.columns:
        df = df[df["Case"] != "AVERAGE"]
    return df


def _finite(df, col):
    if col not in df.columns:
        return np.zeros(0, dtype=float)
    v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    return v[np.isfinite(v)]


def _mean(df, col):
    v = _finite(df, col)
    return float(v.mean()) if v.size else float("nan")


def _total(df, col):
    v = _finite(df, col)
    return int(round(float(v.sum()))) if v.size else 0


def aggregate_scores(table):
    """See the module docstring. Returns a plain dict with exactly AGG_KEYS."""
    df = _rows(table)
    out = {"n_files": int(len(df))}
    for ch in CHANNELS:
        out["n_files_" + ch] = int(_finite(df, "MAE_%s (C)" % ch).size)
        out["MAE_" + ch] = _mean(df, "MAE_%s (C)" % ch)
        out["RMSE_" + ch] = _mean(df, "RMSE_%s (C)" % ch)
        out["MaxErr_" + ch] = _mean(df, "MaxErr_%s (C)" % ch)
        out["Err_step1_" + ch] = _mean(df, "Err_step1_%s (C)" % ch)
        out["MaxErr_1_5_" + ch] = _mean(df, "MaxErr_1_5_%s (C)" % ch)
    maes = np.array([out["MAE_" + ch] for ch in CHANNELS], dtype=float)
    out["MAE_Overall"] = float(np.nanmean(maes)) if np.isfinite(maes).any() else float("nan")
    out["MAE_ToTa"] = 0.5 * (out["MAE_T_outer"] + out["MAE_T_avg"])
    out["blowup"] = bool(np.isfinite(out["MAE_Overall"]) and out["MAE_Overall"] > BLOWUP_C)
    out["JumpSteps_gt2C_ToTa"] = _total(df, "JumpSteps_gt2C_ToTa")
    out["JumpSteps_gt1C_ToTa"] = _total(df, "JumpSteps_gt1C_ToTa")
    out["JumpSteps_gt2C"] = _total(df, "JumpSteps_gt2C")
    assert set(out) == set(AGG_KEYS), sorted(set(out) ^ set(AGG_KEYS))
    return {k: out[k] for k in AGG_KEYS}          # documented order
