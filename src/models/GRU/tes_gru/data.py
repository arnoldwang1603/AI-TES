"""Variant-aware Dataset and the data loader / manual split."""
import os
import glob
import json
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

from .config import *


# ============================================================
# 1. Case detector, numpy twin (Round 7, 2026-09-06)
# ============================================================
# Same rule as rollout._input_case_gate (torch, batched): the input
# temperature peaks in the interior of the run and both the rise into and
# the fall out of that peak exceed 25% of the profile's span. Scale-invariant,
# so it gives the same answer on raw degrees and on MinMax-scaled columns.
# KEEP THE TWO IN SYNC -- run_round7.py's preflight asserts they agree on
# every test case before launching anything.
def input_case_flag_np(inp):
    """True if this run both charges and discharges (per-run flag)."""
    inp = np.asarray(inp, dtype=float).reshape(-1)
    n = inp.shape[0]
    if n == 0:
        return False
    k = int(np.argmax(inp))
    peak = inp[k]
    before = inp[:k + 1].min()
    after = inp[k:].min()
    span = inp.max() - inp.min()
    return bool((peak - after > 0.25 * span) and (peak - before > 0.25 * span)
                and (k > 0.02 * n) and (k < 0.98 * n))


def input_phase_flag_np(inp):
    """Per-timestep flag: 1 once the discharge has begun in a both-phase
    run, 0 elsewhere and 0 throughout every run the case flag rejects.

    "Begun" = the first step AFTER the peak plateau, where the plateau is
    the run of steps from the maximum onward that stay within 1% of the
    span of that maximum. A held peak therefore still counts as charging
    (review finding, 2026-09-06: argmax alone flipped the flag 24-37% of
    the run early on the profiles that hold their top temperature)."""
    inp = np.asarray(inp, dtype=float).reshape(-1)
    n = inp.shape[0]
    if not input_case_flag_np(inp):
        return np.zeros(n, dtype=float)
    k = int(np.argmax(inp))
    thr = inp.max() - 0.01 * (inp.max() - inp.min())
    k_end = k
    while k_end + 1 < n and inp[k_end + 1] >= thr:
        k_end += 1
    return (np.arange(n) > k_end).astype(float)


# ============================================================
# 2. Dataset (variant-aware)
# ============================================================
# Positions of the per-file items in ThermalDataset.__getitem__'s tuple that
# the loops pick out by index (the first five are unpacked by name).
ITEM_INP_LAST = 10      # H9: scaled Input_T(N)
ITEM_LOSS_MASK = 11     # H2: per-file loss mask (3,), T_avg entry 0 when masked
ITEM_PHYS_TARGETS = 12  # H6 only: scaled physical references at target t+1


def load_physics_cache(case_id, times, directory=None, physics_override=None):
    """Physical outer/line-average in raw Celsius at the requested times.

    An explicit mapping supports E3/G11 shifted references without writing
    or overwriting the nominal cache. Cache content is never a target label.
    """
    if physics_override is None:
        name = split_key(case_id).replace('\\', '/').replace('/', '__') + '.npz'
        path = os.path.join(directory or PHYSICS_CACHE_DIR, name)
        if not os.path.isfile(path):
            raise RuntimeError('missing physical reference: %s' % path)
        with np.load(path, allow_pickle=False) as z:
            physical = {k: np.asarray(z[k], dtype=float) for k in ('time', 'T_outer_phys', 'T_avg_line_phys')}
    else:
        physical = physics_override
    source_time = np.asarray(physical['time'], dtype=float).reshape(-1)
    times = np.asarray(times, dtype=float).reshape(-1)
    if len(source_time) < 2 or np.any(np.diff(source_time) <= 0):
        raise RuntimeError('physical reference times must increase')
    if times.min() < source_time[0] - 1e-6 or times.max() > source_time[-1] + 1e-6:
        raise RuntimeError('physical reference does not cover the requested horizon')
    result = np.column_stack([np.interp(times, source_time, physical[k]) for k in ('T_outer_phys', 'T_avg_line_phys')])
    if not np.isfinite(result).all():
        raise RuntimeError('nonfinite physical reference')
    return result


class ThermalDataset(Dataset):
    """
    Per variant returns the appropriate input feature layout.

    Stored arrays (per case):
      X            (N, F)   input feature rows for steps t = 0..N-1
      Y            (N, 3)   targets [T_inner, T_outer, T_avg] for steps t = 1..N
      time_values  (N,)     time at steps 1..N
      init_cond    (4,)     [T_outer, T_inner, T_avg, Input_T] at t=0
      window_obs   (W, 4)   first W rows of [T_outer, T_inner, T_avg, Input_T]
                            (only used by abs_window; otherwise ignored)
      full_*                full series for plotting / inverse_transform
      inp_last     ()       scaled Input_T at step N (the row the [:-1] cut
                            removes from X); the T_inner anchor's reference
                            for the LAST step under ANCHOR_LEAD=1 (H9 fix)

    Item tuple order (see ITEM_* below): X, Y, time_values, init_cond,
    window_obs, full_time, full_t_outer, full_t_inner, full_t_avg,
    full_input_temp, inp_last, loss_mask. New per-file items are appended at
    the END.
      loss_mask    (3,)     1 for every trained channel; T_avg entry 0 when
                            the file's T_avg label is masked (H2, tavg_masked)
    """

    BASE_COLS = ['Time (s)', 'T_outer (C)', 'T_inner (C)', 'T_avg (C)',
                 'Input Temperature (C)']

    def __init__(self, csv_or_df, variant, scaler=None, file_name=None,
                 window_size=WINDOW_SIZE, tavg_masked=False,
                 physics_override=None, physics_cache_dir=None):
        if isinstance(csv_or_df, str):
            self.file_name = os.path.basename(csv_or_df)
            df = pd.read_csv(csv_or_df)
        else:
            self.file_name = file_name or "unknown"
            df = csv_or_df.copy()

        self.variant = variant
        self.window_size = window_size
        self.tavg_masked = bool(tavg_masked)

        df.rename(columns={"T_ave (C)": "T_avg (C)"}, inplace=True)
        if "Input Temperature (C)" not in df.columns:
            df["Input Temperature (C)"] = df["T_inner (C)"]
        df["FileName"] = self.file_name

        if scaler is None:
            self.scaler = MinMaxScaler()
            self.scaler.fit(df[self.BASE_COLS])
        else:
            self.scaler = scaler

        if PHYS_MODE:
            if physics_override is None and self.file_name.startswith('synthetic_rest/'):
                temperature = float(df['T_outer (C)'].iloc[0])
                physics_override = {'time': df['Time (s)'].to_numpy(float),
                                    'T_outer_phys': np.full(len(df), temperature),
                                    'T_avg_line_phys': np.full(len(df), temperature)}
            physical = load_physics_cache(self.file_name, df['Time (s)'].to_numpy(float),
                                         directory=physics_cache_dir, physics_override=physics_override)
            df['_phys_outer'] = physical[:, 0] * self.scaler.scale_[1] + self.scaler.min_[1]
            df['_phys_avg'] = physical[:, 1] * self.scaler.scale_[3] + self.scaler.min_[3]
        df[self.BASE_COLS] = self.scaler.transform(df[self.BASE_COLS])

        # Delta features (GT) computed AFTER scaling. The training rollout
        # OVERRIDES idx 1/2/3 with AR-derived deltas (current - previous,
        # both from the AR chain), so the GT deltas stored here are only
        # used as scaffolding -- not as actual model inputs at runtime.
        # dInput_T is exogenous (Input_T is user-specified at deployment)
        # so it is kept GT and fed unchanged.
        df["dT_outer (C)"] = df["T_outer (C)"].diff().fillna(0)
        df["dT_inner (C)"] = df["T_inner (C)"].diff().fillna(0)
        df["dT_avg (C)"] = df["T_avg (C)"].diff().fillna(0)
        df["dInput Temperature (C)"] = df["Input Temperature (C)"].diff().fillna(0)

        # 5-input refactor (Arnold 2026-05-20). T_inner is now an input
        # (in addition to remaining a prediction target). Feature ordering:
        #   abs / abs_window / abs_sliding:
        #     idx 0 = Time, 1 = T_outer, 2 = T_inner, 3 = T_avg, 4 = Input_T
        #   delta:
        #     idx 0 = Time, 1 = dT_outer, 2 = dT_inner, 3 = dT_avg, 4 = dInput_T
        #   abs+delta:
        #     idx 0 = Time, 1..4 = abs (outer/inner/avg/Input_T),
        #     idx 5..8 = deltas in same channel order.
        if variant == 'delta':
            feat_cols = [
                "Time (s)",
                "dT_outer (C)", "dT_inner (C)", "dT_avg (C)",
                "dInput Temperature (C)",
            ]
        elif variant == 'abs+delta':
            feat_cols = [
                "Time (s)",
                "T_outer (C)", "T_inner (C)", "T_avg (C)", "Input Temperature (C)",
                "dT_outer (C)", "dT_inner (C)", "dT_avg (C)",
                "dInput Temperature (C)",
            ]
        elif variant == 'inverse_delta':
            # Only Time + dT_avg (per Arnold spec: T_avg is the input).
            feat_cols = ["Time (s)", "dT_avg (C)"]
        elif variant == 'inverse_abs+delta':
            feat_cols = ["Time (s)", "T_avg (C)", "dT_avg (C)"]
        elif variant in ('inverse_abs', 'inverse_abs_sliding'):
            # inverse_abs_sliding uses the same 2-d per-step features; the
            # W-length T_avg lookback window is built at rollout time
            # directly from inputs (T_avg is GT exogenous, no AR feedback).
            feat_cols = ["Time (s)", "T_avg (C)"]
        elif variant == 'forward_direct':
            # Exogenous-only seq2seq (2026-07-24): the model sees ONLY the
            # given boundary condition; all state channels are outputs, none
            # feed back. idx 0 = Time, 1 = Input_T.
            # INPUT_LOOKAHEAD (2026-08-06): idx 2..1+k carry the input temperature's next
            # k values (legal: the full curve is a given boundary condition).
            # Built AFTER scaling, so the lead columns share Input_T's scale;
            # the tail holds the last value.
            feat_cols = (["Input Temperature (C)"] if NO_TIME else
                         ["Time (s)", "Input Temperature (C)"])
            for _i in range(1, INPUT_LOOKAHEAD + 1):
                lead = f"InputT_lead{_i}"
                df[lead] = df["Input Temperature (C)"].shift(-_i).ffill()
                feat_cols.append(lead)
            # (the CASE_FLAG_INPUT column is appended after the variant
            #  branches below, so forward_direct and abs_sliding share it)
        elif variant == 'abs_sliding' and TINNER_MODE == 'output_only':
            # v22-style A/B: T_inner is predicted (targets unchanged) but is
            # NOT an input, so its own predictions never feed back.
            #   idx 0 = Time, 1 = T_outer, 2 = T_avg, 3 = Input_T
            feat_cols = [
                "Time (s)",
                "T_outer (C)", "T_avg (C)", "Input Temperature (C)",
            ]
        else:  # abs / abs_window / abs_sliding
            feat_cols = [
                "Time (s)",
                "T_outer (C)", "T_inner (C)", "T_avg (C)", "Input Temperature (C)",
            ]

        if PHYS_MODE in ('input', 'input_residual'):
            feat_cols.extend(['_phys_outer', '_phys_avg'])
        if CASE_FLAG_INPUT:
            # Round 7 (2026-09-06), extended to abs_sliding in round 8: one
            # column from the input-temperature profile alone, appended LAST
            # so every existing column index holds (forward_direct: Input_T
            # at idx 1 and the lead columns; abs_sliding: Input_T at idx 4,
            # the rollout appends this column to each window row itself).
            # Computed per run (the groupby below is one file, but stay
            # correct for several). Already 0/1, so it is NOT scaled.
            assert variant in ('forward_direct', 'abs_sliding'), variant
            flag_col = "CaseFlag_" + CASE_FLAG_INPUT
            df[flag_col] = 0.0
            for _fn, _idx in df.groupby("FileName").groups.items():
                _inp = df.loc[_idx, "Input Temperature (C)"].values
                if CASE_FLAG_INPUT == "case":
                    df.loc[_idx, flag_col] = float(input_case_flag_np(_inp))
                elif CASE_FLAG_INPUT == "phase":
                    df.loc[_idx, flag_col] = input_phase_flag_np(_inp)
            feat_cols.append(flag_col)

        self.X = []
        self.Y = []
        self.time_values = []
        self.init_conditions = []
        self.window_obs = []
        self.full_time = []
        self.full_t_outer = []
        self.full_t_inner = []
        self.full_t_avg = []
        self.full_input_temp = []
        self.inp_last = []
        self.physics_targets = []

        # The model's target depends on whether this is a forward variant
        # (predict the 3 media temperatures) or an inverse variant (predict
        # the driving Input_T + the two outer/inner temperatures, taking
        # T_avg as input instead of output).
        if is_inverse(variant):
            output_cols = ["Input Temperature (C)", "T_inner (C)", "T_outer (C)"]
        else:
            output_cols = ["T_inner (C)", "T_outer (C)", "T_avg (C)"]

        for _, group in df.groupby("FileName"):
            X_seq = group[feat_cols].values[:-1]
            Y_seq = group[output_cols].values[1:]
            time_vals = group["Time (s)"].values[1:]

            init_cond = group[["T_outer (C)", "T_inner (C)", "T_avg (C)",
                               "Input Temperature (C)"]].values[0]

            # Window of first W absolute observations (T_outer, T_inner, T_avg, Input_T).
            # Pads by repeating the last available row if the case is shorter than W.
            full_obs = group[["T_outer (C)", "T_inner (C)", "T_avg (C)",
                              "Input Temperature (C)"]].values
            W = window_size
            if variant == 'forward_direct' and INIT_ENCODER == 'window':
                # G7(b): only the deployment-available initial observation;
                # future measured media temperatures are never encoder inputs.
                window_obs_arr = np.tile(full_obs[:1], (W, 1))
            elif full_obs.shape[0] >= W:
                window_obs_arr = full_obs[:W]
            else:
                pad = np.tile(full_obs[-1:], (W - full_obs.shape[0], 1))
                window_obs_arr = np.concatenate([full_obs, pad], axis=0)

            self.X.append(X_seq)
            self.Y.append(Y_seq)
            self.time_values.append(time_vals)
            self.init_conditions.append(init_cond)
            self.window_obs.append(window_obs_arr)
            self.full_time.append(group["Time (s)"].values)
            self.full_t_outer.append(group["T_outer (C)"].values)
            self.full_t_inner.append(group["T_inner (C)"].values)
            self.full_t_avg.append(group["T_avg (C)"].values)
            self.full_input_temp.append(group["Input Temperature (C)"].values)
            # H9: the true scaled Input_T(N). X holds rows 0..N-1, so without
            # this the lead-1 anchor had to tail-hold Input_T(N-1) at step N.
            self.inp_last.append(group["Input Temperature (C)"].values[-1])
            if PHYS_MODE:
                self.physics_targets.append(group[['_phys_outer', '_phys_avg']].values[1:])

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.Y = torch.tensor(np.array(self.Y), dtype=torch.float32)
        self.time_values = np.array(self.time_values)
        self.init_conditions = torch.tensor(np.array(self.init_conditions),
                                            dtype=torch.float32)
        self.window_obs = torch.tensor(np.array(self.window_obs), dtype=torch.float32)
        self.full_time = np.array(self.full_time)
        self.full_t_outer = np.array(self.full_t_outer)
        self.full_t_inner = np.array(self.full_t_inner)
        self.full_t_avg = np.array(self.full_t_avg)
        self.full_input_temp = np.array(self.full_input_temp)
        self.inp_last = torch.tensor(np.array(self.inp_last), dtype=torch.float32)
        # H2: per-file loss mask over the 3 output channels (T_inner, T_outer,
        # T_avg). Channel 2 is T_avg for the forward variants; the inverse
        # variants have no T_avg output and are never masked.
        mask_row = [1.0, 1.0, 0.0 if (self.tavg_masked and not is_inverse(variant)) else 1.0]
        self.loss_mask = torch.tensor([mask_row] * len(self.X), dtype=torch.float32)
        self.physics_targets = (torch.tensor(np.array(self.physics_targets), dtype=torch.float32)
                                if PHYS_MODE else None)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        item = (
            self.X[i], self.Y[i], self.time_values[i],
            self.init_conditions[i], self.window_obs[i],
            self.full_time[i],
            self.full_t_outer[i], self.full_t_inner[i],
            self.full_t_avg[i], self.full_input_temp[i],
            self.inp_last[i], self.loss_mask[i],
        )
        return item + (self.physics_targets[i],) if PHYS_MODE else item



# ============================================================
# 5b. Split json helpers (Round 10 / H1, 2026-10-11)
# ============================================================
def load_split_spec(path=None):
    """Read and validate a split json (default: the one config resolves).
    Legacy keys: val_split, val_per_type, split_seed, train_subdir, types,
    val, train. Round 10 keys (all optional): drop, tavg_mask, tag,
    split_name. Refuses a json whose lists overlap or whose drop list names
    a validation (or training) file -- a dropped file must be in neither."""
    path = path or split_file_path()
    spec = json.load(open(path, encoding="utf-8"))
    for k in ("val", "train"):
        if not isinstance(spec.get(k), list):
            raise RuntimeError("split file {}: key {!r} missing or not a list".format(path, k))
    val, train = set(spec["val"]), set(spec["train"])
    drop = set(spec.get("drop") or [])
    mask = set(spec.get("tavg_mask") or [])
    if val & train:
        raise RuntimeError("split file {}: {} files in both val and train, e.g. {}".format(
            path, len(val & train), sorted(val & train)[:3]))
    if drop & val:
        raise RuntimeError("split file {}: drop list names {} validation file(s): {} -- refusing".format(
            path, len(drop & val), sorted(drop & val)[:3]))
    if drop & train:
        raise RuntimeError("split file {}: drop list names {} training file(s): {} -- refusing".format(
            path, len(drop & train), sorted(drop & train)[:3]))
    if mask - (val | train):
        raise RuntimeError("split file {}: tavg_mask names {} file(s) in neither list, e.g. {}".format(
            path, len(mask - (val | train)), sorted(mask - (val | train))[:3]))
    return spec


def get_tavg_mask_sets(split=None):
    """(train_masked_keys, val_masked_keys) under the TAVG_MASK knob -- the
    single routing point for every per-file T_avg decision (loss mask, H3
    validation scoring, A4 / H5 / H6 statistics). Keys are split-json keys
    (relative to train_dir, posix); compare dataset ids via split_key().

    split: None (the json config resolves), a basename in splits/, a path,
    or an already-loaded spec dict.
      ""            -> (set(), set())
      "split"       -> (mask & train, mask & val)
      "random:<s>"  -> (|mask & train| files sampled from the json's training
                        list with random.Random(s), mask & val)
      "valonly"     -> (set(), mask & val)
    The draw uses the json's full training list (not the LOTO-filtered one),
    so one seed is one draw whatever TRAIN_EXCLUDE_TYPE says."""
    if not TAVG_MASK:
        return set(), set()
    if isinstance(split, dict):
        spec = split
    else:
        path = split or split_file_path()
        if not os.path.isfile(path) and os.path.basename(path) == path:
            path = os.path.join(BASE_DIR, "splits", path)
        spec = load_split_spec(path)
    train, val = set(spec["train"]), set(spec["val"])
    mask = set(spec.get("tavg_mask") or [])
    if not mask:
        raise RuntimeError("TAVG_MASK=%r but the split json carries no tavg_mask list" % TAVG_MASK)
    val_m = mask & val
    if TAVG_MASK == "split":
        train_m = mask & train
    elif TAVG_MASK == "valonly":
        train_m = set()
    else:                                   # random:<seed>
        n = len(mask & train)
        rng = random.Random(TAVG_MASK_SEED)
        train_m = set(rng.sample(sorted(train), n))
    return train_m, val_m


# ============================================================
# 5c. Algebraic T_avg relabel (Round 10 / A4, 2026-10-11)
# ============================================================
RELABEL_LABELS = ("inner-weighted", "borderline")
TAVG_COLS = ("T_avg (C)", "T_ave (C)")


def load_relabel_table(path=None):
    """key -> inner share s (0..1) for every file the label table marks
    inner-weighted or borderline. Key = basename(folder) + "/" + file, i.e.
    the split-json key relative to train_dir. Rows of the test folder
    (TEST_SUBDIR) are dropped: test files are never relabelled."""
    path = path or RELABEL_CSV_PATH
    if not path:
        raise RuntimeError("load_relabel_table: no RELABEL_CSV")
    tab = pd.read_csv(path)
    need = {"folder", "file", "inner_share_pct", "label"}
    if not need <= set(tab.columns):
        raise RuntimeError("relabel table %s lacks columns %s" % (path, sorted(need - set(tab.columns))))
    out = {}
    for _, r in tab.iterrows():
        folder = str(r["folder"]).replace("\\", "/").strip("/")
        parts = folder.split("/")
        if TEST_SUBDIR in parts:
            continue
        if str(r["label"]).strip() not in RELABEL_LABELS:
            continue
        s = float(r["inner_share_pct"]) / 100.0
        if not (0.0 <= s < 0.9):
            raise RuntimeError("relabel table %s: inner share %.3f out of range for %s/%s" % (
                path, s, folder, r["file"]))
        key = parts[-1] + "/" + str(r["file"])
        if key in out:
            raise RuntimeError("relabel table %s: duplicate key %s" % (path, key))
        out[key] = s
    return out


def relabel_tavg(dfs, table):
    """In place on (df, case_id) pairs: T_avg -> (T_avg - s*T_inner)/(1 - s)
    for the files in `table` (keys matched through split_key). Returns the
    sorted list of relabelled keys. Idempotence is NOT guaranteed -- call once."""
    done = []
    for df, cid in dfs:
        k = split_key(cid)
        if k not in table:
            continue
        s = table[k]
        col = next((c for c in TAVG_COLS if c in df.columns), None)
        if col is None:
            raise RuntimeError("relabel: %s has no T_avg column" % cid)
        ta = df[col].to_numpy(dtype=np.float64)
        ti = df["T_inner (C)"].to_numpy(dtype=np.float64)
        df[col] = (ta - s * ti) / (1.0 - s)
        done.append(k)
    return sorted(done)


def get_relabel_sets(split=None):
    """(train_keys, val_keys) the RELABEL_CSV table would relabel under the
    split json in use -- the routing point for provenance (run_config) and
    later statistics. Empty sets when RELABEL_CSV is off; under a non-json
    split the whole table counts as training."""
    if not RELABEL_CSV_PATH:
        return set(), set()
    table = set(load_relabel_table())
    if VAL_SPLIT != "stratified":
        return table, set()
    if isinstance(split, dict):
        spec = split
    else:
        path = split or split_file_path()
        if not os.path.isfile(path) and os.path.basename(path) == path:
            path = os.path.join(BASE_DIR, "splits", path)
        spec = load_split_spec(path)
    return table & set(spec["train"]), table & set(spec["val"])


def scaler_from_sidecar(d):
    """Round 10 / H4: rebuild the fitted MinMaxScaler from a constants sidecar
    (the whole dict, or its "scaler" block) without touching the training
    CSVs. The arrays round-trip through JSON exactly (repr of float64), so
    transform / inverse_transform are bit-identical to the fitted scaler's."""
    s = d["scaler"] if "scaler" in d else d
    cols = list(s["columns"])
    if cols != list(ThermalDataset.BASE_COLS):
        raise RuntimeError("sidecar scaler columns %r != %r" % (cols, ThermalDataset.BASE_COLS))
    sc = MinMaxScaler(feature_range=tuple(s.get("feature_range", (0, 1))))
    sc.n_features_in_ = len(cols)
    sc.feature_names_in_ = np.array(cols, dtype=object)
    sc.n_samples_seen_ = int(s.get("n_samples_seen_", 0))
    sc.data_min_ = np.asarray(s["data_min_"], dtype=np.float64)
    sc.data_max_ = np.asarray(s["data_max_"], dtype=np.float64)
    rng = s.get("data_range_")
    sc.data_range_ = (np.asarray(rng, dtype=np.float64) if rng is not None
                      else sc.data_max_ - sc.data_min_)
    sc.scale_ = np.asarray(s["scale_"], dtype=np.float64)
    sc.min_ = np.asarray(s["min_"], dtype=np.float64)
    return sc


def is_tavg_masked(case_id, masked_keys):
    """True if a dataset case_id (relative to the data root) is in a key set
    returned by get_tavg_mask_sets."""
    return split_key(case_id) in masked_keys


def split_key(case_id):
    """case_id (relative to the data root, posix) -> split-json key (relative
    to train_dir): strips the leading TRAIN_SUBDIR. Test-set ids are returned
    unchanged (they never match a split key)."""
    pre = TRAIN_SUBDIR + "/"
    return case_id[len(pre):] if case_id.startswith(pre) else case_id


# ============================================================
# 6. Data loading (resolves to the 2026-04-26 newly uploaded dataset)
# ============================================================
def _resolve_new_data_root(script_dir):
    """Return the first NEW_DATA_ROOT_CANDIDATES path that exists, or None."""
    if DATA_ROOT:
        if not os.path.isdir(DATA_ROOT):
            raise RuntimeError("DATA_ROOT does not exist: %s" % DATA_ROOT)
        return os.path.abspath(DATA_ROOT)
    for rel in NEW_DATA_ROOT_CANDIDATES:
        p = os.path.normpath(os.path.join(script_dir, rel))
        if os.path.isdir(p):
            return p
    if NEW_DATA_ROOT_ABSOLUTE and os.path.isdir(NEW_DATA_ROOT_ABSOLUTE):
        return NEW_DATA_ROOT_ABSOLUTE
    return None


def _resolve_legacy_data_root(script_dir):
    """Old layout: raw/ML-model-for-thermal-predictions/data/."""
    project_root = os.path.dirname(script_dir)
    candidates = [
        os.path.join(project_root, "data"),
        os.path.join(script_dir, "..", "..", "data"),
        os.path.join(script_dir, "data"),
        "data",
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    return None


def load_all_data(include_test=None):
    if include_test is None:
        include_test = not SEAL_TEST
    if include_test:
        from paper_report import assert_test_access
        assert_test_access()
    script_dir = BASE_DIR

    new_root = _resolve_new_data_root(script_dir)
    data_root = None
    if new_root is not None:
        # Two layouts are supported (2026-08-06): the repo-local flat layout
        #     <root>/<TRAIN_SUBDIR>            <root>/<TEST_SUBDIR>
        # and the older nested one
        #     <root>/training_data/<TRAIN>     <root>/tests/<TEST>
        # Flat is tried first since that is how AI-TES/data/ is organised.
        flat_train = os.path.join(new_root, TRAIN_SUBDIR)
        flat_test = os.path.join(new_root, TEST_SUBDIR)
        if os.path.isdir(flat_train) and os.path.isdir(flat_test):
            train_dir, test_dir = flat_train, flat_test
        else:
            train_dir = os.path.join(new_root, "training_data", TRAIN_SUBDIR)
            test_dir = os.path.join(new_root, "tests", TEST_SUBDIR)
        if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
            print(f"WARNING: new dataset root resolved to:\n    {new_root}")
            print(f"  but TRAIN_SUBDIR='{TRAIN_SUBDIR}' or TEST_SUBDIR='{TEST_SUBDIR}'")
            print(f"  do not exist there. Falling back to legacy layout.")
            new_root = None
        else:
            data_root = new_root

    if new_root is not None:
        data_source = "NEW (2026-04-26 upload)"
    else:
        legacy_root = _resolve_legacy_data_root(script_dir)
        if legacy_root is None:
            print("ERROR: neither new nor legacy data directory found.")
            return None, None, None, None
        train_dir = os.path.join(legacy_root, "data_in_10s")
        test_dir = os.path.join(legacy_root, "test_in_10s")
        data_root = legacy_root
        data_source = "LEGACY (old data_in_10s/test_in_10s)"

    print(f"Data source: {data_source}")
    print(f"  train_dir = {train_dir}")
    print(f"  test_dir  = {test_dir}")

    raw_train_paths = sorted(glob.glob(os.path.join(train_dir, "**", "*.csv"),
                                       recursive=True))
    raw_test_paths = (sorted(glob.glob(os.path.join(test_dir, "**", "*.csv"),
                                      recursive=True)) if include_test else [])
    print(f"  CSVs in train_dir : {len(raw_train_paths)}")
    print(f"  CSVs in test_dir  : {len(raw_test_paths)}")
    if not raw_train_paths:
        print(f"No training files found in {train_dir}")
        return None, None, None, None

    # ------------------------------------------------------------
    # Decide on the split
    # ------------------------------------------------------------
    if MANUAL_SPLIT_ENABLED:
        # Strategy: keep ALL files in tests/<TEST_SUBDIR>/ as the core test
        # set, then top up to OUTER_TEST_FRAC of the pooled total by sampling
        # additional cases from training_data/<TRAIN_SUBDIR>/. The remaining
        # training_data files form the train+val pool, then 95/5 inner split.
        n_total = len(raw_train_paths) + len(raw_test_paths)
        n_test_target = int(round(n_total * OUTER_TEST_FRAC))
        n_supplement = max(0, n_test_target - len(raw_test_paths))

        if n_supplement > len(raw_train_paths):
            raise RuntimeError(
                f"Cannot build a {n_test_target}-file test set: "
                f"{len(raw_test_paths)} core + only {len(raw_train_paths)} "
                f"available in train_dir."
            )

        rng = random.Random(SPLIT_SEED)
        train_pool_shuffled = list(raw_train_paths)
        rng.shuffle(train_pool_shuffled)

        test_supplement = train_pool_shuffled[:n_supplement]
        trainval_pool = train_pool_shuffled[n_supplement:]

        # Sort each output list for stable ordering (independent of shuffle).
        test_paths = sorted(list(raw_test_paths) + test_supplement)

        # Inner 95/5 split (re-shuffle the surviving train+val pool first
        # so val isn't biased toward early-glob-order files).
        rng.shuffle(trainval_pool)
        n_val = int(round(len(trainval_pool) * INNER_VAL_FRAC))
        val_paths = sorted(trainval_pool[:n_val])
        actual_train_paths = sorted(trainval_pool[n_val:])

        n_train = len(actual_train_paths)
        n_val_actual = len(val_paths)
        n_test_actual = len(test_paths)
        n_trainval = n_train + n_val_actual

        print(f"\nMANUAL split (40 core + supplement to "
              f"{OUTER_TEST_FRAC * 100:.0f}%, seed={SPLIT_SEED}):")
        print(f"  pooled total          : {n_total}")
        print(f"  test     ({n_test_actual:>3}) = "
              f"{n_test_actual / n_total * 100:5.1f}%   "
              f"(target {OUTER_TEST_FRAC * 100:.0f}%)")
        print(f"      core      (tests/{TEST_SUBDIR})        : "
              f"{len(raw_test_paths)}")
        print(f"      supplement (from training_data/{TRAIN_SUBDIR}): "
              f"{len(test_supplement)}")
        print(f"  train+val({n_trainval:>3}) = "
              f"{n_trainval / n_total * 100:5.1f}%   "
              f"(target {(1 - OUTER_TEST_FRAC) * 100:.0f}%)")
        print(f"      val   ({n_val_actual:>3}) = "
              f"{n_val_actual / n_trainval * 100:5.1f}% of train+val   "
              f"(target {INNER_VAL_FRAC * 100:.0f}%)")
        print(f"      train ({n_train:>3}) = "
              f"{n_train / n_trainval * 100:5.1f}% of train+val   "
              f"(target {(1 - INNER_VAL_FRAC) * 100:.0f}%)")
    elif VAL_SPLIT == "stratified":
        # Round 9 (2026-09-17): VAL_PER_TYPE validation runs from EACH
        # case-type folder (the immediate subfolder of train_dir), drawn
        # once with SPLIT_SEED and frozen in splits/<name>.json; test_dir is
        # taken whole. Paths in the file are relative to train_dir, posix.
        split_dir = os.path.join(BASE_DIR, "splits")
        split_name = split_file_name()          # legacy formula, or SPLIT_NAME (H1)
        split_path = split_file_path()
        rel = lambda p: os.path.relpath(p, train_dir).replace(os.sep, "/")
        # Round 10 / H1: a committed json may carry a `drop` list. It is
        # applied FIRST -- before the LOTO filter, by_type, the train-list
        # consistency check and EXCLUDE_BOTH_PHASE -- so a dropped file is
        # invisible to every later stage (and never reaches the test set).
        # Legacy jsons without `drop` behave exactly as before.
        spec, dropped = None, []
        if os.path.isfile(split_path):
            spec = load_split_spec(split_path)
            drop_rel = set(spec.get("drop") or [])
            if drop_rel:
                not_on_disk = sorted(drop_rel - {rel(p) for p in raw_train_paths})
                if not_on_disk:
                    print("  NOTE: {} drop-list entries are not in train_dir anyway, e.g. {}".format(
                        len(not_on_disk), not_on_disk[:2]))
                dropped = sorted(p for p in raw_train_paths if rel(p) in drop_rel)
                raw_train_paths = [p for p in raw_train_paths if rel(p) not in drop_rel]
        elif SPLIT_NAME:
            raise RuntimeError("SPLIT_NAME={!r}: {} does not exist".format(SPLIT_NAME, split_path))
        # Leave-one-type-out: drop the held-out type from train + val before
        # anything else; it joins the test set below. The split file's lists
        # are filtered by the same prefix so the consistency check still
        # holds for the remaining nine types.
        excluded, excl_prefix = [], None
        if TRAIN_EXCLUDE_TYPE:
            excl_prefix = "Case {} (".format(TRAIN_EXCLUDE_TYPE)
            excluded = sorted(p for p in raw_train_paths if rel(p).startswith(excl_prefix))
            folders = sorted({rel(p).split("/")[0] for p in excluded})
            if len(folders) != 1:
                raise RuntimeError("TRAIN_EXCLUDE_TYPE={!r} matches {} case-type folders: {}".format(
                    TRAIN_EXCLUDE_TYPE, len(folders), folders))
            raw_train_paths = [p for p in raw_train_paths if p not in set(excluded)]
            if not os.path.isfile(split_path):
                raise RuntimeError("TRAIN_EXCLUDE_TYPE needs the committed split file {}".format(split_path))
        by_type = {}
        for p in raw_train_paths:
            by_type.setdefault(rel(p).split("/")[0], []).append(p)
        if spec is not None:
            if excl_prefix:
                spec = dict(spec, val=[v for v in spec["val"] if not v.startswith(excl_prefix)],
                            train=[t for t in spec["train"] if not t.startswith(excl_prefix)])
            val_rel = set(spec["val"])
            missing = sorted(val_rel - {rel(p) for p in raw_train_paths})
            if missing:
                raise RuntimeError("split file {} names {} files not in train_dir, "
                                   "e.g. {}".format(split_path, len(missing), missing[:3]))
            # The frozen TRAIN list must match the disk too: a file added to
            # or removed from train_dir since the draw would silently change
            # the training set of every run.
            on_disk = sorted(rel(p) for p in raw_train_paths if rel(p) not in val_rel)
            if on_disk != sorted(spec["train"]):
                extra = sorted(set(on_disk) - set(spec["train"]))
                gone = sorted(set(spec["train"]) - set(on_disk))
                raise RuntimeError("split file {}: training list differs from train_dir "
                                   "({} extra on disk e.g. {}; {} missing e.g. {})".format(
                                       split_path, len(extra), extra[:2], len(gone), gone[:2]))
            src = "read"
        else:
            rng = random.Random(SPLIT_SEED)
            val_rel = set()
            for t in sorted(by_type):
                files = sorted(rel(p) for p in by_type[t])
                if len(files) < VAL_PER_TYPE + 5:
                    raise RuntimeError("case type {!r} has only {} runs".format(t, len(files)))
                val_rel.update(rng.sample(files, VAL_PER_TYPE))
            os.makedirs(split_dir, exist_ok=True)
            spec = {"val_split": "stratified", "val_per_type": VAL_PER_TYPE,
                    "split_seed": SPLIT_SEED, "train_subdir": TRAIN_SUBDIR,
                    "types": sorted(by_type), "val": sorted(val_rel),
                    "train": sorted(rel(p) for p in raw_train_paths if rel(p) not in val_rel)}
            json.dump(spec, open(split_path, "w", encoding="utf-8"), indent=1)
            src = "drawn and written"
        val_paths = sorted(p for p in raw_train_paths if rel(p) in val_rel)
        actual_train_paths = sorted(p for p in raw_train_paths if rel(p) not in val_rel)
        test_paths = list(raw_test_paths) + excluded        # LOTO: held-out type scored too
        assert len(val_paths) == len(val_rel)
        if not spec.get('heldout_family'):
            assert len(val_rel) == VAL_PER_TYPE * len(by_type)
        assert not (set(val_paths) & set(actual_train_paths))
        assert not (set(test_paths) & (set(val_paths) | set(actual_train_paths)))

        print("\nSTRATIFIED split ({} per case type, seed {}; {} {}; tag {}):".format(
            VAL_PER_TYPE, SPLIT_SEED, split_name, src, SPLIT_TAG))
        if dropped:
            print("  DROP LIST: {} runs removed from the training pool before the split: {}".format(
                len(dropped), ", ".join(rel(p) for p in dropped)))
        if excluded:
            print("  LEAVE-ONE-TYPE-OUT: {!r} ({} runs) removed from train and val, appended to test".format(
                rel(excluded[0]).split("/")[0], len(excluded)))
        for t in sorted(by_type):
            n_v = sum(1 for p in by_type[t] if rel(p) in val_rel)
            print("  {:<48} train {:>3}  val {}".format(t[:48], len(by_type[t]) - n_v, n_v))
        print("  train: {}, val: {}, test: {}".format(
            len(actual_train_paths), len(val_paths), len(test_paths)))
    else:
        # Folder-as-given (legacy behaviour). 95/5 split inside train_dir;
        # test_dir is taken whole. NOTE (2026-09-16 audit): sorted, the first
        # 5% are all linear-charging runs; see VAL_SPLIT in config.py.
        val_split = int(0.05 * len(raw_train_paths))
        val_paths = raw_train_paths[:val_split]
        actual_train_paths = raw_train_paths[val_split:]
        test_paths = raw_test_paths

        print(f"\nFOLDER-AS-GIVEN split (legacy):")
        print(f"  train: {len(actual_train_paths)}, val: {len(val_paths)}, "
              f"test: {len(test_paths)}")

    # Build a unique, traceable case_id per file. Pure basenames collide
    # (e.g. 'Case 1.csv' exists under multiple case-type folders), which would
    # corrupt the per-case test summary and overwrite plot files. Use the
    # path relative to the resolved data root, posix-style.
    def _case_id(path):
        try:
            rel = os.path.relpath(path, data_root)
        except ValueError:
            rel = os.path.basename(path)
        return rel.replace(os.sep, "/")

    print("\nLoading training data...")
    train_dfs = [(pd.read_csv(f), _case_id(f)) for f in actual_train_paths]
    print("Loading validation data...")
    val_dfs = [(pd.read_csv(f), _case_id(f)) for f in val_paths]
    print("Loading test data...")
    test_dfs = ([(pd.read_csv(f), _case_id(f)) for f in test_paths] if include_test else [])

    if EXCLUDE_BOTH_PHASE:
        # Round 7 sanity protocol (Arnold, 2026-09-05): remove every run the
        # case detector fires on from train, val AND test, then rerun. The
        # scaler below is fitted on the reduced train set, as it would be
        # had those runs never existed.
        def _fires(df):
            col = ("Input Temperature (C)" if "Input Temperature (C)" in df.columns
                   else "T_inner (C)")           # same fallback as ThermalDataset
            return input_case_flag_np(df[col].values)
        _n0 = (len(train_dfs), len(val_dfs), len(test_dfs))
        train_dfs = [(df, fid) for df, fid in train_dfs if not _fires(df)]
        val_dfs = [(df, fid) for df, fid in val_dfs if not _fires(df)]
        test_dfs = [(df, fid) for df, fid in test_dfs if not _fires(df)]
        print("EXCLUDE_BOTH_PHASE: train {}->{}  val {}->{}  test {}->{}".format(
            _n0[0], len(train_dfs), _n0[1], len(val_dfs), _n0[2], len(test_dfs)))
        if not train_dfs or (include_test and not test_dfs):
            raise RuntimeError("EXCLUDE_BOTH_PHASE removed every case -- "
                               "check the detector / data root")

    if RELABEL_CSV_PATH:
        # Round 10 / A4: algebraic relabel of the inner-weighted / borderline
        # files, BEFORE the scaler fit and before any mask logic (the mask
        # sets are resolved by key later, independently of the values).
        # Training and validation only -- test_dfs are never touched.
        _table = load_relabel_table()
        _rl_tr = relabel_tavg(train_dfs, _table)
        _rl_va = relabel_tavg(val_dfs, _table)
        print("RELABEL_CSV={}: T_avg -> (T_avg - s*T_inner)/(1 - s) on {} training and {} validation "
              "files ({} listed; test files untouched)".format(
                  os.path.basename(RELABEL_CSV_PATH), len(_rl_tr), len(_rl_va), len(_table)))

    # Sanity: case_ids must be globally unique across train/val/test
    all_ids = [fid for _, fid in train_dfs] + [fid for _, fid in val_dfs] \
              + [fid for _, fid in test_dfs]
    if len(set(all_ids)) != len(all_ids):
        from collections import Counter
        dups = [fid for fid, c in Counter(all_ids).items() if c > 1]
        raise RuntimeError(f"Duplicate case_ids after split: {dups[:5]}...")

    print("Fitting scaler on training data...")
    fit_records = list(train_dfs)
    frozen_split = None
    if VAL_SPLIT == 'stratified' and spec.get('frozen_train'):
        frozen_keys = sorted(spec['frozen_train'])
        if len(frozen_keys) != len(set(frozen_keys)) or set(frozen_keys) - (set(spec['train']) | set(spec['val'])):
            raise RuntimeError('invalid frozen_train universe')
        fit_records = [(pd.read_csv(os.path.join(train_dir, key.replace('/', os.sep))),
                        _case_id(os.path.join(train_dir, key.replace('/', os.sep)))) for key in frozen_keys]
        if RELABEL_CSV_PATH:
            relabel_tavg(fit_records, load_relabel_table())
        frozen_split = spec.get('frozen_constants_split')
        print('F1 constants freeze: %d original training files; held-out labels used ONLY for scaler/anchor constants as registered' % len(fit_records))
    cleaned = []
    for df, _ in fit_records:
        d = df.copy()
        d.rename(columns={"T_ave (C)": "T_avg (C)"}, inplace=True)
        if "Input Temperature (C)" not in d.columns:
            d["Input Temperature (C)"] = d["T_inner (C)"]
        # Drop the Convection column if it leaked in (Convection/ files have 6 cols)
        if "Convection (W/m2C)" in d.columns:
            d = d.drop(columns=["Convection (W/m2C)"])
        cleaned.append(d[ThermalDataset.BASE_COLS])
    scaler = MinMaxScaler()
    scaler.fit(pd.concat(cleaned)[ThermalDataset.BASE_COLS])
    if CONSTANTS_FROM:
        with open(CONSTANTS_FROM, encoding='utf-8') as f:
            scaler = scaler_from_sidecar(json.load(f))
    scaler._round10_fit_records = fit_records
    scaler._round10_frozen_split = frozen_split
    if SYNTHETIC_REST_FILES:
        n = len(train_dfs[0][0])
        times = train_dfs[0][0]['Time (s)'].to_numpy(float)
        for i, temperature in enumerate(np.linspace(50, 600, SYNTHETIC_REST_FILES)):
            frame = pd.DataFrame({'Time (s)': times.copy()})
            for col in ThermalDataset.BASE_COLS[1:]:
                frame[col] = float(temperature)
            train_dfs.append((frame, 'synthetic_rest/%03d_T0_%g.csv' % (i, temperature)))
        print('G7(d): appended %d train-only constant initial-temperature files (%d rows); scaler/constants fit precedes augmentation' % (SYNTHETIC_REST_FILES, n))

    return train_dfs, val_dfs, test_dfs, scaler


