"""Variant-aware Dataset and the data loader / manual split."""
import os
import glob
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

from .config import *


# ============================================================
# 2. Dataset (variant-aware)
# ============================================================
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
    """

    BASE_COLS = ['Time (s)', 'T_outer (C)', 'T_inner (C)', 'T_avg (C)',
                 'Input Temperature (C)']

    def __init__(self, csv_or_df, variant, scaler=None, file_name=None,
                 window_size=WINDOW_SIZE):
        if isinstance(csv_or_df, str):
            self.file_name = os.path.basename(csv_or_df)
            df = pd.read_csv(csv_or_df)
        else:
            self.file_name = file_name or "unknown"
            df = csv_or_df.copy()

        self.variant = variant
        self.window_size = window_size

        df.rename(columns={"T_ave (C)": "T_avg (C)"}, inplace=True)
        if "Input Temperature (C)" not in df.columns:
            df["Input Temperature (C)"] = df["T_inner (C)"]
        df["FileName"] = self.file_name

        if scaler is None:
            self.scaler = MinMaxScaler()
            self.scaler.fit(df[self.BASE_COLS])
        else:
            self.scaler = scaler

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
            # INPUT_LOOKAHEAD (2026-08-06): idx 2..1+k carry the inlet's next
            # k values (legal: the full curve is a given boundary condition).
            # Built AFTER scaling, so the lead columns share Input_T's scale;
            # the tail holds the last value.
            feat_cols = ["Time (s)", "Input Temperature (C)"]
            for _i in range(1, INPUT_LOOKAHEAD + 1):
                lead = f"InputT_lead{_i}"
                df[lead] = df["Input Temperature (C)"].shift(-_i).ffill()
                feat_cols.append(lead)
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
            if full_obs.shape[0] >= W:
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

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return (
            self.X[i], self.Y[i], self.time_values[i],
            self.init_conditions[i], self.window_obs[i],
            self.full_time[i],
            self.full_t_outer[i], self.full_t_inner[i],
            self.full_t_avg[i], self.full_input_temp[i],
        )



# ============================================================
# 6. Data loading (resolves to the 2026-04-26 newly uploaded dataset)
# ============================================================
def _resolve_new_data_root(script_dir):
    """Return the first NEW_DATA_ROOT_CANDIDATES path that exists, or None."""
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


def load_all_data():
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
    raw_test_paths = sorted(glob.glob(os.path.join(test_dir, "**", "*.csv"),
                                      recursive=True))
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
    else:
        # Folder-as-given (legacy behaviour). 95/5 split inside train_dir;
        # test_dir is taken whole.
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
    test_dfs = [(pd.read_csv(f), _case_id(f)) for f in test_paths]

    # Sanity: case_ids must be globally unique across train/val/test
    all_ids = [fid for _, fid in train_dfs] + [fid for _, fid in val_dfs] \
              + [fid for _, fid in test_dfs]
    if len(set(all_ids)) != len(all_ids):
        from collections import Counter
        dups = [fid for fid, c in Counter(all_ids).items() if c > 1]
        raise RuntimeError(f"Duplicate case_ids after split: {dups[:5]}...")

    print("Fitting scaler on training data...")
    cleaned = []
    for df, _ in train_dfs:
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

    return train_dfs, val_dfs, test_dfs, scaler


