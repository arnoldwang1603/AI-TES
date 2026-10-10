# =============================================================================
# dataset.py — ThermalDataset for PhysTFv3
#
# Identical to v1 — data format and scaler are unchanged.
# tin_phys and init_phys are still returned (for compatibility) but
# ThermalPredictor no longer uses them in its forward pass.
# The training loop only passes tin_scaled, dtin_scaled, init_scaled.
# =============================================================================

import glob
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

from config import (
    SCALE_COLS, IDX_TIME, IDX_TOUTER, IDX_TINNER, IDX_TAVG, IDX_TIN,
    TRAIN_DATA_DIR, TEST_DATA_DIR, VAL_SPLIT,
)


class ThermalDataset(Dataset):
    """
    Loads a single CSV (or pre-loaded DataFrame) at 10s timestep.

    Per sample (one 4-hour case):
        tin_scaled    [T]    — T_in sequence, MinMax-scaled
        dtin_scaled   [T]    — ΔT_in (scaled diff)
        init_scaled   [4]    — [T_outer, T_inner, T_avg, T_in] at t=0, scaled
        tin_phys      [T]    — T_in in °C  (unused by ThermalPredictor; kept for compatibility)
        init_phys     [3]    — [T_inner, T_outer, T_avg] at t=0 in °C  (unused)
        y_phys        [T, 3] — [T_inner, T_outer, T_avg] in °C  → loss targets
    """

    def __init__(self, csv_or_df, scaler: MinMaxScaler, file_name: str = None):
        if isinstance(csv_or_df, str):
            self.file_name = os.path.basename(csv_or_df)
            df = pd.read_csv(csv_or_df)
        else:
            self.file_name = file_name or "unknown"
            df = csv_or_df.copy()

        df["FileName"] = self.file_name
        df.rename(columns={"T_ave (C)": "T_avg (C)"}, inplace=True)

        self._sc_min   = scaler.data_min_
        self._sc_range = scaler.data_max_ - scaler.data_min_

        (self.tin_scaled, self.dtin_scaled, self.init_scaled,
         self.tin_phys, self.init_phys, self.y_phys) = [], [], [], [], [], []
        self.file_names = []

        for _, group in df.groupby("FileName"):
            scaled = scaler.transform(group[SCALE_COLS])
            T      = len(scaled) - 1

            tin_sc  = scaled[:-1, IDX_TIN]
            dtin_sc = np.zeros(T, dtype=np.float32)
            dtin_sc[1:] = tin_sc[1:] - tin_sc[:-1]

            init_sc = scaled[0, [IDX_TOUTER, IDX_TINNER, IDX_TAVG, IDX_TIN]]

            tin_ph  = (scaled[:-1, IDX_TIN]
                       * self._sc_range[IDX_TIN] + self._sc_min[IDX_TIN])

            init_ph = np.array([
                scaled[0, IDX_TINNER] * self._sc_range[IDX_TINNER] + self._sc_min[IDX_TINNER],
                scaled[0, IDX_TOUTER] * self._sc_range[IDX_TOUTER] + self._sc_min[IDX_TOUTER],
                scaled[0, IDX_TAVG]   * self._sc_range[IDX_TAVG]   + self._sc_min[IDX_TAVG],
            ], dtype=np.float32)

            y_ph = np.column_stack([
                scaled[1:, IDX_TINNER] * self._sc_range[IDX_TINNER] + self._sc_min[IDX_TINNER],
                scaled[1:, IDX_TOUTER] * self._sc_range[IDX_TOUTER] + self._sc_min[IDX_TOUTER],
                scaled[1:, IDX_TAVG]   * self._sc_range[IDX_TAVG]   + self._sc_min[IDX_TAVG],
            ])

            self.tin_scaled.append(tin_sc.astype(np.float32))
            self.dtin_scaled.append(dtin_sc)
            self.init_scaled.append(init_sc.astype(np.float32))
            self.tin_phys.append(tin_ph.astype(np.float32))
            self.init_phys.append(init_ph)
            self.y_phys.append(y_ph.astype(np.float32))
            self.file_names.append(self.file_name)

        self.tin_scaled  = torch.tensor(np.array(self.tin_scaled),  dtype=torch.float32)
        self.dtin_scaled = torch.tensor(np.array(self.dtin_scaled), dtype=torch.float32)
        self.init_scaled = torch.tensor(np.array(self.init_scaled), dtype=torch.float32)
        self.tin_phys    = torch.tensor(np.array(self.tin_phys),    dtype=torch.float32)
        self.init_phys   = torch.tensor(np.array(self.init_phys),   dtype=torch.float32)
        self.y_phys      = torch.tensor(np.array(self.y_phys),      dtype=torch.float32)

    def __len__(self):
        return len(self.tin_scaled)

    def __getitem__(self, idx):
        return (
            self.tin_scaled[idx],
            self.dtin_scaled[idx],
            self.init_scaled[idx],
            self.tin_phys[idx],
            self.init_phys[idx],
            self.y_phys[idx],
        )


def load_data(val_split: float = VAL_SPLIT):
    train_dir = str(TRAIN_DATA_DIR)
    test_dir  = str(TEST_DATA_DIR)

    train_paths = sorted(glob.glob(os.path.join(train_dir, "**", "*.csv"), recursive=True))
    test_paths  = sorted(glob.glob(os.path.join(test_dir, "*.csv")))

    if not train_paths:
        raise FileNotFoundError(f"No training CSVs found under {train_dir}")
    if not test_paths:
        print(f"  Warning: no test files in {test_dir}")

    rng         = np.random.default_rng(42)
    train_paths = list(rng.permutation(train_paths))
    n_val       = max(1, int(val_split * len(train_paths)))
    val_paths   = train_paths[:n_val]
    train_paths = train_paths[n_val:]

    print(f"  Train: {len(train_paths)} files  |  "
          f"Val: {n_val} files  |  Test: {len(test_paths)} files")

    def _load(paths):
        return [(pd.read_csv(p).rename(columns={"T_ave (C)": "T_avg (C)"}),
                 os.path.basename(p)) for p in paths]

    train_dfs = _load(train_paths)
    val_dfs   = _load(val_paths)
    test_dfs  = _load(test_paths)

    all_train = pd.concat([df for df, _ in train_dfs])
    scaler    = MinMaxScaler()
    scaler.fit(all_train[SCALE_COLS])

    return train_dfs, val_dfs, test_dfs, scaler
