import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import random

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True  # 加速卷积/RNN，输入尺寸固定时更明显

# Teacher forcing = 0：训练时始终用模型上一步预测作为下一步输入（不用真实值）
TEACHER_FORCING_RATIO = 0
# rollout_prob = 1 - teacher_forcing：1 表示不用 teacher forcing
ROLLOUT_PROB = 1.0 - TEACHER_FORCING_RATIO

# 4 种 dropout，每种用 3 种初始学习率对比
DROPOUT_LIST = [0, 0.1, 0.2, 0.3]
INITIAL_LR_LIST = [0.01, 0.003, 0.001]
LR_PLOT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']

# 训练速度：增大 batch_size 可明显加速（GPU 利用率更高）；显存不足再改小
BATCH_SIZE = 32
# DataLoader 多进程预取；Windows 下若报错可改为 0
NUM_WORKERS = 4


# === Dataset class ===
class ThermalDataset(Dataset):
    def __init__(self, csv_file_or_df, scaler=None, file_name=None):
        if isinstance(csv_file_or_df, str):
            self.file_name = os.path.basename(csv_file_or_df)
            df = pd.read_csv(csv_file_or_df)
        else:
            self.file_name = file_name if file_name else "unknown"
            df = csv_file_or_df.copy()

        df["FileName"] = self.file_name
        columns_for_scaling = ['Time (s)',
                               'T_outer (C)', 'T_inner (C)', 'T_avg (C)', 'Input Temperature (C)'
                               ]
        rename_map = {"T_ave (C)": "T_avg (C)"}
        df.rename(columns=rename_map, inplace=True)

        if scaler is None:
            self.scaler = MinMaxScaler()
            self.scaler.fit(df[columns_for_scaling])
        else:
            self.scaler = scaler

        df[columns_for_scaling] = self.scaler.transform(df[columns_for_scaling])

        grouped = df.groupby("FileName")
        self.X, self.Y, self.time_values = [], [], []
        self.full_time, self.full_t_min, self.full_t_max, self.full_t_ave = [], [], [], []
        # Store initial condition (t=0) for each sequence
        self.init_conditions = []

        for _, group in grouped:
            # Input: 4 raw features only (delta features removed)
            X_seq = group[["Time (s)",
                           "T_outer (C)", "Input Temperature (C)", "T_avg (C)"]].values[:-1]
            Y_seq = group[["T_inner (C)", "T_outer (C)", "T_avg (C)"]].values[1:]
            time_vals = group["Time (s)"].values[1:]

            # Capture t=0 initial condition: [T_outer, T_inner, T_avg, Input_T]
            init_cond = group[["T_outer (C)", "T_inner (C)", "T_avg (C)",
                                "Input Temperature (C)"]].values[0]

            self.X.append(X_seq)
            self.Y.append(Y_seq)
            self.time_values.append(time_vals)
            self.init_conditions.append(init_cond)
            self.full_time.append(group["Time (s)"].values)
            self.full_t_min.append(group["T_outer (C)"].values)
            self.full_t_max.append(group["T_inner (C)"].values)
            self.full_t_ave.append(group["T_avg (C)"].values)

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.Y = torch.tensor(np.array(self.Y), dtype=torch.float32)
        self.time_values = np.array(self.time_values)
        self.init_conditions = torch.tensor(np.array(self.init_conditions), dtype=torch.float32)
        self.full_time = np.array(self.full_time)
        self.full_t_min = np.array(self.full_t_min)
        self.full_t_max = np.array(self.full_t_max)
        self.full_t_ave = np.array(self.full_t_ave)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            self.X[idx], self.Y[idx], self.time_values[idx],
            self.init_conditions[idx],
            self.full_time[idx],
            self.full_t_min[idx], self.full_t_max[idx], self.full_t_ave[idx]
        )


# ============================================================
# InitStateEncoder for LSTM
# Maps t=0 observations → LSTM initial (h0, c0)
# LSTM needs both hidden state and cell state; encoder outputs both.
# ============================================================
class InitStateEncoder(nn.Module):
    """
    Encodes the known physical state at t=0 into LSTM initial hidden and cell states.
    Input:  x0 [batch, 4]  →  [T_outer, T_inner, T_avg, Input_T] at t=0
    Output: (h0, c0) each [num_layers, batch, hidden_size]
    """
    def __init__(self, obs_size=4, hidden_size=128, num_layers=5):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # Output both h0 and c0: 2 * num_layers * hidden_size
        self.encoder = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * num_layers * hidden_size)
        )

    def forward(self, x0):
        # x0: [batch, 4]
        out = self.encoder(x0)   # [batch, 2 * num_layers * hidden_size]
        half = out.size(-1) // 2
        h = out[:, :half].view(-1, self.num_layers, self.hidden_size)
        c = out[:, half:].view(-1, self.num_layers, self.hidden_size)
        # [num_layers, batch, hidden_size]
        h = h.permute(1, 0, 2).contiguous()
        c = c.permute(1, 0, 2).contiguous()
        return h, c


# === Model definition: LSTM ===
class ThermalLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, output_size=3,
                 num_layers=5, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        self.init_encoder = InitStateEncoder(
            obs_size=4, hidden_size=hidden_size, num_layers=num_layers
        )

    def init_hidden(self, batch_size, x0=None):
        """
        If x0 (initial condition) is provided, encode it into (h0, c0).
        Otherwise return zeros.
        """
        device = next(self.parameters()).device
        if x0 is not None:
            return self.init_encoder(x0.to(device))
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h0, c0)

    def forward(self, x, hidden):
        # hidden is (h, c) for LSTM
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden


# === Weighted loss ===
def weighted_loss(predictions, targets, weights=torch.tensor([1.0, 1.0, 1.0]),
                  time_weights=None):
    weights = weights.to(predictions.device)
    loss = torch.abs(predictions - targets) * weights
    if time_weights is not None:
        time_weights = time_weights.to(predictions.device)
        loss = loss * time_weights.unsqueeze(-1)
    return torch.mean(loss)


# === Load all data once ===
def load_all_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = script_dir

    # 多种可能的数据根目录
    possible_data_dirs = [
        os.path.join(project_root, "data"),
        os.path.join(project_root, "..", "data"),
        os.path.join(project_root, "..", "..", "data"),
        os.path.abspath("data"),
    ]

    data_dir = None
    for dir_path in possible_data_dirs:
        path_abs = os.path.abspath(dir_path)
        if os.path.isdir(path_abs):
            data_dir = path_abs
            break

    if not data_dir:
        print("Data directory not found. Tried:")
        for d in possible_data_dirs:
            print(f"  - {os.path.abspath(d)}")
        print("Please create a 'data' folder and put training CSV in data/data_in_10s/ and test CSV in data/test_in_10s/")
        return None, None, None, None, None

    # 训练/测试子目录：优先 data_in_10s + test_in_10s，再尝试 10s_with_burn_in 或 10s + test/test_in_10s
    train_dir = os.path.join(data_dir, "data_in_10s")
    test_dir = os.path.join(data_dir, "test_in_10s")
    if not os.path.isdir(train_dir):
        train_dir = os.path.join(data_dir, "10s_with_burn_in")
    if not os.path.isdir(train_dir):
        train_dir = os.path.join(data_dir, "10s")
    if not os.path.isdir(test_dir):
        test_dir = os.path.join(data_dir, "test", "test_in_10s")

    train_paths = sorted(glob.glob(os.path.join(train_dir, "**", "*.csv"), recursive=True))
    test_paths = sorted(glob.glob(os.path.join(test_dir, "**", "*.csv"), recursive=True))
    if not test_paths:
        test_paths = sorted(glob.glob(os.path.join(test_dir, "*.csv")))

    print(f"Data dir: {data_dir}")
    print(f"Train dir: {train_dir} -> {len(train_paths)} files")
    print(f"Test dir:  {test_dir} -> {len(test_paths)} files")

    if not train_paths:
        print("No training files found. Put CSV files in one of: data/data_in_10s/, data/10s_with_burn_in/, data/10s/")
        return None, None, None, None, None

    val_split = int(0.05 * len(train_paths))
    val_paths = train_paths[:val_split]
    actual_train_paths = train_paths[val_split:]

    print("Loading training data...")
    train_dfs = [(pd.read_csv(f), os.path.basename(f)) for f in actual_train_paths]
    print("Loading validation data...")
    val_dfs = [(pd.read_csv(f), os.path.basename(f)) for f in val_paths]
    print("Loading test data...")
    test_dfs = [(pd.read_csv(f), os.path.basename(f)) for f in test_paths]

    print("Fitting scaler...")
    scaler = MinMaxScaler()
    all_train_data = pd.concat([df for df, _ in train_dfs])
    scaler.fit(all_train_data[["Time (s)", "T_outer (C)", "T_inner (C)",
                                "T_avg (C)", "Input Temperature (C)"]])

    return train_dfs, val_dfs, test_dfs, scaler, test_paths


# === Training function ===
def train_model(train_dfs, val_dfs, test_dfs, scaler, test_paths,
                max_epochs=300, hidden_size=128, num_layers=5,
                dropout=0.3, rollout_prob=0.2, initial_lr=0.001):

    print("Creating datasets from pre-loaded data...")
    train_datasets = [ThermalDataset(df, scaler=scaler, file_name=fname)
                      for df, fname in train_dfs]
    val_datasets   = [ThermalDataset(df, scaler=scaler, file_name=fname)
                      for df, fname in val_dfs]
    test_datasets  = [ThermalDataset(df, scaler=scaler, file_name=fname)
                      for df, fname in test_dfs]

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    train_loader = DataLoader(
        ConcatDataset(train_datasets),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        ConcatDataset(val_datasets),
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=use_cuda,
    )
    if not use_cuda:
        print("Using CPU")

    model = ThermalLSTM(
        input_size=4, hidden_size=hidden_size,
        num_layers=num_layers, dropout=dropout,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50
    )

    best_val_loss = float('inf')
    early_stop_counter = 0
    patience = 300

    train_loss_history = []
    val_loss_history   = []
    lr_history         = []

    for epoch in range(max_epochs):
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            inputs, targets, _, init_conds, *_ = [
                b.to(device) if torch.is_tensor(b) else b for b in batch
            ]
            batch_size = inputs.shape[0]

            hidden = model.init_hidden(batch_size, x0=init_conds)

            optimizer.zero_grad()

            prev_t_outer = inputs[:, 0, 1]
            prev_t_avg   = inputs[:, 0, 3]

            preds = []
            current_hidden = hidden

            for t in range(inputs.size(1)):
                x_t = inputs[:, t, :].clone()

                if t > 0 and random.random() < rollout_prob:
                    x_t[:, 1] = prev_t_outer.detach()
                    x_t[:, 3] = prev_t_avg.detach()

                x_t = x_t.unsqueeze(1)
                out, current_hidden = model(x_t, current_hidden)
                pred_t = out[:, 0, :]

                preds.append(pred_t)
                prev_t_outer = pred_t[:, 1]
                prev_t_avg   = pred_t[:, 2]

            predictions = torch.stack(preds, dim=1)
            loss = weighted_loss(predictions, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets, _, init_conds, *_ = [
                    b.to(device) if torch.is_tensor(b) else b for b in batch
                ]
                batch_size = inputs.shape[0]

                hidden = model.init_hidden(batch_size, x0=init_conds)
                predictions, _ = model(inputs, hidden)
                loss = weighted_loss(predictions, targets)
                total_val_loss += loss.item()

        avg_val_loss = (total_val_loss / len(val_loader)
                        if len(val_loader) > 0 else float('inf'))
        val_loss_history.append(avg_val_loss)

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        lr_history.append(current_lr)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{max_epochs} | "
                  f"Train Loss: {avg_train_loss:.6f} | "
                  f"Val Loss: {avg_val_loss:.6f}")

        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return model, test_datasets, best_val_loss, train_loss_history, val_loss_history, lr_history


# === Test model ===
def test_model(model, test_datasets, plot_save_dir, return_summary=False):
    """
    逐步自回归预测，保存图像（LSTM_fixed 风格）、每个 case 的预测 CSV、可选返回汇总用列表。
    """
    if not model or not test_datasets:
        print("Model or test_datasets is empty")
        return ([] if return_summary else None)

    device = next(model.parameters()).device
    model.eval()
    os.makedirs(plot_save_dir, exist_ok=True)
    csv_save_dir = plot_save_dir.rstrip(os.sep) + "_csv"
    os.makedirs(csv_save_dir, exist_ok=True)
    print(f"Plots will be saved to: {plot_save_dir}")
    print(f"Prediction CSV will be saved to: {csv_save_dir}")

    summary_data = []

    for idx, dataset in enumerate(test_datasets):
        file_name = os.path.basename(dataset.file_name)
        x, _, _, init_cond, ft, ft_min, ft_max, ft_ave = dataset[0]
        x = x.to(device)
        init_cond = init_cond.unsqueeze(0).to(device)

        hidden = model.init_hidden(1, x0=init_cond)
        seq_len = x.shape[0]
        current_t_outer = x[0, 1]
        current_input_temp = x[0, 2]
        current_t_avg = x[0, 3]
        preds = []

        for t in range(seq_len):
            input_t = torch.tensor([[
                x[t, 0], current_t_outer, current_input_temp, current_t_avg,
            ]], dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                out, hidden = model(input_t, hidden)
                pred = out[0, 0].cpu().numpy()
                preds.append(pred)
                if t < seq_len - 1:
                    current_t_outer = pred[1]
                    current_input_temp = x[t + 1, 2].item()
                    current_t_avg = pred[2]

        pred_seq = np.array(preds)
        n_pred = len(pred_seq)

        dummy_actual = np.zeros((n_pred, 5))
        dummy_actual[:, 0] = ft[:n_pred]
        dummy_actual[:, 1] = ft_min[:n_pred]
        dummy_actual[:, 2] = ft_max[:n_pred]
        dummy_actual[:, 3] = ft_ave[:n_pred]
        dummy_actual[:, 4] = x[:, 2].cpu().numpy()[:n_pred]

        dummy_pred = np.zeros((n_pred, 5))
        dummy_pred[:, 0] = ft[:n_pred]
        dummy_pred[:, 1] = pred_seq[:, 1]
        dummy_pred[:, 2] = pred_seq[:, 0]
        dummy_pred[:, 3] = pred_seq[:, 2]
        dummy_pred[:, 4] = x[:, 2].cpu().numpy()[:n_pred]

        inv_pred = dataset.scaler.inverse_transform(dummy_pred)
        inv_actual = dataset.scaler.inverse_transform(dummy_actual)
        actual_temps = inv_actual[:, 1:4]
        pred_temps = inv_pred[:, 1:4]

        mae_outer = mean_absolute_error(actual_temps[:, 0], pred_temps[:, 0])
        mae_inner = mean_absolute_error(actual_temps[:, 1], pred_temps[:, 1])
        mae_avg = mean_absolute_error(actual_temps[:, 2], pred_temps[:, 2])
        r2_outer = r2_score(actual_temps[:, 0], pred_temps[:, 0])
        r2_inner = r2_score(actual_temps[:, 1], pred_temps[:, 1])
        r2_avg = r2_score(actual_temps[:, 2], pred_temps[:, 2])

        if return_summary:
            summary_data.append({
                "Case": file_name,
                "MAE_T_outer": mae_outer, "MAE_T_inner": mae_inner, "MAE_T_avg": mae_avg,
                "R2_T_outer": r2_outer, "R2_T_inner": r2_inner, "R2_T_avg": r2_avg,
            })

        plot_times = inv_actual[:, 0]
        csv_df = pd.DataFrame({
            "Time (s)": plot_times,
            "T_outer_Actual (C)": inv_actual[:, 1],
            "T_inner_Actual (C)": inv_actual[:, 2],
            "T_avg_Actual (C)": inv_actual[:, 3],
            "T_outer_Pred (C)": inv_pred[:, 1],
            "T_inner_Pred (C)": inv_pred[:, 2],
            "T_avg_Pred (C)": inv_pred[:, 3],
        })
        csv_name = os.path.splitext(file_name)[0] + "_predictions.csv"
        csv_path = os.path.join(csv_save_dir, csv_name)
        csv_df.to_csv(csv_path, index=False)

        plt.figure(figsize=(15, 8))
        plt.plot(plot_times, inv_actual[:, 1], 'b-', label='T_outer Actual', linewidth=2)
        plt.plot(plot_times, inv_actual[:, 2], 'r-', label='T_inner Actual', linewidth=2)
        plt.plot(plot_times, inv_actual[:, 3], 'g-', label='T_avg Actual', linewidth=2)
        plt.plot(plot_times, inv_pred[:, 1], 'b--', label=f'T_outer Pred (MAE={mae_outer:.2f}, R²={r2_outer:.3f})', linewidth=2, alpha=0.8)
        plt.plot(plot_times, inv_pred[:, 2], 'r--', label=f'T_inner Pred (MAE={mae_inner:.2f}, R²={r2_inner:.3f})', linewidth=2, alpha=0.8)
        plt.plot(plot_times, inv_pred[:, 3], 'g--', label=f'T_avg Pred (MAE={mae_avg:.2f}, R²={r2_avg:.3f})', linewidth=2, alpha=0.8)
        plt.plot(plot_times, inv_actual[:, 4], 'k-.', label='Input Temperature', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Temperature (°C)')
        plt.title(f'Temperature Prediction and Input - {file_name}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.5)
        plt.tight_layout()
        plot_filename = os.path.splitext(file_name)[0] + '.png'
        plt.savefig(os.path.join(plot_save_dir, plot_filename), dpi=150, bbox_inches='tight')
        plt.close()

    print(f"\nFinished testing {len(test_datasets)} files. Plots: {plot_save_dir}, CSV: {csv_save_dir}")
    if return_summary:
        return summary_data
    return None


# === Main ===
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(script_dir, "result")
    os.makedirs(result_dir, exist_ok=True)

    print("=" * 60)
    print("LOADING ALL DATA (ONE TIME ONLY)")
    print("=" * 60)
    train_dfs, val_dfs, test_dfs, scaler, test_paths = load_all_data()

    if train_dfs is None:
        print("Failed to load data. Exiting.")
        exit(1)

    print(f"\nData loaded successfully!")
    print(f"Training samples:   {len(train_dfs)}")
    print(f"Validation samples: {len(val_dfs)}")
    print(f"Test samples:       {len(test_dfs)}")
    print(f"Teacher forcing: {TEACHER_FORCING_RATIO} (rollout_prob={ROLLOUT_PROB})")
    print("=" * 60)

    for dropout in DROPOUT_LIST:
        subdir_name = "dropout_0" if dropout == 0 else f"dropout_{dropout}"
        plot_save_dir = os.path.join(result_dir, subdir_name)
        os.makedirs(plot_save_dir, exist_ok=True)
        print(f"\n{'='*60}\n>>> Dropout = {dropout}\n{'='*60}")

        all_lr_results = []
        for lr in INITIAL_LR_LIST:
            print(f"\n--- initial_lr = {lr} ---")
            model, test_sets, val_mae, train_hist, val_hist, lr_hist = train_model(
                train_dfs, val_dfs, test_dfs, scaler, test_paths,
                max_epochs=300,
                hidden_size=128,
                num_layers=5,
                dropout=dropout,
                rollout_prob=ROLLOUT_PROB,
                initial_lr=lr,
            )
            all_lr_results.append({
                "lr": lr,
                "model": model,
                "test_datasets": test_sets,
                "train_loss_history": train_hist,
                "val_loss_history": val_hist,
                "lr_history": lr_hist,
                "best_val_loss": val_mae,
            })
            print(f"  Final validation loss: {val_mae:.6f}")

        # 按验证集最优选一个模型（取整个训练过程中验证损失的最小值对应的 run）
        best_idx = np.argmin([min(r["val_loss_history"]) for r in all_lr_results])
        best = all_lr_results[best_idx]
        print(f"\nBest LR for dropout={dropout}: {best['lr']} (min val loss = {min(best['val_loss_history']):.6f})")

        # Training Loss vs Initial LR（4 条曲线）
        plt.figure(figsize=(8, 5))
        for i, r in enumerate(all_lr_results):
            plt.plot(r["train_loss_history"], label=f"LR={r['lr']}", color=LR_PLOT_COLORS[i], linewidth=2)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Training Loss", fontsize=12)
        plt.title(f"Training Loss vs Initial LR (dropout={dropout}, teacher_forcing=0)", fontsize=13, fontweight="bold")
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_save_dir, "training_loss_vs_initial_lr.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {plot_save_dir}/training_loss_vs_initial_lr.png")

        # Validation Loss vs Initial LR（4 条曲线）
        plt.figure(figsize=(8, 5))
        for i, r in enumerate(all_lr_results):
            plt.plot(r["val_loss_history"], label=f"LR={r['lr']}", color=LR_PLOT_COLORS[i], linewidth=2)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Validation Loss", fontsize=12)
        plt.title(f"Validation Loss vs Initial LR (dropout={dropout}, teacher_forcing=0)", fontsize=13, fontweight="bold")
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_save_dir, "validation_loss_vs_initial_lr.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {plot_save_dir}/validation_loss_vs_initial_lr.png")

        # Learning rate vs epoch（4 个 LR 都画在一张图）
        plt.figure(figsize=(8, 5))
        for i, r in enumerate(all_lr_results):
            plt.plot(r["lr_history"], label=f"LR={r['lr']}", color=LR_PLOT_COLORS[i], linewidth=2)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Learning Rate", fontsize=12)
        plt.title(f"Learning Rate vs Epoch (dropout={dropout}, teacher_forcing=0)", fontsize=13, fontweight="bold")
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_save_dir, "lr_vs_epoch.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {plot_save_dir}/lr_vs_epoch.png")

        # 用最佳模型做测试：图 + 每 case 的 CSV + 汇总 CSV
        print("\nTesting best model...")
        summary_list = test_model(best["model"], best["test_datasets"], plot_save_dir, return_summary=True)
        if summary_list:
            summary_df = pd.DataFrame(summary_list)
            avg_row = {}
            for col in summary_df.columns:
                if col == "Case":
                    avg_row[col] = "Mean"
                elif summary_df[col].dtype in (np.float64, np.float32):
                    avg_row[col] = summary_df[col].mean()
                else:
                    avg_row[col] = ""
            summary_df = pd.concat([summary_df, pd.DataFrame([avg_row])], ignore_index=True)
            summary_path = os.path.join(plot_save_dir, f"summary_{len(summary_list)}cases_MAE_R2.csv")
            summary_df.to_csv(summary_path, index=False, float_format="%.4f")
            print(f"Saved: {summary_path}")
