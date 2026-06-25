import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import MinMaxScaler
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
torch.backends.cudnn.benchmark = False

# Only test dropout = 0.3
configs = [
    {"hidden_size": 128, "num_layers": 5, "max_epochs": 300, "dropout": 0.3},
]


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
        # NEW: store initial condition (t=0) for each sequence
        self.init_conditions = []

        for _, group in grouped:
            # Input: 4 raw features only (delta features removed)
            X_seq = group[["Time (s)",
                           "T_outer (C)", "Input Temperature (C)", "T_avg (C)"]].values[:-1]
            Y_seq = group[["T_inner (C)", "T_outer (C)", "T_avg (C)"]].values[1:]
            time_vals = group["Time (s)"].values[1:]

            # === Capture t=0 initial condition: [T_outer, T_inner, T_avg, Input_T] ===
            # These are the 4 known physical states at the very first timestep
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
            self.init_conditions[idx],          # NEW: initial condition at t=0
            self.full_time[idx],
            self.full_t_min[idx], self.full_t_max[idx], self.full_t_ave[idx]
        )


# ============================================================
# NEW: InitStateEncoder
# Maps t=0 observations → GRU initial hidden state h0
# Replaces the zero initialization entirely — no burn-in needed
# ============================================================
class InitStateEncoder(nn.Module):
    """
    Encodes the known physical state at t=0 into a meaningful
    GRU hidden state, so the model understands the system's
    starting condition without any burn-in steps.

    Input:  x0 [batch, 4]  →  [T_outer, T_inner, T_avg, Input_T] at t=0
    Output: h0 [num_layers, batch, hidden_size]
    """
    def __init__(self, obs_size=4, hidden_size=128, num_layers=5):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.encoder = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_layers * hidden_size)
        )

    def forward(self, x0):
        # x0: [batch, 4]
        h = self.encoder(x0)                              # [batch, num_layers * hidden_size]
        h = h.view(-1, self.num_layers, self.hidden_size) # [batch, num_layers, hidden_size]
        h = h.permute(1, 0, 2).contiguous()               # [num_layers, batch, hidden_size]
        return h


# === Model definition ===
class ThermalGRU(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, output_size=3,
                 num_layers=5, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        # Replaces zero init — trained end-to-end with the GRU
        self.init_encoder = InitStateEncoder(
            obs_size=4, hidden_size=hidden_size, num_layers=num_layers
        )

    def init_hidden(self, batch_size, x0=None):
        """
        If x0 (initial condition) is provided, encode it into h0.
        Otherwise fall back to zeros (for safety, should not be needed).
        """
        device = next(self.parameters()).device
        if x0 is not None:
            return self.init_encoder(x0.to(device))
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

    def forward(self, x, hidden):
        out, hidden = self.gru(x, hidden)
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

    possible_data_dirs = [
        os.path.join(script_dir, "..", "..", "data"),
        os.path.join(script_dir, "data"),
        "data"
    ]

    data_dir = None
    for dir_path in possible_data_dirs:
        if os.path.exists(dir_path):
            data_dir = dir_path
            break

    if not data_dir:
        print("Data directory not found")
        return None, None, None, None, None

    train_dir = os.path.join(data_dir, "data_in_10s")
    test_dir = os.path.join(data_dir, "test_in_10s")

    train_paths = sorted(glob.glob(os.path.join(train_dir, "**", "*.csv"), recursive=True))
    test_paths = sorted(glob.glob(os.path.join(test_dir, "*.csv")))

    print(f"Found training files: {len(train_paths)}")
    print(f"Found test files: {len(test_paths)}")

    if not train_paths:
        print("No training files found")
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
                dropout=0.3, rollout_prob=0.2):

    print("Creating datasets from pre-loaded data...")
    train_datasets = [ThermalDataset(df, scaler=scaler, file_name=fname)
                      for df, fname in train_dfs]
    val_datasets   = [ThermalDataset(df, scaler=scaler, file_name=fname)
                      for df, fname in val_dfs]
    test_datasets  = [ThermalDataset(df, scaler=scaler, file_name=fname)
                      for df, fname in test_dfs]

    train_loader = DataLoader(ConcatDataset(train_datasets), batch_size=16, shuffle=True)
    val_loader   = DataLoader(ConcatDataset(val_datasets),   batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("Using CPU")

    model = ThermalGRU(
        input_size=4, hidden_size=hidden_size,
        num_layers=num_layers, dropout=dropout,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50
    )

    best_val_loss = float('inf')
    early_stop_counter = 0
    patience = 300

    train_loss_history = []
    val_loss_history   = []

    for epoch in range(max_epochs):
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            # Unpack — note init_conditions is now index 3
            inputs, targets, _, init_conds, *_ = [
                b.to(device) if torch.is_tensor(b) else b for b in batch
            ]
            batch_size = inputs.shape[0]

            # === Key change: use InitStateEncoder instead of zeros ===
            hidden = model.init_hidden(batch_size, x0=init_conds)

            optimizer.zero_grad()

            # No burn-in needed — hidden state already encodes t=0 physics
            prev_t_outer = inputs[:, 0, 1]
            prev_t_avg   = inputs[:, 0, 3]

            preds = []
            current_hidden = hidden

            for t in range(inputs.size(1)):
                x_t = inputs[:, t, :].clone()

                # Scheduled sampling
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

    return model, test_datasets, best_val_loss, train_loss_history, val_loss_history


# === Test model ===
def test_model(model, test_datasets):
    device = next(model.parameters()).device
    model.eval()

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    output_dir  = os.path.join(script_dir, "plot_GRU_init_encoder")
    os.makedirs(output_dir, exist_ok=True)

    all_mae_inner, all_mae_outer, all_mae_avg = [], [], []

    for idx, dataset in enumerate(test_datasets):
        file = os.path.basename(dataset.file_name)

        # Unpack with new init_conditions field
        x, _, _, init_cond, ft, ft_min, ft_max, ft_ave = dataset[0]
        x         = x.to(device)
        init_cond = init_cond.unsqueeze(0).to(device)   # [1, 4]

        # === No burn-in: encode t=0 state directly ===
        hidden = model.init_hidden(1, x0=init_cond)

        seq_len = x.shape[0]
        current_t_outer    = x[0, 1]
        current_input_temp = x[0, 2]
        current_t_avg      = x[0, 3]

        preds = []

        for t in range(seq_len):
            input_t = torch.tensor([[
                x[t, 0],            # Time (s)
                current_t_outer,    # T_outer (C)
                current_input_temp, # Input Temperature (C)
                current_t_avg,      # T_avg (C)
            ]], dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                out, hidden = model(input_t, hidden)
                pred = out[0, 0].cpu().numpy()
                preds.append(pred)

                if t < seq_len - 1:
                    current_t_outer    = pred[1]
                    current_input_temp = x[t + 1, 2].item()
                    current_t_avg      = pred[2]

        pred_seq = np.array(preds)

        # Inverse transform
        dummy_actual = np.zeros((len(pred_seq), 5))
        dummy_actual[:, 0] = ft[:len(pred_seq)]
        dummy_actual[:, 1] = ft_min[:len(pred_seq)]
        dummy_actual[:, 2] = ft_max[:len(pred_seq)]
        dummy_actual[:, 3] = ft_ave[:len(pred_seq)]
        dummy_actual[:, 4] = x[:, 2].cpu().numpy()[:len(pred_seq)]

        dummy_pred = np.zeros((len(pred_seq), 5))
        dummy_pred[:, 0] = ft[:len(pred_seq)]
        dummy_pred[:, 1] = pred_seq[:, 1]
        dummy_pred[:, 2] = pred_seq[:, 0]
        dummy_pred[:, 3] = pred_seq[:, 2]
        dummy_pred[:, 4] = x[:, 2].cpu().numpy()[:len(pred_seq)]

        inv_pred   = dataset.scaler.inverse_transform(dummy_pred)
        inv_actual = dataset.scaler.inverse_transform(dummy_actual)

        mae_inner = np.mean(np.abs(inv_actual[:, 2] - inv_pred[:, 2]))
        mae_outer = np.mean(np.abs(inv_actual[:, 1] - inv_pred[:, 1]))
        mae_avg   = np.mean(np.abs(inv_actual[:, 3] - inv_pred[:, 3]))

        # Early-stage MAE (first 10% of sequence) to verify cold-start quality
        early_end = max(10, len(pred_seq) // 10)
        early_mae_inner = np.mean(np.abs(inv_actual[:early_end, 2] - inv_pred[:early_end, 2]))

        all_mae_inner.append(mae_inner)
        all_mae_outer.append(mae_outer)
        all_mae_avg.append(mae_avg)

        # Plot
        plt.figure(figsize=(12, 6))
        time_axis = inv_actual[:, 0]

        plt.plot(time_axis, inv_actual[:, 1], label="T_outer Actual",  color="blue")
        plt.plot(time_axis, inv_pred[:, 1],   "--", label="T_outer Pred", color="blue")
        plt.plot(time_axis, inv_actual[:, 3], label="T_avg Actual",    color="green")
        plt.plot(time_axis, inv_pred[:, 3],   "--", label="T_avg Pred",   color="green")
        plt.plot(time_axis, inv_actual[:, 2], label="T_inner Actual",  color="red")
        plt.plot(time_axis, inv_pred[:, 2],   "--", label="T_inner Pred", color="red")



        plt.xlabel("Time (s)")
        plt.ylabel("Temperature (C)")
        plt.title(f"Prediction (No Burn-in) - {file}")
        plt.legend(fontsize=8)

        mae_text = (f"Full MAE: T_inner={mae_inner:.3f}°C, "
                    f"T_outer={mae_outer:.3f}°C, T_avg={mae_avg:.3f}°C\n"
                    f"Early-stage MAE (first {early_end} steps): T_inner={early_mae_inner:.3f}°C")
        plt.text(0.02, 0.98, mae_text, transform=plt.gca().transAxes,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                 fontsize=9)
        plt.tight_layout()

        output_path = os.path.join(output_dir, f"plot_{file}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    # Summary
    print(f"\n{'='*50}")
    print("Test Results Summary (No Burn-in)")
    print(f"{'='*50}")
    print(f"Mean MAE T_inner: {np.mean(all_mae_inner):.4f} ± {np.std(all_mae_inner):.4f} °C")
    print(f"Mean MAE T_outer: {np.mean(all_mae_outer):.4f} ± {np.std(all_mae_outer):.4f} °C")
    print(f"Mean MAE T_avg:   {np.mean(all_mae_avg):.4f}   ± {np.std(all_mae_avg):.4f} °C")
    print(f"\nAll plots saved to: {output_dir}")


# === Main ===
if __name__ == "__main__":
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
    print("=" * 60)

    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"Running Config: {cfg}")
        print(f"{'='*60}")

        model, test_sets, val_mae, train_history, val_history = train_model(
            train_dfs, val_dfs, test_dfs, scaler, test_paths,
            max_epochs=cfg["max_epochs"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"]
        )

        print(f"\nFinal Validation MAE: {val_mae:.6f}")

        # Plot loss curves
        script_dir = os.path.dirname(os.path.abspath(__file__))

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_history, label='Train Loss', color='steelblue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss (dropout=0.3, no burn-in)')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(val_history, label='Val Loss', color='coral')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Validation Loss (dropout=0.3, no burn-in)')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        loss_path = os.path.join(script_dir, 'loss_curve_init_encoder.png')
        plt.savefig(loss_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Loss curve saved to: {loss_path}")

        # Test
        print("\nTesting model...")
        test_model(model, test_sets)