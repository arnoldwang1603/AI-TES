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


# === Dataset 类 ===
class ThermalDataset(Dataset):
    def __init__(self, csv_file, scaler=None):
        self.file_name = os.path.basename(csv_file)
        df = pd.read_csv(csv_file)
        # 统一列名处理
        if "T_ave (C)" in df.columns:
            df = df.rename(columns={"T_ave (C)": "T_avg (C)"})

        df["FileName"] = self.file_name
        columns_for_scaling = ['Time (s)', 'T_outer (C)', 'T_inner (C)', 'T_avg (C)']

        if scaler is None:
            self.scaler = MinMaxScaler()
            self.scaler.fit(df[columns_for_scaling])
        else:
            self.scaler = scaler

        df[columns_for_scaling] = self.scaler.transform(df[columns_for_scaling])
        grouped = df.groupby("FileName")
        self.X, self.Y, self.time_values = [], [], []
        self.full_time, self.full_t_min, self.full_t_max, self.full_t_ave = [], [], [], []

        for _, group in grouped:
            X_seq = group[["Time (s)", "T_outer (C)", "T_inner (C)", "T_avg (C)"]].values[:-1]
            Y_seq = group[["T_outer (C)", "T_avg (C)"]].values[1:]
            time_vals = group["Time (s)"].values[1:]

            self.X.append(X_seq)
            self.Y.append(Y_seq)
            self.time_values.append(time_vals)
            self.full_time.append(group["Time (s)"].values)
            self.full_t_min.append(group["T_outer (C)"].values)
            self.full_t_max.append(group["T_inner (C)"].values)
            self.full_t_ave.append(group["T_avg (C)"].values)

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.Y = torch.tensor(np.array(self.Y), dtype=torch.float32)
        self.time_values = np.array(self.time_values)
        self.full_time = np.array(self.full_time)
        self.full_t_min = np.array(self.full_t_min)
        self.full_t_max = np.array(self.full_t_max)
        self.full_t_ave = np.array(self.full_t_ave)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            self.X[idx], self.Y[idx], self.time_values[idx],
            self.full_time[idx],
            self.full_t_min[idx], self.full_t_max[idx], self.full_t_ave[idx]
        )


# === 模型定义 ===
class ThermalGRU(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, output_size=2, num_layers=5):
        super(ThermalGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.gru(x, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)


# === 加权损失 ===
def weighted_loss(predictions, targets, weights=torch.tensor([1.0, 1.0]), time_weights=None):
    weights = weights.to(predictions.device)
    loss = torch.abs(predictions - targets) * weights
    if time_weights is not None:
        time_weights = time_weights.to(predictions.device)
        loss = loss * time_weights.unsqueeze(-1)
    return torch.mean(loss)


# === 训练函数 ===
def train_model(max_epochs=500):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(script_dir)
    project_root = os.path.dirname(src_dir)
    data_dir = os.path.join(project_root, "data")

    # ==== 加载 train 和 test 文件夹中的 CSV 文件 ====
    train_dir = os.path.join(data_dir, "180s Time steps")
    test_dir = os.path.join(data_dir, "test")
    train_paths = sorted(glob.glob(os.path.join(train_dir, "**", "*.csv"), recursive=True))
    test_paths = sorted(glob.glob(os.path.join(test_dir, "*.csv")))

    # 数据分割
    val_split = int(0.05 * len(train_paths))
    val_paths = train_paths[:val_split]
    actual_train_paths = train_paths[val_split:]

    # 创建scaler
    train_dfs = [pd.read_csv(f) for f in actual_train_paths]
    scaler = MinMaxScaler()
    # 处理列名统一
    all_train_df = pd.concat([df.rename(columns={"T_ave (C)": "T_avg (C)"}) if "T_ave (C)" in df.columns else df
                              for df in train_dfs])
    scaler.fit(all_train_df[["Time (s)", "T_outer (C)", "T_inner (C)", "T_avg (C)"]])

    # 创建数据集
    train_datasets = [ThermalDataset(f, scaler=scaler) for f in actual_train_paths]
    val_datasets = [ThermalDataset(f, scaler=scaler) for f in val_paths]
    test_datasets = [ThermalDataset(f, scaler=scaler) for f in test_paths]

    train_loader = DataLoader(ConcatDataset(train_datasets), batch_size=16, shuffle=True)
    val_loader = DataLoader(ConcatDataset(val_datasets), batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ThermalGRU().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)

    best_val_loss, early_stop_counter = float('inf'), 0
    patience, burn_in_steps = 300, 1

    for epoch in range(max_epochs):
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            inputs, targets, *_ = [b.to(device) if torch.is_tensor(b) else b for b in batch]
            batch_size, seq_len, _ = inputs.shape
            hidden = model.init_hidden(batch_size)
            optimizer.zero_grad()

            if seq_len > burn_in_steps:
                _, hidden = model(inputs[:, :burn_in_steps], hidden)

            current_t_min = inputs[:, 0, 1]
            current_t_max = inputs[:, 0, 2]
            current_t_ave = inputs[:, 0, 3]
            time_weights = torch.linspace(1, 0, seq_len, device=device)
            batch_loss = 0.0

            for t in range(seq_len):
                input_t = torch.stack([inputs[:, t, 0], current_t_min, current_t_max, current_t_ave], dim=1).unsqueeze(
                    1)
                delta, hidden = model(input_t, hidden)
                current_vals = input_t[:, 0, 1:3]
                output = current_vals + delta[:, 0]
                loss_t = weighted_loss(output, targets[:, t], time_weights=time_weights[t:t + 1])
                batch_loss += loss_t

                if t < seq_len - 1:
                    use_teacher = (torch.rand(batch_size, device=device) < 0.5).float()
                    ground_truth = inputs[:, t + 1, 1:4]
                    current_t_min = use_teacher * ground_truth[:, 0] + (1 - use_teacher) * output[:, 0]
                    current_t_max = ground_truth[:, 1]
                    current_t_ave = use_teacher * ground_truth[:, 2] + (1 - use_teacher) * output[:, 1]

            (batch_loss / seq_len).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_train_loss += batch_loss.item() / seq_len

        # 验证
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for val_batch in val_loader:
                inputs, targets, *_ = [b.to(device) if torch.is_tensor(b) else b for b in val_batch]
                hidden = model.init_hidden(inputs.size(0))

                if seq_len > burn_in_steps:
                    _, hidden = model(inputs[:, :burn_in_steps], hidden)

                current_t_min = inputs[:, 0, 1]
                current_t_max = inputs[:, 0, 2]
                current_t_ave = inputs[:, 0, 3]
                batch_val_loss = 0

                for t in range(seq_len):
                    input_t = torch.stack([inputs[:, t, 0], current_t_min, current_t_max, current_t_ave],
                                          dim=1).unsqueeze(1)
                    delta, hidden = model(input_t, hidden)
                    current_vals = input_t[:, 0, 1:3]
                    output = current_vals + delta[:, 0]
                    loss_t = weighted_loss(output, targets[:, t])
                    batch_val_loss += loss_t

                    if t < seq_len - 1:
                        current_t_min = inputs[:, t + 1, 1]
                        current_t_max = inputs[:, t + 1, 2]
                        current_t_ave = inputs[:, t + 1, 3]

                val_loss += batch_val_loss.item() / seq_len

        ave_val_loss = val_loss / len(val_loader)
        scheduler.step(ave_val_loss)

        print(
            f"[Epoch {epoch + 1}] Train Loss: {total_train_loss / len(train_loader):.4f}, Val Loss: {ave_val_loss:.4f}")

        # 早停和模型保存
        if ave_val_loss < best_val_loss:
            best_val_loss = ave_val_loss
            torch.save(model.state_dict(), os.path.join(script_dir, f"best_gru_{max_epochs}epoch.pth"))
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping.")
                break

    last_val_loss = ave_val_loss
    last_train_loss = total_train_loss / len(train_loader)

    # 加载最佳模型
    model.load_state_dict(torch.load(os.path.join(script_dir, f"best_gru_{max_epochs}epoch.pth")))
    return model, test_datasets, scaler, last_val_loss, last_train_loss


# === 测试函数 ===
def test_model(model, test_datasets, scaler, epoch_tag=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    for dataset in test_datasets:
        file = os.path.basename(dataset.file_name)
        x, _, _, ft, ft_min, ft_max, ft_ave = dataset[0]
        x = x.to(device)
        hidden = model.init_hidden(1)

        current_t_min = x[0, 1]
        current_t_max = x[0, 2]
        current_t_ave = x[0, 3]
        preds = []

        for t in range(x.shape[0]):
            input_t = torch.tensor([[x[t, 0], current_t_min, current_t_max, current_t_ave]],
                                   dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                delta, hidden = model(input_t, hidden)
                current_vals = input_t[:, :, 1:3]
                output = current_vals + delta
                pred = output[0, 0].cpu().numpy()
                preds.append(pred)

                if t < x.shape[0] - 1:
                    if t >= 4:
                        current_t_min, current_t_ave = pred[0], pred[1]
                        current_t_max = x[t + 1, 2]
                    else:
                        current_t_min = x[t + 1, 1]
                        current_t_max = x[t + 1, 2]
                        current_t_ave = x[t + 1, 3]

        pred_seq = np.array(preds)

        # 反归一化
        dummy = np.zeros((len(ft), 4))
        dummy[:, 0], dummy[:, 1], dummy[:, 2], dummy[:, 3] = ft, ft_min, ft_max, ft_ave
        inv_actual = scaler.inverse_transform(dummy)

        dummy_pred = np.zeros((len(pred_seq), 4))
        dummy_pred[:, 1], dummy_pred[:, 3] = pred_seq[:, 0], pred_seq[:, 1]
        inv_pred = scaler.inverse_transform(dummy_pred)


# === 主循环（只保留一个） ===
if __name__ == "__main__":
    epoch_list = list(range(10, 501, 10))  # [5, 10]
    final_val_losses = []
    final_train_losses = []
    for epoch_num in epoch_list:
        print(f"\n{'=' * 50}")
        print(f"Training with {epoch_num} epochs")
        print(f"{'=' * 50}")

        model, test_sets, scaler, last_val_loss, last_train_loss = train_model(max_epochs=epoch_num)
        test_model(model, test_sets, scaler, epoch_tag=f"{epoch_num}epoch")

        final_val_losses.append(last_val_loss)
        final_train_losses.append(last_train_loss)

        print(f"Final results for {epoch_num} epochs:")
        print(f"  Train Loss: {last_train_loss:.4f}")
        print(f"  Val Loss: {last_val_loss:.4f}")

    # === 绘制结果 ===
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_list, final_train_losses, marker="s", linewidth=2, markersize=8, label="Final Train Loss")
    plt.plot(epoch_list, final_val_losses, marker="o", linewidth=2, markersize=8, label="Final Val Loss")
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("180s Final Train & Validation Loss at Different Epochs", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()

    # 保存图表
    plt.savefig("180s train_val_loss_vs_epochs.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 打印最终结果
    print(f"\n{'=' * 50}")
    print("FINAL SUMMARY:")
    print(f"{'=' * 50}")
    for i, epochs in enumerate(epoch_list):
        print(f"Epochs {epochs:2d}: Train Loss = {final_train_losses[i]:.4f}, Val Loss = {final_val_losses[i]:.4f}")