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


def get_data_paths():
    """获取数据路径，自动检测可用的目录"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(script_dir)
    project_root = os.path.dirname(src_dir)
    data_dir = os.path.join(project_root, "data")

    print(f"Looking for data in: {data_dir}")

    # 尝试不同的训练数据目录名称
    possible_train_dirs = [
        "180s Time steps",
        "150s Time steps",
    ]

    train_dir = None
    for dirname in possible_train_dirs:
        potential_dir = os.path.join(data_dir, dirname)
        if os.path.exists(potential_dir):
            train_dir = potential_dir
            print(f"Found training directory: {train_dir}")
            break

    if train_dir is None:
        train_dir = data_dir
        print(f"Using data directory directly: {train_dir}")

    test_dir = os.path.join(data_dir, "test")
    if not os.path.exists(test_dir):
        test_dir = data_dir
        print(f"Test directory not found, using data directory: {test_dir}")

    train_paths = sorted(glob.glob(os.path.join(train_dir, "**", "*.csv"), recursive=True))
    test_paths = sorted(glob.glob(os.path.join(test_dir, "*.csv")))

    print(f"Found {len(train_paths)} training files")
    print(f"Found {len(test_paths)} test files")

    if len(train_paths) == 0:
        print("No training CSV files found. Please check your directory structure.")
        if os.path.exists(data_dir):
            print(f"Contents of {data_dir}:")
            for item in sorted(os.listdir(data_dir)):
                print(f"  {item}")
        raise ValueError("No training data found")

    return train_paths, test_paths


def train_model_with_data_ratio(train_paths, test_paths, data_ratio, num_epochs=300):
    """使用指定比例的数据训练模型"""
    print(f"\n=== Training with {int(data_ratio * 100)}% of data ({int(len(train_paths) * data_ratio)} files) ===")

    # 选择指定比例的训练数据
    num_train = max(1, int(len(train_paths) * data_ratio))
    selected_train_paths = train_paths[:num_train]

    # 验证集从最后的文件中选择
    val_split = max(1, int(0.1 * len(train_paths)))  # 使用10%作为验证集
    val_paths = train_paths[-val_split:] if len(train_paths) > val_split else train_paths[:1]

    print(f"Using {len(selected_train_paths)} files for training")
    print(f"Using {len(val_paths)} files for validation")

    # 创建统一的归一化器
    train_dfs = []
    for f in selected_train_paths:
        try:
            df = pd.read_csv(f)
            if "T_ave (C)" in df.columns:
                df = df.rename(columns={"T_ave (C)": "T_avg (C)"})
            train_dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            continue

    if len(train_dfs) == 0:
        raise ValueError("No valid training data loaded")

    scaler = MinMaxScaler()
    all_train_df = pd.concat(train_dfs)
    required_columns = ["Time (s)", "T_outer (C)", "T_inner (C)", "T_avg (C)"]
    scaler.fit(all_train_df[required_columns])

    # 创建数据集
    train_datasets = [ThermalDataset(f, scaler=scaler) for f in selected_train_paths]
    val_datasets = [ThermalDataset(f, scaler=scaler) for f in val_paths]

    train_loader = DataLoader(ConcatDataset(train_datasets), batch_size=16, shuffle=True)
    val_loader = DataLoader(ConcatDataset(val_datasets), batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ThermalGRU().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=10)

    best_train_loss = float('inf')
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            inputs, targets, *_ = [b.to(device) if torch.is_tensor(b) else b for b in batch]
            batch_size, seq_len, _ = inputs.shape
            hidden = model.init_hidden(batch_size)
            optimizer.zero_grad()

            # Burn-in
            if seq_len > 1:
                _, hidden = model(inputs[:, :1], hidden)

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

                # Teacher forcing
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

        avg_train_loss = total_train_loss / len(train_loader)

        # 验证阶段
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets, *_ = [b.to(device) if torch.is_tensor(b) else b for b in batch]
                batch_size, seq_len, _ = inputs.shape
                hidden = model.init_hidden(batch_size)

                if seq_len > 1:
                    _, hidden = model(inputs[:, :1], hidden)

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

                total_val_loss += batch_val_loss.item() / seq_len

        avg_val_loss = total_val_loss / len(val_loader)

        # 更新最佳损失
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        scheduler.step(avg_val_loss)

        # 每10个epoch打印一次进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch + 1:2d}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f} | Best: Train={best_train_loss:.4f}, Val={best_val_loss:.4f}")

        # 早停
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    return best_train_loss, best_val_loss, len(selected_train_paths)


def run_sample_size_experiment():
    """运行样本数量实验"""
    # 获取数据路径
    train_paths, test_paths = get_data_paths()

    # 定义不同的数据比例
    data_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    results = []

    print(f"Running experiment with {len(data_ratios)} different data sizes")
    print(f"Total available training files: {len(train_paths)}")

    for ratio in data_ratios:
        try:
            best_train, best_val, num_files = train_model_with_data_ratio(
                train_paths, test_paths, ratio, num_epochs=50
            )
            results.append({
                'ratio': ratio,
                'num_files': num_files,
                'best_train_loss': best_train,
                'best_val_loss': best_val
            })
            print(f"✓ Ratio {int(ratio * 100):2d}%: {num_files:2d} files -> Train={best_train:.4f}, Val={best_val:.4f}")
        except Exception as e:
            print(f"✗ Error with ratio {ratio}: {e}")
            continue

    if len(results) == 0:
        print("No successful experiments. Check your data.")
        return

    # 绘制结果
    plot_results(results)

    # 保存结果
    save_results(results)

    return results


def plot_results(results):
    """绘制实验结果"""
    # 提取数据
    ratios = [r['ratio'] * 100 for r in results]  # 转换为百分比
    num_files = [r['num_files'] for r in results]
    train_losses = [r['best_train_loss'] for r in results]
    val_losses = [r['best_val_loss'] for r in results]

    # 创建双图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 图1: Loss vs 数据比例
    ax1.plot(ratios, train_losses, 'o-', label='Best Training Loss', linewidth=2, markersize=8)
    ax1.plot(ratios, val_losses, 's-', label='Best Validation Loss', linewidth=2, markersize=8)
    ax1.set_xlabel('Training Data Size (%)', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('180s Model Performance vs Training Data Size', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.set_xlim(min(ratios) - 5, max(ratios) + 5)

    # 图2: Loss vs 文件数量
    ax2.plot(num_files, train_losses, 'o-', label='Best Training Loss', linewidth=2, markersize=8)
    ax2.plot(num_files, val_losses, 's-', label='Best Validation Loss', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Training Files', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('180s Model Performance vs Number of Training Files', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)

    plt.tight_layout()

    # 保存图片
    plt.savefig('180s sample_size_experiment_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nResults saved as '180s sample_size_experiment_results.png'")


def save_results(results):
    """保存实验结果到CSV文件"""
    import pandas as pd

    df = pd.DataFrame(results)
    df['data_percentage'] = df['ratio'] * 100
    df = df[['data_percentage', 'num_files', 'best_train_loss', 'best_val_loss']]
    df.columns = ['Data_Percentage', 'Num_Files', 'Best_Train_Loss', 'Best_Val_Loss']

    df.to_csv('180s sample_size_experiment_results.csv', index=False)
    print("Results saved as '180s sample_size_experiment_results.csv'")

    # 打印总结
    print(f"\n{'=' * 60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Data%':<6} {'Files':<6} {'Train Loss':<12} {'Val Loss':<12} {'Gap':<12}")
    print(f"{'-' * 60}")
    for _, row in df.iterrows():
        gap = row['Best_Val_Loss'] - row['Best_Train_Loss']
        print(
            f"{row['Data_Percentage']:5.0f}% {row['Num_Files']:5.0f}  {row['Best_Train_Loss']:11.4f}  {row['Best_Val_Loss']:11.4f}  {gap:11.4f}")


# === 主函数 ===
if __name__ == "__main__":
    print("Starting sample size experiment...")
    results = run_sample_size_experiment()
    print("Experiment completed!")