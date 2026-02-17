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
from sklearn.metrics import r2_score, mean_absolute_error


def _collate_thermal_variable_length(batch):
    """Pad variable-length (X, Y, ...) to max length in batch; return mask for loss."""
    X_list, Y_list, ext_list, time_list, full_list = zip(*batch)
    max_len = max(X.size(0) for X in X_list)
    max_ext = max(e.size(0) for e in ext_list)
    B = len(X_list)
    X_pad = torch.zeros(B, max_len, X_list[0].size(1), dtype=X_list[0].dtype)
    Y_pad = torch.zeros(B, max_len, Y_list[0].size(1), dtype=Y_list[0].dtype)
    mask = torch.zeros(B, max_len, dtype=torch.float32)
    for i, (X, Y) in enumerate(zip(X_list, Y_list)):
        L = X.size(0)
        X_pad[i, :L] = X
        Y_pad[i, :L] = Y
        mask[i, :L] = 1.0
    ext_pad = torch.zeros(B, max_ext, ext_list[0].size(1), dtype=ext_list[0].dtype)
    for i, ext in enumerate(ext_list):
        L = ext.size(0)
        ext_pad[i, :L] = ext
        if L < max_ext:
            ext_pad[i, L:] = ext[-1:]
    return X_pad, Y_pad, ext_pad, time_list, full_list, mask


def _convert_extended_time_xlsx_to_csv_dir(xlsx_dir, out_csv_dir):
    """
    将「Testing cases for temperature condition_extended time」下的 xlsx 转为 5 列 csv 到 out_csv_dir。
    返回生成的 csv 路径列表。
    """
    os.makedirs(out_csv_dir, exist_ok=True)
    xlsx_paths = sorted(glob.glob(os.path.join(xlsx_dir, "*.xlsx")))
    xlsx_paths = [p for p in xlsx_paths if "case" in os.path.basename(p).lower()][:10]
    if not xlsx_paths:
        xlsx_paths = sorted(glob.glob(os.path.join(xlsx_dir, "*.xlsx")))[:10]
    csv_paths = []
    for xp in xlsx_paths:
        df = pd.read_excel(xp, header=1)
        time_col = "Time (s).1" if "Time (s).1" in df.columns else "Time (s)"
        if df[time_col].dropna().empty:
            continue
        out = pd.DataFrame()
        out["Time (s)"] = df[time_col].values
        out["T_outer (C)"] = df["T_outer (C)"].values
        out["T_inner (C)"] = df["T_inner (C)"].values
        out["T_avg (C)"] = (df["T_ave (C)"].values if "T_ave (C)" in df.columns else df["T_avg (C)"].values)
        if "T_in (C)" in df.columns and "Time (s)" in df.columns:
            in_df = df[["Time (s)", "T_in (C)"]].dropna(subset=["Time (s)", "T_in (C)"])
            if len(in_df) >= 2:
                out["Input Temperature (C)"] = np.interp(
                    out["Time (s)"].values.astype(float),
                    in_df["Time (s)"].values.astype(float),
                    in_df["T_in (C)"].values.astype(float),
                )
            else:
                out["Input Temperature (C)"] = df["T_inner (C)"].values
        else:
            out["Input Temperature (C)"] = df["T_inner (C)"].values
        out = out.dropna(subset=["Time (s)", "T_outer (C)", "T_inner (C)", "T_avg (C)"])
        base = os.path.splitext(os.path.basename(xp))[0]
        cp = os.path.join(out_csv_dir, base + ".csv")
        out.to_csv(cp, index=False)
        csv_paths.append(cp)
    return sorted(csv_paths)


class UpdatedThermalDataset(Dataset):
    def __init__(self, csv_file, scaler=None, prepend_burn_in=0):
        """
        prepend_burn_in: 若 >0（如 30），在序列前重复第一条数据 prepend_burn_in 遍，用于测试时预热，
                         预测从“原始第一条”之后开始，便于和真实曲线对齐。
        """
        self.file_name = os.path.basename(csv_file)
        self.prepend_burn_in = prepend_burn_in
        if str(csv_file).lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(csv_file)
        else:
            df = pd.read_csv(csv_file)
        df["FileName"] = self.file_name

        # Check and normalize column names
        # print(f"Processing file: {csv_file}")
        # print(f"Original columns: {list(df.columns)}") # Verbose, can be commented out

        # Column name normalization mapping
        column_mapping = {}

        # Handle inconsistency between T_ave vs T_avg
        if 'T_ave (C)' in df.columns and 'T_avg (C)' not in df.columns:
            column_mapping['T_ave (C)'] = 'T_avg (C)'
        
        if column_mapping:
            df = df.rename(columns=column_mapping)
            # print(f"Column standardized: {column_mapping}")

        # Check required columns and handle missing Input Temperature
        expected_cols = ['Time (s)', 'T_outer (C)', 'T_inner (C)', 'T_avg (C)', 'Input Temperature (C)']
        missing_cols = [col for col in expected_cols if col not in df.columns]

        if 'Input Temperature (C)' in missing_cols and 'T_inner (C)' in df.columns:
            df['Input Temperature (C)'] = df['T_inner (C)']
            # print(f"Using T_inner as substitute for Input Temperature")
            missing_cols.remove('Input Temperature (C)')

        if missing_cols:
            raise ValueError(f"File {csv_file} is missing required columns: {missing_cols}")

        # print(f"Final columns: {list(df.columns)}")

        columns_for_scaling = ['Time (s)', 'T_outer (C)', 'T_inner (C)', 'T_avg (C)', 'Input Temperature (C)']

        if scaler is None:
            self.scaler = MinMaxScaler()
            self.scaler.fit(df[columns_for_scaling])
        else:
            self.scaler = scaler

        df[columns_for_scaling] = self.scaler.transform(df[columns_for_scaling])

        # 只添加这两个差分：dT_avg, dInput Temperature
        df["dT_avg (C)"] = df["T_avg (C)"].diff().fillna(0)
        df["dInput Temperature (C)"] = df["Input Temperature (C)"].diff().fillna(0)

        grouped = df.groupby("FileName")
        self.X, self.Y, self.external_conditions, self.time_values, self.full_data = [], [], [], [], []
        self.original_lengths = []  # 每条序列原始长度（用于测试集绘图时只画原始区间）

        for _, group in grouped:
            if self.prepend_burn_in > 0 and len(group) >= 1:
                first_row = group.iloc[[0]]
                prepended = pd.concat([first_row] * self.prepend_burn_in + [group], ignore_index=True)
                prepended["dT_avg (C)"] = prepended["T_avg (C)"].diff().fillna(0)
                prepended["dInput Temperature (C)"] = prepended["Input Temperature (C)"].diff().fillna(0)
                group = prepended
                self.original_lengths.append(len(group) - self.prepend_burn_in)
            else:
                self.original_lengths.append(len(group))

            external_seq = group[["Time (s)", "Input Temperature (C)"]].values
            X_cols = ["Time (s)", "Input Temperature (C)", "T_outer (C)", "T_inner (C)", "T_avg (C)",
                      "dT_avg (C)", "dInput Temperature (C)"]
            X_seq = group[X_cols].values[:-1]
            Y_seq = group[["T_outer (C)", "T_inner (C)", "T_avg (C)"]].values[1:]
            time_vals = group["Time (s)"].values[1:]

            self.X.append(X_seq)
            self.Y.append(Y_seq)
            self.external_conditions.append(external_seq)
            self.time_values.append(time_vals)
            self.full_data.append(group[columns_for_scaling].values)

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.Y = torch.tensor(np.array(self.Y), dtype=torch.float32)
        self.external_conditions = torch.tensor(np.array(self.external_conditions), dtype=torch.float32)
        self.time_values = np.array(self.time_values)
        self.full_data = np.array(self.full_data)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx], self.external_conditions[idx], self.time_values[idx], self.full_data[idx])

class UpdatedThermalLSTM(nn.Module):
    def __init__(self, input_size=7, hidden_size=128, output_size=3, num_layers=3, 
                 encoder_dropout=0.0, lstm_dropout=0.0, output_dropout=0.0):
        super(UpdatedThermalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # External encoder: Time, Input Temperature (2 维)
        self.external_encoder = nn.Sequential(
            nn.Linear(2, 32), 
            nn.ReLU(), 
            nn.Dropout(encoder_dropout),
            nn.Linear(32, 32),
            nn.Dropout(encoder_dropout)
        )
        
        # State encoder: T_outer, T_inner, T_avg, dT_avg, dInput Temp (5 维)
        self.state_encoder = nn.Sequential(
            nn.Linear(5, 32), 
            nn.ReLU(), 
            nn.Dropout(encoder_dropout),
            nn.Linear(32, 32),
            nn.Dropout(encoder_dropout)
        )

        # LSTM with dropout between layers (only applies when num_layers > 1)
        # This helps prevent the model from memorizing specific sequences
        self.lstm = nn.LSTM(64, hidden_size, num_layers, batch_first=True, dropout=lstm_dropout if num_layers > 1 else 0)

        # Output network with dropout
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(output_dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

    # The 'hidden' parameter is a tuple (h, c) ---
    def forward(self, x, hidden_cell):
        batch_size, seq_len, _ = x.shape
        external = x[:, :, :2]
        state = x[:, :, 2:]
        external_encoded = self.external_encoder(external)
        state_encoded = self.state_encoder(state)
        combined = torch.cat([external_encoded, state_encoded], dim=-1)

        # LSTM takes and returns a tuple of (hidden_state, cell_state)
        out, hidden_cell = self.lstm(combined, hidden_cell)
        
        output = self.output_net(out)
        return output, hidden_cell

    # init_hidden returns a TUPLE of two tensors ---
    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        cell_state = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (hidden_state, cell_state)


# === Loss function ===
def thermal_loss(predictions, targets, temp_weights=torch.tensor([1.0, 1.0, 1.0]), 
                 loss_type='mae', huber_delta=1.0, mse_weight=0.0, mask=None):
    """
    改进的损失函数，支持多种损失类型和组合；支持 mask 用于变长序列（只对有效步求平均）。
    """
    temp_weights = temp_weights.to(predictions.device)
    error = predictions - targets

    def _mean(x):
        if mask is not None:
            m = mask.to(x.device)
            # 保证 m 与 x 同形，避免 (B,3)*(B,3,1) 等导致 dim1 上 8 vs 3 的广播错误
            while m.dim() > x.dim():
                m = m.squeeze(-1)
            if m.shape != x.shape:
                m = m.expand(x.shape).clone()
            return (x * m).sum() / (m.sum() * x.size(-1) + 1e-8)
        return torch.mean(x)

    if loss_type == 'mae':
        loss = torch.abs(error) * temp_weights
        loss = _mean(loss)
    elif loss_type == 'mse':
        loss = (error ** 2) * temp_weights
        loss = _mean(loss)
    elif loss_type == 'huber':
        abs_error = torch.abs(error)
        quadratic = torch.clamp(abs_error, max=huber_delta)
        linear = abs_error - quadratic
        loss = (0.5 * quadratic ** 2 / huber_delta + linear) * temp_weights
        loss = _mean(loss)
    elif loss_type == 'combined':
        mae_loss = torch.abs(error) * temp_weights
        mse_loss = (error ** 2) * temp_weights
        loss = (1 - mse_weight) * _mean(mae_loss) + mse_weight * _mean(mse_loss)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Choose from 'mae', 'mse', 'huber', 'combined'")
    return loss


# === Learning Rate Scheduler with Warmup ===
class WarmupCosineAnnealingLR:
    """
    带Warmup的Cosine Annealing学习率调度器。
    先线性warmup，然后使用cosine annealing衰减。
    """
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr_ratio=0.01, 
                 initial_lr=None, warmup_start_lr=None):
        """
        Args:
            optimizer: 优化器
            warmup_epochs: Warmup的epoch数
            max_epochs: 总训练epoch数
            min_lr_ratio: 最小学习率相对于初始学习率的比例
            initial_lr: 初始学习率（如果None则使用optimizer的lr）
            warmup_start_lr: Warmup起始学习率（如果None则为initial_lr * 0.1）
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.initial_lr = initial_lr if initial_lr is not None else optimizer.param_groups[0]['lr']
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr is not None else self.initial_lr * 0.1
        self.min_lr = self.initial_lr * min_lr_ratio
        self.current_epoch = 0
        
    def step(self, epoch=None):
        """更新学习率"""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        if self.current_epoch < self.warmup_epochs:
            # Warmup阶段：线性增加学习率
            lr = self.warmup_start_lr + (self.initial_lr - self.warmup_start_lr) * \
                 (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine Annealing阶段
            progress = (self.current_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.initial_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_last_lr(self):
        """获取当前学习率"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


class WarmupStepLR:
    """
    带Warmup的Step学习率调度器。
    先线性warmup，然后按步长衰减。
    """
    def __init__(self, optimizer, warmup_epochs, step_size, gamma=0.1,
                 initial_lr=None, warmup_start_lr=None):
        """
        Args:
            optimizer: 优化器
            warmup_epochs: Warmup的epoch数
            step_size: 学习率衰减的步长（每step_size个epoch衰减一次）
            gamma: 衰减因子
            initial_lr: 初始学习率
            warmup_start_lr: Warmup起始学习率
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.step_size = step_size
        self.gamma = gamma
        self.initial_lr = initial_lr if initial_lr is not None else optimizer.param_groups[0]['lr']
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr is not None else self.initial_lr * 0.1
        self.current_epoch = 0
        self.base_lr = self.initial_lr
        
    def step(self, epoch=None):
        """更新学习率"""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        if self.current_epoch < self.warmup_epochs:
            # Warmup阶段
            lr = self.warmup_start_lr + (self.initial_lr - self.warmup_start_lr) * \
                 (self.current_epoch / self.warmup_epochs)
            self.base_lr = self.initial_lr
        else:
            # Step衰减阶段
            steps = (self.current_epoch - self.warmup_epochs) // self.step_size
            lr = self.base_lr * (self.gamma ** steps)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_last_lr(self):
        """获取当前学习率"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


# === Training function ===
def train_updated_model(encoder_dropout=0.0, lstm_dropout=0.0, output_dropout=0.0,
                       loss_type='mae', loss_temp_weights=None, huber_delta=1.0, mse_weight=0.0,
                       lr_scheduler_type='plateau', initial_lr=0.001, warmup_epochs=10,
                       lr_patience=15, lr_factor=0.7, lr_step_size=50, lr_gamma=0.1,
                       burn_in_steps=30, teacher_forcing_ratio=1):
    """
    Train the LSTM model with step-by-step autoregressive training (optional burn-in + teacher forcing).
    
    Args:
        encoder_dropout: Dropout rate for encoder networks (default: 0.0)
        lstm_dropout: Dropout rate between LSTM layers (default: 0.0)
        output_dropout: Dropout rate for output network (default: 0.0)
        
        loss_type: 损失函数类型 - 'mae', 'mse', 'huber', 'combined' (default: 'mae')
        loss_temp_weights: 三个温度值的权重 [T_outer, T_inner, T_avg] (default: [1.0, 1.0, 1.0])
        huber_delta: Huber Loss的delta参数 (default: 1.0)
        mse_weight: 组合损失中MSE的权重，MAE权重=1-mse_weight (default: 0.0)
        
        lr_scheduler_type: 学习率调度器类型 - 'plateau', 'cosine', 'step' (default: 'plateau')
        initial_lr: 初始学习率 (default: 0.001)
        warmup_epochs: Warmup的epoch数，仅用于'cosine'和'step' (default: 10)
        lr_patience: ReduceLROnPlateau的patience (default: 15)
        lr_factor: 学习率衰减因子 (default: 0.7)
        lr_step_size: StepLR的步长 (default: 50)
        lr_gamma: StepLR的衰减因子 (default: 0.1)
        
        burn_in_steps: 前若干步只预热隐状态、不参与 loss (default: 30)，0 表示不 burn-in
        teacher_forcing_ratio: 训练时下一步状态用真实值的概率 (default: 1)，0=全用预测、1=全用真实，验证时始终用真实值
    
    Returns:
        model: Trained model
        test_datasets: Test datasets
        script_dir: Script directory path
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    # 与 GRU 一致：项目根目录（LSTM.py 在根目录时 project_root = script_dir）
    project_root = script_dir
    data_dir = os.path.join(project_root, "data")
    fixed_data_dir = data_dir
    fixed_test_dir = os.path.join(project_root, "data", "test")

    # 与 GRU 一致：优先 10s_with_burn_in / test/test_in_10s，回退到 data_in_10s / test_in_10s
    use_burn_in_dir = True
    if use_burn_in_dir:
        train_dir = os.path.join(fixed_data_dir, "10s_with_burn_in")
        test_dir = os.path.join(fixed_test_dir, "test_in_10s")
    else:
        train_dir = os.path.join(fixed_data_dir, "10s")
        test_dir = os.path.join(fixed_test_dir, "test_in_10s")
    if not os.path.isdir(train_dir):
        train_dir = os.path.join(data_dir, "data_in_10s")
    if not os.path.isdir(test_dir):
        test_dir = os.path.join(data_dir, "test_in_10s")

    train_paths = sorted(glob.glob(os.path.join(train_dir, "**", "*.csv"), recursive=True))
    test_paths = sorted(glob.glob(os.path.join(test_dir, "**", "*.csv"), recursive=True))

    if not train_paths:
        raise FileNotFoundError("No training files found. Expected: " + train_dir)

    print(f"Train dir: {train_dir} -> {len(train_paths)} files")
    print(f"Test dir:  {test_dir} -> {len(test_paths)} files")

    # 与 GRU 一致：验证集取训练文件的前 5%
    val_split = int(0.05 * len(train_paths))
    val_paths = train_paths[:val_split]
    actual_train_paths = train_paths[val_split:]

    # Create a unified scaler
    train_dfs = []
    for path in actual_train_paths:
        try:
            df = pd.read_csv(path)
            if 'T_ave (C)' in df.columns and 'T_avg (C)' not in df.columns:
                df = df.rename(columns={'T_ave (C)': 'T_avg (C)'})
            if 'Input Temperature (C)' not in df.columns and 'T_inner (C)' in df.columns:
                df['Input Temperature (C)'] = df['T_inner (C)']
            required_cols = ['Time (s)', 'T_outer (C)', 'T_inner (C)', 'T_avg (C)', 'Input Temperature (C)']
            if all(col in df.columns for col in required_cols):
                train_dfs.append(df)
        except Exception as e:
            print(f"Skipping file {path} due to error: {e}")
            continue

    if not train_dfs:
        raise ValueError("No valid training data files to create scaler.")

    scaler = MinMaxScaler()
    scaler_cols = ['Time (s)', 'T_outer (C)', 'T_inner (C)', 'T_avg (C)', 'Input Temperature (C)']
    scaler.fit(pd.concat(train_dfs)[scaler_cols])

    # Create datasets；测试集在序列前重复第一条 30 步用于预热，预测从“原始第一条”起对齐
    train_datasets = [UpdatedThermalDataset(p, scaler) for p in actual_train_paths]
    val_datasets = [UpdatedThermalDataset(p, scaler) for p in val_paths]
    test_datasets = [UpdatedThermalDataset(p, scaler, prepend_burn_in=30) for p in test_paths]

    train_loader = DataLoader(
        ConcatDataset(train_datasets), batch_size=32, shuffle=True,
        collate_fn=_collate_thermal_variable_length
    )
    val_loader = DataLoader(
        ConcatDataset(val_datasets), batch_size=32, collate_fn=_collate_thermal_variable_length
    ) if val_datasets else None

    # Model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate the LSTM model with dropout parameters
    # Dropout helps reduce overfitting by randomly zeroing neurons during training
    # This prevents the model from memorizing specific sequences
    model = UpdatedThermalLSTM(
        encoder_dropout=encoder_dropout,
        lstm_dropout=lstm_dropout,
        output_dropout=output_dropout
    ).to(device)

    # Setup optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    
    # Setup learning rate scheduler based on type
    if lr_scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=lr_factor, patience=lr_patience
        )
    elif lr_scheduler_type == 'cosine':
        scheduler = WarmupCosineAnnealingLR(
            optimizer, warmup_epochs=warmup_epochs, max_epochs=300, 
            initial_lr=initial_lr
        )
    elif lr_scheduler_type == 'step':
        scheduler = WarmupStepLR(
            optimizer, warmup_epochs=warmup_epochs, step_size=lr_step_size, 
            gamma=lr_gamma, initial_lr=initial_lr
        )
    else:
        raise ValueError(f"Unknown lr_scheduler_type: {lr_scheduler_type}. Choose from 'plateau', 'cosine', 'step'")
    
    # Setup loss function parameters
    if loss_temp_weights is None:
        loss_temp_weights = torch.tensor([1.0, 1.0, 1.0])
    else:
        loss_temp_weights = torch.tensor(loss_temp_weights)
    
    print(f"Using device: {device}")
    print(f"Dropout settings - Encoder: {encoder_dropout}, LSTM: {lstm_dropout}, Output: {output_dropout}")
    print(f"Loss function: {loss_type} (temp_weights={loss_temp_weights.tolist()}, huber_delta={huber_delta}, mse_weight={mse_weight})")
    print(f"Learning rate scheduler: {lr_scheduler_type} (initial_lr={initial_lr}, warmup_epochs={warmup_epochs if lr_scheduler_type != 'plateau' else 0})")
    print(f"Autoregressive: burn_in_steps={burn_in_steps}, teacher_forcing_ratio={teacher_forcing_ratio}")

    num_epochs, best_val_loss, early_stop_counter, patience = 300, float('inf'), 0, 30

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        n_train_batches = 0
        for X, Y, _, _, _, mask in train_loader:
            X, Y = X.to(device), Y.to(device)
            mask = mask.to(device)
            B, T, _ = X.shape
            if T < 2:
                continue
            hidden = model.init_hidden(B)
            optimizer.zero_grad()
            # Burn-in: 前 burn_in_steps 步只跑前向、更新隐状态，不参与 loss
            start_t = 0
            if burn_in_steps > 0 and T > burn_in_steps:
                with torch.no_grad():
                    _, hidden = model(X[:, :burn_in_steps], hidden)
                start_t = burn_in_steps
            current_state = X[:, start_t, 2:5].clone()  # (B, 3) T_outer, T_inner, T_avg
            batch_loss = 0.0
            n_steps = 0
            for t in range(start_t, T):
                input_t = torch.cat([X[:, t, :2], current_state, X[:, t, 5:7]], dim=-1).unsqueeze(1)  # (B, 1, 7)
                pred, hidden = model(input_t, hidden)
                pred = pred.squeeze(1)  # (B, 3)；若为 (1,B,3) 则再 squeeze(0)
                if pred.dim() == 3:
                    pred = pred.squeeze(0)
                mask_t = mask[:, t].unsqueeze(1).expand(-1, 3)
                step_loss = thermal_loss(
                    pred, Y[:, t],
                    temp_weights=loss_temp_weights,
                    loss_type=loss_type,
                    huber_delta=huber_delta,
                    mse_weight=mse_weight,
                    mask=mask_t,
                )
                batch_loss = batch_loss + step_loss
                n_steps += 1
                if t < T - 1:
                    use_tf = torch.rand(1).item() < teacher_forcing_ratio
                    new_state = X[:, t + 1, 2:5].clone() if use_tf else pred.detach().clone()
                    mask_next = mask[:, t + 1].unsqueeze(1).expand(-1, 3)
                    current_state = torch.where(mask_next.bool(), new_state, current_state)
            if n_steps > 0:
                final_loss = batch_loss / n_steps
                final_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_train_loss += final_loss.item()
                n_train_batches += 1

        val_loss = 0.0
        if val_loader:
            model.eval()
            with torch.no_grad():
                for X, Y, _, _, _, mask in val_loader:
                    X, Y = X.to(device), Y.to(device)
                    mask = mask.to(device)
                    B, T, _ = X.shape
                    if T < 2:
                        continue
                    hidden = model.init_hidden(B)
                    start_t = 0
                    if burn_in_steps > 0 and T > burn_in_steps:
                        _, hidden = model(X[:, :burn_in_steps], hidden)
                        start_t = burn_in_steps
                    current_state = X[:, start_t, 2:5].clone()
                    batch_val_loss = 0.0
                    n_steps = 0
                    for t in range(start_t, T):
                        input_t = torch.cat([X[:, t, :2], current_state, X[:, t, 5:7]], dim=-1).unsqueeze(1)
                        pred, hidden = model(input_t, hidden)
                        pred = pred.squeeze(1)
                        if pred.dim() == 3:
                            pred = pred.squeeze(0)
                        mask_t = mask[:, t].unsqueeze(1).expand(-1, 3)
                        batch_val_loss += thermal_loss(
                            pred, Y[:, t],
                            temp_weights=loss_temp_weights,
                            loss_type=loss_type,
                            huber_delta=huber_delta,
                            mse_weight=mse_weight,
                            mask=mask_t,
                        ).item()
                        n_steps += 1
                        if t < T - 1:
                            current_state = X[:, t + 1, 2:5].clone()
                    if n_steps > 0:
                        val_loss += batch_val_loss / n_steps
                val_loss /= len(val_loader)
        
        # Update learning rate scheduler based on type
        avg_train_loss = total_train_loss / max(1, n_train_batches)
        if not val_loader:
            val_loss = avg_train_loss
        if lr_scheduler_type == 'plateau':
            # ReduceLROnPlateau uses validation loss if available, otherwise training loss
            scheduler.step(val_loss if val_loader else avg_train_loss)
            current_lr = optimizer.param_groups[0]['lr']
        else:
            # WarmupCosineAnnealingLR and WarmupStepLR update based on epoch
            current_lr = scheduler.step(epoch)
        
        print(f"[Epoch {epoch + 1:3d}] Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(script_dir, "updated_best_lstm.pth"))
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(torch.load(os.path.join(script_dir, "updated_best_lstm.pth")))
    # --- CHANGE 1: Return script_dir for use in the main block ---
    return model, test_datasets, script_dir


# --- CHANGE 2: Add plot_save_dir as an argument ---
def test_updated_model(model, test_datasets, plot_save_dir, burn_in_steps=30):
    """
    逐步自回归预测：先 burn-in 预热隐状态，之后每步只用上一步的预测作为当前状态输入。
    """
    if not model or not test_datasets:
        print("Model or dataset is empty")
        return
    
    # --- CHANGE 3: Create the plot directory if it doesn't exist ---
    os.makedirs(plot_save_dir, exist_ok=True)
    csv_save_dir = plot_save_dir.rstrip(os.sep) + "_csv"
    os.makedirs(csv_save_dir, exist_ok=True)
    print(f"Plots will be saved to: {plot_save_dir}")
    print(f"Prediction CSV will be saved to: {csv_save_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    print(f"\nStart testing {len(test_datasets)} files (burn_in_steps={burn_in_steps}, autoregressive)...")

    for dataset in test_datasets:
        try:
            file_name = dataset.file_name
            X, Y, external_conditions, time_values, full_data = dataset[0]

            X = X.to(device)
            L = X.size(0)
            if L < 2:
                continue
            hidden = model.init_hidden(1)
            predictions = []

            with torch.no_grad():
                start_t = 0
                if burn_in_steps > 0 and L > burn_in_steps:
                    _, hidden = model(X[:burn_in_steps].unsqueeze(0), hidden)
                    start_t = burn_in_steps
                current_state = X[start_t, 2:5].clone()
                for t in range(start_t, L):
                    input_t = torch.cat([X[t, :2], current_state, X[t, 5:7]], dim=-1).unsqueeze(0).unsqueeze(0)
                    pred, hidden = model(input_t, hidden)
                    pred_state = pred[0, 0]
                    predictions.append(pred_state.cpu().numpy())
                    if t < L - 1:
                        current_state = pred_state.clone()

            if not predictions:
                print("No predictions generated")
                continue

            pred_array = np.array(predictions)
            n_pred = len(pred_array)
            prepend = getattr(dataset, 'prepend_burn_in', 0)
            if prepend > 0:
                # 测试集前 30 步是重复的第一条，只画原始区间；预测从原始第 1 步起
                R = len(full_data) - prepend
                dummy_pred = np.zeros((n_pred, 5))
                dummy_pred[:, 0] = full_data[prepend + 1:prepend + 1 + n_pred, 0]
                dummy_pred[:, 1:4] = pred_array
                dummy_pred[:, 4] = full_data[prepend + 1:prepend + 1 + n_pred, 4]
                inv_pred = dataset.scaler.inverse_transform(dummy_pred)
                full_data_original = full_data[prepend:prepend + R]
                inv_actual = dataset.scaler.inverse_transform(full_data_original)
                actual_temps = inv_actual[1:R, 1:4]
            else:
                dummy_pred = np.zeros((n_pred, 5))
                dummy_pred[:, 0] = full_data[start_t + 1:start_t + 1 + n_pred, 0]
                dummy_pred[:, 1:4] = pred_array
                dummy_pred[:, 4] = full_data[start_t + 1:start_t + 1 + n_pred, 4]
                inv_pred = dataset.scaler.inverse_transform(dummy_pred)
                inv_actual = dataset.scaler.inverse_transform(full_data)
                actual_temps = inv_actual[start_t + 1:start_t + 1 + n_pred, 1:4]
            pred_temps = inv_pred[:, 1:4]

            # 输出预测表为 CSV（时间、真实三温、预测三温）
            csv_df = pd.DataFrame({
                "Time (s)": inv_pred[:, 0],
                "T_outer_Actual (C)": actual_temps[:, 0],
                "T_inner_Actual (C)": actual_temps[:, 1],
                "T_avg_Actual (C)": actual_temps[:, 2],
                "T_outer_Pred (C)": pred_temps[:, 0],
                "T_inner_Pred (C)": pred_temps[:, 1],
                "T_avg_Pred (C)": pred_temps[:, 2],
            })
            csv_name = os.path.splitext(file_name)[0] + "_predictions.csv"
            csv_path = os.path.join(csv_save_dir, csv_name)
            csv_df.to_csv(csv_path, index=False)
            
            # Calculate R² scores
            r2_outer = r2_score(actual_temps[:, 0], pred_temps[:, 0])
            r2_inner = r2_score(actual_temps[:, 1], pred_temps[:, 1])
            r2_avg   = r2_score(actual_temps[:, 2], pred_temps[:, 2])

            # Calculate MAE scores
            mae_outer = mean_absolute_error(actual_temps[:, 0], pred_temps[:, 0])
            mae_inner = mean_absolute_error(actual_temps[:, 1], pred_temps[:, 1])
            mae_avg   = mean_absolute_error(actual_temps[:, 2], pred_temps[:, 2])

            # Plot results（若做了 prepend，inv_actual 已是原始 R 点，无重复段）
            plt.figure(figsize=(15, 8))

            plt.plot(inv_actual[:, 0], inv_actual[:, 1], 'b-', label='T_outer Actual', linewidth=2)
            plt.plot(inv_actual[:, 0], inv_actual[:, 2], 'r-', label='T_inner Actual', linewidth=2)
            plt.plot(inv_actual[:, 0], inv_actual[:, 3], 'g-', label='T_avg Actual', linewidth=2)

            pred_times = inv_pred[:, 0]
            plt.plot(pred_times, inv_pred[:, 1], 'b--', label=f'T_outer Pred (MAE={mae_outer:.2f}, R²={r2_outer:.3f})', linewidth=2, alpha=0.8)
            plt.plot(pred_times, inv_pred[:, 2], 'r--', label=f'T_inner Pred (MAE={mae_inner:.2f}, R²={r2_inner:.3f})', linewidth=2, alpha=0.8)
            plt.plot(pred_times, inv_pred[:, 3], 'g--', label=f'T_avg Pred (MAE={mae_avg:.2f}, R²={r2_avg:.3f})', linewidth=2, alpha=0.8)

            plt.plot(inv_actual[:, 0], inv_actual[:, 4], 'k-.', label='Input Temperature', linewidth=2)
            
            plt.xlabel('Time (s)')
            plt.ylabel('Temperature (°C)')
            plt.title(f'Temperature Prediction and Input - {file_name}')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.5)
            plt.tight_layout()

            # --- CHANGE 4: Save the figure instead of showing it ---
            plot_filename = os.path.splitext(file_name)[0] + '.png'
            full_plot_path = os.path.join(plot_save_dir, plot_filename)
            plt.savefig(full_plot_path, bbox_inches='tight')
            
            # --- CHANGE 5: Close the plot to free memory ---
            plt.close()

        except Exception as e:
            print(f"Error testing file {dataset.file_name}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\nFinished testing {len(test_datasets)} files...")

# === Main function ===
if __name__ == "__main__":
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
        result_dir = os.path.join(script_dir, "result")
        os.makedirs(result_dir, exist_ok=True)
        dropout_list = [0, 0.1, 0.2, 0.3]
        for dropout in dropout_list:
            print(f"\n{'='*60}\n>>> Training & testing with dropout = {dropout}\n{'='*60}")
            model, test_datasets, script_dir = train_updated_model(
                encoder_dropout=dropout,
                lstm_dropout=dropout,
                output_dropout=dropout,
            )
            subdir_name = "dropout_0" if dropout == 0 else f"dropout_{dropout}"
            plot_save_dir = os.path.join(result_dir, subdir_name)
            os.makedirs(plot_save_dir, exist_ok=True)
            if model and test_datasets:
                test_updated_model(model, test_datasets, plot_save_dir)
            elif model:
                print("No test set; skip testing.")
        print(f"\nAll results saved under: {result_dir}")

    except Exception as e:
        print(f"Program error: {e}")
        import traceback; traceback.print_exc()
