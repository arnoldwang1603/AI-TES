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


class UpdatedThermalDataset(Dataset):
    def __init__(self, csv_file, scaler=None):
        self.file_name = os.path.basename(csv_file)
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

        grouped = df.groupby("FileName")
        self.X, self.Y, self.external_conditions, self.time_values, self.full_data = [], [], [], [], []

        for _, group in grouped:
            external_seq = group[["Time (s)", "Input Temperature (C)"]].values
            X_seq = group[["Time (s)", "Input Temperature (C)", "T_outer (C)", "T_inner (C)", "T_avg (C)"]].values[:-1]
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
    def __init__(self, input_size=5, hidden_size=128, output_size=3, num_layers=3):
        super(UpdatedThermalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.external_encoder = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32))
        self.state_encoder = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 32))

        # LSTM
        self.lstm = nn.LSTM(64, hidden_size, num_layers, batch_first=True, dropout=0.1)

        self.output_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
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
def thermal_loss(predictions, targets, temp_weights=torch.tensor([1.0, 1.0, 1.0])):
    temp_weights = temp_weights.to(predictions.device)
    loss = torch.abs(predictions - targets) * temp_weights
    return torch.mean(loss)


# === Training function ===
def train_updated_model():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    possible_data_dirs = [os.path.join(script_dir, "..", "..", "data"), os.path.join(script_dir, "data"), "data"]
    data_dir = next((d for d in possible_data_dirs if os.path.exists(d)), None)

    if not data_dir:
        raise FileNotFoundError("Data directory not found")

    train_dir = os.path.join(data_dir, "150s Time steps")
    test_dir = os.path.join(data_dir, "test")

    train_paths = sorted(glob.glob(os.path.join(train_dir, "**", "*.csv"), recursive=True))
    test_paths = sorted(glob.glob(os.path.join(test_dir, "*.csv")))
    
    if not train_paths:
        raise FileNotFoundError("No training files found.")

    val_split = max(1, int(0.1 * len(train_paths)))
    val_paths, actual_train_paths = train_paths[:val_split], train_paths[val_split:]

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

    # Create datasets
    train_datasets = [UpdatedThermalDataset(p, scaler) for p in actual_train_paths]
    val_datasets = [UpdatedThermalDataset(p, scaler) for p in val_paths]
    test_datasets = [UpdatedThermalDataset(p, scaler) for p in test_paths]

    train_loader = DataLoader(ConcatDataset(train_datasets), batch_size=8, shuffle=True)
    val_loader = DataLoader(ConcatDataset(val_datasets), batch_size=8) if val_datasets else None

    # Model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate the LSTM model ---
    model = UpdatedThermalLSTM().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=15)
    print(f"Using device: {device}")

    num_epochs, best_val_loss, early_stop_counter, patience = 300, float('inf'), 0, 30

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for X, Y, _, _, _ in train_loader:
            X, Y = X.to(device), Y.to(device)
            hidden = model.init_hidden(X.size(0))
            optimizer.zero_grad()
            predictions, hidden = model(X, hidden)
            loss = thermal_loss(predictions, Y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_train_loss += loss.item()

        val_loss = 0
        if val_loader:
            model.eval()
            with torch.no_grad():
                for X, Y, _, _, _ in val_loader:
                    X, Y = X.to(device), Y.to(device)
                    hidden = model.init_hidden(X.size(0))
                    predictions, hidden = model(X, hidden)
                    val_loss += thermal_loss(predictions, Y).item()
            val_loss /= len(val_loader)
            scheduler.step(val_loss)

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"[Epoch {epoch + 1:3d}] Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")

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
def test_updated_model(model, test_datasets, plot_save_dir):
    if not model or not test_datasets:
        print("Model or dataset is empty")
        return
    
    # --- CHANGE 3: Create the plot directory if it doesn't exist ---
    os.makedirs(plot_save_dir, exist_ok=True)
    print(f"Plots will be saved to: {plot_save_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    print(f"\nStart testing {len(test_datasets)} files...")

    for dataset in test_datasets:
        try:
            file_name = dataset.file_name
            X, Y, external_conditions, time_values, full_data = dataset[0]

            # print(f"\n=== Test file: {file_name} ===")

            # Prediction Logic
            X = X.to(device)
            external_conditions = external_conditions.to(device)
            hidden = model.init_hidden(1)
            predictions = []
            current_state = X[0, 2:].clone()

            with torch.no_grad():
                for t in range(len(external_conditions) - 1):
                    input_t = torch.cat([external_conditions[t], current_state]).unsqueeze(0).unsqueeze(0)
                    pred, hidden = model(input_t, hidden)
                    pred_state = pred[0, 0]
                    predictions.append(pred_state.cpu().numpy())
                    if t < len(X) - 1:
                        current_state = pred_state.clone() if t >= 3 else X[t + 1, 2:].clone()

            if not predictions:
                print("No predictions generated")
                continue

            pred_array = np.array(predictions)
            dummy_pred = np.zeros((len(pred_array), 5))
            dummy_pred[:, 0] = full_data[1:len(pred_array) + 1, 0]
            dummy_pred[:, 1:4] = pred_array
            dummy_pred[:, 4] = full_data[1:len(pred_array) + 1, 4]

            inv_pred = dataset.scaler.inverse_transform(dummy_pred)
            inv_actual = dataset.scaler.inverse_transform(full_data)
            
            # Align actual and predicted data for metrics
            actual_temps = inv_actual[1:len(inv_pred) + 1, 1:4]
            pred_temps = inv_pred[:, 1:4]
            
            # Calculate R² scores
            r2_outer = r2_score(actual_temps[:, 0], pred_temps[:, 0])
            r2_inner = r2_score(actual_temps[:, 1], pred_temps[:, 1])
            r2_avg   = r2_score(actual_temps[:, 2], pred_temps[:, 2])

            # Calculate MAE scores
            mae_outer = mean_absolute_error(actual_temps[:, 0], pred_temps[:, 0])
            mae_inner = mean_absolute_error(actual_temps[:, 1], pred_temps[:, 1])
            mae_avg   = mean_absolute_error(actual_temps[:, 2], pred_temps[:, 2])

            # Plot results
            plt.figure(figsize=(15, 8))

            plt.plot(inv_actual[:, 0], inv_actual[:, 1], 'b-', label='T_outer Actual', linewidth=2)
            plt.plot(inv_actual[:, 0], inv_actual[:, 2], 'r-', label='T_inner Actual', linewidth=2)
            plt.plot(inv_actual[:, 0], inv_actual[:, 3], 'g-', label='T_avg Actual', linewidth=2)
            
            pred_times = inv_actual[1:len(inv_pred) + 1, 0]
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
        # --- CHANGE 6: Capture the script_dir returned from training ---
        model, test_datasets, script_dir = train_updated_model()
        if model and test_datasets:
            # --- CHANGE 7: Define the target plot directory and pass it to the test function ---
            # Assuming src/models is the structure
            plot_save_dir = os.path.join(script_dir, 'src/plot_lstm_150')
            test_updated_model(model, test_datasets, plot_save_dir)

    except Exception as e:
        print(f"Program error: {e}")
        import traceback; traceback.print_exc()
