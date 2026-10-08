"""Encoders, the stacked-GRU model, and the loss functions."""
import torch
import torch.nn as nn

from .config import *


# ============================================================
# 3. Encoders
# ============================================================
class InitStateEncoder(nn.Module):
    """4-d t=0 observation -> h0 (num_layers, batch_size, hidden_size)."""

    def __init__(self, obs_size=4, hidden_size=128, num_layers=5):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
            nn.Linear(obs_size, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, num_layers * hidden_size),
        )

    def forward(self, x0):
        h = self.encoder(x0).view(-1, self.num_layers, self.hidden_size)
        return h.permute(1, 0, 2).contiguous()


class WindowInitStateEncoder(nn.Module):
    """Maps a fixed initial window (W, 4) -> h0 (num_layers, batch_size, hidden_size)."""

    def __init__(self, obs_size=4, window_size=WINDOW_SIZE,
                 hidden_size=128, num_layers=5,
                 enc_layers=2, enc_hidden=128):
        super().__init__()
        self.window_size = window_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.encoder_gru = nn.GRU(input_size=obs_size, hidden_size=enc_hidden,
                                  num_layers=enc_layers, batch_first=True)
        self.proj = nn.Linear(enc_hidden, num_layers * hidden_size)

    def forward(self, x_window):
        out, _ = self.encoder_gru(x_window)
        last = out[:, -1, :]
        h = self.proj(last).view(-1, self.num_layers, self.hidden_size)
        return h.permute(1, 0, 2).contiguous()


# ============================================================
# 4. Model
# ============================================================
class ThermalGRU(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=3,
                 num_layers=5, dropout=0.3,
                 encoder='point', window_size=WINDOW_SIZE):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.encoder_kind = encoder

        self.gru = nn.GRU(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

        if encoder == 'point':
            self.init_encoder = InitStateEncoder(
                obs_size=4, hidden_size=hidden_size, num_layers=num_layers,
            )
        elif encoder == 'window':
            self.init_encoder = WindowInitStateEncoder(
                obs_size=4, window_size=window_size,
                hidden_size=hidden_size, num_layers=num_layers,
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder}")

    def init_hidden(self, batch_size, x0=None, x_window=None):
        device = next(self.parameters()).device
        if self.encoder_kind == 'point':
            assert x0 is not None, "point encoder needs x0"
            return self.init_encoder(x0.to(device))
        else:
            assert x_window is not None, "window encoder needs x_window"
            return self.init_encoder(x_window.to(device))

    def forward(self, x, hidden):
        out, hidden = self.gru(x, hidden)
        return self.fc(out), hidden


# ============================================================
# 5. Loss
# ============================================================
# Channel order is (T_inner, T_outer, T_avg) -- see VARIANT_OUTPUT_CHANNELS.
# Weights come from config (default 1/6/3, mirroring the LSTM line's recipe).
_DEFAULT_WEIGHTS = torch.tensor(LOSS_WEIGHTS)


def weighted_loss(predictions, targets, weights=None, raw_affine=None, model=None):
    """Per-channel weighted L1 + Early-Horizon Transient Loss + PINN Smoothness.

    predictions / targets are in the MinMax-SCALED space.
    """
    if weights is None:
        weights = _DEFAULT_WEIGHTS
    weights = weights.to(predictions.device)
    
    # Dynamic Weight Annealing (DWA) task coefficients
    use_dwa = getattr(model, 'use_dwa', False)
    dwa_w1 = getattr(model, 'dwa_w1', 1.0) if use_dwa else 1.0
    dwa_w2 = getattr(model, 'dwa_w2', 1.0) if use_dwa else 1.0
    dwa_w3 = getattr(model, 'dwa_w3', 1.0) if use_dwa else 1.0

    # ------------------------------------------------------------
    # 1. Primary Data Loss with Transient Penalty
    # ------------------------------------------------------------
    if predictions.ndim >= 3:
        diff = torch.abs(predictions - targets) * weights.view(1, 1, -1)
        seq_len = predictions.size(1)
        t_steps = torch.arange(seq_len, device=predictions.device, dtype=torch.float32)

        # Arm L: Channel-Decoupled Early Loss Multiplier
        if EARLY_CH_WEIGHTS is not None and EARLY_CH_TAUS is not None:
            w_ch = torch.tensor(EARLY_CH_WEIGHTS, device=predictions.device, dtype=torch.float32)
            tau_ch = torch.tensor(EARLY_CH_TAUS, device=predictions.device, dtype=torch.float32)
            # temp_scale shape: (seq_len, 3) -> broadcast to (1, seq_len, 3)
            temp_scale = 1.0 + (w_ch - 1.0) * torch.exp(-t_steps.unsqueeze(1) / torch.clamp(tau_ch, min=1e-4))
            diff = diff * temp_scale.unsqueeze(0)

        # Arm I & Standard Early Loss Multiplier (Uniform Tau)
        elif EARLY_LOSS_WEIGHT > 1.0:
            temp_scale = 1.0 + (EARLY_LOSS_WEIGHT - 1.0) * torch.exp(-t_steps / max(EARLY_LOSS_TAU, 1e-4))
            diff = diff * temp_scale.view(1, -1, 1)
            
        loss_data_val = torch.mean(diff)
    else:
        loss_data_val = torch.mean(torch.abs(predictions - targets) * weights)

    loss = dwa_w1 * loss_data_val

    # ------------------------------------------------------------
    # 2. Arm M: Early Transient Derivative/Slope Matching
    # ------------------------------------------------------------
    if EARLY_DERIV_WEIGHT > 0.0 and predictions.ndim >= 3 and predictions.size(1) > 1:
        diff_pred = predictions[:, 1:, :] - predictions[:, :-1, :]
        diff_targ = targets[:, 1:, :] - targets[:, :-1, :]
        t_deriv_steps = torch.arange(diff_pred.size(1), device=predictions.device, dtype=torch.float32)
        deriv_scale = torch.exp(-t_deriv_steps / max(EARLY_DERIV_TAU, 1e-4)).view(1, -1, 1)
        loss_deriv = torch.mean(torch.abs(diff_pred - diff_targ) * deriv_scale * weights.view(1, 1, -1))
        loss = loss + EARLY_DERIV_WEIGHT * loss_deriv

    # ------------------------------------------------------------
    # 3. PINN Physical Regularizations (In Physical Celsius Space)
    # ------------------------------------------------------------
    loss_phys_tracked = 0.0          # energy-residual raw value (Z4)
    loss_smooth_to_tracked = 0.0
    loss_smooth_ta_tracked = 0.0
    # Sum of all PINN terms at their BASE weights (before the balancing scale
    # k). The actual contribution to `loss` is pinn_scale * pinn_base.
    pinn_base = predictions.new_zeros(())
    pinn_scale = float(getattr(model, 'pinn_scale', 1.0))

    if raw_affine is not None and predictions.ndim >= 3:
        (s_in, m_in), (s_out, m_out), (s_avg, m_avg) = raw_affine

        # Map predictions from normalized [0, 1] space to Celsius physical space
        t_in = (predictions[..., 0] - m_in) / s_in
        t_out = (predictions[..., 1] - m_out) / s_out
        t_avg = (predictions[..., 2] - m_avg) / s_avg

        # Standard PINN Smoothness (Arm F / G) -- DWA-weighted, not balanced
        if SMOOTH_TO_WEIGHT > 0.0 and predictions.size(1) > 1:
            diff_to = t_out[:, 1:] - t_out[:, :-1]
            loss_smooth_to_val = torch.mean(diff_to ** 2)
            loss = loss + dwa_w2 * SMOOTH_TO_WEIGHT * loss_smooth_to_val
            loss_smooth_to_tracked = float(loss_smooth_to_val.item())

        if SMOOTH_TA_WEIGHT > 0.0 and predictions.size(1) > 1:
            diff_ta = t_avg[:, 1:] - t_avg[:, :-1]
            loss_smooth_ta_val = torch.mean(diff_ta ** 2)
            loss = loss + dwa_w3 * SMOOTH_TA_WEIGHT * loss_smooth_ta_val
            loss_smooth_ta_tracked = float(loss_smooth_ta_val.item())

        # Arm O: Transient-Localized PINN Smoothness (2nd-Order Curvature Damping)
        if (EARLY_SMOOTH_TO_WEIGHT > 0.0 or EARLY_SMOOTH_TA_WEIGHT > 0.0) and predictions.size(1) > 2:
            t_pinn_steps = torch.arange(predictions.size(1) - 2, device=predictions.device, dtype=torch.float32)
            pinn_decay = torch.exp(-t_pinn_steps / max(EARLY_SMOOTH_TAU, 1e-4)).view(1, -1)

            def _decayed(x):
                # "mean": legacy -- averaged over the full horizon (~220x dilution).
                # "weighted": true weighted mean over the decay window.
                if PINN_NORM == "weighted":
                    return torch.sum(x * pinn_decay) / (x.size(0) * pinn_decay.sum())
                return torch.mean(x * pinn_decay)

            if EARLY_SMOOTH_TO_WEIGHT > 0.0:
                d2_to = t_out[:, 2:] - 2 * t_out[:, 1:-1] + t_out[:, :-2]
                loss_early_smooth_to = _decayed(d2_to ** 2)
                pinn_base = pinn_base + EARLY_SMOOTH_TO_WEIGHT * loss_early_smooth_to
                loss_smooth_to_tracked += float(loss_early_smooth_to.item())

            if EARLY_SMOOTH_TA_WEIGHT > 0.0:
                d2_ta = t_avg[:, 2:] - 2 * t_avg[:, 1:-1] + t_avg[:, :-2]
                loss_early_smooth_ta = _decayed(d2_ta ** 2)
                pinn_base = pinn_base + EARLY_SMOOTH_TA_WEIGHT * loss_early_smooth_ta
                loss_smooth_ta_tracked += float(loss_early_smooth_ta.item())

        # Arm Z4: lumped energy-balance residual on T_avg (full horizon).
        energy = getattr(model, 'energy_coef', None)
        if PINN_ENERGY_WEIGHT > 0.0 and energy is not None and predictions.size(1) > 1:
            a, b, c, sigma = energy
            d_ta = t_avg[:, 1:] - t_avg[:, :-1]
            drive = (a * (t_in[:, :-1] - t_avg[:, :-1])
                     + b * (t_out[:, :-1] - t_avg[:, :-1]) + c)
            loss_energy = torch.mean((d_ta - drive) ** 2) / (sigma ** 2)
            pinn_base = pinn_base + PINN_ENERGY_WEIGHT * loss_energy
            loss_phys_tracked = float(loss_energy.item())

        loss = loss + pinn_scale * pinn_base

    # Attach diagnostic metrics to the returned loss tensor
    loss.loss_data = float(loss_data_val.item())
    loss.loss_phys = loss_phys_tracked
    loss.loss_smooth_to = loss_smooth_to_tracked
    loss.loss_smooth_ta = loss_smooth_ta_tracked
    loss.loss_pinn_base = float(pinn_base.item())
    loss.loss_pinn = pinn_scale * loss.loss_pinn_base

    return loss
