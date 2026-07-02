"""Encoders, the stacked-GRU model, and the loss."""
import torch
import torch.nn as nn

from .config import *


# ============================================================
# 3. Encoders
# ============================================================
class InitStateEncoder(nn.Module):
    """4-d t=0 obs -> h0 (L, B, H). Same as V3/V4."""

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
    """Updated InitStateEncoder: maps a fixed initial window (W, 4) -> h0 (L, B, H).

    Unlike the V3/V4 point-wise InitStateEncoder (which uses only the t=0
    observation), this version processes the first W observations through a
    small GRU encoder before producing h0 for the main GRU.

    NOTE: this is a FIXED initial window -- it is applied exactly once at the
    start of the rollout to produce h0, not recomputed at each step.
    Conceptually it's a trainable counterpart of V2's burn-in (an internal
    GRU warm-up over the first W steps) bolted onto the V3 InitStateEncoder
    contract: returns an h0 of the same shape and is used only at the start
    of the autoregressive rollout.
    """

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
_DEFAULT_WEIGHTS = torch.tensor([1.0, 1.0, 1.0])


def weighted_loss(predictions, targets, weights=None):
    if weights is None:
        weights = _DEFAULT_WEIGHTS
    weights = weights.to(predictions.device)
    return torch.mean(torch.abs(predictions - targets) * weights)


