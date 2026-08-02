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
# Channel order is (T_inner, T_outer, T_avg) -- see VARIANT_OUTPUT_CHANNELS.
# Weights come from config (default 1/6/3, mirroring the LSTM line's recipe).
_DEFAULT_WEIGHTS = torch.tensor(LOSS_WEIGHTS)


def weighted_loss(predictions, targets, weights=None, raw_affine=None):
    """Per-channel weighted L1 + optional T_avg physics-bound hinge.

    predictions / targets are in the MinMax-SCALED space.

    raw_affine: optional ((s_in, m_in), (s_out, m_out), (s_avg, m_avg)) from
    the fitted scaler (scaled = raw * s + m). When given together with a
    non-zero PHYSICS_BOUND_WEIGHT, a hinge penalty is added for predictions
    that put T_avg outside [min(T_inner, T_outer), max(T_inner, T_outer)].
    The check MUST be done in raw temperature space: MinMax scaling is
    per-channel, so channel ordering is not preserved after scaling.
    """
    if weights is None:
        weights = _DEFAULT_WEIGHTS
    weights = weights.to(predictions.device)
    loss = torch.mean(torch.abs(predictions - targets) * weights)

    if raw_affine is not None and PHYSICS_BOUND_WEIGHT > 0.0:
        (s_in, m_in), (s_out, m_out), (s_avg, m_avg) = raw_affine
        t_in = (predictions[..., 0] - m_in) / s_in
        t_out = (predictions[..., 1] - m_out) / s_out
        t_avg = (predictions[..., 2] - m_avg) / s_avg
        # Bounds are DETACHED: the penalty must pull T_avg back inside the
        # bracket, never widen the bracket by distorting T_inner / T_outer
        # (T_inner in particular is anchored and accurate -- letting the
        # hinge push it outward would corrupt the channel we just fixed).
        lo = torch.minimum(t_in, t_out).detach()
        hi = torch.maximum(t_in, t_out).detach()
        # Hinge in raw degrees, rescaled by the T_avg scale so it is
        # commensurate with the scaled-space L1 term above.
        viol = torch.relu(t_avg - hi) + torch.relu(lo - t_avg)
        loss = loss + PHYSICS_BOUND_WEIGHT * torch.mean(viol) * s_avg
    return loss


