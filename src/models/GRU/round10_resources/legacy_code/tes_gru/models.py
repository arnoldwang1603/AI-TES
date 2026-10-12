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


class IndependentThermalGRU(nn.Module):
    """MODEL_SPLIT=independent / avg_only: several complete ThermalGRUs
    behind the ThermalGRU interface, one per output-channel group.

    groups, e.g. ((0,), (1,), (2, 3)): parts[i] emits the channels of
    groups[i], in order; the groups partition 0..output_size-1 in order.
    independent = ((0,), (1,), (2..)): T_inner, T_outer, T_avg (+ the
    absolute / gate channels). avg_only = ((0, 1), (2..)): the wall pair
    shares one trunk, T_avg has its own. forward() concatenates the parts'
    outputs into the (B, T, OUTPUT_SIZE) layout every caller expects, and the
    hidden state is a tuple of the parts' hidden states, so the rollouts, the
    anchors (which live as attributes on THIS object, fitted by
    train.fit_model_constants) and evaluate.py run unchanged.

    Which part trains is the training loop's business (train.py stages);
    this class only guarantees that nothing is shared between the parts.
    """

    NAMES = ("T_inner", "T_outer", "T_avg")

    def __init__(self, input_size, hidden_size=128, output_size=3,
                 num_layers=5, dropout=0.3,
                 encoder='point', window_size=WINDOW_SIZE, groups=None):
        super().__init__()
        assert output_size >= 3, output_size
        groups = tuple(tuple(g) for g in (groups or ((0,), (1,), tuple(range(2, output_size)))))
        assert [c for g in groups for c in g] == list(range(output_size)), groups
        self.groups = groups
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.encoder_kind = encoder
        # one label per part: "T_inner", "T_outer", "T_avg" or "T_inner+T_outer"
        self.channel_names = tuple("+".join(self.NAMES[c] for c in g if c < 3) for g in groups)
        self.parts = nn.ModuleList([
            ThermalGRU(input_size, hidden_size=hidden_size, output_size=len(g),
                       num_layers=num_layers, dropout=dropout,
                       encoder=encoder, window_size=window_size)
            for g in groups
        ])

    def init_hidden(self, batch_size, x0=None, x_window=None):
        return tuple(p.init_hidden(batch_size, x0=x0, x_window=x_window)
                     for p in self.parts)

    def forward(self, x, hidden):
        outs, hs = [], []
        for p, h in zip(self.parts, hidden):
            o, h2 = p(x, h)
            outs.append(o)
            hs.append(h2)
        return torch.cat(outs, dim=-1), tuple(hs)


def build_model(variant, params, encoder):
    """The one place a model is constructed (train.py, eval_checkpoint.py):
    ThermalGRU, or the multi-model wrapper under MODEL_SPLIT != shared."""
    kw = dict(input_size=INPUT_DIMS[variant], output_size=OUTPUT_SIZE,
              hidden_size=params['hidden_size'], num_layers=params['num_layers'],
              dropout=params['dropout'], encoder=encoder, window_size=WINDOW_SIZE)
    if MODEL_GROUPS is None:
        return ThermalGRU(**kw)
    return IndependentThermalGRU(groups=MODEL_GROUPS, **kw)


def model_parts(model):
    """The ThermalGRU(s) inside `model`: [model] for the shared model, the
    parts for the wrapper. For init and parameter counting."""
    return list(model.parts) if isinstance(model, IndependentThermalGRU) else [model]


def stage_loss_weights(group, full_weights):
    """Loss-weight vector (length 3, the rollout's channel layout) for one
    training stage of a split model: the configured weight on the stage's
    channels (channels >= 3 are consumed inside the rollout and carry no
    loss of their own), zero elsewhere. Shared by train.py and the runner's
    preflight, which checks the stage vectors sum to the full vector."""
    w = torch.zeros(3)
    for c in group:
        if c < 3:
            w[c] = float(full_weights[c])
    return w


# ============================================================
# 5. Loss
# ============================================================
# Channel order is (T_inner, T_outer, T_avg) -- see VARIANT_OUTPUT_CHANNELS.
# Weights come from config (default 1/6/3, mirroring the LSTM line's recipe).
_DEFAULT_WEIGHTS = torch.tensor(LOSS_WEIGHTS)


def masked_loss_terms(predictions, targets, weights, mask):
    """(numerator, denominator) of the masked weighted L1 (H2):
        num = sum(|p - t| * w * m),  den = sum over (B, T, 3) of m,
    with mask (B, 3) or (B, 1, 3) broadcast over time. The denominator counts
    the unmasked (file, step, channel) entries, so with an all-ones mask
    num / den == mean(|p - t| * w), the legacy loss, exactly. Both terms are
    returned so train.validate_epoch can accumulate them across batches."""
    m = mask.to(predictions.device, predictions.dtype)
    if m.dim() == 2:
        m = m.unsqueeze(1)                     # (B, 1, 3)
    ae = torch.abs(predictions - targets)
    num = (ae * weights * m).sum()
    den = m.expand_as(ae).sum()
    return num, den


def weighted_loss(predictions, targets, weights=None, raw_affine=None, mask=None):
    """Per-channel weighted L1 + optional T_avg physics-bound hinge.

    predictions / targets are in the MinMax-SCALED space.

    mask: optional per-file (B, 3) loss mask (H2, ThermalDataset.loss_mask).
    None keeps the legacy mean(|p - t| * w) bit-for-bit; with a mask the loss
    is sum(|p - t| * w * m) / sum(m) (see masked_loss_terms), which reduces
    to the legacy value when m is all ones.

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
    if mask is None:
        loss = torch.mean(torch.abs(predictions - targets) * weights)
    else:
        # sum(|p-t| w m) / sum(m), written as the mean over ALL entries
        # (masked ones contribute 0) divided by the unmasked fraction: the
        # same reduction kernel as the legacy line, and the fraction is
        # exactly 1.0 for an all-ones mask, so the legacy value is
        # reproduced BIT-FOR-BIT rather than to rounding.
        m = mask.to(predictions.device, predictions.dtype)
        if m.dim() == 2:
            m = m.unsqueeze(1)                 # (B, 1, 3)
        ae = torch.abs(predictions - targets)
        frac = m.expand_as(ae).sum() / float(ae.numel())
        loss = torch.mean(ae * weights * m) / frac

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


