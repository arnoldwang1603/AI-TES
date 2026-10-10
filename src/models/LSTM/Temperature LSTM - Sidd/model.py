import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    D_MODEL, N_HEADS, N_LAYERS, FFN_DIM, TF_DROPOUT, SEQ_LEN,
    LSTM_DIM, LSTM_LAYERS, N_ATTN_LAYERS, ATTN_WINDOW,
    ALPHA1_MIN, ALPHA1_MAX,
    ALPHA2_MIN, ALPHA2_MAX,
    ALPHA3_MIN, ALPHA3_MAX,
    X_INIT, X_MIN, X_MAX,
    D, W,
    TA_DELTA_MAX, PHYS_REG_WEIGHT,
    T_INNER_WEIGHT, T_OUTER_WEIGHT, T_AVG_WEIGHT,
    ALPHA_SMOOTH_WEIGHT, SMOOTH_TA_WEIGHT,
    IDX_TINNER, IDX_TOUTER,
    DT,
)


class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class AlphaTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(2, D_MODEL)
        self.init_bias  = nn.Sequential(
            nn.Linear(4, D_MODEL),
            nn.LayerNorm(D_MODEL),
        )
        self.pos_enc = SinusoidalPE(D_MODEL, max_len=max(SEQ_LEN + 1, 2000))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=FFN_DIM,
            dropout=TF_DROPOUT, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=N_LAYERS, enable_nested_tensor=False,
        )
        self.head1 = nn.Linear(D_MODEL, 1)
        self.head2 = nn.Linear(D_MODEL, 1)
        self.head3 = nn.Linear(D_MODEL, 1)
        self.register_buffer('a1_min', torch.tensor(ALPHA1_MIN))
        self.register_buffer('a1_rng', torch.tensor(ALPHA1_MAX - ALPHA1_MIN))
        self.register_buffer('a2_min', torch.tensor(ALPHA2_MIN))
        self.register_buffer('a2_rng', torch.tensor(ALPHA2_MAX - ALPHA2_MIN))
        self.register_buffer('a3_min', torch.tensor(ALPHA3_MIN))
        self.register_buffer('a3_rng', torch.tensor(ALPHA3_MAX - ALPHA3_MIN))

    def forward(self, tin_scaled, dtin_scaled, init_scaled):
        x = torch.stack([tin_scaled, dtin_scaled], dim=-1)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = x + self.init_bias(init_scaled).unsqueeze(1)
        h = self.transformer(x)
        a1 = torch.sigmoid(self.head1(h)) * self.a1_rng + self.a1_min
        a2 = torch.sigmoid(self.head2(h)) * self.a2_rng + self.a2_min
        a3 = torch.sigmoid(self.head3(h)) * self.a3_rng + self.a3_min
        return a1, a2, a3, h


@torch.jit.script
def _ode_loop(alpha1: torch.Tensor, alpha2: torch.Tensor, alpha3: torch.Tensor,
              tin_phys: torch.Tensor, init_phys: torch.Tensor,
              dt: float, w2: float, D2: float, x_val: float, D_val: float) -> torch.Tensor:
    x2  = x_val * x_val
    Dx  = D_val - x_val
    Dx2 = Dx * Dx
    Ti  = init_phys[:, 0].clone()
    To  = init_phys[:, 1].clone()
    Tm  = (Ti + To) * 0.5
    T   = tin_phys.size(1)
    out = torch.zeros(tin_phys.size(0), T, 2,
                      dtype=tin_phys.dtype, device=tin_phys.device)
    for t in range(T):
        a1  = alpha1[:, t, 0]
        a2  = alpha2[:, t, 0]
        a3  = alpha3[:, t, 0]
        Tin = tin_phys[:, t]
        Ti_new = (Ti + dt * a1 * Tin / w2) / (1.0 + dt * a1 / w2)
        A11 = 1.0 + dt * a2 * (1.0 / x2 + 1.0 / Dx2)
        A12 = -dt * a2 / Dx2
        A21 = -dt * a3 / Dx2
        A22 = 1.0 + dt * a3 * (1.0 / D2 + 1.0 / Dx2)
        b1  = Tm + dt * a2 * Ti_new / x2
        b2  = To + dt * a3 * Ti_new / D2
        det    = A11 * A22 - A12 * A21
        Tm_new = (b1 * A22 - b2 * A12) / det
        To_new = (A11 * b2 - A21 * b1) / det
        out[:, t, 0] = Ti_new
        out[:, t, 1] = To_new
        Ti  = Ti_new
        To  = To_new
        Tm  = Tm_new
    return out


class ODEIntegrator(nn.Module):
    def __init__(self, dt: float = DT):
        super().__init__()
        self.dt  = dt
        self.w2  = W ** 2
        self.D2  = D ** 2
        self._D  = D
        self._x_min = X_MIN
        self._x_rng = X_MAX - X_MIN
        frac = (X_INIT - X_MIN) / (X_MAX - X_MIN)
        self.raw_x = nn.Parameter(torch.tensor(
            math.log(frac / (1.0 - frac)), dtype=torch.float32))

    @property
    def x(self):
        return torch.sigmoid(self.raw_x) * self._x_rng + self._x_min

    def forward(self, alpha1, alpha2, alpha3, tin_phys, init_phys):
        return _ode_loop(alpha1, alpha2, alpha3, tin_phys, init_phys,
                         self.dt, self.w2, self.D2, self.x.item(), self._D)


class PhysicsLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=4, hidden_size=LSTM_DIM, num_layers=LSTM_LAYERS,
            batch_first=True, bidirectional=True,
            dropout=TF_DROPOUT if LSTM_LAYERS > 1 else 0.0,
        )
        n_state = LSTM_LAYERS * 2 * LSTM_DIM
        self.init_h   = nn.Linear(4, n_state)
        self.init_c   = nn.Linear(4, n_state)
        self.lstm_proj = nn.Sequential(
            nn.Linear(2 * LSTM_DIM, D_MODEL),
            nn.LayerNorm(D_MODEL),
        )
        self.pos_enc = SinusoidalPE(D_MODEL, max_len=max(SEQ_LEN + 1, 2000))
        attn_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=FFN_DIM,
            dropout=TF_DROPOUT, batch_first=True, norm_first=True,
        )
        self.attention = nn.TransformerEncoder(
            attn_layer, num_layers=N_ATTN_LAYERS, enable_nested_tensor=False,
        )
        self.head_ti = self._make_head(2 * D_MODEL, D_MODEL,      TF_DROPOUT)
        self.head_to = self._make_head(    D_MODEL, D_MODEL // 2, TF_DROPOUT)

    @staticmethod
    def _make_head(in_features, hidden, dropout):
        layers = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        nn.init.normal_(layers[3].weight, std=0.001)
        nn.init.zeros_(layers[3].bias)
        return layers

    @staticmethod
    def _make_window_mask(seq_len, window, device):
        idx  = torch.arange(seq_len, device=device)
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        return dist > window

    def forward(self, tin_scaled, dtin_scaled, init_scaled, ti_ode_sc, to_ode_sc):
        B, T = tin_scaled.shape
        h0 = (self.init_h(init_scaled)
                  .view(B, LSTM_LAYERS * 2, LSTM_DIM)
                  .permute(1, 0, 2).contiguous())
        c0 = (self.init_c(init_scaled)
                  .view(B, LSTM_LAYERS * 2, LSTM_DIM)
                  .permute(1, 0, 2).contiguous())

        x = torch.stack([tin_scaled, dtin_scaled, ti_ode_sc, to_ode_sc], dim=-1)
        lstm_out, _ = self.lstm(x, (h0, c0))

        h_lstm = self.lstm_proj(lstm_out)
        h_pe   = self.pos_enc(h_lstm)

        win_mask = self._make_window_mask(T, ATTN_WINDOW, h_pe.device)
        h_attn   = self.attention(h_pe, mask=win_mask)

        delta_ti = self.head_ti(torch.cat([h_lstm, h_attn], dim=-1))
        delta_to = self.head_to(h_attn)

        return delta_ti, delta_to


class ThermalPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha_tf  = AlphaTransformer()
        self.ode       = ODEIntegrator()
        self.phys_lstm = PhysicsLSTM()

        self.head_ta = nn.Sequential(
            nn.Linear(D_MODEL + 2, D_MODEL // 2),
            nn.GELU(),
            nn.Dropout(TF_DROPOUT),
            nn.Linear(D_MODEL // 2, 1),
        )
        nn.init.normal_(self.head_ta[3].weight, std=0.001)
        nn.init.zeros_(self.head_ta[3].bias)

        self.register_buffer('ta_delta_max', torch.tensor(TA_DELTA_MAX))
        self.register_buffer('loss_weights',
            torch.tensor([T_INNER_WEIGHT, T_OUTER_WEIGHT, T_AVG_WEIGHT]))
        self.register_buffer('sc_min_ti', torch.zeros(1))
        self.register_buffer('sc_min_to', torch.zeros(1))
        self.register_buffer('sc_rng_ti', torch.ones(1))
        self.register_buffer('sc_rng_to', torch.ones(1))

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.ode = self.ode.cpu()  # sequential for-loop; no GPU benefit
        return self

    def register_scaler(self, scaler):
        sc_min = scaler.data_min_
        sc_rng = scaler.data_max_ - sc_min
        self.sc_min_ti.fill_(sc_min[IDX_TINNER])
        self.sc_min_to.fill_(sc_min[IDX_TOUTER])
        self.sc_rng_ti.fill_(sc_rng[IDX_TINNER])
        self.sc_rng_to.fill_(sc_rng[IDX_TOUTER])

    def forward(self, tin_scaled, dtin_scaled, init_scaled, tin_phys, init_phys):
        device = tin_scaled.device

        a1, a2, a3, h = self.alpha_tf(tin_scaled, dtin_scaled, init_scaled)

        ode_out = self.ode(a1.cpu(), a2.cpu(), a3.cpu(),
                           tin_phys.cpu(), init_phys.cpu())
        Ti_ode = ode_out[:, :, 0:1]
        To_ode = ode_out[:, :, 1:2]

        Ti_ode_sc = ((Ti_ode.to(device) - self.sc_min_ti) / self.sc_rng_ti).squeeze(-1)
        To_ode_sc = ((To_ode.to(device) - self.sc_min_to) / self.sc_rng_to).squeeze(-1)

        delta_ti, delta_to = self.phys_lstm(
            tin_scaled, dtin_scaled, init_scaled, Ti_ode_sc, To_ode_sc)

        Ti = Ti_ode.to(device) + delta_ti
        To = To_ode.to(device) + delta_to

        ta_input = torch.cat(
            [h, Ti_ode_sc.unsqueeze(-1), To_ode_sc.unsqueeze(-1)], dim=-1)
        ta_delta = torch.tanh(self.head_ta(ta_input)) * self.ta_delta_max
        Ta0 = init_phys[:, 2:3].unsqueeze(1).cpu()
        Ta  = Ta0 + ta_delta.cpu()

        return torch.cat([Ti, To, Ta.to(device)], dim=-1), (a1, a2, a3)

    def compute_loss(self, preds, targets, params, tavg_w=None):
        w = self.loss_weights.clone()
        if tavg_w is not None:
            w = w.clone()
            w[2] = tavg_w
        L_data = (torch.abs(preds - targets) * w).mean()

        L_alpha_smooth = torch.tensor(0.0, device=preds.device)
        if ALPHA_SMOOTH_WEIGHT > 0:
            for p in params:
                L_alpha_smooth = L_alpha_smooth + ((p[:, 1:] - p[:, :-1]) ** 2).mean()
            L_alpha_smooth = L_alpha_smooth / len(params)

        L_ta_smooth = torch.tensor(0.0, device=preds.device)
        if SMOOTH_TA_WEIGHT > 0 and (tavg_w is None or tavg_w > 0):
            L_ta_smooth = ((preds[:, 1:, 2] - preds[:, :-1, 2]) ** 2).mean()

        Ti, To, Ta = preds[:, :, 0], preds[:, :, 1], preds[:, :, 2]
        L_phys = (F.relu(Ta - torch.max(Ti, To)) + F.relu(torch.min(Ti, To) - Ta)).mean()

        L_total = (L_data
                   + ALPHA_SMOOTH_WEIGHT * L_alpha_smooth
                   + SMOOTH_TA_WEIGHT    * L_ta_smooth
                   + PHYS_REG_WEIGHT     * L_phys)

        return L_total, L_data, L_alpha_smooth + L_ta_smooth, L_phys


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
