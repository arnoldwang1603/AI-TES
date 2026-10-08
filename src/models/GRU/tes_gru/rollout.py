"""Variant-aware autoregressive rollout helpers (train-time)."""
import numpy as np
import torch

from .config import *


# ============================================================
# 7. Variant rollout helpers
# ============================================================
def variant_to_encoder(variant):
    # abs_sliding still uses the point InitStateEncoder (h0 built from
    # the single t=0 observation, which is production-available).
    # The "sliding window" in abs_sliding is in the per-step INPUT, not
    # in the h0 computation. Inverse variants likewise use the point
    # encoder (init_cond is 4-d, h0 from t=0 absolute state).
    return 'window' if variant == 'abs_window' else 'point'


def _rewrite_step_input(variant, x_t,
                        cur_outer, cur_inner, cur_avg,
                        prev_outer, prev_inner, prev_avg):
    """
    Variant-aware input rewrite at one AR step (5-input refactor, 2026-05-20).

    Conventions across variants (column indices in x_t):
      delta     (5-d): [Time, dT_outer, dT_inner, dT_avg, dInput_T]
                       idx 1, 2, 3 overwritten with AR-derived deltas.
                       idx 4 (dInput_T) stays GT exogenous.
      abs+delta (9-d): [Time, T_outer, T_inner, T_avg, Input_T,
                       dT_outer, dT_inner, dT_avg, dInput_T]
                       idx 1, 2, 3 = AR-fed abs T_outer / T_inner / T_avg.
                       idx 5, 6, 7 = AR-derived deltas.
                       idx 4 Input_T and idx 8 dInput_T stay GT exogenous.
      abs       (5-d): [Time, T_outer, T_inner, T_avg, Input_T]
      abs_window(5-d): same as abs (the window enters via h0, not the input)
                       idx 1, 2, 3 = AR-fed abs values.

    cur_X / prev_X are the AR-tracked absolute T_outer / T_inner / T_avg
    at the current and previous step. Deltas = cur - prev, computed from
    the AR chain only (never future GT).
    """
    x_new = x_t.clone()
    dT_outer = (cur_outer - prev_outer)
    dT_inner = (cur_inner - prev_inner)
    dT_avg = (cur_avg - prev_avg)

    if variant == 'delta':
        x_new[:, 1] = dT_outer
        x_new[:, 2] = dT_inner
        x_new[:, 3] = dT_avg
        # idx 4 dInput_T stays GT
    elif variant == 'abs+delta':
        x_new[:, 1] = cur_outer
        x_new[:, 2] = cur_inner
        x_new[:, 3] = cur_avg
        # idx 4 Input_T stays GT
        x_new[:, 5] = dT_outer
        x_new[:, 6] = dT_inner
        x_new[:, 7] = dT_avg
        # idx 8 dInput_T stays GT
    else:  # abs / abs_window
        x_new[:, 1] = cur_outer
        x_new[:, 2] = cur_inner
        x_new[:, 3] = cur_avg
    return x_new


def _compute_step_offsets(t, inputs, idx_inp, default_W):
    """Computes the list of historical temporal offsets relative to step t."""
    mode = DYNAMIC_WINDOW_MODE
    if mode == "step":
        w = WINDOW_EARLY if t < WINDOW_SPLIT_STEP else WINDOW_LATE
        return list(range(-(w - 1), 1))
    elif mode == "ramp":
        frac = min(1.0, float(t) / max(1.0, float(WINDOW_RAMP_STEPS)))
        w = int(WINDOW_MIN + frac * (WINDOW_MAX - WINDOW_MIN))
        w = max(WINDOW_MIN, min(WINDOW_MAX, w))
        return list(range(-(w - 1), 1))
    elif mode == "flux_adaptive":
        t_prev = max(0, t - 1)
        v_in = torch.mean(torch.abs(inputs[:, t, idx_inp] - inputs[:, t_prev, idx_inp])).item()
        w = int(round(WINDOW_MAX - (WINDOW_MAX - WINDOW_MIN) * np.tanh(v_in / max(1e-4, FLUX_SCALE))))
        w = max(WINDOW_MIN, min(WINDOW_MAX, w))
        return list(range(-(w - 1), 1))
    elif mode == "dilated":
        k_tokens = WINDOW_TOKENS
        d = 1 if t <= 10 else (2 if t <= 30 else 4)
        return [-(k_tokens - 1 - i) * d for i in range(k_tokens)]
    elif mode == "asymmetric":
        return list(range(-(WINDOW_SLOW - 1), 1))
    else:
        return list(range(-(default_W - 1), 1))


def run_rollout_train(model, variant, inputs, targets, init_conds, win_obs,
                      tf_prob=0.0, curriculum_horizon=None):
    # Arm P: Truncate rollout horizon during curriculum warmup
    if curriculum_horizon is not None and curriculum_horizon < inputs.size(1):
        inputs = inputs[:, :curriculum_horizon, :]
        targets = targets[:, :curriculum_horizon, :]

    batch_size = inputs.shape[0]
    if model.encoder_kind == 'point':
        hidden = model.init_hidden(batch_size, x0=init_conds)
    else:
        hidden = model.init_hidden(batch_size, x_window=win_obs)

    if variant == 'forward_direct':
        out, _ = model(inputs, hidden)
        return out

    if variant == 'abs_sliding':
        return _rollout_sliding(model, inputs, targets, hidden, batch_size,
                                tf_prob=tf_prob, init_conds=init_conds)

    if is_inverse(variant):
        return _rollout_inverse(model, variant, inputs, hidden, batch_size)

    # --- Initialize the AR-history triple (T_outer, T_inner, T_avg) ---
    if variant == 'delta':
        cur_outer = init_conds[:, 0].clone()    # T_outer(0)
        cur_inner = init_conds[:, 1].clone()    # T_inner(0)
        cur_avg = init_conds[:, 2].clone()      # T_avg(0)
    else:  # abs+delta / abs / abs_window
        cur_outer = inputs[:, 0, 1].clone()
        cur_inner = inputs[:, 0, 2].clone()
        cur_avg = inputs[:, 0, 3].clone()
    prev_outer = cur_outer.clone()
    prev_inner = cur_inner.clone()
    prev_avg = cur_avg.clone()

    # For Soft-Start blending: stack initial condition at t=0 in target prediction order
    t0_state = torch.stack([init_conds[:, 1], init_conds[:, 0], init_conds[:, 2]], dim=1)  # (B, 3)

    preds = []
    cur_h = hidden

    # Apply Recurrent Hidden-State Warmup if configured (Arm K)
    warmup_steps = getattr(model, 'warmup_steps', 0)
    if warmup_steps > 0:
        warmup_input = inputs[:, :1, :].repeat(1, warmup_steps, 1)
        _, cur_h = model.gru(warmup_input, cur_h)

    for t in range(inputs.size(1)):
        if t > 0 and tf_prob > 0.0:
            tf_mask = (torch.rand(batch_size, device=inputs.device) < tf_prob)
            gt_inner = targets[:, t - 1, 0]
            gt_outer = targets[:, t - 1, 1]
            gt_avg = targets[:, t - 1, 2]
            cur_outer = torch.where(tf_mask, gt_outer, cur_outer)
            cur_inner = torch.where(tf_mask, gt_inner, cur_inner)
            cur_avg = torch.where(tf_mask, gt_avg, cur_avg)

        x_t = _rewrite_step_input(
            variant,
            inputs[:, t, :],
            cur_outer.detach(), cur_inner.detach(), cur_avg.detach(),
            prev_outer.detach(), prev_inner.detach(), prev_avg.detach(),
        )
        out, cur_h = model(x_t.unsqueeze(1), cur_h)
        pred_t = out[:, 0, :]    # (B, 3) = (T_inner_pred, T_outer_pred, T_avg_pred)

        # Arm N: Convex Spatial Bounding for T_avg
        if USE_SPATIAL_ANCHOR:
            beta = torch.sigmoid(pred_t[:, 2])
            t_avg_bounded = pred_t[:, 1] + beta * (pred_t[:, 0] - pred_t[:, 1])
            pred_t = torch.stack([pred_t[:, 0], pred_t[:, 1], t_avg_bounded], dim=1)

        # Apply Soft-Start Blending if active
        soft_start_tau = getattr(model, 'soft_start_tau', None)
        if soft_start_tau is not None and soft_start_tau > 0.0:
            step = t + 1
            alpha = torch.exp(torch.tensor(-step / soft_start_tau, device=inputs.device, dtype=torch.float32))
            pred_t = alpha * t0_state + (1.0 - alpha) * pred_t

        preds.append(pred_t)

        prev_outer = cur_outer
        prev_inner = cur_inner
        prev_avg = cur_avg
        cur_outer = pred_t[:, 1]    # T_outer
        cur_inner = pred_t[:, 0]    # T_inner
        cur_avg = pred_t[:, 2]      # T_avg
    return torch.stack(preds, dim=1)


def _rollout_sliding(model, inputs, targets, hidden, batch_size,
                     tf_prob=0.0, W=None, init_conds=None):
    if W is None:
        W = WINDOW_SIZE
    seq_len = inputs.size(1)
    device = inputs.device

    out_only = (TINNER_MODE == "output_only")
    if out_only:
        IDX_AVG, IDX_INP, FEAT_DIM = 2, 3, 4
    else:
        IDX_AVG, IDX_INP, FEAT_DIM = 3, 4, 5

    outer_hist = [inputs[:, 0, 1]]
    inner_hist = None if out_only else [inputs[:, 0, 2]]
    avg_hist = [inputs[:, 0, IDX_AVG]]

    anchor = model.tinner_anchor if TINNER_MODE == "anchor" else None
    step_anchor = getattr(model, 'step_anchor', None) \
        if OTHER_CH_MODE == "persistence" else None

    # For Soft-Start blending: stack initial condition at t=0 in target prediction order
    t0_state = None
    if init_conds is not None:
        t0_state = torch.stack([init_conds[:, 1], init_conds[:, 0], init_conds[:, 2]], dim=1)  # (B, 3)

    preds = []
    cur_h = hidden

    # Apply Recurrent Hidden-State Warmup if configured (Arm K)
    warmup_steps = getattr(model, 'warmup_steps', 0)
    if warmup_steps > 0:
        warmup_seq = inputs[:, :1, :FEAT_DIM].repeat(1, warmup_steps, 1)
        _, cur_h = model.gru(warmup_seq, cur_h)

    for t in range(seq_len):
        if t > 0 and tf_prob > 0.0:
            tf_mask = (torch.rand(batch_size, device=device) < tf_prob)
            gt_inner = targets[:, t - 1, 0]
            gt_outer = targets[:, t - 1, 1]
            gt_avg = targets[:, t - 1, 2]
            outer_hist[t] = torch.where(tf_mask, gt_outer, outer_hist[t]).detach()
            avg_hist[t] = torch.where(tf_mask, gt_avg, avg_hist[t]).detach()
            if inner_hist is not None:
                inner_hist[t] = torch.where(tf_mask, gt_inner, inner_hist[t]).detach()

        # Dynamic Window Horizon Offset Resolution (Arms Q, R, S, T, U)
        offsets = _compute_step_offsets(t, inputs, IDX_INP, W)

        window_steps = []
        for offset in offsets:
            k = t + offset
            if k < 0:
                if SLIDING_PAD_MODE == "variable":
                    continue
                elif SLIDING_PAD_MODE == "zero":
                    window_steps.append(
                        torch.zeros(batch_size, FEAT_DIM, device=device))
                else:  # "init": repeat the t=0 steady-state row
                    window_steps.append(inputs[:, 0, :FEAT_DIM])
            else:
                # Arm U: Asymmetric lookback clamping for fast-responding convective channels
                k_fast = k
                if DYNAMIC_WINDOW_MODE == "asymmetric" and offset < -(WINDOW_FAST - 1):
                    k_fast = max(0, t - (WINDOW_FAST - 1))

                if out_only:
                    step_feats = torch.stack([
                        inputs[:, k_fast, 0],          # Time (Fast)
                        outer_hist[k].detach(),        # T_outer (Slow)
                        avg_hist[k].detach(),          # T_avg (Slow)
                        inputs[:, k_fast, IDX_INP],    # Input_T (Fast)
                    ], dim=1)
                else:
                    step_feats = torch.stack([
                        inputs[:, k_fast, 0],          # Time (Fast)
                        outer_hist[k].detach(),        # T_outer (Slow)
                        inner_hist[k_fast].detach(),   # T_inner (Fast)
                        avg_hist[k].detach(),          # T_avg (Slow)
                        inputs[:, k_fast, IDX_INP],    # Input_T (Fast)
                    ], dim=1)
                window_steps.append(step_feats)
        window = torch.stack(window_steps, dim=1)

        out, cur_h = model(window, cur_h)
        pred_t = out[:, -1, :]

        if step_anchor is not None:
            (kb_o, kc_o), (kb_a, kc_a) = step_anchor
            t_out_s = outer_hist[t].detach() + kb_o + kc_o * pred_t[:, 1]
            t_avg_s = avg_hist[t].detach() + kb_a + kc_a * pred_t[:, 2]
            pred_t = torch.stack(
                [pred_t[:, 0], t_out_s, t_avg_s], dim=1)

        if anchor is not None:
            ka, kb, kc = anchor
            t_a = min(t + 1, seq_len - 1) if ANCHOR_LEAD else t
            tin_s = ka * inputs[:, t_a, IDX_INP] + kb + kc * pred_t[:, 0]
            pred_t = torch.cat([tin_s.unsqueeze(1), pred_t[:, 1:]], dim=1)

        # Arm N: Convex Spatial Bounding for T_avg (bounds T_avg between T_outer and T_inner)
        if USE_SPATIAL_ANCHOR:
            beta = torch.sigmoid(pred_t[:, 2])
            t_avg_bounded = pred_t[:, 1] + beta * (pred_t[:, 0] - pred_t[:, 1])
            pred_t = torch.stack([pred_t[:, 0], pred_t[:, 1], t_avg_bounded], dim=1)

        # Apply Soft-Start Blending if active
        soft_start_tau = getattr(model, 'soft_start_tau', None)
        if soft_start_tau is not None and soft_start_tau > 0.0 and t0_state is not None:
            step = t + 1
            alpha = torch.exp(torch.tensor(-step / soft_start_tau, device=device, dtype=torch.float32))
            pred_t = alpha * t0_state + (1.0 - alpha) * pred_t

        preds.append(pred_t)

        outer_hist.append(pred_t[:, 1])
        avg_hist.append(pred_t[:, 2])
        if inner_hist is not None:
            inner_hist.append(pred_t[:, 0])

    return torch.stack(preds, dim=1)


def _rollout_inverse(model, variant, inputs, hidden, batch_size, W=None):
    """Inverse rollout (Arnold 2026-05-21).

    For all inverse variants, the per-step input contains only Time +
    T_avg (and dT_avg / T_avg-window for delta / sliding flavors). T_avg
    is GT EXOGENOUS at every step (the model assumes the user can
    measure T_avg in deployment, like a sensor reading at the centre of
    the media). No AR feedback in the input -- the model just maps the
    observed T_avg trajectory to the unknown driving Input_T plus T_inner
    and T_outer.

    Inputs already contain the full feat_cols values (Time + T_avg [+
    dT_avg]). For inverse_abs_sliding we additionally build a per-step
    window over the past W T_avg values (current included at the last
    slot), zero-padded at the oldest positions when t < W-1.

    Returns predictions of shape (batch, seq_len, 3). The 3 channels are
    (Input_T, T_inner, T_outer) for inverse variants (see
    VARIANT_OUTPUT_CHANNELS).
    """
    if variant != 'inverse_abs_sliding':
        # Non-sliding inverse variants: all inputs are GT, single forward.
        out, _ = model(inputs, hidden)
        return out

    # inverse_abs_sliding (sequence-input, 2026-05-25 -- per Arnold):
    # at each rollout step t, the GRU sees a W-step sequence
    #     window[k] = [Time(t-W+1+k), T_avg(t-W+1+k)]   for k = 0..W-1
    # Steps before 0 are zero-padded. All values are GT (no AR in the
    # inverse problem). Hidden state CARRIES OVER between rollout steps;
    # the GRU's last-step output is the prediction for step t+1.
    if W is None:
        W = WINDOW_SIZE
    seq_len = inputs.size(1)
    device = inputs.device

    preds = []
    cur_h = hidden

    # Apply Recurrent Hidden-State Warmup if configured (Arm K)
    warmup_steps = getattr(model, 'warmup_steps', 0)
    if warmup_steps > 0:
        warmup_seq = inputs[:, :1, :2].repeat(1, warmup_steps, 1)
        _, cur_h = model.gru(warmup_seq, cur_h)

    for t in range(seq_len):
        window_steps = []
        for offset in range(-(W - 1), 1):   # offset = -(W-1) .. 0
            k = t + offset
            if k < 0:
                # See SLIDING_PAD_MODE: "variable" skips (variable-length window),
                # "init" repeats the t=0 observation (Time, T_avg), "zero" legacy.
                if SLIDING_PAD_MODE == "variable":
                    continue
                elif SLIDING_PAD_MODE == "zero":
                    window_steps.append(torch.zeros(batch_size, 2, device=device))
                else:  # "init"
                    window_steps.append(inputs[:, 0, :2])
            else:
                # inputs has 2 features per step for inverse_abs_sliding:
                # idx 0 = Time, idx 1 = T_avg. Both GT.
                window_steps.append(inputs[:, k, :2])
        window = torch.stack(window_steps, dim=1)   # (B, W, 2)

        out, cur_h = model(window, cur_h)
        preds.append(out[:, -1, :])   # last step's output = pred for t+1

    return torch.stack(preds, dim=1)
