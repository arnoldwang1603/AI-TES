"""Variant-aware autoregressive rollout helpers (train-time)."""
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


def run_rollout_train(model, variant, inputs, targets, init_conds, win_obs,
                      tf_prob=0.0):
    """Forward pass during training/validation, variant-aware.

    abs_sliding has its own logic in _rollout_sliding. The other 4 variants
    share a unified per-step loop that maintains a two-step AR history for
    T_outer / T_inner / T_avg, so dT slots can be derived from the AR chain
    without any future GT.

    tf_prob: probability of using the GROUND-TRUTH absolute state instead
    of the model's previous-step prediction when stepping forward in the
    AR chain. Linear-decay schedule is computed by the caller (typically
    `teacher_forcing_prob(epoch)`). At test/inference time pass tf_prob=0.

    targets: (B, N, 3) tensor of GT (T_inner, T_outer, T_avg) at steps 1..N.
    Used only when tf_prob > 0.
    """
    batch_size = inputs.shape[0]
    if model.encoder_kind == 'point':
        hidden = model.init_hidden(batch_size, x0=init_conds)
    else:
        hidden = model.init_hidden(batch_size, x_window=win_obs)

    if variant == 'forward_direct':
        # Exogenous-only seq2seq: inputs are [Time, Input_T] (all GT given
        # boundary conditions), one forward pass over the whole sequence.
        # No AR feedback exists, so the TF gate is a no-op. h0 still comes
        # from the InitStateEncoder (initial state is given).
        out, _ = model(inputs, hidden)
        return out

    if variant == 'abs_sliding':
        return _rollout_sliding(model, inputs, targets, hidden, batch_size,
                                tf_prob=tf_prob)

    if is_inverse(variant):
        # Inverse rollout: T_avg is GT exogenous (in inputs[:, :, 1] for
        # the no-AR inverse variants), no AR feedback in the input.
        # Single-shot forward for non-sliding flavors; per-step window
        # build for inverse_abs_sliding. TF gate is a no-op (no AR chain).
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

    preds = []
    cur_h = hidden
    for t in range(inputs.size(1)):
        # --- Teacher-forcing gate ---
        # At t > 0, with probability tf_prob, override the AR-fed cur_X
        # with the corresponding GT abs value at step t (= targets[t-1]).
        # Per-batch-element Bernoulli decision.
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
        preds.append(pred_t)

        # Shift AR history forward: new "previous" is the old "current",
        # new "current" is this step's prediction.
        prev_outer = cur_outer
        prev_inner = cur_inner
        prev_avg = cur_avg
        cur_outer = pred_t[:, 1]    # model output index 1 = T_outer
        cur_inner = pred_t[:, 0]    # index 0 = T_inner
        cur_avg = pred_t[:, 2]      # index 2 = T_avg
    return torch.stack(preds, dim=1)


def _rollout_sliding(model, inputs, targets, hidden, batch_size,
                     tf_prob=0.0, W=None):
    """abs_sliding rollout (sequence-input, 2026-05-25 -- per Arnold).

    Earlier (now removed) version flattened the past W steps into a
    (5 + 3*W)-d FEATURE vector at each rollout step (seq_len = 1 for the
    GRU; the lookback was a static feature stack). Per Arnold's spec on
    2026-05-25, this is wrong: the model should see the past W steps as
    an actual TEMPORAL sequence so the GRU's recurrence can model them.

    New rollout shape:
        per rollout step t, GRU input = (B, W, 5):
            window[k] = [Time(t-W+1+k),
                         T_outer_AR(t-W+1+k),
                         T_inner_AR(t-W+1+k),
                         T_avg_AR(t-W+1+k),
                         Input_T(t-W+1+k)]
        for k = 0 .. W-1. Steps before 0 are zero-padded.

    Decisions (Arnold + user 2026-05-25):
      * Hidden state CARRIES OVER between rollout steps (the GRU's final
        hidden state after the W-step window becomes the starting hidden
        state for the next rollout step's window). This means the GRU is
        advanced W steps per rollout iteration; with stride=1 the windows
        overlap by W-1.
      * Stride = 1: one prediction per input step (output[..., -1, :]
        of each W-step window is the prediction for step t+1).
      * AR in window: the T_outer / T_inner / T_avg values at each step
        in the window come from the model's previous predictions (or
        GT-via-TF at the time those steps were produced), NOT from
        ground truth. Time and Input_T at each step ARE GT (Time is
        always exogenous, Input_T is exogenous in the forward problem).

    Teacher-forcing schedule: with probability tf_prob (per batch element)
    at step t > 0, the most-recent AR value (used as window[..., -1, :]
    for THIS step's prediction AND saved into the history for future
    windows) is replaced by GT. Earlier history entries are not
    retroactively altered -- TF affects each step only at the moment it
    is added to history, matching the per-step semantics used by the
    forward `abs` variant.
    """
    if W is None:
        W = WINDOW_SIZE
    seq_len = inputs.size(1)
    device = inputs.device

    # Column layout of `inputs` depends on TINNER_MODE (see data.py):
    #   arfed / anchor : [Time, T_outer, T_inner, T_avg, Input_T]   (5-d)
    #   output_only    : [Time, T_outer, T_avg, Input_T]            (4-d, v22)
    out_only = (TINNER_MODE == "output_only")
    if out_only:
        IDX_AVG, IDX_INP, FEAT_DIM = 2, 3, 4
    else:
        IDX_AVG, IDX_INP, FEAT_DIM = 3, 4, 5

    # AR history per step k = 0 .. seq_len (one extra slot for the final
    # prediction we never use). Initialised from the GT t=0 row.
    # output_only keeps NO inner history: T_inner is predicted but its
    # values are never consumed anywhere.
    outer_hist = [inputs[:, 0, 1]]   # k=0: T_outer GT initial
    inner_hist = None if out_only else [inputs[:, 0, 2]]
    avg_hist = [inputs[:, 0, IDX_AVG]]

    # "anchor": head channel 0 is a z-scored residual; reconstruct
    # T_inner_s = KA * InputT_s(t) + KB + KC * z (constants from train_model;
    # AttributeError here means train_model never attached them -- fail loud).
    anchor = model.tinner_anchor if TINNER_MODE == "anchor" else None
    step_anchor = getattr(model, 'step_anchor', None) \
        if OTHER_CH_MODE == "persistence" else None

    preds = []
    cur_h = hidden

    for t in range(seq_len):
        # Teacher-forcing gate (t > 0). Replace the t-step AR values with
        # GT for the masked batch elements. This is what gets written to
        # history AND used in this iteration's window. (In output_only mode
        # there is no inner history to force.)
        if t > 0 and tf_prob > 0.0:
            tf_mask = (torch.rand(batch_size, device=device) < tf_prob)
            gt_inner = targets[:, t - 1, 0]
            gt_outer = targets[:, t - 1, 1]
            gt_avg = targets[:, t - 1, 2]
            outer_hist[t] = torch.where(tf_mask, gt_outer, outer_hist[t]).detach()
            avg_hist[t] = torch.where(tf_mask, gt_avg, avg_hist[t]).detach()
            if inner_hist is not None:
                inner_hist[t] = torch.where(tf_mask, gt_inner, inner_hist[t]).detach()

        # Build the W-step window: steps [t-W+1, t-W+2, ..., t].
        # Steps with k < 0 are zero-padded.
        window_steps = []
        for offset in range(-(W - 1), 1):   # offset = -(W-1) .. 0 inclusive
            k = t + offset
            if k < 0:
                # Pre-rollout step (no real data). SLIDING_PAD_MODE decides:
                #   "variable": skip -> variable-length window (t=0 -> a single
                #     real step == `abs`; the clean h0 is used, no cold-pad drift)
                #   "init": repeat the t=0 steady-state row (Yiming's fix; a zero
                #     in scaled space is the coldest temp -> ~9 C t=0 undershoot)
                #   "zero": legacy cold-system zero-pad (2026-05-25 baseline)
                if SLIDING_PAD_MODE == "variable":
                    continue
                elif SLIDING_PAD_MODE == "zero":
                    window_steps.append(
                        torch.zeros(batch_size, FEAT_DIM, device=device))
                else:  # "init"
                    window_steps.append(inputs[:, 0, :FEAT_DIM])
            else:
                # AR slots at step k come from history (detached, so the
                # window does NOT backprop through the AR chain -- mirrors
                # forward `abs` semantics).
                if out_only:
                    step_feats = torch.stack([
                        inputs[:, k, 0],          # Time(k)        GT
                        outer_hist[k].detach(),   # T_outer(k)     AR
                        avg_hist[k].detach(),     # T_avg(k)       AR
                        inputs[:, k, IDX_INP],    # Input_T(k)     GT exogenous
                    ], dim=1)
                else:
                    step_feats = torch.stack([
                        inputs[:, k, 0],          # Time(k)        GT
                        outer_hist[k].detach(),   # T_outer(k)     AR (or GT-via-TF)
                        inner_hist[k].detach(),   # T_inner(k)     AR
                        avg_hist[k].detach(),     # T_avg(k)       AR
                        inputs[:, k, IDX_INP],    # Input_T(k)     GT exogenous
                    ], dim=1)
                window_steps.append(step_feats)
        window = torch.stack(window_steps, dim=1)   # (B, W, FEAT_DIM)

        # GRU forward over the W-step window. Hidden state advances W
        # steps and carries to the next rollout iteration.
        out, cur_h = model(window, cur_h)
        # Last step's output is the prediction for step t+1.
        pred_t = out[:, -1, :]                       # (B, 3) = (T_inner, T_outer, T_avg)

        if step_anchor is not None:
            # T_outer(t+1) = T_outer_AR(t) + KB + KC*z ; same for T_avg.
            (kb_o, kc_o), (kb_a, kc_a) = step_anchor
            t_out_s = outer_hist[t].detach() + kb_o + kc_o * pred_t[:, 1]
            t_avg_s = avg_hist[t].detach() + kb_a + kc_a * pred_t[:, 2]
            pred_t = torch.stack(
                [pred_t[:, 0], t_out_s, t_avg_s], dim=1)

        if anchor is not None:
            # Reconstruct T_inner in its scaled space from the z-scored
            # residual head, anchored on the GT Input_T at step t. Gradients
            # flow through KC * z; the anchor term is constant w.r.t. params.
            ka, kb, kc = anchor
            # ANCHOR_LEAD=1 uses Input_T(t+1) (exogenous, known in advance);
            # falls back to Input_T(t) at the last step.
            t_a = min(t + 1, seq_len - 1) if ANCHOR_LEAD else t
            tin_s = ka * inputs[:, t_a, IDX_INP] + kb + kc * pred_t[:, 0]
            pred_t = torch.cat([tin_s.unsqueeze(1), pred_t[:, 1:]], dim=1)

        preds.append(pred_t)

        # Save prediction as AR value at step t+1. (In anchor mode the
        # fed-back T_inner is the RECONSTRUCTED value -- GT-anchor dominated,
        # so feedback error resets every step instead of compounding.)
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


