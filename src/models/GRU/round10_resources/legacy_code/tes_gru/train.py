"""Variant-aware training loop with resume support."""
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

from .config import *
from .data import *
from .models import *
from .rollout import *
from .runio import *


# ============================================================
# 8. Training (variant-aware)
# ============================================================
def fit_model_constants(model, variant, train_dfs, scaler):
    """Fit every data-derived constant the reconstructions need and attach it
    to the model: the T_inner anchor (KA/KB/KC), the persistence and the
    round-4 fixed-reference anchors, the position head (mu/sd + affine
    triples), the anchored-fallback midpoint stats, and the round-8 T_outer
    reference (KA/KB/KC + the two scalers the EMA seed crosses between).

    Split out of train_model (2026-09-19) so a saved checkpoint can be
    evaluated without retraining: build the model, call this with the SAME
    train_dfs/scaler the run used, load the state_dict, run test_model.
    Nothing here depends on the weights; it is a pure function of the
    training data and the config."""
    # ---- T_inner anchor reparameterization (TINNER_MODE == "anchor") ----
    # The head's channel 0 predicts a z-scored residual
    #     delta(t+1) = T_inner_raw(t+1) - Input_T_raw(t)
    # and the rollout reconstructs, in T_inner's MinMax-scaled space,
    #     T_inner_s(t+1) = KA * InputT_s(t) + KB + KC * z .
    # KA/KB map Input_T's scaled space into T_inner's scaled space; KC folds
    # in the train-set delta std so the head trains on O(1) values (a raw
    # ~0.6 C delta is only ~0.0015 MinMax units -- too small for the loss).
    # Recomputed on every train_model call, so resume is unaffected.
    # ---- Raw-space affine constants for the T_avg physics bound ----
    # (scaled = raw * scale_ + min_, per channel). Order matches the model's
    # output channels: T_inner, T_outer, T_avg.
    # Round 10 / H9: the triple is fitted for EVERY forward variant and always
    # handed to weighted_loss; the hinge itself stays gated on
    # PHYSICS_BOUND_WEIGHT > 0 inside the loss, so the value is unchanged at 0.
    if not is_inverse(variant):
        _i = {c: ThermalDataset.BASE_COLS.index(c) for c in
              ("T_inner (C)", "T_outer (C)", "T_avg (C)")}
        model.raw_affine = tuple(
            (float(scaler.scale_[_i[c]]), float(scaler.min_[_i[c]]))
            for c in ("T_inner (C)", "T_outer (C)", "T_avg (C)"))
        if PHYSICS_BOUND_WEIGHT > 0.0:
            print(f"[{variant}] physics bound ON (w={PHYSICS_BOUND_WEIGHT}): "
                  f"T_avg constrained to [min(T_inner,T_outer), max(...)]")
    else:
        model.raw_affine = None
    print(f"[{variant}] loss weights (T_inner,T_outer,T_avg) = {LOSS_WEIGHTS}")

    # ---- T_outer / T_avg persistence parameterization (Arnold 2026-07-23) ----
    # Head predicts a z-scored per-step change; the rollout adds it to the
    # channel's own previous (AR) value:  T(t+1) = T_AR(t) + mu + sigma*z,
    # expressed in scaled space as  T_s(t+1) = T_s_AR(t) + KB + KC*z.
    if variant == 'abs_sliding' and OTHER_CH_MODE == 'persistence':
        step_stats = {}
        for name, col in (("T_outer", "T_outer (C)"), ("T_avg", "T_avg (C)")):
            d = []
            for df, _ in train_dfs:
                dd = df.copy()
                dd.rename(columns={"T_ave (C)": "T_avg (C)"}, inplace=True)
                d.append(np.diff(dd[col].values))
            d = np.concatenate(d)
            mu, sd = float(d.mean()), float(max(d.std(), 1e-3))
            s = float(scaler.scale_[ThermalDataset.BASE_COLS.index(col)])
            step_stats[name] = (s * mu, s * sd)   # (KB, KC) in scaled space
            print(f"[{variant}] {name} persistence: step mean={mu:+.4f} C "
                  f"std={sd:.4f} C -> KB={s*mu:+.6f} KC={s*sd:.6f}")
        model.step_anchor = (step_stats["T_outer"], step_stats["T_avg"])
    else:
        model.step_anchor = None

    # ---- T_outer / T_avg fixed-reference anchors (OTHER_CH_MODE='anchor') ----
    # Reconstruction, in MinMax-scaled space:
    #   T_outer_s(t) = T_outer0_s + KBo + KCo*z_o        (T_outer(0) is GT)
    #   T_avg_s(t)   = s_a*( w*(T_in_s-m_i)/s_i + (1-w)*(T_out_s-m_o)/s_o
    #                        + mu_a ) + m_a + KCa*z_a
    # Constants fitted on the train set here (same pattern as the T_inner
    # anchor), so resume is unaffected.
    # ---- T_avg position head (OTHER_CH_MODE == 'pos_head') ----
    # T_avg = T_outer + pos*(T_inner - T_outer), pos = mu + sd*z, fitted on
    # the train set. Only steps with a meaningful gradient contribute to the
    # fit (|T_inner - T_outer| > 5 C); elsewhere pos is ill-conditioned.
    if variant in ('abs_sliding', 'forward_direct') and OTHER_CH_MODE == 'pos_head':
        ps = []
        for df, _ in train_dfs:
            d = df.copy()
            d.rename(columns={"T_ave (C)": "T_avg (C)"}, inplace=True)
            ti = d["T_inner (C)"].values
            to = d["T_outer (C)"].values
            ta = d["T_avg (C)"].values
            gap = ti - to
            if POS_GAP_FLOOR:
                # Floor the denominator, then EVERY step is usable -- the
                # ill-conditioned ones are exactly what the floor repairs.
                sgn = np.where(gap < 0, -1.0, 1.0)
                if POS_FLOOR_SOFT:
                    gd = sgn * np.sqrt(gap * gap + POS_GAP_FLOOR ** 2)
                else:
                    gd = sgn * np.maximum(np.abs(gap), POS_GAP_FLOOR)
                ps.append((ta - to) / gd)
            else:
                ok = np.abs(gap) > 5.0
                if POS_FIT_CLEAN:
                    # Excursion steps carry pos far outside the normal
                    # band and inflate sd; the head should not pay
                    # resolution for steps a gate handles anyway.
                    ok &= ((ta <= np.maximum(ti, to) + 0.5)
                           & (ta >= np.minimum(ti, to) - 0.5))
                if ok.sum():
                    ps.append((ta[ok] - to[ok]) / gap[ok])
        ps = np.concatenate(ps) if ps else np.array([0.27])
        model.pos_time_prior = None
        if POS_TIME_PRIOR == "median":
            # Elapsed-time prior: the per-step MEDIAN of the training runs'
            # position (usable steps only), indexed by step number; the head
            # then predicts the residual, so mu/sd become the residual's.
            # The step index at apply time comes from the scaled Time input
            # (column 0): k = (t_s - m_t) / s_t / dt.
            i_t = ThermalDataset.BASE_COLS.index("Time (s)")
            s_t, m_t = float(scaler.scale_[i_t]), float(scaler.min_[i_t])
            rows, dts = [], []
            for df, _ in train_dfs:
                d = df.copy()
                d.rename(columns={"T_ave (C)": "T_avg (C)"}, inplace=True)
                ti = d["T_inner (C)"].values; to = d["T_outer (C)"].values
                ta = d["T_avg (C)"].values; gap = ti - to
                ok = np.abs(gap) > 5.0
                if POS_FIT_CLEAN:
                    ok &= ((ta <= np.maximum(ti, to) + 0.5)
                           & (ta >= np.minimum(ti, to) - 0.5))
                p = np.where(ok, (ta - to) / np.where(ok, gap, 1.0), np.nan)
                rows.append(p)
                tv = d["Time (s)"].values
                dts.append(float(np.median(np.diff(tv))) if len(tv) > 1 else 10.0)
            L = max(len(r) for r in rows)
            Mx = np.full((len(rows), L), np.nan)
            for r_i, r in enumerate(rows):
                Mx[r_i, :len(r)] = r
            with np.errstate(all="ignore"):
                curve = np.nanmedian(Mx, axis=0)
            n_ok = np.sum(~np.isnan(Mx), axis=0)
            idx = np.arange(L)
            good = ~np.isnan(curve)
            if not good.all():
                # steps no run has a usable gap at (typically the first
                # few, before the walls part) take the nearest fitted value
                curve = np.interp(idx, idx[good], curve[good])
            dt = float(np.median(dts))
            resid = Mx - curve[None, :]
            ps = resid[~np.isnan(resid)]                 # what the head predicts
            model.pos_time_prior = torch.tensor(curve, dtype=torch.float32)
            model.pos_time_affine = (s_t, m_t, dt)
            print(f"[{variant}] position time prior (median over {len(rows)} runs, {L} steps, "
                  f"dt={dt:g}s): pos at step 1/50/200/500/last = "
                  + "/".join("{:.3f}".format(curve[k]) for k in (0, min(49, L - 1), min(199, L - 1), min(499, L - 1), L - 1))
                  + f"; usable runs at step 1/50: {n_ok[0]}/{n_ok[min(49, L - 1)]}; "
                  f"residual mean={ps.mean():+.4f} sd={ps.std():.4f} "
                  f"(t<200: {np.nanstd(resid[:, :200]):.4f})")
        p_mu = float(ps.mean())
        # Widen beyond the raw spread so the head can reach the tails; the
        # between-case spread is what the fixed-w anchor could not follow.
        p_sd = float(max(ps.std(), 1e-3) * 2.0)
        i_o = ThermalDataset.BASE_COLS.index("T_outer (C)")
        i_i = ThermalDataset.BASE_COLS.index("T_inner (C)")
        i_a = ThermalDataset.BASE_COLS.index("T_avg (C)")
        if POS_ANCHORED_FALLBACK:
            # Midpoint-anchored fallback constants: T_avg deviation from
            # the surface midpoint, over ALL steps (the fallback must
            # serve both between-surface and excursion regimes).
            res = []
            for df, _ in train_dfs:
                dd = df.copy()
                dd.rename(columns={'T_ave (C)': 'T_avg (C)'}, inplace=True)
                res.append(dd['T_avg (C)'].values
                           - 0.5 * (dd['T_inner (C)'].values
                                    + dd['T_outer (C)'].values))
            res = np.concatenate(res)
            model.pos_mid = (float(res.mean()),
                             float(max(res.std(), 1.0)))
            print(f"[{variant}] anchored fallback: midpoint residual "
                  f"mean={model.pos_mid[0]:+.2f} C sd={model.pos_mid[1]:.2f} C")
        else:
            model.pos_mid = None
        model.pos_head = (
            p_mu, p_sd,
            (float(scaler.scale_[i_i]), float(scaler.min_[i_i])),
            (float(scaler.scale_[i_o]), float(scaler.min_[i_o])),
            (float(scaler.scale_[i_a]), float(scaler.min_[i_a])),
        )
        # Plain concatenation, NOT nested f-strings: quote reuse and
        # multi-line expressions inside f-strings are Python 3.12+ syntax
        # (PEP 701) and crash older interpreters at import (2026-08-22,
        # found the hard way on the lab server).
        _tags = ""
        if POS_GAP_FLOOR:
            _tags += " (gap floor {:g} C)".format(POS_GAP_FLOOR)
        if POS_CASE_GATE:
            _tags += " (case gate ON)"
        if POS_TEMP_GATE:
            _soft = (", soft {:g} C".format(POS_TEMP_SOFT)
                     if POS_TEMP_SOFT else ", hard")
            _tags += " (temp gate {:g} C{})".format(POS_TEMP_GATE, _soft)
        if POS_LEARNED_GATE:
            _tags += " (learned gate, bias {:g})".format(POS_GATE_BIAS)
        if OUTPUT_SIZE != 3:
            _tags += " [{} output channels]".format(OUTPUT_SIZE)
        if POS_TIME_PRIOR != "0":
            _tags += " (time prior {}: head = residual)".format(POS_TIME_PRIOR)
        print(f"[{variant}] T_avg position head{_tags}: pos mean={p_mu:.4f} "
              f"raw sd={ps.std():.4f} -> head sd={p_sd:.4f} "
              f"(covers pos {p_mu-3*p_sd:.3f}..{p_mu+3*p_sd:.3f} at 3 sigma)")
    else:
        model.pos_head = None

    if variant in ('abs_sliding', 'forward_direct') and OTHER_CH_MODE.startswith('anchor'):
        d_out, num, den, parts = [], 0.0, 0.0, []
        for df, _ in train_dfs:
            d = df.copy()
            d.rename(columns={"T_ave (C)": "T_avg (C)"}, inplace=True)
            ti = d["T_inner (C)"].values
            to = d["T_outer (C)"].values
            ta = d["T_avg (C)"].values
            d_out.append(to - to[0])
            x = ti - to
            num += float((x * (ta - to)).sum())
            den += float((x * x).sum())
            parts.append((ti, to, ta))
        w = num / den if den > 1e-9 else 0.5
        d_out = np.concatenate(d_out)
        d_avg = np.concatenate([ta - (w * ti + (1 - w) * to) for ti, to, ta in parts])
        mo, so = float(d_out.mean()), float(max(d_out.std(), 1e-3))
        ma, sa_ = float(d_avg.mean()), float(max(d_avg.std(), 1e-3))
        idx = {c: ThermalDataset.BASE_COLS.index(c) for c in
               ("T_inner (C)", "T_outer (C)", "T_avg (C)")}
        aff = {k: (float(scaler.scale_[i]), float(scaler.min_[i])) for k, i in idx.items()}
        model.other_anchor = {
            'w': w,
            'out': (aff["T_outer (C)"][0] * mo, aff["T_outer (C)"][0] * so),
            'avg': (ma, aff["T_avg (C)"][0] * sa_),
            'aff': (aff["T_inner (C)"], aff["T_outer (C)"], aff["T_avg (C)"]),
        }
        print(f"[{variant}] T_outer anchored on T_outer(0): delta mean={mo:+.2f} std={so:.2f} C")
        print(f"[{variant}] T_avg anchored on {w:.3f}*T_inner+{1-w:.3f}*T_outer: "
              f"delta mean={ma:+.2f} std={sa_:.2f} C")
    else:
        model.other_anchor = None

    # forward_direct also gets the anchor: its inputs carry the exogenous
    # Input_T at every step, so the same reconstruction applies. Without this
    # the E-arm comparison was confounded -- E lost BOTH the AR feedback and
    # the anchor, so its T_inner (2.33) measured the missing anchor, not the
    # formulation.
    if variant in ('abs_sliding', 'forward_direct') and TINNER_MODE == 'anchor':
        deltas = []
        for df, _ in train_dfs:
            d = df.copy()
            d.rename(columns={"T_ave (C)": "T_avg (C)"}, inplace=True)
            if "Input Temperature (C)" not in d.columns:
                d["Input Temperature (C)"] = d["T_inner (C)"]
            tin = d["T_inner (C)"].values
            inp = d["Input Temperature (C)"].values
            # ANCHOR_LEAD=1 -> residual vs Input_T(t+1); =0 -> vs Input_T(t)
            deltas.append(tin[1:] - (inp[1:] if ANCHOR_LEAD else inp[:-1]))
        deltas = np.concatenate(deltas)
        d_mean = float(deltas.mean())
        # Floor the std: with the Input_T:=T_inner fallback data the delta
        # degenerates to ~0 and would kill the head's gradient.
        d_std = float(max(deltas.std(), 0.05))
        # ANCHOR_SCALE widens the head's usable output range (2026-08-09).
        # Round 4 showed the residual +8.4 C on input-jump cases (40, 54) is
        # NOT an information problem -- feeding the input temperature's future (lookahead
        # 1/3/5) left it at 8.38/8.46/8.49, unchanged. With d_std ~0.65 C, an
        # 8.5 C correction is ~13 sigma: the head simply cannot reach it. This
        # multiplier rescales KC so the same z range spans a wider correction
        # (the trade is coarser resolution in the common near-zero regime).
        d_std *= ANCHOR_SCALE
        if ANCHOR_SCALE != 1.0:
            print(f"[{variant}] ANCHOR_SCALE={ANCHOR_SCALE} -> effective delta "
                  f"std {d_std:.3f} C (max |delta| in train set "
                  f"{np.abs(deltas - d_mean).max():.1f} C = "
                  f"{np.abs(deltas - d_mean).max()/d_std:.1f} sigma)")
        i_tin = ThermalDataset.BASE_COLS.index("T_inner (C)")
        i_inp = ThermalDataset.BASE_COLS.index("Input Temperature (C)")
        # sklearn MinMaxScaler: scaled = raw * scale_ + min_
        s_tin, m_tin = float(scaler.scale_[i_tin]), float(scaler.min_[i_tin])
        s_inp, m_inp = float(scaler.scale_[i_inp]), float(scaler.min_[i_inp])
        ka = s_tin / s_inp
        kb = -ka * m_inp + s_tin * d_mean + m_tin
        kc = s_tin * d_std
        model.tinner_anchor = (ka, kb, kc)
        print(f"[{variant}] T_inner anchor on Input_T: delta mean={d_mean:+.3f} C "
              f"std={d_std:.3f} C  ->  KA={ka:.5f} KB={kb:+.5f} KC={kc:.6f}")

    # ---- Round 8: T_outer anchored on the live input temperature (Arnold) ----
    # The T_inner recipe applied verbatim to head 1: delta_o = T_outer(t+1) -
    # Input_T(t+ANCHOR_LEAD), z-scored on the train set; the rollout
    # reconstructs T_outer_s = KA*InputT_s + KB + KC*z.
    if variant in ('abs_sliding', 'forward_direct') and TOUTER_MODE != 'abs':
        def _ema(x, tau, e0):
            # numpy twin of rollout.touter_reference (raw degrees; the EMA is
            # affine-equivariant so the scaled-space reconstruction matches);
            # seeded at T_outer(0), the given initial condition. Same r as the
            # rollout (config.touter_ema_r: euler or exact form).
            r = touter_ema_r()
            e = np.empty_like(x, dtype=float)
            e[0] = e0
            for t in range(1, len(x)):
                e[t] = r * e[t - 1] + (1.0 - r) * x[t]
            return e
        deltas_o, absol_o = [], []
        for df, _ in train_dfs:
            d = df.copy()
            d.rename(columns={"T_ave (C)": "T_avg (C)"}, inplace=True)
            if "Input Temperature (C)" not in d.columns:
                d["Input Temperature (C)"] = d["T_inner (C)"]
            tout = d["T_outer (C)"].values.astype(float)
            inp = d["Input Temperature (C)"].values.astype(float)
            ref = _ema(inp, TOUTER_TAU, tout[0]) if TOUTER_MODE == 'anchor_ema' else inp
            deltas_o.append(tout[1:] - (ref[1:] if ANCHOR_LEAD else ref[:-1]))
            absol_o.append(tout)
        deltas_o = np.concatenate(deltas_o)
        do_mean = float(deltas_o.mean())
        do_std = float(max(deltas_o.std(), 0.05)) * TOUTER_SCALE  # range multiplier, own knob
        i_to = ThermalDataset.BASE_COLS.index("T_outer (C)")
        i_inp = ThermalDataset.BASE_COLS.index("Input Temperature (C)")
        s_to, m_to = float(scaler.scale_[i_to]), float(scaler.min_[i_to])
        s_inp, m_inp = float(scaler.scale_[i_inp]), float(scaler.min_[i_inp])
        ka_o = s_to / s_inp
        kb_o = -ka_o * m_inp + s_to * do_mean + m_to
        kc_o = s_to * do_std
        model.touter_anchor = (ka_o, kb_o, kc_o)
        # Scaler constants for the two columns the EMA seed crosses between
        # (T_outer(0) is given in T_outer's scale, the EMA runs in Input_T's).
        model.touter_ref_affine = ((s_to, m_to), (s_inp, m_inp))
        print("[{}] T_outer anchor on {}: delta mean={:+.2f} C std={:.2f} C (x{:g} range) "
              "(T_outer itself: std {:.2f} C)  ->  KA={:.5f} KB={:+.5f} KC={:.6f}".format(
                  variant, ("EMA(tau=%d, %s) of Input_T" % (TOUTER_TAU, TOUTER_EMA_FORM)
                            if TOUTER_MODE == 'anchor_ema' else "Input_T"),
                  do_mean, do_std, TOUTER_SCALE, float(np.concatenate(absol_o).std()),
                  ka_o, kb_o, kc_o))
    else:
        model.touter_anchor = None
        model.touter_ref_affine = None

    return model


def _tupled(x):
    if isinstance(x, list):
        return tuple(_tupled(v) for v in x)
    if isinstance(x, dict):
        return {k: _tupled(v) for k, v in x.items()}
    return x


def set_model_constants(model, consts):
    """Round 10 / H4: the inverse of fit_model_constants -- attach the
    constants a sidecar recorded (runio.load_constants_sidecar(p)["constants"])
    to `model`. JSON lists come back as tuples (the reconstructions unpack
    them), pos_time_prior as a float32 tensor, absent modes as None. The
    constants belong to the checkpoint's own split; never pair a sidecar
    with another run's data."""
    for k in MODEL_CONSTANT_ATTRS:
        v = consts.get(k)
        if k == "pos_time_prior":
            v = None if v is None else torch.tensor(v, dtype=torch.float32)
        else:
            v = _tupled(v)
        setattr(model, k, v)
    return model


def train_model(variant, train_dfs, val_dfs, test_dfs, scaler, params):
    print(f"\n--- Building datasets for variant '{variant}' ---")
    # H2: per-file T_avg loss mask. Empty sets under the legacy TAVG_MASK="".
    tr_masked, va_masked = get_tavg_mask_sets()
    train_datasets = [ThermalDataset(df, variant, scaler=scaler, file_name=fn,
                                     tavg_masked=is_tavg_masked(fn, tr_masked))
                      for df, fn in train_dfs]
    val_datasets = [ThermalDataset(df, variant, scaler=scaler, file_name=fn,
                                   tavg_masked=is_tavg_masked(fn, va_masked))
                    for df, fn in val_dfs]
    if TAVG_MASK:
        n_tr = sum(ds.tavg_masked for ds in train_datasets)
        n_va = sum(ds.tavg_masked for ds in val_datasets)
        print(f"[{variant}] TAVG_MASK={TAVG_MASK}: T_avg label masked on {n_tr}/{len(train_datasets)} "
              f"training files and {n_va}/{len(val_datasets)} validation files "
              f"(validation T_avg scored on {len(val_datasets) - n_va})")
    # H3: under SEAL_TEST the test files are never turned into datasets --
    # the run scores its validation files instead (main.py).
    test_datasets = ([] if SEAL_TEST else
                     [ThermalDataset(df, variant, scaler=scaler, file_name=fn)
                      for df, fn in test_dfs])

    pin = (DEVICE.type == "cuda")
    train_loader = DataLoader(ConcatDataset(train_datasets),
                              batch_size=params['batch_size'], shuffle=True,
                              pin_memory=pin, num_workers=0)
    val_loader = DataLoader(ConcatDataset(val_datasets),
                            batch_size=params['batch_size'],
                            pin_memory=pin, num_workers=0)

    model = build_model(variant, params, variant_to_encoder(variant)).to(DEVICE)
    parts = model_parts(model)      # [model], or the three nets of the wrapper

    # ---- Modern RNN init: orthogonal weight_hh for the main GRU
    # (Chen 2019). PyTorch's default uniform init is suboptimal for
    # stacked depth >= 3. Done BEFORE any resume-from-checkpoint so a
    # resumed run uses the saved (already-trained) weights.
    if ORTHOGONAL_INIT_RECURRENT:
        for part in parts:
            apply_orthogonal_init_gru(part.gru)
            # Also apply to the small window-encoder GRU if this variant uses it.
            if getattr(part, 'encoder_kind', 'point') == 'window':
                apply_orthogonal_init_gru(part.init_encoder.encoder_gru)
        print(f"[{variant}] orthogonal init applied to main GRU "
              f"(and window encoder if present)"
              + (f" -- {len(parts)} independent nets" if len(parts) > 1 else ""))

    fit_model_constants(model, variant, train_dfs, scaler)

    # ---- Trainable parameter counts (broken down) ----
    def _count(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    n_params_total = _count(model)
    n_params_gru = sum(_count(part.gru) for part in parts)
    n_params_fc = sum(_count(part.fc) for part in parts)
    n_params_encoder = sum(_count(part.init_encoder) for part in parts)
    print(f"[{variant}] trainable params: total={n_params_total:,}  "
          f"gru={n_params_gru:,}  fc={n_params_fc:,}  "
          f"encoder({model.encoder_kind})={n_params_encoder:,}")

    os.makedirs(checkpoints_dir(), exist_ok=True)
    ckpt_path = os.path.join(checkpoints_dir(), f"best_gru_variant_{variant}.pth")
    full_weights = torch.tensor(LOSS_WEIGHTS, dtype=torch.float32)

    # Round 10 / H4: the constants sidecar next to the best checkpoint --
    # written now, right after fit_model_constants (a crashed run still has
    # it), and again at every best-checkpoint save so the pair always agrees.
    sidecar_path = constants_sidecar_path(ckpt_path)

    def _write_sidecar(**kw):
        info = dict(variant=variant, checkpoint=os.path.basename(ckpt_path),
                    loss_weights_realised=[float(w) for w in full_weights.tolist()])
        info.update(kw)
        return write_constants_sidecar(sidecar_path, model, scaler, extra=info)

    _write_sidecar(stage="after_fit")
    print(f"[{variant}] constants sidecar -> {sidecar_path}")

    if MODEL_GROUPS is not None:
        # One stage per part, in channel order (independent: T_inner,
        # T_outer, T_avg; avg_only: walls, T_avg). The finished nets are
        # frozen (no grad, eval mode) while the next trains; every stage has
        # its own optimiser, scheduler, early stopping and best epoch;
        # ckpt_path always holds the best state of every part so far. Loss =
        # the configured weights on that stage's channels only.
        stages, train_hist, val_hist, lr_hist, val_hist_ch = [], [], [], [], []
        best_epoch = total_epoch = 0
        best_val = float('nan')
        for k, part in enumerate(parts):
            w = stage_loss_weights(model.groups[k], full_weights)
            label = f"{variant}/{model.channel_names[k]}"
            for j, other in enumerate(parts):
                other.requires_grad_(j == k)
            r = _train_stage(model, variant, params, train_loader, val_loader,
                             ckpt_path, label, w, trainable=list(part.parameters()),
                             frozen=[q for j, q in enumerate(parts) if j != k],
                             resume_ok=False, on_best=_write_sidecar)
            if r['best_epoch'] == 0:
                raise RuntimeError(f"[{label}] stage recorded no validation improvement "
                                   f"(best_val={r['best_val']}); refusing to continue")
            stages.append({'channel': model.channel_names[k],
                           'best_epoch': r['best_epoch'], 'total_epoch': r['total_epoch'],
                           'best_val': r['best_val']})
            if k == len(parts) - 1:
                best_epoch = total_epoch + r['best_epoch']   # index into the concatenated curve
                best_val = r['best_val']
            total_epoch += r['total_epoch']
            train_hist += r['train_hist']; val_hist += r['val_hist']; lr_hist += r['lr_hist']
            val_hist_ch += r['val_hist_ch']
        for part in parts:
            part.requires_grad_(True)
        print(f"[{variant}] {MODEL_SPLIT} stages: " + ", ".join(
            f"{st['channel']} best ep{st['best_epoch']}/{st['total_epoch']} (val {st['best_val']:.6f})"
            for st in stages))
    else:
        stages = None
        r = _train_stage(model, variant, params, train_loader, val_loader,
                         ckpt_path, variant, full_weights,
                         trainable=list(model.parameters()), frozen=[],
                         resume_ok=True, on_best=_write_sidecar)
        best_val, best_epoch, total_epoch = r['best_val'], r['best_epoch'], r['total_epoch']
        train_hist, val_hist, lr_hist = r['train_hist'], r['val_hist'], r['lr_hist']
        val_hist_ch = r['val_hist_ch']

    param_breakdown = {
        'total': n_params_total,
        'gru': n_params_gru,
        'fc': n_params_fc,
        'encoder': n_params_encoder,
    }

    # ---- Persist full training history (always overwrites; small file) ----
    save_train_history(
        variant, train_hist, val_hist, lr_hist,
        best_epoch=best_epoch, total_epoch=total_epoch,
        val_mae=best_val, param_breakdown=param_breakdown, stages=stages,
        val_hist_ch=val_hist_ch,
    )

    return (model, test_datasets, best_val, best_epoch, total_epoch,
            param_breakdown, train_hist, val_hist, lr_hist, stages)


def validate_epoch(model, variant, val_loader, weights, raw_affine, use_mask=None):
    """One full-autoregressive pass over val_loader -> (scalar, [per-channel]).

    Legacy (use_mask False, the default under TAVG_MASK=""): the exact mean
    over validation FILES of the weighted loss -- each batch's mean loss
    weighted by its size (with batch 16 the 20-file stratified set is 16 + 4,
    where a plain per-batch average gave the last four files 1/8 each) -- and
    the unweighted per-channel MAE over all files. Arithmetic unchanged from
    Round 9.

    Masked (H2): numerator sum(|p-t| w m) and denominator sum(m) accumulated
    across batches, so the scalar is the masked loss over the whole set; the
    per-channel MAE counts unmasked files only (vs3: T_avg on 17 files,
    T_inner / T_outer on 20). The scalar drives early stopping and
    ReduceLROnPlateau. (The physics hinge is excluded here; config asserts
    PHYSICS_BOUND_WEIGHT == 0 under TAVG_MASK.)"""
    if use_mask is None:
        use_mask = bool(TAVG_MASK)
    model.eval()
    vloss, n_val = 0.0, 0
    vch = torch.zeros(3, dtype=torch.float64)
    vnum, vden = 0.0, 0.0
    vch_num = torch.zeros(3, dtype=torch.float64)
    vch_den = torch.zeros(3, dtype=torch.float64)
    with torch.no_grad():
        for batch in val_loader:
            items = [b.to(DEVICE, non_blocking=True) if torch.is_tensor(b) else b
                     for b in batch]
            inputs, targets, init_conds, win_obs = items[0], items[1], items[3], items[4]
            inp_last = items[ITEM_INP_LAST]
            preds = run_rollout_train(model, variant, inputs, targets,
                                      init_conds, win_obs,
                                      tf_prob=0.0, inp_last=inp_last)
            nb = int(inputs.shape[0])
            if use_mask:
                m = items[ITEM_LOSS_MASK]
                num, den = masked_loss_terms(preds, targets, weights.to(preds.device), m)
                vnum += float(num.item())
                vden += float(den.item())
                m3 = m.to(preds.device, preds.dtype).unsqueeze(1)          # (B, 1, 3)
                ae = (preds - targets).abs()
                vch_num += (ae * m3).sum(dim=(0, 1)).double().cpu()
                vch_den += m3.expand_as(ae).sum(dim=(0, 1)).double().cpu()
            else:
                vloss += weighted_loss(preds, targets, weights=weights,
                                       raw_affine=raw_affine).item() * nb
                vch += (preds - targets).abs().mean(dim=(0, 1)).double().cpu()[:3] * nb
            n_val += nb
    if use_mask:
        avg_val = vnum / max(vden, 1.0)
        vch_list = (vch_num / vch_den.clamp_min(1.0)).tolist()
    else:
        avg_val = vloss / max(1, n_val)
        vch_list = (vch / max(1, n_val)).tolist()
    return avg_val, vch_list


def _train_stage(model, variant, params, train_loader, val_loader, ckpt_path,
                 label, weights, trainable, frozen, resume_ok, on_best=None):
    """One early-stopped training run of `trainable` (a parameter list) with
    the per-channel loss `weights`; `frozen` modules are kept in eval mode
    (dropout off) while `model.train()` is on. Saves the FULL model state to
    ckpt_path at every validation improvement and restores it at the end.
    Shared mode calls this once with everything trainable; independent mode
    once per net. Resume-from-interruption only when resume_ok. on_best
    (H4): called with best_epoch / best_val / stage after each best save
    (the constants sidecar)."""
    optimizer = optim.Adam(trainable, lr=params['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        factor=params['sched_factor'], patience=params['sched_patience'],
    )

    best_val = float('inf')
    best_epoch = 0           # 1-based epoch at which best_val was achieved
    total_epoch = 0          # 1-based final epoch reached (early-stop or max_epochs)
    early = 0

    train_hist, val_hist, lr_hist = [], [], []
    val_hist_ch = []         # per-epoch unweighted validation MAE per channel
    start_epoch = 0          # epoch index from which to (re)start the loop

    # ---- Resume from saved state if available ----
    if RESUME_ENABLED and resume_ok:
        resume = load_resume_state(variant)
        if resume is not None:
            try:
                model.load_state_dict(resume['model_state'])
                optimizer.load_state_dict(resume['optimizer_state'])
                scheduler.load_state_dict(resume['scheduler_state'])
                best_val = resume['best_val']
                best_epoch = resume['best_epoch']
                early = resume['early_stop_counter']
                total_epoch = resume['total_epoch']
                train_hist = list(resume['train_hist'])
                val_hist = list(resume['val_hist'])
                lr_hist = list(resume['lr_hist'])
                val_hist_ch = list(resume.get('val_hist_ch', []))
                start_epoch = resume['epoch']  # epochs already completed
                # Restore RNG state for deterministic continuation.
                random.setstate(resume['rng_python'])
                np.random.set_state(resume['rng_numpy'])
                torch.set_rng_state(resume['rng_torch'].cpu()
                                    if torch.is_tensor(resume['rng_torch'])
                                    else resume['rng_torch'])
                if (resume.get('rng_torch_cuda') is not None
                        and torch.cuda.is_available()):
                    try:
                        torch.cuda.set_rng_state_all(resume['rng_torch_cuda'])
                    except Exception as e:
                        print(f"[{label}] WARNING: could not restore CUDA RNG: {e}")
                print(f"[{label}] RESUMED from epoch {start_epoch} "
                      f"(best_val={best_val:.6f} @ ep{best_epoch})")
            except Exception as e:
                print(f"[{label}] WARNING: resume_state.pt found but could not be "
                      f"loaded ({e}). Starting fresh.")
                start_epoch = 0
                best_val = float('inf')
                best_epoch = 0
                total_epoch = 0
                early = 0
                train_hist, val_hist, lr_hist = [], [], []

    raw_affine = getattr(model, 'raw_affine', None)
    use_mask = bool(TAVG_MASK)          # H2
    for epoch in range(start_epoch, params['max_epochs']):
        # Teacher-forcing probability for this epoch (decay 0.5 -> 0 over
        # the first TF_DECAY_EPOCHS, then 0 thereafter).
        tf_prob_train = teacher_forcing_prob(epoch)

        # ---- LR warmup (Liu 2019 RAdam motivation) ----
        # If still inside the warmup window, override the optimizer's LR
        # with a linearly-ramped value. ReduceLROnPlateau is suspended
        # during warmup (we don't call scheduler.step) so it can't
        # interfere with the ramp. After warmup completes the scheduler
        # takes over from base_lr.
        wlr = warmup_lr(epoch, params['lr'])
        in_warmup = wlr is not None
        if in_warmup:
            for pg in optimizer.param_groups:
                pg['lr'] = wlr

        # ---- train ----
        model.train()
        for m in frozen:
            m.eval()
        tloss = 0.0
        for batch in train_loader:
            items = [b.to(DEVICE, non_blocking=True) if torch.is_tensor(b) else b
                     for b in batch]
            inputs, targets, init_conds, win_obs = items[0], items[1], items[3], items[4]
            inp_last = items[ITEM_INP_LAST]
            # H2: the mask is handed over ONLY under TAVG_MASK, so the legacy
            # path keeps its exact mean() arithmetic (bit-for-bit Round 9).
            mask = items[ITEM_LOSS_MASK] if use_mask else None
            optimizer.zero_grad()
            preds = run_rollout_train(model, variant, inputs, targets,
                                       init_conds, win_obs,
                                       tf_prob=tf_prob_train, inp_last=inp_last)
            loss = weighted_loss(preds, targets, weights=weights, raw_affine=raw_affine,
                                 mask=mask)
            loss.backward()
            # ---- Gradient clipping (Pascanu 2013) ----
            if GRADIENT_CLIP_NORM is not None and GRADIENT_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(
                    trainable, max_norm=GRADIENT_CLIP_NORM,
                )
            optimizer.step()
            tloss += loss.item()
        avg_train = tloss / max(1, len(train_loader))
        train_hist.append(avg_train)

        # ---- val (always full autoregressive, tf_prob=0) ----
        avg_val, vch_list = validate_epoch(model, variant, val_loader, weights,
                                           raw_affine, use_mask=use_mask)
        val_hist.append(avg_val)
        val_hist_ch.append(vch_list)

        # ReduceLROnPlateau only runs AFTER warmup is over.
        if not in_warmup:
            scheduler.step(avg_val)

        cur_lr = optimizer.param_groups[0]['lr']
        lr_hist.append(cur_lr)

        total_epoch = epoch + 1
        if avg_val < best_val:
            best_val = avg_val
            best_epoch = epoch + 1
            early = 0
            torch.save(model.state_dict(), ckpt_path)
            if on_best is not None:
                try:
                    on_best(best_epoch=best_epoch, best_val=float(best_val), stage=label)
                except Exception as e:
                    print(f"[{label}] WARNING: constants sidecar write failed: {e}")
        else:
            early += 1

        if (epoch + 1) % 10 == 0:
            phase = 'warmup' if in_warmup else 'main  '
            print(f"[{label}] epoch {epoch + 1:>4} ({phase}): "
                  f"train={avg_train:.4f}  val={avg_val:.4f}  "
                  f"lr={cur_lr:.2e}  tf={tf_prob_train:.3f}  "
                  f"best={best_val:.4f} @ ep{best_epoch}")

        # ---- Periodic resume-state save (atomic) ----
        if resume_ok and (epoch + 1) % CHECKPOINT_EVERY_N_EPOCHS == 0:
            try:
                save_resume_state(variant, {
                    'variant': variant,
                    'epoch': epoch + 1,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'best_val': best_val,
                    'best_epoch': best_epoch,
                    'early_stop_counter': early,
                    'total_epoch': total_epoch,
                    'train_hist': train_hist,
                    'val_hist': val_hist,
                    'lr_hist': lr_hist,
                    'val_hist_ch': val_hist_ch,
                    'rng_python': random.getstate(),
                    'rng_numpy': np.random.get_state(),
                    'rng_torch': torch.get_rng_state(),
                    'rng_torch_cuda': (torch.cuda.get_rng_state_all()
                                       if torch.cuda.is_available() else None),
                })
            except Exception as e:
                print(f"[{label}] WARNING: resume save failed at epoch "
                      f"{epoch + 1}: {e}")

        if early >= params['early_stop_patience']:
            print(f"[{label}] early stop at epoch {epoch + 1}")
            break

    if best_epoch == 0:
        # No epoch ever improved on inf: NaN/inf validation loss throughout.
        # Nothing was saved for this stage; do not restore silently.
        raise RuntimeError(f"[{label}] no validation improvement in {total_epoch} epochs "
                           f"(best_val={best_val}) -- checkpoint not written")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    print(f"[{label}] restored best (val={best_val:.6f} @ ep{best_epoch} / "
          f"{total_epoch} total) from {ckpt_path}")
    return dict(best_val=best_val, best_epoch=best_epoch, total_epoch=total_epoch,
                train_hist=train_hist, val_hist=val_hist, lr_hist=lr_hist,
                val_hist_ch=val_hist_ch)


