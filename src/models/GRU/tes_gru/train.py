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
def train_model(variant, train_dfs, val_dfs, test_dfs, scaler, params):
    print(f"\n--- Building datasets for variant '{variant}' ---")
    train_datasets = [ThermalDataset(df, variant, scaler=scaler, file_name=fn)
                      for df, fn in train_dfs]
    val_datasets = [ThermalDataset(df, variant, scaler=scaler, file_name=fn)
                    for df, fn in val_dfs]
    test_datasets = [ThermalDataset(df, variant, scaler=scaler, file_name=fn)
                     for df, fn in test_dfs]

    pin = (DEVICE.type == "cuda")
    train_loader = DataLoader(ConcatDataset(train_datasets),
                              batch_size=params['batch_size'], shuffle=True,
                              pin_memory=pin, num_workers=0)
    val_loader = DataLoader(ConcatDataset(val_datasets),
                            batch_size=params['batch_size'],
                            pin_memory=pin, num_workers=0)

    model = ThermalGRU(
        input_size=INPUT_DIMS[variant],
        output_size=OUTPUT_SIZE,
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        dropout=params['dropout'],
        encoder=variant_to_encoder(variant),
        window_size=WINDOW_SIZE,
    ).to(DEVICE)

    # ---- Modern RNN init: orthogonal weight_hh for the main GRU
    # (Chen 2019). PyTorch's default uniform init is suboptimal for
    # stacked depth >= 3. Done BEFORE any resume-from-checkpoint so a
    # resumed run uses the saved (already-trained) weights.
    if ORTHOGONAL_INIT_RECURRENT:
        apply_orthogonal_init_gru(model.gru)
        # Also apply to the small window-encoder GRU if this variant uses it.
        if getattr(model, 'encoder_kind', 'point') == 'window':
            apply_orthogonal_init_gru(model.init_encoder.encoder_gru)
        print(f"[{variant}] orthogonal init applied to main GRU "
              f"(and window encoder if present)")

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
    if PHYSICS_BOUND_WEIGHT > 0.0 and not is_inverse(variant):
        _i = {c: ThermalDataset.BASE_COLS.index(c) for c in
              ("T_inner (C)", "T_outer (C)", "T_avg (C)")}
        model.raw_affine = tuple(
            (float(scaler.scale_[_i[c]]), float(scaler.min_[_i[c]]))
            for c in ("T_inner (C)", "T_outer (C)", "T_avg (C)"))
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
        print(f"[{variant}] T_avg position head"
              f"{f' (gap floor {POS_GAP_FLOOR:g} C)' if POS_GAP_FLOOR else ''}"
              f"{' (case gate ON)' if POS_CASE_GATE else ''}"
              f"{f' (temp gate {POS_TEMP_GATE:g} C'
                 f'{f", soft {POS_TEMP_SOFT:g} C" if POS_TEMP_SOFT else ", hard"})'
                 if POS_TEMP_GATE else ''}"
              f"{f' (learned gate, bias {POS_GATE_BIAS:g})' if POS_LEARNED_GATE else ''}"
              f"{f' [{OUTPUT_SIZE} output channels]' if OUTPUT_SIZE != 3 else ''}"
              f": pos mean={p_mu:.4f} "
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

    # ---- Trainable parameter counts (broken down) ----
    def _count(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    n_params_total = _count(model)
    n_params_gru = _count(model.gru)
    n_params_fc = _count(model.fc)
    n_params_encoder = _count(model.init_encoder)
    print(f"[{variant}] trainable params: total={n_params_total:,}  "
          f"gru={n_params_gru:,}  fc={n_params_fc:,}  "
          f"encoder({model.encoder_kind})={n_params_encoder:,}")

    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        factor=params['sched_factor'], patience=params['sched_patience'],
    )

    best_val = float('inf')
    best_epoch = 0           # 1-based epoch at which best_val was achieved
    total_epoch = 0          # 1-based final epoch reached (early-stop or max_epochs)
    early = 0
    os.makedirs(checkpoints_dir(), exist_ok=True)
    ckpt_path = os.path.join(checkpoints_dir(), f"best_gru_variant_{variant}.pth")

    train_hist, val_hist, lr_hist = [], [], []
    start_epoch = 0          # epoch index from which to (re)start the loop

    # ---- Resume from saved state if available ----
    if RESUME_ENABLED:
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
                        print(f"[{variant}] WARNING: could not restore CUDA RNG: {e}")
                print(f"[{variant}] RESUMED from epoch {start_epoch} "
                      f"(best_val={best_val:.6f} @ ep{best_epoch})")
            except Exception as e:
                print(f"[{variant}] WARNING: resume_state.pt found but could not be "
                      f"loaded ({e}). Starting fresh.")
                start_epoch = 0
                best_val = float('inf')
                best_epoch = 0
                total_epoch = 0
                early = 0
                train_hist, val_hist, lr_hist = [], [], []

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
        tloss = 0.0
        for batch in train_loader:
            inputs, targets, _, init_conds, win_obs, *_ = [
                b.to(DEVICE, non_blocking=True) if torch.is_tensor(b) else b
                for b in batch
            ]
            optimizer.zero_grad()
            preds = run_rollout_train(model, variant, inputs, targets,
                                       init_conds, win_obs,
                                       tf_prob=tf_prob_train)
            loss = weighted_loss(preds, targets, raw_affine=getattr(model, 'raw_affine', None))
            loss.backward()
            # ---- Gradient clipping (Pascanu 2013) ----
            if GRADIENT_CLIP_NORM is not None and GRADIENT_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=GRADIENT_CLIP_NORM,
                )
            optimizer.step()
            tloss += loss.item()
        avg_train = tloss / max(1, len(train_loader))
        train_hist.append(avg_train)

        # ---- val (always full autoregressive, tf_prob=0) ----
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets, _, init_conds, win_obs, *_ = [
                    b.to(DEVICE, non_blocking=True) if torch.is_tensor(b) else b
                    for b in batch
                ]
                preds = run_rollout_train(model, variant, inputs, targets,
                                          init_conds, win_obs,
                                          tf_prob=0.0)
                vloss += weighted_loss(preds, targets,
                                       raw_affine=getattr(model, 'raw_affine', None)).item()
        avg_val = vloss / max(1, len(val_loader))
        val_hist.append(avg_val)

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
        else:
            early += 1

        if (epoch + 1) % 10 == 0:
            phase = 'warmup' if in_warmup else 'main  '
            print(f"[{variant}] epoch {epoch + 1:>4} ({phase}): "
                  f"train={avg_train:.4f}  val={avg_val:.4f}  "
                  f"lr={cur_lr:.2e}  tf={tf_prob_train:.3f}  "
                  f"best={best_val:.4f} @ ep{best_epoch}")

        # ---- Periodic resume-state save (atomic) ----
        if (epoch + 1) % CHECKPOINT_EVERY_N_EPOCHS == 0:
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
                    'rng_python': random.getstate(),
                    'rng_numpy': np.random.get_state(),
                    'rng_torch': torch.get_rng_state(),
                    'rng_torch_cuda': (torch.cuda.get_rng_state_all()
                                       if torch.cuda.is_available() else None),
                })
            except Exception as e:
                print(f"[{variant}] WARNING: resume save failed at epoch "
                      f"{epoch + 1}: {e}")

        if early >= params['early_stop_patience']:
            print(f"[{variant}] early stop at epoch {epoch + 1}")
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    print(f"[{variant}] restored best (val={best_val:.6f} @ ep{best_epoch} / "
          f"{total_epoch} total) from {ckpt_path}")

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
        val_mae=best_val, param_breakdown=param_breakdown,
    )

    return (model, test_datasets, best_val, best_epoch, total_epoch,
            param_breakdown, train_hist, val_hist, lr_hist)


