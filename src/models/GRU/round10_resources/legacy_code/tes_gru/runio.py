"""Run-directory layout + checkpoint / resume / meta / history I/O."""
import os
import json
import shutil
import time

import numpy as np
import torch

from . import config            # live access to the mutable config.RUN_NAME
from .config import *
from .utils import _to_jsonable, sha256_file, sha256_lines


def run_dir():
    """Top-level directory for this run. e.g. src/runs/<RUN_NAME>/."""
    script_dir = BASE_DIR
    return os.path.join(script_dir, RUNS_ROOT_DIR, config.RUN_NAME)


def checkpoints_dir():
    return os.path.join(run_dir(), "checkpoints")


def comparison_dir():
    return os.path.join(run_dir(), "comparison")


def variant_dir(variant):
    """Per-variant subdir: runs/<RUN_NAME>/variants/<variant>/."""
    return os.path.join(run_dir(), "variants", variant)


def plots_dir(variant):
    """All per-case plots for this variant go here."""
    return os.path.join(variant_dir(variant), "plots")


def ensure_run_layout():
    """Create the canonical empty directory structure for the run."""
    for d in (run_dir(), checkpoints_dir(), comparison_dir(),
              os.path.join(run_dir(), "variants")):
        os.makedirs(d, exist_ok=True)
    for v in VARIANTS:
        os.makedirs(variant_dir(v), exist_ok=True)
        os.makedirs(plots_dir(v), exist_ok=True)


def run_config_dict():
    """The run_config.json payload (every knob config.py reads, plus data /
    numerics provenance). Shared by save_run_config_snapshot and the H4
    constants sidecar so the two can never disagree."""
    # H2: which files lose their T_avg label (digests of the sorted key lists).
    from .data import get_tavg_mask_sets, get_relabel_sets
    tr_m, va_m = get_tavg_mask_sets()
    rl_tr, rl_va = get_relabel_sets()          # A4
    cfg = {
        'run_name': config.RUN_NAME,
        'started_or_resumed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        # Read the LIVE value (main loop rebinds config.SEED per seed). A bare
        # star-imported SEED would be the frozen import-time constant (7).
        'seed': config.SEED,
        'window_size': WINDOW_SIZE,
        'sliding_pad_mode': SLIDING_PAD_MODE,
        'tinner_mode': TINNER_MODE,
        'touter_mode': TOUTER_MODE,
        'touter_tau': TOUTER_TAU,
        'touter_scale': TOUTER_SCALE,
        'touter_ema_form': TOUTER_EMA_FORM,
        'anchor_lead': ANCHOR_LEAD,
        'loss_weights': LOSS_WEIGHTS,
        'physics_bound_weight': PHYSICS_BOUND_WEIGHT,
        'other_ch_mode': OTHER_CH_MODE,
        'input_lookahead': INPUT_LOOKAHEAD,
        'anchor_scale': ANCHOR_SCALE,
        'pos_gap_floor': POS_GAP_FLOOR,
        'pos_case_gate': POS_CASE_GATE,
        'pos_floor_soft': POS_FLOOR_SOFT,
        'pos_temp_gate': POS_TEMP_GATE,
        'pos_temp_soft': POS_TEMP_SOFT,
        'pos_abs_head': POS_ABS_HEAD,
        'pos_learned_gate': POS_LEARNED_GATE,
        'pos_gate_bias': POS_GATE_BIAS,
        'pos_learned_pool': POS_LEARNED_POOL,
        'pos_anchored_fallback': POS_ANCHORED_FALLBACK,
        'pos_fit_clean': POS_FIT_CLEAN,
        'case_flag_input': CASE_FLAG_INPUT,
        'exclude_both_phase': EXCLUDE_BOTH_PHASE,
        'output_size': OUTPUT_SIZE,
        'model_split': MODEL_SPLIT,
        'pos_time_prior': POS_TIME_PRIOR,
        'variants': VARIANTS,
        'input_dims': INPUT_DIMS,
        'latest_params': LATEST_PARAMS,
        # H9: numerics provenance (TF32 knob + the versions / GPU behind it).
        'tf32': TF32,                     # "0" | "1" | "legacy"
        'tavg_mask': TAVG_MASK,           # H2: "" | split | valonly | random:<seed>
        'tavg_mask_n_train': len(tr_m),
        'tavg_mask_n_val': len(va_m),
        'tavg_mask_train_sha256': sha256_lines(tr_m) if tr_m else None,
        'tavg_mask_val_sha256': sha256_lines(va_m) if va_m else None,
        'sealed': bool(SEAL_TEST),        # H3: True = validation-scored, test set never read
        # A4: algebraic T_avg relabel -- the table (as given, re-resolvable by
        # eval_checkpoint), its digest and how many files it touched.
        'relabel_csv': RELABEL_CSV or None,
        'relabel_path': RELABEL_CSV_PATH,
        'relabel_sha256': sha256_file(RELABEL_CSV_PATH) if RELABEL_CSV_PATH else None,
        'relabel_n_train': len(rl_tr),
        'relabel_n_val': len(rl_va),
        'runtime': runtime_env_info(),
        'data': {
            'train_subdir': TRAIN_SUBDIR,
            'test_subdir': TEST_SUBDIR,
            'manual_split_enabled': MANUAL_SPLIT_ENABLED,
            'val_split': VAL_SPLIT,
            'val_per_type': VAL_PER_TYPE,
            'train_exclude_type': TRAIN_EXCLUDE_TYPE,
            'outer_test_frac': OUTER_TEST_FRAC,
            'inner_val_frac': INNER_VAL_FRAC,
            'split_seed': SPLIT_SEED,
            # Round 10 / H1: which json, and its digest.
            'split_name': split_file_name() if VAL_SPLIT == "stratified" else None,
            'split_tag': SPLIT_TAG if VAL_SPLIT == "stratified" else None,
            'split_sha256': (sha256_file(split_file_path())
                             if VAL_SPLIT == "stratified" and os.path.isfile(split_file_path()) else None),
        },
        'persistence': {
            'resume_enabled': RESUME_ENABLED,
            'save_predictions': SAVE_PREDICTIONS,
            'checkpoint_every_n_epochs': CHECKPOINT_EVERY_N_EPOCHS,
        },
    }
    return cfg


def save_run_config_snapshot():
    """One-time JSON dump of the config used for this run, written to
    runs/<RUN_NAME>/run_config.json. Helpful provenance for any reviewer
    who later inspects the output folder."""
    cfg = run_config_dict()
    path = os.path.join(run_dir(), "run_config.json")
    # A finished run's provenance is frozen: once every variant carries a
    # done.flag the snapshot is not rewritten, so a later launch (with
    # whatever env it happens to have) cannot relabel what was trained
    # (2026-09-12 review: the MAX_EPOCHS guard depended on this file).
    if os.path.isfile(path) and all(is_variant_complete(v) for v in VARIANTS):
        return path
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path


def is_variant_complete(variant):
    return os.path.isfile(os.path.join(variant_dir(variant), "done.flag"))


def has_resume_state(variant):
    return os.path.isfile(os.path.join(variant_dir(variant), "resume_state.pt"))


def save_resume_state(variant, payload):
    """Atomic save of the in-progress training state."""
    var_dir = variant_dir(variant)
    os.makedirs(var_dir, exist_ok=True)
    final = os.path.join(var_dir, "resume_state.pt")
    tmp = final + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, final)


def load_resume_state(variant):
    path = os.path.join(variant_dir(variant), "resume_state.pt")
    if not os.path.isfile(path):
        return None
    try:
        # weights_only=False: we are explicitly loading our own RNG / optimizer
        # objects (PyTorch >=2.6 default would block these as 'unsafe').
        return torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        # weights_only kwarg unsupported on older PyTorch versions.
        return torch.load(path, map_location=DEVICE)


def remove_resume_state(variant):
    path = os.path.join(variant_dir(variant), "resume_state.pt")
    if os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            pass


def save_train_history(variant, train_hist, val_hist, lr_hist,
                       best_epoch, total_epoch, val_mae, param_breakdown,
                       stages=None, val_hist_ch=None):
    var_dir = variant_dir(variant)
    os.makedirs(var_dir, exist_ok=True)
    payload = {
        'variant': variant,
        'train_hist': list(train_hist),
        'val_hist': list(val_hist),
        'lr_hist': list(lr_hist),
        'best_epoch': int(best_epoch),
        'total_epoch': int(total_epoch),
        'best_val_mae': float(val_mae),
        'param_breakdown': param_breakdown,
        'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    if val_hist_ch is not None:
        # Per-epoch UNWEIGHTED validation MAE per channel (T_inner, T_outer,
        # T_avg; scaled units), the exact mean over validation files -- so
        # which channel drove the stopping decision can be read off the run
        # (round 9's question). val_hist itself stays the weighted scalar
        # the stop is decided on.
        payload['val_hist_ch'] = [list(map(float, v)) for v in val_hist_ch]
    if stages is not None:
        # MODEL_SPLIT != shared: the stages' own numbers. The top-level
        # histories are the stages concatenated; best_epoch is the last
        # stage's best epoch offset by the earlier stages' lengths, so it
        # indexes the concatenated curve; total_epoch is the sum.
        payload['stages'] = stages
    path = os.path.join(var_dir, "train_history.json")
    with open(path, "w") as f:
        json.dump(_to_jsonable(payload), f, indent=2)


def load_train_history(variant):
    path = os.path.join(variant_dir(variant), "train_history.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_predictions_npz(variant, raw_predictions, path=None):
    """raw_predictions: list of dicts with keys 'case_id', 'inv_actual', 'inv_pred'.
    path: explicit destination (H3: predictions_val.npz, or eval_checkpoint's
    --out-dir); None = the legacy variants/<v>/predictions.npz, gated on
    SAVE_PREDICTIONS."""
    if path is None:
        if not SAVE_PREDICTIONS:
            return
        var_dir = variant_dir(variant)
        os.makedirs(var_dir, exist_ok=True)
        path = os.path.join(var_dir, "predictions.npz")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez_compressed(
        path,
        case_ids=np.array([p['case_id'] for p in raw_predictions], dtype=object),
        inv_actuals=np.array([p['inv_actual'] for p in raw_predictions],
                             dtype=object),
        inv_preds=np.array([p['inv_pred'] for p in raw_predictions], dtype=object),
    )


def save_variant_meta(variant, meta):
    var_dir = variant_dir(variant)
    os.makedirs(var_dir, exist_ok=True)
    with open(os.path.join(var_dir, "meta.json"), "w") as f:
        json.dump(_to_jsonable(meta), f, indent=2)


def load_variant_meta(variant):
    path = os.path.join(variant_dir(variant), "meta.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def mark_variant_done(variant):
    var_dir = variant_dir(variant)
    os.makedirs(var_dir, exist_ok=True)
    with open(os.path.join(var_dir, "done.flag"), "w") as f:
        f.write(f"completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    # Resume state no longer needed once the variant is fully done.
    remove_resume_state(variant)


# ============================================================
# Round 10 / H4: constants sidecar
# ============================================================
# Everything a checkpoint needs besides its weights, written next to it as
# checkpoints/best_gru_variant_<v>.constants.json: the MinMax scaler (so the
# training CSVs never have to be re-read), the data-derived constants
# train.fit_model_constants attaches to the model, the loss weights, the
# knobs, the split / drop / mask digests, git HEAD and the numerics. RULE:
# the constants belong to the checkpoint's OWN split -- a sidecar is never
# combined with another run's data, and eval_checkpoint prefers it over a
# refit for that reason.
SIDECAR_FORMAT = "tes_gru.constants/1"
SIDECAR_KEYS = ("format", "written_at", "variant", "checkpoint", "scaler", "constants",
                "loss_weights", "loss_weights_realised", "physics_bound_weight", "knobs",
                "split", "git_head", "runtime", "tf32", "seed")
# Model attributes fit_model_constants sets (None when a mode does not use one).
MODEL_CONSTANT_ATTRS = ("tinner_anchor", "touter_anchor", "touter_ref_affine", "raw_affine",
                        "step_anchor", "other_anchor", "pos_head", "pos_mid",
                        "pos_time_prior", "pos_time_affine")
_GIT_HEAD = {}


def constants_sidecar_path(ckpt_path):
    """checkpoints/best_gru_variant_<v>.pth -> .../best_gru_variant_<v>.constants.json"""
    base = ckpt_path[:-4] if ckpt_path.endswith(".pth") else ckpt_path
    return base + ".constants.json"


def model_constants_dict(model):
    return {k: getattr(model, k, None) for k in MODEL_CONSTANT_ATTRS}


def git_head():
    """HEAD of the checkout that contains this package, or None (tolerated:
    the lab server may run from an export without .git)."""
    if "head" not in _GIT_HEAD:
        head = None
        try:
            import subprocess
            r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BASE_DIR,
                               capture_output=True, text=True, timeout=15)
            if r.returncode == 0 and r.stdout.strip():
                head = r.stdout.strip()
        except Exception:
            head = None
        _GIT_HEAD["head"] = head
    return _GIT_HEAD["head"]


def write_constants_sidecar(path, model, scaler, extra=None):
    """Write the constants sidecar for `model` / `scaler` to `path` (atomic).
    extra: free-form provenance merged under "extra" (variant, checkpoint,
    best_epoch, ...); an extra 'loss_weights_realised' overrides the recorded
    realised weights (H8: channel-0 weight / kc)."""
    from .data import ThermalDataset, load_split_spec, get_tavg_mask_sets
    extra = dict(extra or {})
    cols = list(ThermalDataset.BASE_COLS)
    assert len(scaler.scale_) == len(cols), (len(scaler.scale_), cols)
    tr_m, va_m = get_tavg_mask_sets()
    split = {'val_split': VAL_SPLIT, 'split_name': None, 'split_tag': None, 'split_sha256': None,
             'n_train': None, 'n_val': None, 'n_drop': 0, 'drop_sha256': None,
             'train_exclude_type': TRAIN_EXCLUDE_TYPE,
             'tavg_mask': TAVG_MASK, 'tavg_mask_n_train': len(tr_m), 'tavg_mask_n_val': len(va_m),
             'tavg_mask_train_sha256': sha256_lines(tr_m) if tr_m else None,
             'tavg_mask_val_sha256': sha256_lines(va_m) if va_m else None}
    if VAL_SPLIT == "stratified" and os.path.isfile(split_file_path()):
        spec = load_split_spec(split_file_path())
        drop = list(spec.get("drop") or [])
        split.update(split_name=split_file_name(), split_tag=SPLIT_TAG,
                     split_sha256=sha256_file(split_file_path()),
                     n_train=len(spec["train"]), n_val=len(spec["val"]),
                     n_drop=len(drop), drop_sha256=sha256_lines(drop) if drop else None)
    knobs = run_config_dict()
    for k in ("run_name", "started_or_resumed_at", "runtime", "persistence"):
        knobs.pop(k, None)
    d = {
        'format': SIDECAR_FORMAT,
        'written_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'variant': extra.get('variant'),
        'checkpoint': extra.get('checkpoint'),
        'scaler': {
            'columns': cols,
            'feature_range': list(scaler.feature_range),
            'data_min_': scaler.data_min_, 'data_max_': scaler.data_max_,
            'data_range_': scaler.data_range_,
            'scale_': scaler.scale_, 'min_': scaler.min_,
            'n_samples_seen_': int(getattr(scaler, 'n_samples_seen_', 0)),
        },
        'constants': model_constants_dict(model),
        'loss_weights': list(LOSS_WEIGHTS),
        'loss_weights_realised': list(extra.pop('loss_weights_realised', LOSS_WEIGHTS)),
        'physics_bound_weight': PHYSICS_BOUND_WEIGHT,
        'knobs': knobs,
        'split': split,
        'git_head': git_head(),
        'runtime': runtime_env_info(),
        'tf32': TF32,
        'seed': config.SEED,
        'extra': extra,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_to_jsonable(d), f, indent=2)
    os.replace(tmp, path)
    return path


def load_constants_sidecar(path):
    """Read a sidecar written by write_constants_sidecar; validates the format
    and the documented top-level keys."""
    with open(path) as f:
        d = json.load(f)
    if d.get('format') != SIDECAR_FORMAT:
        raise RuntimeError("%s: not a constants sidecar (format %r)" % (path, d.get('format')))
    missing = [k for k in SIDECAR_KEYS if k not in d]
    if missing:
        raise RuntimeError("%s: sidecar lacks keys %s" % (path, missing))
    return d


