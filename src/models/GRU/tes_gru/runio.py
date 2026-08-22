"""Run-directory layout + checkpoint / resume / meta / history I/O."""
import os
import json
import shutil
import time

import numpy as np
import torch

from . import config            # live access to the mutable config.RUN_NAME
from .config import *
from .utils import _to_jsonable


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


def save_run_config_snapshot():
    """One-time JSON dump of the config used for this run, written to
    runs/<RUN_NAME>/run_config.json. Helpful provenance for any reviewer
    who later inspects the output folder."""
    cfg = {
        'run_name': config.RUN_NAME,
        'started_or_resumed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        # Read the LIVE value (main loop rebinds config.SEED per seed). A bare
        # star-imported SEED would be the frozen import-time constant (7).
        'seed': config.SEED,
        'window_size': WINDOW_SIZE,
        'sliding_pad_mode': SLIDING_PAD_MODE,
        'tinner_mode': TINNER_MODE,
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
        'output_size': OUTPUT_SIZE,
        'variants': VARIANTS,
        'input_dims': INPUT_DIMS,
        'latest_params': LATEST_PARAMS,
        'data': {
            'train_subdir': TRAIN_SUBDIR,
            'test_subdir': TEST_SUBDIR,
            'manual_split_enabled': MANUAL_SPLIT_ENABLED,
            'outer_test_frac': OUTER_TEST_FRAC,
            'inner_val_frac': INNER_VAL_FRAC,
            'split_seed': SPLIT_SEED,
        },
        'persistence': {
            'resume_enabled': RESUME_ENABLED,
            'save_predictions': SAVE_PREDICTIONS,
            'checkpoint_every_n_epochs': CHECKPOINT_EVERY_N_EPOCHS,
        },
    }
    path = os.path.join(run_dir(), "run_config.json")
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
                       best_epoch, total_epoch, val_mae, param_breakdown):
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
    path = os.path.join(var_dir, "train_history.json")
    with open(path, "w") as f:
        json.dump(_to_jsonable(payload), f, indent=2)


def load_train_history(variant):
    path = os.path.join(variant_dir(variant), "train_history.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_predictions_npz(variant, raw_predictions):
    """raw_predictions: list of dicts with keys 'case_id', 'inv_actual', 'inv_pred'."""
    if not SAVE_PREDICTIONS:
        return
    var_dir = variant_dir(variant)
    os.makedirs(var_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(var_dir, "predictions.npz"),
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


