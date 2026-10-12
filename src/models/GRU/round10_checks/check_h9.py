"""Round 10 / H9 check: anchor last-step fix, raw_affine pass-through, dropout
guard, TF32 knob. Run from src/models/GRU:  python round10_checks/check_h9.py

No training. Prints PASS / FAIL per item, exits non-zero on any FAIL.
Needs the data under AI-TES/data and the Round 9 champion run directory
(packaged in round10_resources/round9; override with CHAMPION_RUN_DIR).
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # src/models/GRU
sys.path.insert(0, HERE)
os.chdir(HERE)

CHAMP = os.environ.get("CHAMPION_RUN_DIR") or os.path.normpath(os.path.join(
    HERE, "round10_resources", "round9",
    "2026-08-06_forward_direct_W10_Tin-anchor_l1_w1-6-3_oth-abs_h160x1dp0.2_ep2000_vs2_seed7"))
TOL_STEPS = 1e-6          # C, old tail-hold vs true-last-input on validation
TOL_LAST = 1e-4           # C, last step vs ka*Input_T(N)+kb+kc*z

RESULTS = []


def report(item, ok, detail=""):
    RESULTS.append(bool(ok))
    print("%s  %s  %s" % ("PASS" if ok else "FAIL", item, detail), flush=True)


ENV_VARS_EXTRA = ["LOSS_WEIGHTS", "VARIANTS", "HIDDEN_SIZE", "NUM_LAYERS", "DROPOUT", "MAX_EPOCHS",
                  "VAL_SPLIT", "VAL_PER_TYPE", "TRAIN_EXCLUDE_TYPE", "SEEDS", "SPLIT_NAME", "TAVG_MASK", "TF32"]


def clean_env(ec):
    for var in list(ec.KEYS.values()) + ENV_VARS_EXTRA:
        os.environ.pop(var, None)


def fresh_import(env):
    block_test_csv_reads()
    for k, v in env.items():
        os.environ[k] = v
    for m in [m for m in list(sys.modules) if m.startswith("tes_gru")]:
        del sys.modules[m]
    import tes_gru.config as config
    return config


def block_test_csv_reads():
    """Make accidental reopening of the real sealed test CSVs fail loudly.

    Historical Round 9 NPZ/CSV report replay is permitted; the loader's
    training and validation CSVs remain readable.
    """
    import pandas as pd
    if getattr(pd.read_csv, "_round10_sealed_guard", False):
        return
    original = pd.read_csv

    def guarded(path, *args, **kwargs):
        name = getattr(path, "name", path)
        if "test_70_cases" in str(name).replace("\\", "/").lower():
            raise RuntimeError("P0 checks may not reopen sealed test CSVs: %s" % name)
        return original(path, *args, **kwargs)

    guarded._round10_sealed_guard = True
    pd.read_csv = guarded


def score_validation_case(model, ds, variant, config, _ev):
    """Run evaluate.test_model on the single dataset, return inv_pred (N, 5)."""
    captured = []
    _ev.SAVE_PREDICTIONS = True
    _ev.save_predictions_npz = lambda v, rp: captured.extend(rp)
    config.RUN_NAME = "_check_h9_tmp"
    out_dir = tempfile.mkdtemp(prefix="check_h9_")
    try:
        _ev.test_model(variant, model, [ds], output_dir=out_dir, set_name="check_h9")
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
    return captured[0]["inv_pred"].astype("float64")


def main():
    import eval_checkpoint as ec
    import numpy as np
    import torch

    # ------------------------------------------------------------------ (a)
    c = json.load(open(os.path.join(CHAMP, "run_config.json")))
    env = ec.env_from_config(c)
    clean_env(ec)
    env["TF32"] = "legacy"
    config = fresh_import(env)
    from tes_gru.data import load_all_data, ThermalDataset
    from tes_gru.models import build_model, weighted_loss
    from tes_gru.rollout import variant_to_encoder, run_rollout_train
    from tes_gru.train import fit_model_constants
    import tes_gru.evaluate as _ev
    from tes_gru.config import DEVICE, LATEST_PARAMS, set_seed

    variant = c["variants"][0]
    set_seed(c["seed"])
    train_dfs, val_dfs, test_dfs, scaler = load_all_data(include_test=False)
    model = build_model(variant, LATEST_PARAMS, variant_to_encoder(variant)).to(DEVICE)
    fit_model_constants(model, variant, train_dfs, scaler)
    ck = os.path.join(CHAMP, "checkpoints", "best_gru_variant_%s.pth" % variant)
    model.load_state_dict(torch.load(ck, map_location=DEVICE))
    model.eval()

    # The regression input is a validation file. Replaying the historical
    # tail-hold on the same weights isolates exactly the H9 reconstruction.
    df_val, fn_val = max(val_dfs, key=lambda pair: abs(float(pair[0]["Input Temperature (C)"].iloc[-1])
                                                   - float(pair[0]["Input Temperature (C)"].iloc[-2])))
    ds = ThermalDataset(df_val, variant, scaler=scaler, file_name=fn_val)
    assert len(ds[0]) >= 11, "dataset item tuple has %d entries (inp_last missing?)" % len(ds[0])
    report("(a) validation loader returns no test files", not test_dfs)
    with np.load(os.path.join(CHAMP, "variants", variant, "predictions.npz"), allow_pickle=True) as archive:
        archive_ok = (len(archive["case_ids"]) == 70 and len(archive["inv_preds"]) == 70
                      and all(np.asarray(a).ndim == 2 and np.asarray(a).shape[1] == 5
                              and np.isfinite(np.asarray(a, dtype=float)).all() for a in archive["inv_preds"]))
    report("(a) historical prediction payload has 70 finite five-column cases (archive integrity only)", archive_ok)
    COLS = {"T_outer": 1, "T_inner": 2, "T_avg": 3}

    def compare(tag):
        inv_pred = score_validation_case(model, ds, variant, config, _ev)
        true_last = ds.inp_last.clone()
        ds.inp_last[:] = ds.X[0, -1, 1].item()
        try:
            ref = score_validation_case(model, ds, variant, config, _ev)
        finally:
            ds.inp_last[:] = true_last
        assert inv_pred.shape == ref.shape, (inv_pred.shape, ref.shape)
        d = {ch: float(np.abs(inv_pred[:-1, j] - ref[:-1, j]).max()) for ch, j in COLS.items()}
        worst = max(d.values())
        print("   [%s] validation old/new max |delta|, steps 1..N-1: %s" % (
            tag, "  ".join("%s %.2e" % kv for kv in d.items())))
        # last step: the fix target
        x = ds.X[0].to(DEVICE)
        ic = ds.init_conditions[0:1].to(DEVICE)
        with torch.no_grad():
            hidden = model.init_hidden(1, x0=ic)
            out, _ = model(x.unsqueeze(0), hidden)
        ka, kb, kc = model.tinner_anchor
        zz = float(out[0, -1, 0].item())
        tin_s = ka * float(ds.inp_last[0]) + kb + kc * zz
        col = ThermalDataset.BASE_COLS.index("T_inner (C)")
        tin_c = (tin_s - float(scaler.min_[col])) / float(scaler.scale_[col])
        last_new = float(inv_pred[-1, 2])
        expected_delta = ka * (float(true_last[0]) - float(ds.X[0, -1, 1])) / float(scaler.scale_[col])
        delta_error = abs((last_new - float(ref[-1, 2])) - expected_delta)
        print("   [%s] validation last T_inner new %.4f, old-tail %.4f, expected %.4f C" %
              (tag, last_new, ref[-1, 2], tin_c))
        return worst, abs(last_new - tin_c), delta_error

    def set_tf32(mode):
        config.TF32 = mode
        config.TF32_MATMUL, config.TF32_CUDNN = config.tf32_switches(mode)
        set_seed(c["seed"])
        assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == config.tf32_switches(mode)

    worst1, dlast1, delta_error = compare("TF32=legacy")
    report("(a) TF32=legacy: old/new validation steps 1..N-1 match to <= %g C" % TOL_STEPS,
           worst1 <= TOL_STEPS, "max %.2e C" % worst1)
    set_tf32("1")
    worst_on, _, _ = compare("TF32=1")
    report("(a) TF32=1: old/new validation steps before the tail agree", worst_on <= TOL_STEPS)
    report("(a) last step T_inner == ka*Input_T(N)+kb+kc*z (<= %g C)" % TOL_LAST,
           dlast1 <= TOL_LAST, "|delta| %.2e C" % dlast1)
    report("(a) validation last-step change equals ka times the true-last inlet difference",
           delta_error <= 2e-4, "|delta - expected| %.2e C" % delta_error)

    # TF32=0: report only (the archived numbers were made with cuDNN TF32 on)
    set_tf32("0")
    worst0, dlast0, _ = compare("TF32=0")
    print("   (report) TF32=0: max |delta| %.2e C steps 1..N-1; last-step identity |delta| %.2e C" % (worst0, dlast0))
    report("(a) TF32=0 last step T_inner == ka*Input_T(N)+kb+kc*z (<= %g C)" % TOL_LAST,
           dlast0 <= TOL_LAST, "|delta| %.2e C" % dlast0)
    set_tf32("legacy")

    # ------------------------------------------------------------------ (b)
    from tes_gru import config as _cfg
    assert _cfg.PHYSICS_BOUND_WEIGHT == 0.0, _cfg.PHYSICS_BOUND_WEIGHT
    assert model.raw_affine is not None, "fit_model_constants no longer attaches raw_affine"
    g = torch.Generator().manual_seed(0)
    p = torch.rand(4, 50, 3, generator=g, dtype=torch.float64)
    t = torch.rand(4, 50, 3, generator=g, dtype=torch.float64)
    l_a = weighted_loss(p, t).item()
    l_b = weighted_loss(p, t, raw_affine=model.raw_affine).item()
    ok_rand = abs(l_a - l_b) <= 1e-12
    # real batch: 4 validation files through the champion
    from torch.utils.data import DataLoader, ConcatDataset
    vds = [ThermalDataset(df, variant, scaler=scaler, file_name=fn) for df, fn in val_dfs[:4]]
    batch = next(iter(DataLoader(ConcatDataset(vds), batch_size=4)))
    items = [b.to(DEVICE) if torch.is_tensor(b) else b for b in batch]
    with torch.no_grad():
        preds = run_rollout_train(model, variant, items[0], items[1], items[3], items[4],
                                  tf_prob=0.0, inp_last=items[10])
    w = torch.tensor(_cfg.LOSS_WEIGHTS, dtype=torch.float32)
    l_c = weighted_loss(preds, items[1], weights=w).item()
    l_d = weighted_loss(preds, items[1], weights=w, raw_affine=model.raw_affine).item()
    ok_real = abs(l_c - l_d) <= 1e-12
    report("(b) weighted_loss unchanged by raw_affine at bound weight 0 (1e-12)", ok_rand and ok_real,
           "random |d| %.1e, real batch |d| %.1e" % (abs(l_a - l_b), abs(l_c - l_d)))

    # ------------------------------------------------------------------ (c)
    base_env = dict(os.environ)
    for var in list(ec.KEYS.values()) + ENV_VARS_EXTRA:
        base_env.pop(var, None)
    code = "from tes_gru import config; print('RNB=' + config.RUN_NAME_BASE)"

    def run_name(**kw):
        e = dict(base_env, VARIANTS="forward_direct", **kw)
        pr = subprocess.run([sys.executable, "-c", code], env=e, cwd=HERE,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        lines = [l for l in pr.stdout.splitlines() if l.startswith("RNB=")]
        return lines[-1][4:] if lines else "<import failed: %s>" % pr.stdout[-400:]

    rn1 = run_name(NUM_LAYERS="1", DROPOUT="0.3")
    rn2 = run_name(NUM_LAYERS="2", DROPOUT="0.3")
    rn3 = run_name(NUM_LAYERS="1", DROPOUT="0.0")
    report("(c) NUM_LAYERS=1 DROPOUT=0.3 -> RUN_NAME_BASE carries dp0.0",
           "dp0.0" in rn1 and "dp0.3" not in rn1, rn1)
    report("(c) NUM_LAYERS=2 DROPOUT=0.3 keeps dp0.3 (legacy name)", "dp0.3" in rn2, rn2)
    report("(c) NUM_LAYERS=1 DROPOUT=0.0 and DROPOUT=0.3 resolve to the SAME name", rn1 == rn3, rn3)

    # ------------------------------------------------------------------ (d)
    got = {}
    for mode in ("1", "0", "legacy"):
        set_tf32(mode)
        got[mode] = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    report("(d) set_seed applies the TF32 knob to matmul and cudnn (1 / 0 / legacy)",
           got == {"1": (True, True), "0": (False, False), "legacy": (False, True)}, str(got))
    rn_lg = run_name(NUM_LAYERS="1", DROPOUT="0.0", TF32="legacy")
    report("(d) TF32=legacy adds the tf32c tag", rn_lg.endswith("_tf32c"), rn_lg)
    env_old = ec.env_from_config(c)
    report("(d) eval_checkpoint maps a pre-Round-10 run_config (no tf32 key) to TF32=legacy",
           env_old.get("TF32") == "legacy", str(env_old.get("TF32")))
    rn_tf = run_name(NUM_LAYERS="1", DROPOUT="0.0", TF32="1")
    report("(d) TF32=1 adds the tf32 tag to RUN_NAME_BASE; TF32=0 does not",
           rn_tf.endswith("_tf32") or "_tf32_" in rn_tf, rn_tf)
    report("(d) legacy env has no tf32 tag", "tf32" not in rn3, rn3)

    # ------------------------------------------------------------------ run_config keys
    from tes_gru.config import runtime_env_info
    info = runtime_env_info()
    report("(d) runtime_env_info has torch/cuda/cudnn/gpu/tf32 flags",
           all(k in info for k in ("torch", "cuda", "cudnn", "gpu", "tf32_matmul", "tf32_cudnn")), str(info))

    shutil.rmtree(os.path.join(HERE, "runs", "_check_h9_tmp"), ignore_errors=True)
    n_fail = RESULTS.count(False)
    print("\ncheck_h9: %d items, %d FAIL" % (len(RESULTS), n_fail))
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
