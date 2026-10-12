"""Round 10 / H2 check: per-file T_avg loss mask (split / random:<seed> (B7) /
valonly (B8)). Run from src/models/GRU:  python round10_checks/check_h2.py

No training. PASS / FAIL per item, non-zero exit on any FAIL.
"""
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # src/models/GRU
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(HERE)

from check_h9 import clean_env, fresh_import   # noqa: E402

VS3 = "round10_vs3_s42.json"
RESULTS = []


def report(item, ok, detail=""):
    RESULTS.append(bool(ok))
    print("%s  %s  %s" % ("PASS" if ok else "FAIL", item, detail), flush=True)


def base_env(extra):
    env = {"VARIANTS": "forward_direct", "VAL_SPLIT": "stratified", "VAL_PER_TYPE": "2",
           "HIDDEN_SIZE": "128", "NUM_LAYERS": "1", "DROPOUT": "0.0", "TRAIN_EXCLUDE_TYPE": "",
           "TINNER_MODE": "anchor", "ANCHOR_LEAD": "1", "OTHER_CH_MODE": "abs", "LOSS_WEIGHTS": "1,6,3",
           "PHYSICS_BOUND_WEIGHT": "0", "TF32": "0", "SPLIT_NAME": VS3}
    env.update(extra)
    return env


def main():
    import eval_checkpoint as ec
    import torch
    from torch.utils.data import DataLoader, ConcatDataset
    clean_env(ec)

    # ================================================================== split mode
    config = fresh_import(base_env({"TAVG_MASK": "split"}))
    from tes_gru.data import (load_all_data, ThermalDataset, get_tavg_mask_sets, is_tavg_masked,
                              split_key, load_split_spec, ITEM_LOSS_MASK, ITEM_INP_LAST)
    from tes_gru.models import build_model, weighted_loss, masked_loss_terms
    from tes_gru.rollout import variant_to_encoder, run_rollout_train
    from tes_gru.train import fit_model_constants, validate_epoch
    from tes_gru.config import DEVICE, LATEST_PARAMS, set_seed
    variant = "forward_direct"
    spec = load_split_spec(config.split_file_path())
    mask = set(spec["tavg_mask"])
    exp_tr = mask & set(spec["train"])
    exp_va = mask & set(spec["val"])
    want_va = {"Case 1 (linear charging)/Case 16.csv",
               "Case 9 (non-linear discharging)/Case New 1 (Tejas).csv",
               "Case 9 (non-linear discharging)/Case New 4 (Tejas).csv"}

    tr_m, va_m = get_tavg_mask_sets()
    report("(b) split mode: %d masked training files (from the json)" % len(exp_tr),
           tr_m == exp_tr and len(tr_m) == 115, "%d" % len(tr_m))
    report("(b) split mode: masked validation files = Case 1/Case 16, Case 9/New 1, Case 9/New 4",
           va_m == exp_va == want_va, str(sorted(va_m)))

    set_seed(7)
    train_dfs, val_dfs, test_dfs, scaler = load_all_data(include_test=False)
    tr_sets = [ThermalDataset(df, variant, scaler=scaler, file_name=fn, tavg_masked=is_tavg_masked(fn, tr_m))
               for df, fn in train_dfs]
    va_sets = [ThermalDataset(df, variant, scaler=scaler, file_name=fn, tavg_masked=is_tavg_masked(fn, va_m))
               for df, fn in val_dfs]
    plain_sets = [ThermalDataset(df, variant, scaler=scaler, file_name=fn) for df, fn in val_dfs[:3]]
    report("(b) sealed loader excludes all test CSVs", not test_dfs)
    n_tr0 = sum(int(ds.loss_mask[0, 2].item() == 0.0) for ds in tr_sets)
    n_va0 = sum(int(ds.loss_mask[0, 2].item() == 0.0) for ds in va_sets)
    keys_va0 = {split_key(ds.file_name) for ds in va_sets if ds.loss_mask[0, 2].item() == 0.0}
    report("(b) datasets: loss_mask T_avg entry 0 on exactly 115 train / 3 val files, 1 elsewhere",
           n_tr0 == 115 and n_va0 == 3 and keys_va0 == want_va
           and all(ds.loss_mask[0, :2].tolist() == [1.0, 1.0] for ds in tr_sets + va_sets)
           and all(ds.loss_mask[0].tolist() == [1.0, 1.0, 1.0] for ds in plain_sets),
           "train %d val %d" % (n_tr0, n_va0))
    report("(b) item tuple: inp_last at %d, loss_mask (3,) at %d (last)" % (ITEM_INP_LAST, ITEM_LOSS_MASK),
           len(tr_sets[0][0]) == ITEM_LOSS_MASK + 1 and tuple(tr_sets[0][0][ITEM_LOSS_MASK].shape) == (3,))

    model = build_model(variant, LATEST_PARAMS, variant_to_encoder(variant)).to(DEVICE)
    fit_model_constants(model, variant, train_dfs, scaler)
    w = torch.tensor(config.LOSS_WEIGHTS, dtype=torch.float32)

    # ================================================================== (a) all-ones mask
    g = torch.Generator().manual_seed(0)
    p = torch.rand(5, 40, 3, generator=g)
    t = torch.rand(5, 40, 3, generator=g)
    ones = torch.ones(5, 3)
    d_rand = abs(weighted_loss(p, t, weights=w).item() - weighted_loss(p, t, weights=w, mask=ones).item())
    va_loader = DataLoader(ConcatDataset(va_sets), batch_size=16)
    batch = next(iter(va_loader))
    items = [b.to(DEVICE) if torch.is_tensor(b) else b for b in batch]
    model.eval()
    with torch.no_grad():
        preds = run_rollout_train(model, variant, items[0], items[1], items[3], items[4],
                                  tf_prob=0.0, inp_last=items[ITEM_INP_LAST])
    l_leg = weighted_loss(preds, items[1], weights=w, raw_affine=model.raw_affine).item()
    l_one = weighted_loss(preds, items[1], weights=w, raw_affine=model.raw_affine,
                          mask=torch.ones_like(items[ITEM_LOSS_MASK])).item()
    report("(a) all-ones mask reproduces the legacy loss (1e-7): random + real batch",
           d_rand <= 1e-7 and abs(l_leg - l_one) <= 1e-7,
           "random |d| %.1e  real |d| %.1e (loss %.6f)" % (d_rand, abs(l_leg - l_one), l_leg))
    num, den = masked_loss_terms(preds, items[1], w.to(DEVICE), items[ITEM_LOSS_MASK])
    B, T = items[1].shape[0], items[1].shape[1]
    n_masked_in_batch = int((items[ITEM_LOSS_MASK][:, 2] == 0).sum().item())
    report("(a) masked denominator counts unmasked (file, step, channel) entries",
           abs(den.item() - (B * T * 3 - n_masked_in_batch * T)) < 0.5,
           "den %.0f, B %d T %d masked files in batch %d" % (den.item(), B, T, n_masked_in_batch))

    # ================================================================== (c) gradient
    masked_train = [ds for ds in tr_sets if ds.tavg_masked][:4]
    assert len(masked_train) == 4
    mb = next(iter(DataLoader(ConcatDataset(masked_train), batch_size=4)))
    mitems = [b.to(DEVICE) if torch.is_tensor(b) else b for b in mb]
    model.train()
    model.zero_grad()
    mpreds = run_rollout_train(model, variant, mitems[0], mitems[1], mitems[3], mitems[4],
                               tf_prob=0.0, inp_last=mitems[ITEM_INP_LAST])
    loss = weighted_loss(mpreds, mitems[1], weights=w, raw_affine=model.raw_affine, mask=mitems[ITEM_LOSS_MASK])
    loss.backward()
    gw = model.fc.weight.grad
    gb = model.fc.bias.grad
    report("(c) masked-only batch: grad of fc.weight[2] / fc.bias[2] exactly 0, rows 0-1 non-zero",
           gw is not None and float(gw[2].abs().max()) == 0.0 and float(gb[2].abs()) == 0.0
           and float(gw[0].abs().max()) > 0 and float(gw[1].abs().max()) > 0,
           "|g| rows: %.2e %.2e %.2e" % tuple(float(gw[i].abs().max()) for i in range(3)))
    model.zero_grad()

    # ================================================================== (e) validation scalar
    model.eval()
    va_loader = DataLoader(ConcatDataset(va_sets), batch_size=16)      # 20 files -> 16 + 4
    assert len(va_loader) == 2, len(va_loader)
    s_loop, ch_loop = validate_epoch(model, variant, va_loader, w, model.raw_affine, use_mask=True)
    num = den = 0.0
    ch_num = torch.zeros(3, dtype=torch.float64)
    ch_den = torch.zeros(3, dtype=torch.float64)
    w64 = w.double()
    with torch.no_grad():
        for ds in va_sets:
            it = ds[0]
            pr = run_rollout_train(model, variant, it[0][None].to(DEVICE), it[1][None].to(DEVICE),
                                   it[3][None].to(DEVICE), it[4][None].to(DEVICE),
                                   tf_prob=0.0, inp_last=it[ITEM_INP_LAST].reshape(1).to(DEVICE))
            ae = (pr[0].double().cpu() - it[1].double())           # (T, 3)
            ae = ae.abs()
            m = ds.loss_mask[0].double()
            num += float((ae * w64 * m).sum())
            den += float(ae.shape[0] * m.sum())
            ch_num += (ae * m).sum(0)
            ch_den += ae.shape[0] * m
    s_off = num / den
    ch_off = (ch_num / ch_den).tolist()
    d_s = abs(s_loop - s_off)
    d_ch = max(abs(a - b) for a, b in zip(ch_loop, ch_off))
    report("(e) validate_epoch scalar == offline masked recomputation (1e-6)", d_s <= 1e-6,
           "loop %.8f offline %.8f |d| %.1e" % (s_loop, s_off, d_s))
    report("(e) validate_epoch per-channel == offline over unmasked files (1e-6)", d_ch <= 1e-6,
           "|d| %.1e; T_avg over %d files, T_inner/T_outer over %d" % (
               d_ch, int(ch_den[2].item() // ae.shape[0]), int(ch_den[0].item() // ae.shape[0])))
    report("(e) T_avg channel averaged over 17 files, the others over 20",
           int(round(ch_den[2].item() / ae.shape[0])) == 17 and int(round(ch_den[0].item() / ae.shape[0])) == 20)
    # legacy arithmetic vs masked arithmetic on an UNMASKED set: same number to 1e-7
    va_plain = [ThermalDataset(df, variant, scaler=scaler, file_name=fn) for df, fn in val_dfs]
    pl_loader = DataLoader(ConcatDataset(va_plain), batch_size=16)
    s_leg, ch_leg = validate_epoch(model, variant, pl_loader, w, model.raw_affine, use_mask=False)
    s_msk, ch_msk = validate_epoch(model, variant, pl_loader, w, model.raw_affine, use_mask=True)
    report("(e) all-ones masks: masked validation arithmetic == legacy (1e-7)",
           abs(s_leg - s_msk) <= 1e-7 and max(abs(a - b) for a, b in zip(ch_leg, ch_msk)) <= 1e-7,
           "|d| %.1e" % abs(s_leg - s_msk))

    # ================================================================== run_config / plumbing (split)
    import tes_gru.runio as runio
    tmp = tempfile.mkdtemp(prefix="check_h2_")
    runio.run_dir = lambda: tmp
    try:
        rc = json.load(open(runio.save_run_config_snapshot()))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    from tes_gru.utils import sha256_lines
    report("(f) run_config carries tavg_mask + counts + sha256 of the sorted mask lists",
           rc.get("tavg_mask") == "split" and rc.get("tavg_mask_n_train") == 115 and rc.get("tavg_mask_n_val") == 3
           and rc.get("tavg_mask_train_sha256") == sha256_lines(tr_m)
           and rc.get("tavg_mask_val_sha256") == sha256_lines(va_m),
           "%s %s/%s %s" % (rc.get("tavg_mask"), rc.get("tavg_mask_n_train"), rc.get("tavg_mask_n_val"),
                            str(rc.get("tavg_mask_train_sha256"))[:12]))
    report("(f) eval_checkpoint.env_from_config passes TAVG_MASK", ec.env_from_config(rc).get("TAVG_MASK") == "split")
    import export_results as er
    expected_family = "sealed_vs3_mk_baseline" if rc.get("sealed") else "vs3_mk_baseline"
    report("(f) export_results.family carries the mask tag and the recorded seal",
           er.family(rc) == expected_family, er.family(rc))
    report("(f) run name carries mk", config.RUN_NAME_BASE.endswith("_vs3_mk"), config.RUN_NAME_BASE)

    # ================================================================== (d) random:7 and valonly
    clean_env(ec)
    config = fresh_import(base_env({"TAVG_MASK": "random:7"}))
    from tes_gru.data import get_tavg_mask_sets as gm_r
    tr_r, va_r = gm_r()
    report("(d) random:7 masks exactly 115 training files, none a validation file",
           len(tr_r) == 115 and tr_r <= set(spec["train"]) and not (tr_r & set(spec["val"])),
           "%d masked, %d overlap with the json's mask" % (len(tr_r), len(tr_r & exp_tr)))
    report("(d) random:7 validation mask unchanged (= the json's 3 files)", va_r == want_va)
    tr_r2, _ = gm_r()
    report("(d) random:7 is deterministic (same draw twice)", tr_r2 == tr_r)
    report("(f) run name carries mkr7", config.RUN_NAME_BASE.endswith("_vs3_mkr7"), config.RUN_NAME_BASE)

    clean_env(ec)
    config = fresh_import(base_env({"TAVG_MASK": "valonly"}))
    from tes_gru.data import get_tavg_mask_sets as gm_v
    tr_v, va_v = gm_v()
    report("(d) valonly (B8): no training file masked, validation mask = the json's 3 files",
           tr_v == set() and va_v == want_va)
    report("(f) run name carries mkv", config.RUN_NAME_BASE.endswith("_vs3_mkv"), config.RUN_NAME_BASE)

    clean_env(ec)
    config = fresh_import(base_env({}))
    from tes_gru.data import get_tavg_mask_sets as gm_0
    report("(f) legacy TAVG_MASK='' -> empty sets, no tag",
           gm_0() == (set(), set()) and config.RUN_NAME_BASE.endswith("_vs3"), config.RUN_NAME_BASE)

    n_fail = RESULTS.count(False)
    print("\ncheck_h2: %d items, %d FAIL" % (len(RESULTS), n_fail))
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
