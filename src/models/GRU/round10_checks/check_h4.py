"""Round 10 / H4 check: constants sidecar + loader. Run from src/models/GRU:
python round10_checks/check_h4.py

No training. PASS / FAIL per item, non-zero exit on any FAIL. Needs the data
under AI-TES/data and the Round 9 champion run directory (CHAMPION_RUN_DIR
override). The champion directory is never written to: the sidecar goes to a
temp copy of the run.
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # src/models/GRU
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(HERE)

from check_h9 import CHAMP, clean_env, fresh_import   # noqa: E402

RESULTS = []
VARIANT = "forward_direct"


def report(item, ok, detail=""):
    RESULTS.append(bool(ok))
    print("%s  %s  %s" % ("PASS" if ok else "FAIL", item, detail), flush=True)


def copy_run(tmp, name):
    rd = os.path.join(tmp, name)
    os.makedirs(os.path.join(rd, "checkpoints"))
    os.makedirs(os.path.join(rd, "variants", VARIANT))
    shutil.copy2(os.path.join(CHAMP, "run_config.json"), rd)
    shutil.copy2(os.path.join(CHAMP, "checkpoints", "best_gru_variant_%s.pth" % VARIANT),
                 os.path.join(rd, "checkpoints"))
    shutil.copy2(os.path.join(CHAMP, "variants", VARIANT, "train_history.json"),
                 os.path.join(rd, "variants", VARIANT))
    return rd


def main():
    import eval_checkpoint as ec
    import torch
    from torch.utils.data import DataLoader, ConcatDataset

    c = json.load(open(os.path.join(CHAMP, "run_config.json")))
    env = ec.env_from_config(c)
    clean_env(ec)
    env["TF32"] = "legacy"
    config = fresh_import(env)
    from tes_gru.data import load_all_data, ThermalDataset, scaler_from_sidecar, ITEM_INP_LAST
    from tes_gru.models import build_model
    from tes_gru.rollout import variant_to_encoder, run_rollout_train
    from tes_gru.train import fit_model_constants, set_model_constants
    from tes_gru.runio import (write_constants_sidecar, load_constants_sidecar, SIDECAR_KEYS,
                               MODEL_CONSTANT_ATTRS, constants_sidecar_path)
    from tes_gru.config import DEVICE, LATEST_PARAMS, set_seed
    import tes_gru.data as _data

    tmp = tempfile.mkdtemp(prefix="check_h4_")
    try:
        set_seed(c["seed"])
        train_dfs, val_dfs, test_dfs, scaler = load_all_data(include_test=False)
        report("(a) sealed loader excludes all test CSVs", not test_dfs)
        ck = os.path.join(CHAMP, "checkpoints", "best_gru_variant_%s.pth" % VARIANT)
        state = torch.load(ck, map_location=DEVICE)

        # ---- A: refit (the legacy path)
        model_a = build_model(VARIANT, LATEST_PARAMS, variant_to_encoder(VARIANT)).to(DEVICE)
        fit_model_constants(model_a, VARIANT, train_dfs, scaler)
        model_a.load_state_dict(state); model_a.eval()

        # ---- sidecar to a temp copy of the run
        run_sc = copy_run(tmp, "run_with_sidecar")
        sc_path = constants_sidecar_path(os.path.join(run_sc, "checkpoints", "best_gru_variant_%s.pth" % VARIANT))
        write_constants_sidecar(sc_path, model_a, scaler, extra={"variant": VARIANT, "checkpoint": os.path.basename(ck)})
        d = load_constants_sidecar(sc_path)
        report("sidecar written at checkpoints/best_gru_variant_<v>.constants.json and reloads",
               os.path.basename(sc_path) == "best_gru_variant_%s.constants.json" % VARIANT and d["format"].startswith("tes_gru.constants"),
               "%d bytes" % os.path.getsize(sc_path))

        # ---- B: from the sidecar only
        model_b = build_model(VARIANT, LATEST_PARAMS, variant_to_encoder(VARIANT)).to(DEVICE)
        set_model_constants(model_b, d["constants"])
        model_b.load_state_dict(state); model_b.eval()
        scaler_b = scaler_from_sidecar(d)

        # ============================================================== (a)
        va_a = [ThermalDataset(df, VARIANT, scaler=scaler, file_name=fn) for df, fn in val_dfs]
        va_b = [ThermalDataset(df, VARIANT, scaler=scaler_b, file_name=fn) for df, fn in val_dfs]
        dX = max(float((a.X - b.X).abs().max()) for a, b in zip(va_a, va_b))
        dY = max(float((a.Y - b.Y).abs().max()) for a, b in zip(va_a, va_b))
        report("(a) sidecar scaler reproduces the fitted scaler's scaled inputs / targets on the 20 validation files",
               dX == 0.0 and dY == 0.0 and len(va_a) == 20, "max |dX| %.1e  |dY| %.1e" % (dX, dY))
        for col in ("scale_", "min_", "data_min_", "data_max_"):
            assert np.array_equal(getattr(scaler, col), getattr(scaler_b, col)), col

        def predict(model, sets):
            loader = DataLoader(ConcatDataset(sets), batch_size=20)
            outs = []
            with torch.no_grad():
                for batch in loader:
                    items = [b.to(DEVICE) if torch.is_tensor(b) else b for b in batch]
                    outs.append(run_rollout_train(model, VARIANT, items[0], items[1], items[3], items[4],
                                                  tf_prob=0.0, inp_last=items[ITEM_INP_LAST]).cpu())
            return torch.cat(outs)
        pa, pb = predict(model_a, va_a), predict(model_b, va_b)
        dp = float((pa - pb).abs().max())
        report("(a) predictions from the sidecar-built model equal the refit model's on the 20 validation files (< 1e-6 scaled)",
               dp < 1e-6 and tuple(pa.shape) == (20, 1440, 3), "max |delta| %.2e (shape %s)" % (dp, tuple(pa.shape)))
        inv_a = va_a[0].scaler.inverse_transform(np.column_stack([va_a[0].full_time[0], va_a[0].full_t_outer[0],
                                                                  va_a[0].full_t_inner[0], va_a[0].full_t_avg[0],
                                                                  va_a[0].full_input_temp[0]]))
        inv_b = va_b[0].scaler.inverse_transform(np.column_stack([va_b[0].full_time[0], va_b[0].full_t_outer[0],
                                                                  va_b[0].full_t_inner[0], va_b[0].full_t_avg[0],
                                                                  va_b[0].full_input_temp[0]]))
        report("(a) inverse_transform identical through the sidecar scaler", np.array_equal(inv_a, inv_b))

        # ============================================================== (b)
        ka, kb, kc = model_a.tinner_anchor
        report("(b) sidecar kc equals model.tinner_anchor[2] (and ka, kb)",
               d["constants"]["tinner_anchor"][2] == kc and d["constants"]["tinner_anchor"][0] == ka
               and d["constants"]["tinner_anchor"][1] == kb and model_b.tinner_anchor == (ka, kb, kc),
               "kc %.9g" % kc)
        report("(b) raw_affine round-trips as a tuple of pairs; unused modes are None",
               model_b.raw_affine == model_a.raw_affine and isinstance(model_b.raw_affine, tuple)
               and model_b.touter_anchor is None and model_b.pos_head is None and model_b.other_anchor is None)

        # ============================================================== (c)
        missing = [k for k in SIDECAR_KEYS if k not in d]
        sc_keys = ("columns", "data_min_", "data_max_", "scale_", "min_", "data_range_", "n_samples_seen_")
        split_keys = ("split_name", "split_sha256", "drop_sha256", "n_drop", "tavg_mask",
                      "tavg_mask_train_sha256", "tavg_mask_val_sha256")
        rt_keys = ("torch", "cuda", "cudnn", "gpu", "tf32")
        report("(c) every documented key present: top level %s" % (SIDECAR_KEYS,), not missing, "missing %s" % missing)
        report("(c) scaler block (with column names), constants (%d attrs), split digests, runtime, tf32, git_head, timestamp" % len(MODEL_CONSTANT_ATTRS),
               all(k in d["scaler"] for k in sc_keys) and d["scaler"]["columns"] == ThermalDataset.BASE_COLS
               and all(k in d["constants"] for k in MODEL_CONSTANT_ATTRS)
               and all(k in d["split"] for k in split_keys)
               and all(k in d["runtime"] for k in rt_keys)
               and d["tf32"] == "legacy" and bool(d["written_at"])
               and (d["git_head"] is None or (isinstance(d["git_head"], str) and len(d["git_head"]) == 40))
               and d["split"]["split_name"] == "stratified_v2_s42.json" and d["split"]["split_sha256"]
               and d["loss_weights_realised"] == [1.0, 6.0, 3.0] and "tinner_mode" in d["knobs"],
               "git_head %s  split %s  n_drop %s" % (d["git_head"], d["split"]["split_name"], d["split"]["n_drop"]))

        # ============================================================== (d)
        _orig = _data.load_all_data
        def _boom():
            raise AssertionError("load_all_data must not be called when a sidecar is present")
        _data.load_all_data = _boom
        try:
            model_c = build_model(VARIANT, LATEST_PARAMS, variant_to_encoder(VARIANT)).to(DEVICE)
            scaler_c, src_c = ec.load_or_refit_constants(model_c, VARIANT, run_sc)
            model_c.load_state_dict(state); model_c.eval()
            pc = predict(model_c, [ThermalDataset(df, VARIANT, scaler=scaler_c, file_name=fn) for df, fn in val_dfs])
            ok_d = src_c == "sidecar" and float((pa - pc).abs().max()) < 1e-6
        except AssertionError as e:
            ok_d = False
            print("   ", e)
        finally:
            _data.load_all_data = _orig
        report("(d) eval_checkpoint.load_or_refit_constants takes the sidecar without load_all_data (monkeypatched to raise)",
               ok_d)
        run_plain = copy_run(tmp, "run_without_sidecar")
        model_e = build_model(VARIANT, LATEST_PARAMS, variant_to_encoder(VARIANT)).to(DEVICE)
        _, src_e = ec.load_or_refit_constants(model_e, VARIANT, run_plain, train_dfs, scaler)
        raised = False
        try:
            ec.load_or_refit_constants(build_model(VARIANT, LATEST_PARAMS, variant_to_encoder(VARIANT)).to(DEVICE),
                                       VARIANT, run_plain)
        except RuntimeError:
            raised = True
        report("(d) without a sidecar: refit fallback (with warning) when training data is given, error otherwise",
               src_e == "refit" and model_e.tinner_anchor == (ka, kb, kc) and raised)

        # ============================================================== (e) end to end
        out1 = os.path.join(tmp, "o1"); out2 = os.path.join(tmp, "o2")
        r1 = ec.evaluate_one(run_sc, "best", which_set="val", out_dir=out1)
        r2 = ec.evaluate_one(run_plain, "best", which_set="val", out_dir=out2)
        dd = max(abs(r1[k] - r2[k]) for k in ("T_inner", "T_outer", "T_avg"))
        report("(e) eval_checkpoint --set val: sidecar run == refit run on the 20 validation files (per-channel MAE, 1e-9 C)",
               r1["constants"] == "sidecar" and r2["constants"] == "refit" and dd < 1e-9,
               "max |d| %.2e; val MAE Tin %.4f Tout %.4f Tavg %.4f" % (dd, r1["T_inner"], r1["T_outer"], r1["T_avg"]))
        report("(e) champion directory untouched",
               not os.path.exists(os.path.join(CHAMP, "checkpoints", "best_gru_variant_%s.constants.json" % VARIANT)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    n_ok = sum(RESULTS)
    print("\ncheck_h4: %d/%d PASS" % (n_ok, len(RESULTS)))
    sys.exit(0 if n_ok == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
