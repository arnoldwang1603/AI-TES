"""Compression probe: how much of a trained configuration can be cut or rounded without
retraining, and what inference costs on GPU and CPU (2026-09-28).

For each seed's best checkpoint: global magnitude pruning of the GRU weight matrices, of the
init-encoder weights, and of both (the 516-parameter output head is always kept), at 30-95 %;
per-output-channel symmetric weight quantization at 16/8/6/4 bits (activations stay fp32).
No fine-tuning -- the harshest version, i.e. what the trained weights tolerate as they are.
Evaluator = the training forward path (run_rollout_train) per test case, MAE in raw degrees;
it is checked against each run's meta.json at 0 % before anything else is reported.

    python compress_probe.py <runs root> "<glob of one config's seed dirs>" [--seeds 7,21,42,123,1]
"""
import argparse
import contextlib
import glob
import io
import json
import os
import statistics as st
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import eval_checkpoint as E

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("pattern")
ap.add_argument("--seeds", default="7,21,42,123,1")
A = ap.parse_args()
SEEDS = {int(s) for s in A.seeds.split(",") if s.strip()}
dirs = [d for d in sorted(glob.glob(os.path.join(A.root, A.pattern)))
        if int(d.rsplit("seed", 1)[1]) in SEEDS]
if not dirs:
    sys.exit("no seed dirs match")

# Re-import tes_gru under the runs' own knobs (all seeds of one config share them).
c0 = json.load(open(os.path.join(dirs[0], "run_config.json")))
for var in list(E.KEYS.values()) + ["LOSS_WEIGHTS", "VARIANTS", "HIDDEN_SIZE", "NUM_LAYERS", "DROPOUT",
                                    "MAX_EPOCHS", "VAL_SPLIT", "VAL_PER_TYPE", "TRAIN_EXCLUDE_TYPE", "SEEDS"]:
    os.environ.pop(var, None)
os.environ.update(E.env_from_config(c0))

import torch
with contextlib.redirect_stdout(io.StringIO()):
    from tes_gru.data import load_all_data, ThermalDataset
    from tes_gru.models import build_model
    from tes_gru.rollout import variant_to_encoder, run_rollout_train
    from tes_gru.train import fit_model_constants
    from tes_gru.config import DEVICE, LATEST_PARAMS, set_seed
    train_dfs, val_dfs, test_dfs, scaler = load_all_data()
V = "forward_direct"
tests = [ThermalDataset(df, V, scaler=scaler, file_name=fn) for df, fn in test_dfs]
BC = ThermalDataset.BASE_COLS
IDX = [BC.index("T_inner (C)"), BC.index("T_outer (C)"), BC.index("T_avg (C)")]   # target order
S = torch.tensor([scaler.scale_[i] for i in IDX], dtype=torch.float32)
M = torch.tensor([scaler.min_[i] for i in IDX], dtype=torch.float32)


def items_on(dev):
    out = []
    for ds in tests:
        it = ds[0]
        out.append((it[0][None].to(dev), it[1][None].to(dev), it[3][None].to(dev), it[4][None].to(dev)))
    return out


ITEMS = items_on(DEVICE)


def evaluate(model, items=ITEMS):
    model.eval()
    rows = []
    with torch.no_grad():
        for x, y, ic, wo in items:
            p = run_rollout_train(model, V, x, y, ic, wo, tf_prob=0.0)[0, :, :3].float().cpu()
            rows.append((((p - M) / S) - ((y[0].cpu() - M) / S)).abs().mean(dim=0))
    ch = torch.stack(rows).mean(0)
    return dict(T_inner=float(ch[0]), T_outer=float(ch[1]), T_avg=float(ch[2]), overall=float(ch.mean()))


def build(d, dev=DEVICE):
    c = json.load(open(os.path.join(d, "run_config.json")))
    set_seed(c["seed"])
    with contextlib.redirect_stdout(io.StringIO()):
        m = build_model(V, LATEST_PARAMS, variant_to_encoder(V)).to(dev)
        fit_model_constants(m, V, train_dfs, scaler)
    sd = torch.load(os.path.join(d, "checkpoints", "best_gru_variant_%s.pth" % V), map_location=dev)
    m.load_state_dict(sd)
    m.eval()
    return m, {k: v.clone() for k, v in sd.items()}, json.load(open(os.path.join(d, "variants", V, "meta.json")))


is_gru = lambda k: k.startswith("gru.weight_") or ".gru.weight_" in k
is_enc = lambda k: ("init_encoder." in k) and k.endswith(".weight")
GROUPS = {"GRU weights": is_gru,
          "init encoder weights": is_enc,
          "GRU + encoder": lambda k: is_gru(k) or is_enc(k)}      # the output head is never pruned
FRACS = (0.3, 0.5, 0.7, 0.8, 0.9, 0.95)
BITS = (16, 8, 6, 4)


def prune(sd, sel, frac):
    keys = [k for k in sd if sel(k) and sd[k].dim() == 2]
    w = torch.cat([sd[k].abs().flatten() for k in keys])
    kth = int(frac * w.numel())
    thr = w.kthvalue(kth).values if kth > 0 else torch.tensor(-1.0, device=w.device)
    out = dict(sd)
    for k in keys:
        out[k] = sd[k] * (sd[k].abs() > thr)
    return out


def quant(sd, bits):
    out = dict(sd)
    for k, v in sd.items():
        if v.dim() == 2 and (is_gru(k) or k.endswith(".weight")):
            if bits == 16:
                out[k] = v.half().float()
            else:
                q = 2 ** (bits - 1) - 1
                sc = v.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / q
                out[k] = torch.round(v / sc).clamp(-q, q) * sc
    return out


res = {"base": []}
for g in GROUPS:
    for f in FRACS:
        res[(g, f)] = []
for b in BITS:
    res[("quant", b)] = []

for d in dirs:
    m, sd, meta = build(d)
    base = evaluate(m)
    print("seed %-4s check: evaluator %.4f vs meta.json %.4f (T_avg %.4f vs %.4f)" % (
        d.rsplit("seed", 1)[1], base["overall"], meta["Test_MAE_Overall"], base["T_avg"], meta["Test_MAE_T_avg"]), flush=True)
    res["base"].append(base)
    for g, sel in GROUPS.items():
        for f in FRACS:
            m.load_state_dict(prune(sd, sel, f))
            res[(g, f)].append(evaluate(m))
    for b in BITS:
        m.load_state_dict(quant(sd, b))
        res[("quant", b)].append(evaluate(m))
    m.load_state_dict(sd)


def mean(lst, k):
    return st.mean(r[k] for r in lst)


b0 = res["base"]
print("\n%d seeds of %s. Mean overall MAE and change vs the intact network (C):" % (len(dirs), A.pattern))
print("  %-22s overall %.3f  T_avg %.3f" % ("intact", mean(b0, "overall"), mean(b0, "T_avg")))
for g in GROUPS:
    print("  %-22s " % g + "  ".join("%d%%: %+.3f" % (round(100 * f), mean(res[(g, f)], "overall") - mean(b0, "overall")) for f in FRACS))
print("  %-22s " % "weight quantization" + "  ".join("%d-bit: %+.3f" % (b, mean(res[("quant", b)], "overall") - mean(b0, "overall")) for b in BITS))

m, sd, meta = build(dirs[0])
cnt = lambda mod: sum(p.numel() for p in mod.parameters())
tot = cnt(m)
n_gru = sum(v.numel() for k, v in sd.items() if k.startswith("gru.") or ".gru." in k)
n_enc = sum(v.numel() for k, v in sd.items() if "init_encoder." in k)
print("\nparameters: total %s | GRU %s (%.0f%%) | init encoder %s (%.0f%%) | head %s" % (
    format(tot, ","), format(n_gru, ","), 100 * n_gru / tot, format(n_enc, ","), 100 * n_enc / tot,
    format(tot - n_gru - n_enc, ",")))
print("size: fp32 %.2f MB, fp16 %.2f MB, int8 %.2f MB" % (tot * 4 / 2 ** 20, tot * 2 / 2 ** 20, tot / 2 ** 20))


def time_it(model, items, n_rep=3):
    model.eval()
    with torch.no_grad():
        for x, y, ic, wo in items[:5]:
            run_rollout_train(model, V, x, y, ic, wo, tf_prob=0.0)
        if items[0][0].is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_rep):
            for x, y, ic, wo in items:
                run_rollout_train(model, V, x, y, ic, wo, tf_prob=0.0)
        if items[0][0].is_cuda:
            torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t0) / (n_rep * len(items))


print("\ninference per case (one full run, batch 1, anchors and gate included):")
if torch.cuda.is_available():
    print("  GPU (%s): %.2f ms" % (torch.cuda.get_device_name(0), time_it(m, ITEMS)))
mc, _, _ = build(dirs[0], dev="cpu")
cpu_items = items_on("cpu")
print("  CPU, %d threads: %.1f ms" % (torch.get_num_threads(), time_it(mc, cpu_items, 1)))
torch.set_num_threads(1)
print("  CPU, 1 thread: %.1f ms" % time_it(mc, cpu_items, 1))
