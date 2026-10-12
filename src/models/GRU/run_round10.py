"""Round 10 compatibility runner and full-pipeline command aliases.

The ordinary launch and --dry-run inspect/launch the nine registered P1
arms. --all and --smoke delegate to round10_pipeline; --gate delegates to
the independent validation-only P0 regression gate. The teammate runbook
uses round10_pipeline.py for the complete round.

Every Round 10 run is SEALED: SEAL_TEST=1 is forced into every child (the
runner refuses to start if the shell sets SEAL_TEST to anything else). A
run never scores the 70 test cases; main.py scores the run's own validation
files through the same test_model path and writes Val_* keys, and the test
set is read only by paper_report.py at B6 / F11.

Defaults pinned into every arm (BASE below; run_round9.BASE supplies every
other knob): direct model (OTHER_CH_MODE=abs), T_inner anchor l1, loss
1-6-3, one GRU layer (dropout is a no-op at one layer and the name carries
dp0.0), W10, MAX_EPOCHS=2000, stratified split round10_vs3_s42.json
(284 / 20 / 7 dropped, tag vs3), TF32=0 (exact fp32), no mask, no relabel.

P1 arms (seeds S10 = 1,2,3,5,7,11,13,21,42,123):
    B1_mask     h160x1, TAVG_MASK=split                 (tag mk)
    B1_orig     h160x1
    RETRAIN     h160x1, SPLIT_NAME=stratified_v2_s42.json (vs2, 291 files,
                original labels: the Round 9 champion retrained on this
                machine; pairs with the server's h160x1 seeds by label)
    B2_mask     h128x1, TAVG_MASK=split
    B2_orig     h128x1
    B2_relabel  h128x1, RELABEL_CSV=tavg_convention_by_file.csv (tag rl)
    B2_drop9    h128x1, TAVG_MASK=split + TRAIN_EXCLUDE_TYPE=9 (tag xt9;
                validation keeps the other nine types, 18 files)
    B7          h128x1, TAVG_MASK=random:20261011 (tag mkr20261011; one draw
                of 115 training files shared by every seed)
    B8          h128x1, TAVG_MASK=valonly (tag mkv)
The primary comparison and the controls are in the prereg file; the
statistics are computed by paper_stats.py from each run's
summary_errors_val.csv (or eval_checkpoint --set val).

Usage (from src/models/GRU):
    python run_round10.py --dry-run                 # env table + resolved run names, nothing launched
    python run_round10.py                           # all nine arms, 90 runs
    python run_round10.py --only B1_mask,B1_orig    # a subset by arm name
    python run_round10.py --parallel 3              # seeds per arm trained concurrently (default 3)
    python run_round10.py --all --parallel 3         # complete registered pipeline, resume-safe
    python run_round10.py --gate                    # frozen P0/new defaults on 20 validation files, TF32=legacy
    python run_round10.py --smoke [--keep-smoke] [--only ...]
        MAX_EPOCHS=2, one seed per arm, all core and conditional paths;
        each completed run is verified (sealed meta
        without any Test_* key, Val_MAE_* present, constants sidecar on
        disk, run_config provenance, and eval_checkpoint --set val
        reproduces the run's own Val_MAE per channel); a table is printed
        and the ep2 directories are deleted unless --keep-smoke.
    --tf32 0|1|legacy   numerics for every child (default 0; legacy = the
                        pre-Round-10 torch default, tag tf32c)
Env: SEEDS overrides the P1-only seed list; PROGRESS_EVERY tunes
the status cadence; CHAMPION_RUN_DIR points the gate at the Round 9
champion run directory (default: the packaged round10_resources/round9 run).
Resumable per seed (done.flag); Ctrl+C safe. Logs -> sweep_logs/r10-<arm>.w<k>.log
"""
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_round9 as r9   # noqa: E402  (shared launch machinery; importing it has no side effects beyond reading splits/)

S10 = "1,2,3,5,7,11,13,21,42,123"
SMOKE_SEEDS = "7,21"
VS3 = "round10_vs3_s42.json"
VS2 = "stratified_v2_s42.json"
RELABEL = "tavg_convention_by_file.csv"
B7_SEED = 20261011
RUN_PREFIX = r9.RUN_PREFIX
VARIANT = "forward_direct"
CH = ("T_inner", "T_outer", "T_avg")
SMOKE_TOL_C = 1e-3          # eval_checkpoint --set val vs the run's own Val_MAE (same weights, same numerics)
PARALLEL_DEFAULT = int(os.environ.get("PARALLEL_SEEDS", "3"))
CHAMP = os.environ.get("CHAMPION_RUN_DIR") or os.path.normpath(os.path.join(
    HERE, "round10_resources", "round9",
    "2026-08-06_forward_direct_W10_Tin-anchor_l1_w1-6-3_oth-abs_h160x1dp0.2_ep2000_vs2_seed7"))

# Every knob the child reads is pinned (run_round9.BASE pins the position-head
# and reference knobs at their off values); these are the Round 10 values.
BASE = dict(r9.BASE,
            OTHER_CH_MODE="abs", HIDDEN_SIZE="160", NUM_LAYERS="1", DROPOUT="0.0",
            LOSS_WEIGHTS="1,6,3", MAX_EPOCHS="2000", MODEL_SPLIT="shared",
            VAL_SPLIT="stratified", VAL_PER_TYPE="2", TRAIN_EXCLUDE_TYPE="",
            SPLIT_NAME=VS3, TAVG_MASK="", RELABEL_CSV="", TF32="0", SEAL_TEST="1")


def cfg(**kw):
    d = dict(BASE)
    d.update({k: str(v) for k, v in kw.items()})
    return d


def mins(h):
    """Minutes per seed at the 2000-epoch cap (dev 4090, one worker)."""
    return int(round(1.5 * r9.mins(h, 1)))


ARMS = [
    ("B1_mask", cfg(TAVG_MASK="split")),
    ("B1_orig", cfg()),
    ("RETRAIN", cfg(SPLIT_NAME=VS2)),
    ("B2_mask", cfg(HIDDEN_SIZE=128, TAVG_MASK="split")),
    ("B2_orig", cfg(HIDDEN_SIZE=128)),
    ("B2_relabel", cfg(HIDDEN_SIZE=128, RELABEL_CSV=RELABEL)),
    ("B2_drop9", cfg(HIDDEN_SIZE=128, TAVG_MASK="split", TRAIN_EXCLUDE_TYPE="9")),
    ("B7", cfg(HIDDEN_SIZE=128, TAVG_MASK="random:%d" % B7_SEED)),
    ("B8", cfg(HIDDEN_SIZE=128, TAVG_MASK="valonly")),
]
ARM_ENV = dict(ARMS)
SMOKE_ARMS = ["B2_orig", "B2_mask", "B2_relabel", "B7", "B8", "RETRAIN"]
STAGE = "P1"

# Run-name tags preflight_stale_dirs must recognise: a done-flagged dir that
# carries the tag but whose run_config lacks the matching provenance was
# written by older code and would be silently resumed. Future tags (rt, ph,
# enc, w0ikc, h0t, h0z) have no run_config key yet, so any such dir is stale
# until the hook that writes the key lands.
STALE_TAGS = [
    (r"_vs3(_|$)", "vs3", lambda c, m: (c.get("data") or {}).get("split_tag") == "vs3"),
    (r"_mk(_|$)", "mk", lambda c, m: c.get("tavg_mask") == "split"),
    (r"_mkr(\d+)(_|$)", "mkr<seed>", lambda c, m: c.get("tavg_mask") == "random:" + m.group(1)),
    (r"_mkv(_|$)", "mkv", lambda c, m: c.get("tavg_mask") == "valonly"),
    (r"_rl(_|$)", "rl", lambda c, m: bool(c.get("relabel_csv"))),
    (r"_rt[123][kc](_|$)", "rt<R><k|c>", lambda c, m: c.get("other_ch_mode") == "ratio_input" and bool(c.get("ratio_channels"))),
    (r"_ph[a-z]*(_|$)", "ph", lambda c, m: bool(c.get("phys_mode"))),
    (r"_enc[0-9x-]*(_|$)", "enc<...>", lambda c, m: "enc_widths" in c),
    (r"_w0ikc(_|$)", "w0ikc", lambda c, m: c.get("loss_w0_mode") == "inv_kc"),
    (r"_h0t(_|$)", "h0t", lambda c, m: c.get("h0_mode") == "tanh"),
    (r"_h0z(_|$)", "h0z", lambda c, m: c.get("h0_mode") == "zero"),
]


# ----------------------------------------------------------------- preflight
def check_split_json(name, expect_drop=None):
    p = os.path.join(HERE, "splits", name)
    if not os.path.isfile(p):
        sys.exit("missing splits/%s -- it ships with the code; do not let the machine draw its own" % name)
    spec = json.load(open(p, encoding="utf-8"))
    per = int(BASE["VAL_PER_TYPE"])
    n_types = len(spec["types"])
    if spec.get("val_per_type") != per or len(spec["val"]) != per * n_types:
        sys.exit("%s: expected %d x %d validation files, found %d" % (name, n_types, per, len(spec["val"])))
    if set(spec["val"]) & set(spec["train"]):
        sys.exit("%s: validation and training lists overlap" % name)
    drop = set(spec.get("drop") or [])
    if drop & (set(spec["val"]) | set(spec["train"])):
        sys.exit("%s: drop list overlaps val/train" % name)
    if expect_drop is not None and len(drop) != expect_drop:
        sys.exit("%s: expected %d dropped files, found %d" % (name, expect_drop, len(drop)))
    mask = set(spec.get("tavg_mask") or [])
    if mask - (set(spec["val"]) | set(spec["train"])):
        sys.exit("%s: tavg_mask names a file outside train/val" % name)
    print("  split %s: %d types x %d = %d validation, %d training, %d dropped, %d masked (%d val), tag %s" % (
        name, n_types, per, len(spec["val"]), len(spec["train"]), len(drop), len(mask),
        len(mask & set(spec["val"])), spec.get("tag") or "vs%d" % per))


def preflight_split_files(plan):
    names = sorted({e["SPLIT_NAME"] for _, _, e, _, _, _ in plan})
    for n in names:
        check_split_json(n, expect_drop=7 if n == VS3 else None)
    if any(e.get("RELABEL_CSV") for _, _, e, _, _, _ in plan):
        p = os.path.join(HERE, "splits", RELABEL)
        if not os.path.isfile(p):
            sys.exit("missing splits/%s (A4 label table)" % RELABEL)
        print("  relabel table: splits/%s present" % RELABEL)


def preflight_stale_dirs(ignore=()):
    """Refuse leftover smoke dirs (ep<151 tag or Total_Epoch < 151) and
    tagged dirs without the matching run_config provenance (run_round9's
    vs / xt / ind rules plus the Round 10 tags in STALE_TAGS). `ignore`:
    run names this invocation is about to (re)create itself."""
    stale = []
    runs_root = os.path.join(HERE, "runs")
    if os.path.isdir(runs_root):
        for d in sorted(os.listdir(runs_root)):
            if not d.startswith(RUN_PREFIX) or re.sub(r"_seed\d+$", "", d) in ignore:
                continue
            rdir = os.path.join(runs_root, d)
            rc = os.path.join(rdir, "run_config.json")
            flags = glob.glob(os.path.join(rdir, "variants", "*", "done.flag"))
            if not flags or not os.path.isfile(rc):
                continue
            m = re.search(r"_ep(\d+)(_|$)", d)
            if m and int(m.group(1)) < 151:
                stale.append(d + "  (MAX_EPOCHS smoke-test dir; delete it)")
                continue
            short = []
            for f in flags:
                try:
                    tot = int(json.load(open(os.path.join(os.path.dirname(f), "meta.json"))).get("Total_Epoch", 1000))
                except Exception:
                    tot = 1000
                if tot < 151:
                    short.append(tot)
            if short:
                stale.append(d + "  (finished after only %d epochs -- smoke test?)" % short[0])
                continue
            try:
                keys = json.load(open(rc))
            except Exception:
                stale.append(d + "  (unreadable run_config)")
                continue
            data = keys.get("data") or {}
            if "_vs" in d and data.get("val_split") != "stratified":
                stale.append(d + "  (vs tag but no stratified provenance)")
            if re.search(r"_xt\d+(_|$)", d) and not data.get("train_exclude_type"):
                stale.append(d + "  (xt tag but no train_exclude_type provenance)")
            if "_ind_" in d and keys.get("model_split") != "independent":
                stale.append(d + "  (ind tag but no model_split provenance)")
            for pat, tag, ok in STALE_TAGS:
                mm = re.search(pat, d)
                if mm and not ok(keys, mm):
                    stale.append(d + "  (%s tag but no matching run_config provenance)" % tag)
    if stale:
        print("\nREFUSING TO LAUNCH -- stale done-flagged run dirs would be")
        print("silently reused by these arms. Delete or rename them first:")
        for d in stale:
            print("   runs/" + d)
        sys.exit(2)
    print("  no stale run dirs under runs/")


def preflight_child():
    from round10_pipeline import isolated_env
    env = isolated_env(BASE)
    env.update(PYTHONUNBUFFERED="1", SEAL_TEST="1")
    code = r'''
import json
from tes_gru.config import VAL_SPLIT, SEAL_TEST, split_file_path
from tes_gru.data import load_all_data, ThermalDataset, input_case_flag_np
from tes_gru.rollout import _input_case_gate
assert VAL_SPLIT == "stratified" and SEAL_TEST
spec = json.load(open(split_file_path(), encoding="utf-8"))
tr, va, te, scaler = load_all_data(include_test=False)
assert len(tr) == len(spec["train"]) and len(va) == len(spec["val"]) and not te, (len(tr), len(va), len(te))
counts = []
for name, frames in (("train", tr), ("val", va)):
    fired = 0
    for frame, case_id in frames:
        column = "Input Temperature (C)" if "Input Temperature (C)" in frame else "T_inner (C)"
        expected = bool(input_case_flag_np(frame[column].values))
        dataset = ThermalDataset(frame, "forward_direct", scaler=scaler, file_name=case_id)
        actual = bool(_input_case_gate(dataset.X[:, :, 1])[0].item())
        assert expected == actual, (name, case_id, expected, actual)
        fired += int(actual)
    counts.append((name, fired, len(frames)))
assert counts[1][1] > 0, "no both-phase validation file"
print("sealed split %d/%d; test excluded; detector twins: %s" % (len(tr), len(va), counts))
print("CHECK_OK")
'''
    r9._run_check(code, env, "sealed vs3 train/validation split + detector twins")


def preflight_all(plan, resolve_names=True):
    print("\nPre-flight:")
    r9.preflight_compile()
    preflight_split_files(plan)
    preflight_stale_dirs()
    preflight_child()
    if resolve_names:
        print("  [config import under every arm's knobs]")
        r9.preflight_arm_configs(plan)


def refuse_unsealed():
    v = os.environ.get("SEAL_TEST")
    if v is not None and v.strip() not in ("1", "true", "True"):
        sys.exit("run_round10 refuses SEAL_TEST=%r: every Round 10 run is sealed (SEAL_TEST=1 is forced)" % v)


# ----------------------------------------------------------------- plan
def build_plan(names, seeds, overrides=None):
    plan = []
    for name, env in ARMS:
        if names is not None and name not in names:
            continue
        env = dict(env, **(overrides or {}))
        n = len([s for s in seeds.split(",") if s.strip()])
        per = mins(int(env["HIDDEN_SIZE"])) * (2.0 / 2000.0 * 50 if env["MAX_EPOCHS"] == "2" else 1.0)
        plan.append((STAGE, name, env, seeds, n, n * per))
    return plan


def print_plan(plan, title):
    tot_runs = sum(n for _, _, _, _, n, _ in plan)
    tot_min = sum(m for _, _, _, _, _, m in plan)
    print("#" * 74)
    print("# %s  arms=%d  runs=%d  ~%.0f GPU-h one worker  (finished seeds are skipped; SEAL_TEST=1)" % (
        title, len(plan), tot_runs, tot_min / 60))
    print("#" * 74)
    print("  base: %s" % {k: BASE[k] for k in ("OTHER_CH_MODE", "HIDDEN_SIZE", "NUM_LAYERS", "LOSS_WEIGHTS",
                                                "MAX_EPOCHS", "SPLIT_NAME", "TAVG_MASK", "RELABEL_CSV", "TF32", "SEAL_TEST")})
    for st, name, env, seeds, n, m in plan:
        knobs = {k: v for k, v in env.items() if BASE.get(k) != v}
        print("  [%s] %-11s x%-3d seeds=%s  %s" % (st, name, n, seeds, knobs))


def run_dir_of(name, seed):
    return os.path.join(HERE, "runs", "%s_seed%s" % (r9.RUN_NAMES[name], str(seed).strip()))


# ----------------------------------------------------------------- smoke verification
def transport_env(overrides=None):
    """Preserve legacy knobs while resolving paths before child cwd changes."""
    from round10_pipeline import resolved_data_root, resolved_resource_paths
    env = dict(os.environ)
    env.update(overrides or {})
    env["DATA_ROOT"] = resolved_data_root(env)
    env.update(resolved_resource_paths(env))
    return env


def verify_sealed_run(run_dir, env):
    """Checks on a finished sealed run; returns a dict with ok + details.
    eval_checkpoint --set val runs in a child (its own env isolation) and
    writes to a scratch dir, so the run dir is not touched."""
    out = dict(run=os.path.basename(run_dir), ok=False, sealed=False, no_test_key=False, val_keys=False,
               sidecar=False, artefacts=False, provenance=False, repro=False, max_delta=float("nan"),
               n_tavg=None, val={}, msg="")
    vd = os.path.join(run_dir, "variants", VARIANT)
    try:
        meta = json.load(open(os.path.join(vd, "meta.json")))
        rc = json.load(open(os.path.join(run_dir, "run_config.json")))
    except Exception as e:
        out["msg"] = "meta/run_config unreadable: %r" % (e,)
        return out
    out["sealed"] = bool(meta.get("sealed") is True and rc.get("sealed") is True)
    out["no_test_key"] = not any(k.startswith("Test_") for k in meta)
    out["val_keys"] = all(("Val_MAE_" + c) in meta for c in CH) and "Val_MAE_Overall" in meta
    out["val"] = {c: meta.get("Val_MAE_" + c) for c in CH}
    out["n_tavg"] = meta.get("Scored_N_files_T_avg")
    out["sidecar"] = os.path.isfile(os.path.join(run_dir, "checkpoints", "best_gru_variant_%s.constants.json" % VARIANT))
    out["artefacts"] = (os.path.isfile(os.path.join(vd, "summary_errors_val.csv"))
                        and os.path.isfile(os.path.join(vd, "predictions_val.npz"))
                        and not os.path.isfile(os.path.join(vd, "summary_errors.csv"))
                        and not os.path.isfile(os.path.join(vd, "predictions.npz")))
    data = rc.get("data") or {}
    out["provenance"] = (rc.get("tf32") == env["TF32"] and rc.get("tavg_mask", "") == env["TAVG_MASK"]
                         and data.get("split_name") == env["SPLIT_NAME"]
                         and (rc.get("relabel_csv") or "") == env["RELABEL_CSV"]
                         and str(data.get("train_exclude_type") or "") == env["TRAIN_EXCLUDE_TYPE"]
                         and int(rc["latest_params"]["max_epochs"]) == int(env["MAX_EPOCHS"]))
    tmp = tempfile.mkdtemp(prefix="r10smoke_")
    try:
        agg_p = os.path.join(tmp, "agg.json")
        p = subprocess.run([sys.executable, os.path.join(HERE, "eval_checkpoint.py"), run_dir, "--which", "best",
                            "--set", "val", "--out-dir", os.path.join(tmp, "val"), "--out", agg_p],
                           env=dict(transport_env(env), PYTHONUNBUFFERED="1"), cwd=HERE,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        if os.path.isfile(agg_p):
            agg = json.load(open(agg_p))[0]
            deltas = []
            channels_match = out["val_keys"]
            for c in CH:
                expected = meta.get("Val_MAE_" + c)
                actual = agg.get(c)
                expected_missing = expected is None or not math.isfinite(float(expected))
                actual_missing = actual is None or not math.isfinite(float(actual))
                if expected_missing or actual_missing:
                    allowed_unscored = (c == "T_avg" and agg.get("n_files_T_avg") == 0
                                        and meta.get("Scored_N_files_T_avg") == 0)
                    channels_match = channels_match and expected_missing and actual_missing and allowed_unscored
                else:
                    deltas.append(abs(float(actual) - float(expected)))
            out["max_delta"] = max(deltas) if deltas else float("nan")
            out["repro"] = bool(deltas) and channels_match and out["max_delta"] <= SMOKE_TOL_C and \
                agg["n_files_T_avg"] == meta.get("Scored_N_files_T_avg") and agg["n_files"] == meta.get("Scored_N_files")
            if not out["repro"]:
                out["msg"] = "eval_checkpoint val: %s vs meta %s (n %s/%s)" % (
                    {c: round(agg[c], 5) if agg.get(c) is not None else None for c in CH},
                    {c: round(meta["Val_MAE_" + c], 5) if meta.get("Val_MAE_" + c) is not None else None for c in CH},
                    agg["n_files_T_avg"], meta.get("Scored_N_files_T_avg"))
        else:
            out["msg"] = "eval_checkpoint failed (exit %s): %s" % (p.returncode, p.stdout[-700:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    out["ok"] = all(out[k] for k in ("sealed", "no_test_key", "val_keys", "sidecar", "artefacts", "provenance", "repro"))
    return out


def print_smoke_table(rows):
    print("\n" + "#" * 74 + "\n# SMOKE TABLE  (Val MAE in C from meta.json; |d| = max per-channel gap to eval_checkpoint --set val)")
    print("# %-10s %-5s %-7s %-7s %-7s %-4s %-9s %-6s %-7s %-7s %-7s %-6s %-3s" % (
        "arm", "seed", "Tin", "Tout", "Tavg", "nTa", "|d| C", "sealed", "noTest", "sidecar", "proven", "repro", "OK"))
    for arm, seed, r in rows:
        v = r["val"]
        print("# %-10s %-5s %-7s %-7s %-7s %-4s %-9s %-6s %-7s %-7s %-7s %-6s %-3s" % (
            arm, seed, *["%.3f" % v[c] if isinstance(v.get(c), (int, float)) else "-" for c in CH],
            r["n_tavg"] if r["n_tavg"] is not None else "-",
            ("%.1e" % r["max_delta"]) if r["max_delta"] == r["max_delta"] else "-",
            "yes" if r["sealed"] else "NO", "yes" if r["no_test_key"] else "NO", "yes" if r["sidecar"] else "NO",
            "yes" if r["provenance"] else "NO", "yes" if r["repro"] else "NO", "OK" if r["ok"] else "FAIL"))
        if r["msg"]:
            print("#     " + r["msg"].replace("\n", "\n#     ")[:900])
    print("#" * 74)


def delete_smoke_dirs(dirs):
    for d in dirs:
        base = os.path.basename(d)
        if not re.search(r"_ep2(_|$)", base):
            print("  NOT deleting %s (no ep2 tag)" % base)
            continue
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
            print("  deleted runs/%s" % base)


# ----------------------------------------------------------------- regression gate
def regression_gate():
    """Independent P0/new-default validation comparison; absent input fails."""
    import round10_validation_gate
    from paper_report import atomic_json
    result = round10_validation_gate.run_gate()
    atomic_json(os.path.join(HERE, "runs", "_round10_results", "S0", "regression.json"), result)
    if not result["ok"]:
        raise RuntimeError("Independent validation regression gate failed")
    print("Independent validation gate PASS; no test access")
    return True


# ----------------------------------------------------------------- launching
def launch(plan, n_workers, log_tag, res, tot_runs, tot_min):
    done, spent = 0, 0.0
    for st, name, env_over, seeds, n, est in plan:
        nw = min(n_workers, n)
        print("\n%s\n=== [%s] %s  (%d seeds on %d worker%s, ~%.1f GPU-h) ===" % (
            "=" * 74, st, name, n, nw, "s" if nw > 1 else "", est / 60))
        lps = [os.path.join(r9.LOG_DIR, "%s-%s.w%d.log" % (log_tag, name, k + 1)) for k in range(nw)]
        print("  console -> " + (lps[0] if nw == 1 else lps[0].replace(".w1.log", ".w<k>.log")))
        stop = threading.Event()
        threading.Thread(target=r9.tail_progress,
                         args=(lps, "[%s] %s" % (st, name), n, done, tot_runs, spent / 60, tot_min / 60, stop),
                         daemon=True).start()
        t0 = time.time()
        rc = r9.run_arm(name, env_over, seeds, nw, lps)
        stop.set()
        hrs = (time.time() - t0) / 3600.0
        spent += hrs * 60
        done += n
        missing = [s for s in seeds.split(",") if s.strip() and not glob.glob(
            os.path.join(run_dir_of(name, s), "variants", "*", "done.flag"))]
        res[name] = (not missing, rc, hrs, missing)
        print("  %s %s after %.2f h" % (name, "OK (%d seeds done)" % n if not missing
                                       else "INCOMPLETE: seeds %s have no done.flag (exit %s)" % (",".join(missing), rc), hrs))


def smoke(names, n_workers, keep, gate):
    seeds = os.environ.get("SEEDS", SMOKE_SEEDS)
    arms = [a for a in SMOKE_ARMS if names is None or a in names]
    if not arms:
        sys.exit("no smoke arm selected; smoke arms: " + ", ".join(SMOKE_ARMS))
    plan = build_plan(arms, seeds, overrides=dict(MAX_EPOCHS="2", HIDDEN_SIZE="128"))
    plan.sort(key=lambda t: arms.index(t[1]))
    print_plan(plan, "ROUND 10 SMOKE  (MAX_EPOCHS=2, h128x1, tag ep2)")
    os.makedirs(r9.LOG_DIR, exist_ok=True)
    print("\nPre-flight:")
    r9.preflight_compile()
    preflight_split_files(plan)
    print("  [config import under every smoke arm's knobs]")
    r9.preflight_arm_configs(plan)
    targets = [run_dir_of(name, s) for _, name, _, sd, _, _ in plan for s in sd.split(",") if s.strip()]
    old = [d for d in targets if os.path.isdir(d)]
    if old:
        print("  removing %d leftover smoke dir(s) of this plan:" % len(old))
        delete_smoke_dirs(old)
    preflight_stale_dirs(ignore={r9.RUN_NAMES[n] for n in arms})
    preflight_child()
    tot_runs = sum(n for _, _, _, _, n, _ in plan)
    tot_min = sum(m for _, _, _, _, _, m in plan)
    res, rows = {}, []
    try:
        for item in plan:
            launch([item], n_workers, "r10-smoke", res, tot_runs, tot_min)
            _, name, env, sd, _, _ = item
            for s in [x.strip() for x in sd.split(",") if x.strip()]:
                rd = run_dir_of(name, s)
                if not glob.glob(os.path.join(rd, "variants", "*", "done.flag")):
                    rows.append((name, s, dict(run=os.path.basename(rd), ok=False, sealed=False, no_test_key=False,
                                               val_keys=False, sidecar=False, artefacts=False, provenance=False,
                                               repro=False, max_delta=float("nan"), n_tavg=None, val={},
                                               msg="no done.flag")))
                    continue
                print("  verifying %s ..." % os.path.basename(rd), flush=True)
                rows.append((name, s, verify_sealed_run(rd, env)))
    except KeyboardInterrupt:
        print("\nInterrupted; smoke dirs kept for inspection: %s" % [os.path.basename(t) for t in targets if os.path.isdir(t)])
        sys.exit(130)
    print_smoke_table(rows)
    all_ok = bool(rows) and all(r["ok"] for _, _, r in rows)
    gate_ok = regression_gate() if gate else None
    if keep:
        print("\n--keep-smoke: ep2 directories kept: %s" % [os.path.basename(t) for t in targets])
    else:
        print("\ndeleting the smoke directories:")
        delete_smoke_dirs(targets)
    print("\nSMOKE %s (%d/%d runs verified)%s" % ("PASS" if all_ok else "FAIL", sum(r["ok"] for _, _, r in rows), len(rows),
                                                  "" if gate_ok is None else ";  regression gate %s" % ("PASS" if gate_ok else "FAIL")))
    sys.exit(0 if all_ok and gate_ok is not False else 1)


def main():
    os.environ.update(transport_env())
    # Full-round smoke/gate use the same entrypoints as the teammate's
    # package. The historical P1-only flags remain for inspecting its table.
    if "--smoke" in sys.argv[1:] or "--all" in sys.argv[1:]:
        args = sys.argv[1:]
        import round10_pipeline
        round10_pipeline.main()
        return
    if "--gate" in sys.argv[1:]:
        import round10_validation_gate
        sys.argv = [sys.argv[0], "--out", os.path.join(HERE, "runs", "_round10_results", "S0", "regression.json")]
        round10_validation_gate.main()
        return
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass
    argv = sys.argv[1:]

    def after(flag):
        if flag not in argv:
            return None
        i = argv.index(flag)
        if i + 1 >= len(argv) or argv[i + 1].startswith("--"):
            sys.exit("%s needs a value" % flag)
        return argv[i + 1]

    refuse_unsealed()
    tf32 = after("--tf32")
    if tf32 is not None:
        if tf32 not in ("0", "1", "legacy"):
            sys.exit("--tf32 takes 0, 1 or legacy")
        BASE["TF32"] = tf32
        for _, e in ARMS:
            e["TF32"] = tf32
    only = after("--only")
    names = {s.strip() for s in only.split(",") if s.strip()} if only else None
    if names:
        bad = names - set(ARM_ENV)
        if bad:
            sys.exit("unknown arm(s) %s; arms: %s" % (sorted(bad), ", ".join(n for n, _ in ARMS)))
    par = after("--parallel")
    n_workers = max(1, int(par)) if par else PARALLEL_DEFAULT
    if "--gate" in argv and "--smoke" not in argv:
        ok = regression_gate()
        sys.exit(0 if ok else 1)
    if "--smoke" in argv:
        smoke(names, n_workers, keep="--keep-smoke" in argv, gate="--no-gate" not in argv)
        return

    seeds = os.environ.get("SEEDS", S10)
    plan = build_plan(names, seeds)
    if not plan:
        sys.exit("nothing to run")
    print_plan(plan, "ROUND 10 / %s" % STAGE)
    tot_runs = sum(n for _, _, _, _, n, _ in plan)
    tot_min = sum(m for _, _, _, _, _, m in plan)
    if "--dry-run" in argv:
        print("\nResolving run names (tes_gru.config imported under each arm's knobs):")
        r9.preflight_arm_configs(plan)
        for _, name, _, _, _, _ in plan:
            print("  %-11s -> runs/%s_seed<N>" % (name, r9.RUN_NAMES[name]))
        print("\nDry run complete -- nothing launched.")
        return
    os.makedirs(r9.LOG_DIR, exist_ok=True)
    preflight_all(plan)
    if n_workers > 1:
        free = r9.gpu_free_gb()
        fit = max(1, int(free // r9.GB_PER_WORKER)) if free > 0 else n_workers
        if fit < n_workers:
            print("  GPU has %.1f GB free: %d workers at %s GB each do not fit -> --parallel %d" % (free, n_workers, r9.GB_PER_WORKER, fit))
            n_workers = fit
        else:
            print("  GPU has %.1f GB free: %d workers x %s GB fit" % (free, n_workers, r9.GB_PER_WORKER))
    res = {}
    try:
        launch(plan, n_workers, "r10", res, tot_runs, tot_min)
    except KeyboardInterrupt:
        print("\nInterrupted:", {k: "%.2fh" % v[2] for k, v in res.items()})
        print("Re-running resumes where it left off.")
        sys.exit(130)
    print("\n" + "#" * 74 + "\n# ROUND 10 SUMMARY")
    for nm, (ok, rc, h, missing) in res.items():
        print("#   %-12s %-12s %.2f h" % (nm, "OK" if ok else "INCOMPLETE", h))
    bad = [nm for nm, v in res.items() if not v[0]]
    if bad:
        print("# %d arm(s) incomplete -- re-run the same command; finished seeds are skipped" % len(bad))
    print("#" * 74)
    print("# Statistics: python paper_stats.py --from-runs runs " + " ".join(
        "--arm %s=%s" % (nm, r9.RUN_NAMES[nm]) for _, nm, _, _, _, _ in plan)
        + " --seeds %s --pair B1_mask:B1_orig --pair B2_mask:B2_orig --family P1 --out r10_stats" % seeds)
    print("# Export:     python export_results.py runs Round10_upload --with-plots 2")


if __name__ == "__main__":
    main()
