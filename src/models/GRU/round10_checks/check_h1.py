"""Round 10 / H1 check: SPLIT_NAME knob, drop list, frozen vs3 split.
Run from src/models/GRU:  python round10_checks/check_h1.py

No training. PASS / FAIL per item, non-zero exit on any FAIL.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # src/models/GRU
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(HERE)

from check_h9 import clean_env, fresh_import, ENV_VARS_EXTRA   # noqa: E402

VS3 = "round10_vs3_s42.json"
LEGACY = "stratified_v2_s42.json"
RESULTS = []


def report(item, ok, detail=""):
    RESULTS.append(bool(ok))
    print("%s  %s  %s" % ("PASS" if ok else "FAIL", item, detail), flush=True)


def base_env(extra):
    env = {"VARIANTS": "forward_direct", "VAL_SPLIT": "stratified", "VAL_PER_TYPE": "2",
           "HIDDEN_SIZE": "128", "NUM_LAYERS": "1", "DROPOUT": "0.0", "TRAIN_EXCLUDE_TYPE": ""}
    env.update(extra)
    return env


def main():
    import eval_checkpoint as ec
    clean_env(ec)

    # ------------------------------------------------------------------ (a) vs3
    config = fresh_import(base_env({"SPLIT_NAME": VS3}))
    from tes_gru.data import load_all_data, split_key, load_split_spec
    from tes_gru.utils import sha256_file
    spec = load_split_spec(config.split_file_path())
    tr, va, te, sc = load_all_data(include_test=False)
    tr_keys = sorted(split_key(fn) for _, fn in tr)
    va_keys = sorted(split_key(fn) for _, fn in va)
    te_ids = [fn for _, fn in te]
    drop = set(spec["drop"])
    report("(a) vs3 sizes train 284 / val 20, sealed test CSVs excluded",
           (len(tr), len(va), len(te)) == (284, 20, 0), "%d/%d/%d" % (len(tr), len(va), len(te)))
    report("(a) vs3 train/val lists equal the json", tr_keys == sorted(spec["train"]) and va_keys == sorted(spec["val"]))
    leak = [k for k in tr_keys + va_keys if k in drop] + [i for i in te_ids if split_key(i) in drop]
    report("(a) no dropped key in train / val", not leak, str(leak[:3]))
    train_dir = None
    # every key on disk (reuse the loader's resolution)
    from tes_gru.data import _resolve_new_data_root
    root = _resolve_new_data_root(config.BASE_DIR)
    train_dir = os.path.join(root, config.TRAIN_SUBDIR)
    missing = [k for k in spec["train"] + spec["val"] + spec["drop"] if not os.path.isfile(os.path.join(train_dir, k))]
    report("(a) every train / val / drop key exists on disk", not missing, str(missing[:3]))
    allk = set(tr_keys) | set(va_keys)
    report("(a) Case 10/Case 8 and Case 1/Case new 20 absent",
           "Case 10 (Discharging + Constant)/Case 8.csv" not in allk
           and "Case 1 (linear charging)/Case new 20.csv" not in allk)
    report("(a) Case 1/Case new 18 (1 second step) in train",
           "Case 1 (linear charging)/Case new 18 (1 second step).csv" in set(tr_keys))
    want_val = ["Case 1 (linear charging)/Case 16.csv", "Case 9 (non-linear discharging)/Case New 1 (Tejas).csv",
                "Case 9 (non-linear discharging)/Case New 4 (Tejas).csv"]
    report("(a) Case 1/Case 16, Case 9/New 1, Case 9/New 4 in val", all(k in set(va_keys) for k in want_val))
    report("(a) RUN_NAME_BASE / SPLIT_TAG = vs3", config.SPLIT_TAG == "vs3" and "_vs3" in config.RUN_NAME_BASE
           and "vs2" not in config.RUN_NAME_BASE, config.RUN_NAME_BASE)

    # ------------------------------------------------------------------ (e) run_config data block
    import tes_gru.runio as runio
    tmp = tempfile.mkdtemp(prefix="check_h1_")
    runio.run_dir = lambda: tmp
    try:
        p = runio.save_run_config_snapshot()
        rc = json.load(open(p))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    d = rc.get("data", {})
    report("(e) run_config data block has split_name / split_tag / split_sha256",
           d.get("split_name") == VS3 and d.get("split_tag") == "vs3"
           and d.get("split_sha256") == sha256_file(config.split_file_path()),
           "%s %s %s" % (d.get("split_name"), d.get("split_tag"), str(d.get("split_sha256"))[:12]))
    report("(e) eval_checkpoint.env_from_config passes SPLIT_NAME", ec.env_from_config(rc).get("SPLIT_NAME") == VS3)
    import export_results as er
    export_prefix = "sealed_vs3_" if rc.get("sealed") else "vs3_"
    report("(e) export_results.family uses the vs3 tag and the recorded seal",
           er.family(rc).startswith(export_prefix), er.family(rc))

    # ------------------------------------------------------------------ (b) legacy default
    clean_env(ec)
    config = fresh_import(base_env({}))
    from tes_gru.data import load_all_data, split_key
    leg = json.load(open(os.path.join(HERE, "splits", LEGACY), encoding="utf-8"))
    tr, va, te, sc = load_all_data(include_test=False)
    tr_keys = [split_key(fn) for _, fn in tr]
    va_keys = [split_key(fn) for _, fn in va]
    report("(b) legacy default: train 291 / val 20, sealed test CSVs excluded",
           (len(tr), len(va), len(te)) == (291, 20, 0), "%d/%d/%d" % (len(tr), len(va), len(te)))
    report("(b) legacy default lists identical to stratified_v2_s42.json",
           tr_keys == leg["train"] and va_keys == leg["val"]
           and sorted(tr_keys) == sorted(leg["train"]) and sorted(va_keys) == sorted(leg["val"]))
    report("(b) legacy split_file_name() resolves to the legacy json", config.split_file_name() == LEGACY)
    report("(d) legacy RUN_NAME_BASE carries vs2", "_vs2" in config.RUN_NAME_BASE and "vs3" not in config.RUN_NAME_BASE,
           config.RUN_NAME_BASE)

    # ------------------------------------------------------------------ (d) run names via subprocess
    env0 = dict(os.environ)
    for var in list(ec.KEYS.values()) + ENV_VARS_EXTRA:
        env0.pop(var, None)
    code = "from tes_gru import config; print('RNB=' + config.RUN_NAME_BASE)"

    def run(extra, code=code):
        e = dict(env0, **base_env(extra))
        pr = subprocess.run([sys.executable, "-c", code], env=e, cwd=HERE,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        return pr.returncode, pr.stdout

    rc3, out3 = run({"SPLIT_NAME": VS3})
    rc2, out2 = run({})
    n3 = [l for l in out3.splitlines() if l.startswith("RNB=")]
    n2 = [l for l in out2.splitlines() if l.startswith("RNB=")]
    report("(d) subprocess: SPLIT_NAME=vs3 json -> _vs3 tag", bool(n3) and n3[-1].endswith("_vs3"), n3[-1] if n3 else out3[-300:])
    report("(d) subprocess: legacy -> _vs2 tag", bool(n2) and n2[-1].endswith("_vs2"), n2[-1] if n2 else out2[-300:])

    # ------------------------------------------------------------------ (c) refused drop list
    bad_name = "_tmp_check_h1_bad.json"
    bad = dict(json.load(open(os.path.join(HERE, "splits", VS3), encoding="utf-8")))
    bad["drop"] = list(bad["drop"]) + [bad["val"][0]]
    bad["tag"] = "vsbad"
    bad_path = os.path.join(HERE, "splits", bad_name)
    json.dump(bad, open(bad_path, "w", encoding="utf-8"))
    try:
        rcb, outb = run({"SPLIT_NAME": bad_name},
                        code="from tes_gru.data import load_all_data; load_all_data(include_test=False); print('LOADED_OK')")
    finally:
        os.remove(bad_path)
    report("(c) a json whose drop list names a validation file is refused",
           rcb != 0 and "LOADED_OK" not in outb.splitlines() and "drop list names" in outb,   # 3.13 tracebacks echo the -c source
           "exit %d; %s" % (rcb, [l for l in outb.splitlines() if "drop list" in l][-1:] or outb[-200:]))

    n_fail = RESULTS.count(False)
    print("\ncheck_h1: %d items, %d FAIL" % (len(RESULTS), n_fail))
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
