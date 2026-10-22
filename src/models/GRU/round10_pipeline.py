"""Registered Round 10 workflow, with deterministic decisions and resume.

The arm protocol is separate from outcomes. Validation alone selects a
candidate; B6 cannot change it; the final recipe is frozen before F9.
"""
import argparse
import ast
import concurrent.futures
import glob
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time

from paper_report import atomic_json, read_json, sha256, pid_alive
from round10_integrity import matches_frozen_sha256
from round10_migration import ensure_v4_restart, restart_v4

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANT = "forward_direct"
STAGES = ["S0", "S1", "S1b", "S2", "S3", "S4", "S5", "S5b", "S6", "S7", "S7b", "S8", "S9"]
CH = ("T_inner", "T_outer", "T_avg")


def protocol():
    return read_json(os.path.join(HERE, "prereg", "round10_v4.json"))


def arm(name, stage, env, seeds, condition=None):
    return dict(arm=name, stage=stage, env=env, seeds=list(map(int, seeds)), condition=condition)


def base_env():
    from run_round10 import BASE
    e = dict(BASE)
    e.update(RUN_PREFIX="r10v4", SEAL_TEST="1", LOSS_W0_MODE="", ENC_WIDTHS="256,256",
             H0_MODE="linear", FEAT_COLS="Time (s),Input Temperature (C)",
             EARLY_STEP_WEIGHT="1", RATIO_CHANNELS="R3", RATIO_REF="kelvin", PHYS_MODE="",
             INIT_ENCODER="point", INIT_BURNIN_STEPS="0", SYNTHETIC_REST_FILES="0",
             FROZEN_CONSTANTS="")
    e["PHYSICS_CACHE_DIR"] = resolved_resource_paths()["PHYSICS_CACHE_DIR"]
    return e


def full_arm_table(prereg, candidate=None, winners=None, frozen_candidate=None):
    """One table for planning, real launching and the all-arm smoke."""
    from run_round10 import ARMS
    seeds = prereg["seeds"]["S20"]
    cross_seeds = prereg["seeds"]["S10"]
    base = base_env()
    table = []
    for name, env in ARMS:
        table.append(arm(name, "S1", dict(dict(base, **env), RUN_PREFIX="r10v4"),
                         cross_seeds if name == "RETRAIN" else seeds))
    direct = dict(base, HIDDEN_SIZE="128", TAVG_MASK="split")
    def add(name, stage, overrides, sd=seeds, condition=None):
        table.append(arm(name, stage, dict(direct, **{k: str(v) for k, v in overrides.items()}), sd, condition))
    # These are the five cells explicitly enumerated by the registered P2.
    cells = re.findall(r"\((6|12|1),([01])\)", prereg["batches"]["P2"]["arms"])
    if len(cells) != 5:
        raise RuntimeError("P2 arm protocol no longer identifies exactly five cells")
    for scale, lead in cells:
        add("H8_a%s_l%s" % (scale, lead), "S2", dict(ANCHOR_SCALE=scale, INPUT_LOOKAHEAD=lead))
    add("H8_inv_kc", "S2", dict(LOSS_W0_MODE="inv_kc"))
    add("C14", "S2", dict(TINNER_MODE="arfed"))
    add("C15", "S2", dict(ANCHOR_LEAD="0"))
    for ratio in ("R3", "R1", "R2"):
        add(ratio, "S3", dict(OTHER_CH_MODE="ratio_input", RATIO_CHANNELS=ratio,
                             TINNER_MODE="arfed" if ratio == "R3" else "anchor"))
    add("R2C", "S3", dict(OTHER_CH_MODE="ratio_input", RATIO_CHANNELS="R2", RATIO_REF="celsius20"))
    add("C3", "S3", dict(TOUTER_MODE="anchor_input"))
    add("C2", "S3", dict(OTHER_CH_MODE="anchor_initial"))
    # Supplementary *already registered* G7 parameters are shipped verbatim
    # with provenance. No choices are made from validation outcomes here.
    extra = read_json(os.path.join(HERE, "prereg", "round10_execution_v4.json"))
    add("G7a", "S4", dict(EARLY_STEP_WEIGHT=extra["G7a"]["weight"]))
    add("G7b", "S4", dict(INIT_ENCODER="window", WINDOW_SIZE=extra["G7b"]["window"]))
    add("ENC_linear", "S4", dict(ENC_WIDTHS=""))
    add("ENC_16_16", "S4", dict(ENC_WIDTHS="16,16"))
    add("H0_tanh", "S4", dict(H0_MODE="tanh"))
    add("H0_zero", "S4", dict(H0_MODE="zero"))
    add("NO_TIME", "S4", dict(FEAT_COLS="Input Temperature (C)"))
    add("G7c", "S4", dict(INIT_BURNIN_STEPS=extra["G7c"]["steps"]))
    add("G7d", "S4", dict(SYNTHETIC_REST_FILES=extra["G7d"]["files"]))
    for name, mode in (("C11", "residual"), ("C12", "input_residual"), ("C13", "input")):
        add(name, "S5", dict(PHYS_MODE=mode))
    lookup = {a["arm"]: a for a in table}
    confirm = winners if winners is not None else ["H8_a6_l0", "R3", "C11", "ENC_linear"]
    for name in confirm:
        source = lookup[name]
        table.append(arm("C17_" + name, "S6", dict(source["env"], HIDDEN_SIZE="160"), seeds,
                         "family winner meets its registered adoption rule"))
    for split_seed in (43, 44):
        add("F4_direct_%d" % split_seed, "S7", dict(SPLIT_NAME="round10_vs3_s%d.json" % split_seed), cross_seeds)
        if candidate:
            source = lookup[candidate]
            table.append(arm("F4_%s_%d" % (candidate, split_seed), "S7",
                dict(source["env"], SPLIT_NAME="round10_vs3_s%d.json" % split_seed), cross_seeds, "confirmed challenger"))
    effective = candidate if frozen_candidate is None else (None if frozen_candidate == "direct" else frozen_candidate)
    for fold in range(1, 11):
        fold_env = dict(SPLIT_NAME="loto_v4_fold%d.json" % fold, FROZEN_CONSTANTS="", CONSTANTS_FROM="")
        add("F1_%d" % fold, "S7b", fold_env, extra["F1"]["seeds"])
        if effective:
            source = lookup[effective]
            table.append(arm("F1_%s_%d" % (effective, fold), "S7b",
                dict(source["env"], **fold_env), extra["F1"]["seeds"], "frozen recipe differs from direct"))
    frozen_env = lookup[effective]["env"] if effective else direct
    table.append(arm("F9", "S8", dict(frozen_env, HIDDEN_SIZE="160"), prereg["seeds"]["F9_fresh_primes"]))
    if effective:
        table.append(arm("F10", "S8", dict(base, TAVG_MASK="split"), prereg["seeds"]["F9_fresh_primes"], "recipe differs from direct"))
    return table


def resolved_data_root(env=None):
    env = os.environ if env is None else env
    return os.path.abspath(env.get("DATA_ROOT") or os.path.join(HERE, "..", "..", "..", "data"))


def resolved_resource_paths(env=None):
    """Resolve caller-relative paths before any child changes its cwd."""
    env = os.environ if env is None else env
    resource_root = os.path.join(HERE, "round10_resources", "round9")
    defaults = dict(PHYSICS_CACHE_DIR=os.path.join(HERE, "round10_resources", "physics", "nominal"),
                    ROUND9_RUNS_ROOT=resource_root,
                    CHAMPION_RUN_DIR=os.path.join(resource_root,
                        "2026-08-06_forward_direct_W10_Tin-anchor_l1_w1-6-3_oth-abs_h160x1dp0.2_ep2000_vs2_seed7"))
    return {key: os.path.abspath(env.get(key) or default) for key, default in defaults.items()}


def set_cpu_threads(n):
    """Cap the CPU thread pools of every child run (isolated_env inherits os.environ).

    n <= 0 keeps the PyTorch default (one pool thread per core). 2026-10-22,
    round10_posthoc/bench_threads.py: the pool threads mostly spin while the GRU's
    sequential GPU steps pace each epoch. On the lab server (48 threads, RTX 4090)
    3 workers x unlimited gave 7.0 epochs/s in total, 3 x 1 thread 15.6, 8 x 1 18.0,
    10 x 1 18.2 (GPU-bound beyond ~8); the loss history was identical in every setting.
    """
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if n > 0:
            os.environ[var] = str(n)
        else:
            os.environ.pop(var, None)


def isolated_env(overrides):
    env = dict(os.environ)
    # A pipeline started from a research shell must not inherit experimental
    # knobs absent from its explicitly frozen protocol.
    from run_round9 import BASE as legacy
    import eval_checkpoint
    transport = {k: env[k] for k in ("DATA_ROOT", "CHAMPION_RUN_DIR", "ROUND9_RUNS_ROOT", "ROUND10_NO_PLOTS") if k in env}
    with open(os.path.join(HERE, "tes_gru", "config.py"), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    read_knobs = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
            if isinstance(node.func, ast.Attribute) and node.func.attr in ("get", "getenv"):
                if isinstance(node.args[0].value, str):
                    read_knobs.add(node.args[0].value)
    for key in set(legacy) | set(base_env()) | read_knobs | set(eval_checkpoint.KEYS.values()) | set(eval_checkpoint.ENV_VARS_EXTRA):
        env.pop(key, None)
    env.update(transport)
    env.update(overrides)
    env["DATA_ROOT"] = resolved_data_root(env)
    env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", MPLBACKEND="Agg",
               TEMP=os.path.join(HERE, "scratch"), TMP=os.path.join(HERE, "scratch"))
    env.update(resolved_resource_paths(env))
    # Statistical replay is part of this frozen package, not a developer's
    # sibling wiki checkout or a research-shell fixture override.
    env["BAND_CAL_DIR"] = os.path.join(HERE, "round10_resources", "band_calibration")
    os.makedirs(env["TEMP"], exist_ok=True)
    return env


def resolve_names(table):
    # Import config repeatedly inside ONE isolated process; avoid loading
    # CUDA dozens of times solely to format run names.
    script = """import json,os,sys,importlib
arms=json.loads(sys.stdin.read()); keys=set(k for a in arms for k in a['env'])
out=[]
for a in arms:
 for k in keys: os.environ.pop(k,None)
 os.environ.update(a['env']);os.environ['SEEDS']='7'
 sys.modules.pop('tes_gru.config',None)
 from tes_gru import config
 # from-package cached attribute must also be refreshed
 config=importlib.import_module('tes_gru.config')
 out.append({'arm':a['arm'],'name':config.RUN_NAME_BASE})
print('ROUND10_NAMES='+json.dumps(out))
"""
    p = subprocess.run([sys.executable, "-c", script], input=json.dumps(table),
                       env=isolated_env(base_env()), cwd=HERE, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True)
    lines = [s[len("ROUND10_NAMES="):] for s in p.stdout.splitlines() if s.startswith("ROUND10_NAMES=")]
    if p.returncode or not lines:
        raise RuntimeError("Config name resolution failed:\n" + p.stdout[-4000:])
    names = {r["arm"]: r["name"] for r in json.loads(lines[-1])}
    occupied = {}
    for a in table:
        a["run_name"] = names[a["arm"]]
        for seed in a["seeds"]:
            key = (a["run_name"], seed)
            if key in occupied:
                # F9/direct and earlier baselines have disjoint fresh seeds.
                raise RuntimeError("Run-name collision: %s and %s" % (a["arm"], occupied[key]))
            occupied[key] = a["arm"]
    return table


def run_record(a, seed):
    return dict(arm=a["arm"], seed=int(seed), run_dir=os.path.join(HERE, "runs", "%s_seed%d" % (a["run_name"], seed)))


def verify_done(record, env, reproduce=False):
    from run_round10 import verify_sealed_run
    rd = record["run_dir"]
    flag = os.path.join(rd, "variants", VARIANT, "done.flag")
    if not os.path.isfile(flag):
        return False
    if reproduce:
        result = verify_sealed_run(rd, env)
        if not result["ok"]:
            raise RuntimeError("Smoke validation failed for %s: %s" % (record, result))
        return result
    rc = read_json(os.path.join(rd, "run_config.json"))
    meta = read_json(os.path.join(rd, "variants", VARIANT, "meta.json"))
    if not rc.get("sealed") or not meta.get("sealed") or any(k.startswith("Test_") for k in meta):
        raise RuntimeError("Unsealed done.flag cannot be reused: " + rd)
    if int(rc["seed"]) != int(record["seed"]):
        raise RuntimeError("Resume seed disagrees with directory/manifest: " + rd)
    import eval_checkpoint
    saved = eval_checkpoint.env_from_config(rc)
    for key in set(saved) & set(env):
        if key in ("SEEDS", "VARIANTS", "DROPOUT"):
            continue
        if str(saved[key]) != str(env[key]):
            try:
                same_number = float(saved[key]) == float(env[key])
            except (TypeError, ValueError):
                same_number = False
            if not same_number:
                raise RuntimeError("Resume config mismatch %s: %s != %s in %s" % (key, saved[key], env[key], rd))
    required = [os.path.join(rd, "checkpoints", "best_gru_variant_%s.pth" % VARIANT),
                os.path.join(rd, "checkpoints", "best_gru_variant_%s.constants.json" % VARIANT)]
    required += [os.path.join(rd, "variants", VARIANT, s) for s in ("summary_errors_val.csv", "predictions_val.npz", "meta.json")]
    if not all(os.path.isfile(p) for p in required):
        raise RuntimeError("Done flag has incomplete artifacts: " + rd)
    return True


def launch_arm(a, workers, result_dir, smoke=False):
    os.makedirs(os.path.join(HERE, "sweep_logs"), exist_ok=True)
    active = set()
    active_lock = threading.Lock()
    stopped = threading.Event()
    def one(seed):
        if stopped.is_set():
            raise RuntimeError("Arm cancelled after another worker failed")
        record = run_record(a, seed)
        env = dict(a["env"], SEEDS=str(seed), SEAL_TEST="1")
        if verify_done(record, env):
            return record
        log = os.path.join(HERE, "sweep_logs", "pipeline-v4-%s-seed%d.log" % (a["arm"], seed))
        with open(log, "a", encoding="utf-8") as f:
            p = subprocess.Popen([sys.executable, os.path.join(HERE, "GRU_input_ablation.py")],
                env=isolated_env(env), cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
            with active_lock:
                active.add(p)
            try:
                code = p.wait()
            except BaseException:
                p.terminate()
                p.wait()
                raise
            finally:
                with active_lock:
                    active.discard(p)
        if not verify_done(record, env):
            raise RuntimeError("%s seed%d stopped (exit %d); see %s" % (a["arm"], seed, code, log))
        # CUDA teardown on this Windows machine can fault after completion;
        # accept ONLY the known code and fully verified completion artifacts.
        if code not in (0, -1073740791, 3221226505):
            raise RuntimeError("%s seed%d exited %d despite artifacts; see %s" % (a["arm"], seed, code, log))
        return record
    records = []
    print("[%s] %s: %d seeds (%d workers)" % (a["stage"], a["arm"], len(a["seeds"]), workers), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, len(a["seeds"]))) as pool:
        jobs = {pool.submit(one, seed): seed for seed in a["seeds"]}
        try:
            for job in concurrent.futures.as_completed(jobs):
                record = job.result()
                records.append(record)
                print("  %s seed%d complete (%d/%d)" % (a["arm"], record["seed"], len(records), len(a["seeds"])), flush=True)
        except BaseException:
            stopped.set()
            for job in jobs:
                job.cancel()
            with active_lock:
                for process in list(active):
                    if process.poll() is None:
                        process.terminate()
            raise
    return sorted(records, key=lambda r: r["seed"])


def write_manifest(path, records, **kwargs):
    resources = resolved_resource_paths()
    obj = dict(runs=records, data_root=resolved_data_root(),
               physics_cache_dir=resources["PHYSICS_CACHE_DIR"],
               round9_root=resources["ROUND9_RUNS_ROOT"])
    obj.update(kwargs)
    atomic_json(path, obj)
    return obj


def command(script, args, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, os.path.basename(script) + ".log")
    with open(path, "a", encoding="utf-8") as f:
        p = subprocess.run([sys.executable, os.path.join(HERE, script)] + args,
                           env=isolated_env(base_env()),
                           cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode:
        raise RuntimeError("%s failed (exit%d): %s" % (script, p.returncode, path))


def preflight(out):
    import torch
    if sys.version_info < (3, 9):
        raise RuntimeError("Python >=3.9 required; lab servers should use3.10/3.11")
    for path in glob.glob(os.path.join(HERE, "*.py")) + glob.glob(os.path.join(HERE, "tes_gru", "*.py")):
        with open(path, encoding="utf-8-sig") as f:
            ast.parse(f.read(), filename=path, feature_version=(3, 9))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable in this Python environment")
    dataset_identity = verify_dataset_hashes()
    free = shutil.disk_usage(HERE).free / (1024 ** 3)
    if free < 30:
        raise RuntimeError("Need >=30GiB free disk, found%.1f" % free)
    lock = read_json(os.path.join(HERE, "prereg", "round10_package_hashes.json"))
    for relative, digest in lock["files"].items():
        if not matches_frozen_sha256(os.path.join(HERE, relative), digest):
            raise RuntimeError("Frozen package input changed: " + relative)
    p = protocol()
    split_path = os.path.join(HERE, "splits", "round10_vs3_s42.json")
    if not matches_frozen_sha256(split_path, p["data"]["hashes"]["split_json_sha256"]):
        raise RuntimeError("vs3 split hash differs from prereg")
    for script in sorted(glob.glob(os.path.join(HERE, "round10_checks", "check_*.py"))):
        command(os.path.relpath(script, HERE), [], out)
    command("round10_validation_gate.py", ["--out", os.path.join(out, "regression.json")], out)
    result = dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(0), free_disk_GiB=free,
                  prereg_sha256=sha256(os.path.join(HERE, "prereg", "round10_v4.json")),
                  test_access=False, package_hashes_verified=len(lock["files"]))
    result["dataset_sha256"] = dataset_identity
    result["training_source_files_verified"] = 311
    atomic_json(os.path.join(out, "preflight.json"), result)
    print("Preflight PASS: %s, free disk%.1fGiB; no test access" % (result["gpu"], free))


def verify_dataset_hashes():
    path = os.path.join(HERE, "round10_resources", "training_source_hashes.json")
    expected = read_json(path)
    root = resolved_data_root()
    train = os.path.join(root, expected["train_subdir"])
    for key, digest in expected["files"].items():
        if not matches_frozen_sha256(os.path.join(train, *key.split("/")), digest):
            raise RuntimeError("Training source changed or differs from the sealed dataset: " + key)
    return sha256(path)


def comparison(records, arms, pairs, family, out):
    import pandas as pd
    import paper_stats as stats
    frames = []
    for record in records:
        if record["arm"] not in arms:
            continue
        path = os.path.join(record["run_dir"], "variants", VARIANT, "summary_errors_val.csv")
        rows = pd.read_csv(path)
        for _, row in rows[rows.Case.astype(str) != "AVERAGE"].iterrows():
            for ch in CH:
                # Case is the full relative path; CaseFile is display-only
                # and can repeat across different thermal-profile folders.
                frames.append(dict(arm=record["arm"], seed=record["seed"], file=row["Case"],
                                   channel=ch, MAE=row["MAE_%s (C)" % ch]))
            for ch in stats.JUMP_COLS:
                frames.append(dict(arm=record["arm"], seed=record["seed"], file=row["Case"], channel=ch, MAE=row[ch]))
    long = pd.DataFrame(frames)
    os.makedirs(out, exist_ok=True)
    long.to_csv(os.path.join(out, family + "_long.csv"), index=False)
    tidy, verdicts = stats.compare_family(long, pairs, family=family,
        bands=protocol()["bands"]["fixed_bands_degC"], seed=0)
    stats.write_outputs(tidy, verdicts, out, family)
    return tidy, verdicts, long


def adoption_flags(tidy, verdict, arm_name, source_stage, manifest, out):
    """Additional registered family predicates are evaluated from readouts."""
    rows = {r["channel"]: r for r in tidy.to_dict("records") if r["armA"] == arm_name}
    v = next(r for r in verdict.to_dict("records") if r["armA"] == arm_name)
    p = protocol()
    allowed = v["verdict"] == "BETTER" and not v["vetoed"]
    if source_stage == "S2":
        rule = p["batches"]["P2"]["rule"]
        # P2 explicitly requires overall+inner, despite the general statement
        # excluding overall. This explicit family rule is logged verbatim.
        from round10_decisions import overall_difference
        overall = overall_difference(manifest["runs"], arm_name, "B2_mask")
        allowed = (overall["hi"] < 0 and rows["T_inner"]["ci_hi"] < 0 and
                   all(rows[c]["ci_hi"] <= rows[c]["band"] for c in ("T_outer", "T_avg")) and not v["vetoed"])
    elif source_stage == "S5" and arm_name in ("C11", "C12"):
        from round10_decisions import physics_adoption
        allowed = allowed and physics_adoption(manifest["runs"], arm_name, p)
        rule = p["batches"]["P5"]["rule"]
    elif source_stage == "S4":
        from round10_decisions import p4_adoption
        allowed, diagnostic = p4_adoption(manifest["runs"], arm_name, p, out)
        rule = p["batches"]["P4"]
        return bool(allowed), dict(verdict=v, diagnostic=diagnostic, rule=rule)
    else:
        rule = p["batches"]["P3"]["rule"] if source_stage == "S3" else p["verdict_classes"]
    return bool(allowed), dict(verdict=v, rows=rows, rule=rule)


class Pipeline:
    def __init__(self, args):
        self.args = args
        self.out = os.path.join(HERE, "runs", "_round10_v4_results")
        os.makedirs(self.out, exist_ok=True)
        self.state_path = os.path.join(self.out, "pipeline_state.json")
        self.identity = sha256(os.path.join(HERE, "prereg", "round10_v4.json"))
        if not os.path.isfile(self.state_path):
            ensure_v4_restart(HERE)
        package_path = os.path.join(HERE, "prereg", "round10_package_hashes.json")
        package_identity = sha256(package_path)
        package = read_json(package_path)
        for relative, expected in package["files"].items():
            if not matches_frozen_sha256(os.path.join(HERE, relative), expected):
                raise RuntimeError("Package input changed during a pipeline: " + relative)
        dataset_identity = verify_dataset_hashes()
        self.state = read_json(self.state_path) if os.path.isfile(self.state_path) else dict(
            prereg_sha256=self.identity, package_sha256=package_identity,
            dataset_sha256=dataset_identity, stages={}, runs=[], decisions=[], candidate=None, winners=[])
        if self.state["prereg_sha256"] != self.identity:
            raise RuntimeError("Prereg changed; cannot combine outcomes across protocols")
        if self.state.get("package_sha256") != package_identity:
            raise RuntimeError("Package/splits/protocol changed; cannot resume old stages")
        if self.state.get("dataset_sha256") != dataset_identity:
            raise RuntimeError("Training source seal changed; cannot combine stages")
        self.records = self.state["runs"]

    def save(self):
        self.state["runs"] = self.records
        atomic_json(self.state_path, self.state)

    def decide(self, stage, key, data):
        entry = dict(stage=stage, key=key, evidence=data)
        self.state["decisions"].append(entry)
        self.save()
        path = os.path.join(self.out, "decision_log.md")
        # Rewrite from durable journal: crash recovery cannot duplicate entries.
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Registered automatic decisions\n\n")
            for d in self.state["decisions"]:
                f.write("## %s — %s\n\n```json\n%s\n```\n\n" % (d["stage"], d["key"],
                    json.dumps(d["evidence"], indent=2, default=str)))

    def add_runs(self, table, stage):
        table = resolve_names([a for a in table if a["stage"] == stage])
        for a in table:
            records = launch_arm(a, self.args.parallel, self.out)
            existing = {(r["arm"], r["seed"]) for r in self.records}
            self.records.extend(r for r in records if (r["arm"], r["seed"]) not in existing)
            self.save()
        return table

    def manifest(self, stage, subset=None, **extra):
        path = os.path.join(self.out, stage, "manifest.json")
        records = self.records if subset is None else [r for r in self.records if r["arm"] in subset]
        return path, write_manifest(path, records, **extra)

    def analyses(self, stage, scripts, subset=None):
        path, _ = self.manifest(stage, subset)
        for script in scripts:
            command(script + ".py", ["--manifest", path, "--out", os.path.join(self.out, stage, script)],
                    os.path.join(self.out, stage))

    def run_stage(self, stage):
        out = os.path.join(self.out, stage)
        os.makedirs(out, exist_ok=True)
        table = full_arm_table(protocol(), self.state.get("candidate"), self.state.get("winners"),
                               self.state.get("frozen_recipe", {}).get("arm"))
        if stage == "S0":
            preflight(out)
        elif stage in ("S1", "S2", "S3", "S5"):
            self.add_runs(table, stage)
            from round10_decisions import descriptive_tables
            descriptive_tables([r for r in self.records if r["arm"] in {a["arm"] for a in table if a["stage"] == stage}], out, stage)
        elif stage == "S1b":
            from round10_decisions import p1_readouts, descriptive_tables
            descriptive_tables(self.records, out, "P1")
            comparison(self.records, {"B1_mask", "B1_orig", "B2_mask", "B2_orig", "B7", "B8"},
                       [("B1_mask", "B1_orig"), ("B2_mask", "B2_orig"), ("B7", "B2_orig"), ("B8", "B2_orig")], "P1", out)
            evidence = p1_readouts(self.records, protocol(), out)
            self.decide(stage, "P1 label/control interpretation (does not select a recipe)", evidence)
            path, _ = self.manifest(stage + "/B6", {"B1_mask", "B1_orig"})
            command("paper_report.py", ["--b6", "--manifest", path, "--out", os.path.join(out, "B6")], out)
            self.analyses(stage, ["g1_probe", "a7_bridge"], {"B1_mask", "B1_orig", "B2_mask", "B2_orig", "B2_relabel", "B2_drop9", "B7", "B8", "RETRAIN"})
        elif stage == "S4":
            self.add_runs(table, stage)
            self.decide(stage, "All nine P4 arms completed before adoption decisions",
                        dict(rule=protocol()["batches"]["P4"]["arms"], screening=False))
            from round10_decisions import descriptive_tables
            descriptive_tables([r for r in self.records if r["arm"] in {a["arm"] for a in table if a["stage"] == stage}], out, stage)
        elif stage == "S5b":
            relevant = {a["arm"] for a in table if a["stage"] in ("S2", "S3", "S4", "S5")} | {"B1_mask", "B2_mask"}
            self.analyses(stage, ["c16_curves", "e3_g11_shift", "linear_sysid", "error_budget"], relevant)
            import pandas as pd
            from round10_decisions import per_seed
            summary = pd.read_csv(os.path.join(out, "linear_sysid", "linear_validation_summary.csv"))
            linear = float(summary.loc[(summary.model == "linear_binned") & (summary.channel == "T_outer"), "mean"].iloc[0])
            gru = float(per_seed(self.records, "B2_mask", "MAE_T_outer (C)").mean())
            rule = protocol()["batches"]["linear_sysID"]["rule"]
            limit = re.search(r"within\s+([0-9.]+)\s+degC", rule)
            if not limit:
                raise RuntimeError("Unsupported linear-baseline interpretation rule")
            close = abs(linear - gru) <= float(limit.group(1))
            self.decide(stage, "linear baseline and paper ML-value wording", dict(rule=rule,
                binned_linear_T_outer=linear, direct_GRU_T_outer=gru, difference=linear - gru,
                within_registered_margin=close, wording="Narrow the ML-value claim: the linear baseline is competitive on T_outer" if close
                else "Report measured wall advantage with its validation scope"))
        elif stage == "S6":
            self.select_and_confirm(table, out)
        elif stage == "S7":
            self.add_runs(table, stage)
        elif stage == "S7b":
            from round10_decisions import freeze_recipe
            candidate = self.state.get("candidate")
            frozen = self.state.get("frozen_recipe")
            if frozen is None:
                frozen = freeze_recipe(self.records, candidate, protocol(), out)
                self.state["frozen_recipe"] = frozen
                self.decide(stage, "final recipe frozen before F1 and fresh seed launch", frozen)
                atomic_json(os.path.join(out, "frozen_recipe.json"), frozen)
            self.add_runs(full_arm_table(protocol(), candidate, [], frozen_candidate=frozen["arm"]), stage)
            path, _ = self.manifest(stage, frozen_recipe=frozen)
            command("f1_envelope.py", ["--manifest", path, "--out", os.path.join(out, "f1_envelope")], out)
        elif stage == "S8":
            frozen = self.state.get("frozen_recipe")
            if not frozen:
                raise RuntimeError("S8 requires the recipe frozen before F1")
            self.add_runs(full_arm_table(protocol(), self.state.get("candidate"), [],
                          frozen_candidate=frozen["arm"]), stage)
            from round10_decisions import descriptive_tables
            descriptive_tables([r for r in self.records if r["arm"] in ("F9", "F10")], out, stage)
        elif stage == "S9":
            frozen = self.state["frozen_recipe"]
            names = {"F9", "F10", "B1_mask", "B1_orig", "RETRAIN"}
            if frozen["arm"] != "direct":
                names.add("C17_" + frozen["arm"])
                names.add(frozen["arm"])
            # Only confirmed reference families enter C16's test side.
            for winner in self.state.get("confirmed", []):
                names.add("C17_" + winner)
            path, manifest = self.manifest(stage, names, frozen_recipe=frozen,
                    analyses_root=os.path.join(self.out, "S5b"), confirmed=self.state.get("confirmed", []))
            from round10_decisions import round9_twins
            manifest["runs"].extend(round9_twins(manifest["round9_root"], protocol()))
            atomic_json(path, manifest)
            command("paper_report.py", ["--f11", "--manifest", path, "--out", out], out)

    def select_and_confirm(self, table, out):
        if "candidate_frozen" in self.state:
            self.add_runs(table, "S6")
        else:
            winners = []
            diagnostics = {}
            from round10_decisions import choose_family_winner
            registered_candidates = {"S2": [a["arm"] for a in table if a["stage"] == "S2" and a["arm"].startswith("H8_")],
                                     "S3": ["R3"],
                                     "S4": [a["arm"] for a in table if a["stage"] == "S4"],
                                     "S5": ["C11", "C12"]}
            for stage, family in (("S2", "anchor"), ("S3", "ratio"), ("S4", "encoder_G7"), ("S5", "physics")):
                candidates = [n for n in registered_candidates[stage] if any(r["arm"] == n for r in self.records)]
                candidate_pairs = [(name, "B2_mask") for name in candidates]
                if stage == "S2":
                    import pandas as pd
                    pieces = [comparison(self.records, set(names + ["B2_mask"]), [(n, "B2_mask") for n in names],
                        subgroup, os.path.join(out, subgroup)) for subgroup, names in
                        (("anchor_grid", [n for n in candidates if n != "H8_inv_kc"]), ("anchor_share", ["H8_inv_kc"]))]
                    tidy = pd.concat([piece[0] for piece in pieces], ignore_index=True)
                    verdict = pd.concat([piece[1] for piece in pieces], ignore_index=True)
                else:
                    tidy, verdict, _ = comparison(self.records, set(candidates + ["B2_mask"]), candidate_pairs,
                                                family, os.path.join(out, family))
                path, manifest = self.manifest("S6/" + family)
                flags = {}
                for name in candidates:
                    allowed, detail = adoption_flags(tidy, verdict, name, stage, manifest, out)
                    flags[name] = allowed
                    diagnostics[name] = detail
                if stage == "S4":
                    # The registration independently asks for the smallest
                    # encoder, a startup treatment, and NO_TIME confirmation.
                    groups = [[n for n in candidates if n.startswith("ENC_") or n.startswith("H0_")],
                              [n for n in candidates if n.startswith("G7")], ["NO_TIME"]]
                    for names in groups:
                        winner = choose_family_winner(names, flags, tidy, family, self.records, protocol())
                        if winner and winner not in winners:
                            winners.append(winner)
                else:
                    winner = choose_family_winner(candidates, flags, tidy, family, self.records, protocol())
                    if winner:
                        winners.append(winner)
            self.state["winners"] = winners
            self.state["candidate_frozen"] = True
            self.decide("S6", "family candidates selected before confirmation/F4", dict(winners=winners, evidence=diagnostics))
            self.add_runs(full_arm_table(protocol(), None, winners), "S6")
        confirmed = []
        deployment_confirmed = []
        scores = {}
        for winner in self.state["winners"]:
            name = "C17_" + winner
            tidy, verdict, _ = comparison(self.records, {name, "B1_mask"}, [(name, "B1_mask")],
                                         name, os.path.join(out, name))
            v = verdict.iloc[0].to_dict()
            scores[winner] = v
            p4 = winner.startswith("ENC_") or winner.startswith("H0_") or winner.startswith("G7") or winner == "NO_TIME"
            if p4:
                from round10_decisions import p4_adoption
                allowed, detail = p4_adoption(self.records, name, protocol(), out)
                scores[winner]["P4_confirmation"] = detail
                if allowed:
                    deployment_confirmed.append(winner)
            else:
                allowed = True
            # A P4 equivalence adoption is separately reported for deployment.
            # The paper's recipe requires BETTER at both capacities.
            _, initial_v = __import__("round10_decisions").compare(self.records, winner, "B2_mask", protocol())
            if allowed and initial_v["verdict"] == "BETTER" and v["verdict"] == "BETTER" and not v["vetoed"]:
                confirmed.append(winner)
        confirmed.sort(key=lambda n: (scores[n]["ToTa_diff"], n))
        self.state["confirmed"] = confirmed
        self.state["deployment_confirmed"] = deployment_confirmed
        self.state["candidate"] = confirmed[0] if confirmed else None
        self.decide("S6", "C17 confirmations and frozen F4 top two", dict(
            candidate=self.state["candidate"], confirmed=confirmed, deployment_confirmed=deployment_confirmed, evidence=scores,
            recipe_rule=protocol()["recipe_change_rule"]))

    def run(self):
        chosen = STAGES if not self.args.stage else [self.args.stage]
        if self.args.until:
            chosen = chosen[:chosen.index(self.args.until) + 1] if self.args.until in chosen else chosen
        for stage in chosen:
            prior = STAGES[:STAGES.index(stage)]
            if any(self.state["stages"].get(s, {}).get("status") != "done" for s in prior):
                raise RuntimeError("%s requires completed stages: %s" % (stage, prior))
            if self.state["stages"].get(stage, {}).get("status") == "done":
                print("[%s] already complete" % stage, flush=True)
                continue
            self.state["stages"][stage] = dict(status="running", started=time.time())
            self.save()
            try:
                self.run_stage(stage)
            except BaseException as e:
                self.state["stages"][stage].update(status="failed", reason=str(e))
                self.save()
                raise
            self.state["stages"][stage].update(status="done", completed=time.time())
            self.save()
            print("[%s] complete" % stage, flush=True)


def smoke(args):
    # Includes every named core/conditional arm, all ten F1 fold layouts,
    # redraws, fresh-seed paths and every possible C17 family implementation.
    p = protocol()
    core = full_arm_table(p, candidate="R3", winners=[])
    possible = [a for a in core if a["arm"].startswith("H8_") or a["arm"] == "R3" or a["stage"] == "S4" or a["arm"] in ("C11", "C12")]
    for a in possible:
        core.append(arm("C17_" + a["arm"], "S6", dict(a["env"], HIDDEN_SIZE="160"), [7], "smoke path"))
    # Every eligible recipe gets one strict-fold smoke, in addition to all
    # ten direct and R3 folds. This covers each representation's fitted
    # constants and reconstruction without multiplying the full design.
    for a in possible:
        if a["arm"] != "R3":
            core.append(arm("F1_" + a["arm"] + "_1", "S7b", dict(a["env"],
                SPLIT_NAME="loto_v4_fold1.json", FROZEN_CONSTANTS="", CONSTANTS_FROM=""), [7], "smoke path"))
    # Smoke must not collide with a production run or another smoke alias.
    selected = []
    for a in core:
        if args.only and a["arm"] not in args.only.split(","):
            continue
        a = dict(a, env=dict(a["env"], MAX_EPOCHS="2", RUN_PREFIX="r10v4smoke_" + a["arm"], SEAL_TEST="1", ROUND10_NO_PLOTS="1"), seeds=[7])
        selected.append(a)
    if not selected:
        raise RuntimeError("No smoke arms selected")
    selected = resolve_names(selected)
    out = os.path.join(HERE, "runs", "_round10_v4_smoke")
    os.makedirs(out, exist_ok=True)
    rows, targets = [], []
    try:
        for a in selected:
            for seed in a["seeds"]:
                target = run_record(a, seed)["run_dir"]
                # A previous smoke is reviewed, never silently promoted to
                # a formal run; a fresh invocation actually trains two epochs.
                if os.path.isdir(target):
                    safe_delete_smoke(target)
                targets.append(target)
            records = launch_arm(a, args.parallel, out, smoke=True)
            for record in records:
                result = verify_done(record, dict(a["env"], SEEDS=str(record["seed"])), reproduce=True)
                holdout = None
                if a["arm"].startswith("F1_"):
                    from f1_envelope import check_holdout_roundtrip
                    holdout = check_holdout_roundtrip(record["run_dir"], os.path.join(out, a["arm"] + "_holdout"))
                rows.append(dict(arm=a["arm"], seed=record["seed"], ok=result["ok"], holdout=holdout,
                                 max_delta=result["max_delta"], sealed=result["sealed"],
                                 no_test_key=result["no_test_key"], sidecar=result["sidecar"], repro=result["repro"]))
                atomic_json(os.path.join(out, "smoke_results.json"), rows)
    finally:
        if not args.keep_smoke:
            for target in targets:
                safe_delete_smoke(target)
    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(out, "smoke_results.csv"), index=False)
    if len(rows) != len(selected) or not all(r["ok"] for r in rows):
        raise RuntimeError("All-arm smoke incomplete")
    print("SMOKE PASS %d/%d; two epochs each, validation and strict F1 holdout round-trips; no test access" % (len(rows), len(selected)))


def safe_delete_smoke(path):
    root = os.path.realpath(os.path.join(HERE, "runs"))
    target = os.path.realpath(path)
    name = os.path.basename(target)
    if os.path.commonpath([root, target]) != root or not name.startswith("r10v4smoke_") or "_ep2_" not in name:
        raise RuntimeError("Unsafe smoke cleanup target: " + path)
    if os.path.isdir(target):
        shutil.rmtree(target)


def dry_run(args):
    p = protocol()
    # Counts are exact for the named unconditional graph; conditional
    # choices cannot have actual checkpoint names before their decisions.
    table = resolve_names(full_arm_table(p, winners=[]))
    minimum = 0
    counts = {}
    for a in table:
        counts[a["stage"]] = counts.get(a["stage"], 0) + len(a["seeds"])
        if not a["condition"]:
            minimum += len(a["seeds"])
        for seed in a["seeds"]:
            print("[%s] %-20s %s%s" % (a["stage"], a["arm"], os.path.basename(run_record(a, seed)["run_dir"]),
                " CONDITIONAL: " + a["condition"] if a["condition"] else ""))
    print("Unconditional graph: %d runs; all nine P4 arms unconditional; per-stage:%s" % (minimum, counts))
    # Enumerate *every possible* conditional name: per-family selections,
    # best-challenger redraws, recipe-specific F9 and optional F10. The
    # maximum is a bound on one executable branch, not this enumerated union.
    variants = []
    sources = [a for a in table if a["arm"].startswith("H8_") or a["arm"] == "R3" or a["stage"] == "S4" or a["arm"] in ("C11", "C12")]
    for source in sources:
        env = source["env"]
        variants.append(arm("C17_" + source["arm"], "S6", dict(env, HIDDEN_SIZE="160"), p["seeds"]["S20"], "eligible family winner"))
        for seed in (43, 44):
            variants.append(arm("F4_%s_%d" % (source["arm"], seed), "S7", dict(env, SPLIT_NAME="round10_vs3_s%d.json" % seed),
                                p["seeds"]["S10"], "best confirmed challenger"))
        for fold in range(1, 11):
            variants.append(arm("F1_%s_%d" % (source["arm"], fold), "S7b",
                dict(env, SPLIT_NAME="loto_v4_fold%d.json" % fold, FROZEN_CONSTANTS="", CONSTANTS_FROM=""),
                p["seeds"]["S10"], "frozen recipe differs from direct"))
        variants.append(arm("F9_" + source["arm"], "S8", dict(env, HIDDEN_SIZE="160"), p["seeds"]["F9_fresh_primes"], "frozen challenger recipe"))
    variants.append(arm("F10", "S8", dict(base_env(), TAVG_MASK="split"), p["seeds"]["F9_fresh_primes"], "recipe differs from direct"))
    variants = resolve_names(variants)
    for a in variants:
        for seed in a["seeds"]:
            print("[%s] %-20s %s CONDITIONAL: %s" % (a["stage"], a["arm"], os.path.basename(run_record(a, seed)["run_dir"]), a["condition"]))
    maximum = minimum + 6 * len(p["seeds"]["S20"]) + 12 * len(p["seeds"]["S10"]) + len(p["seeds"]["F9_fresh_primes"])
    print("Total actual executable branch: %d..%d runs. All possible names printed; no training or test access." % (minimum, maximum))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    modes = ap.add_mutually_exclusive_group(required=True)
    modes.add_argument("--all", action="store_true")
    modes.add_argument("--restart-v4", action="store_true", help="Audit a stopped v3 before starting a fresh v4 namespace; preserve all runs and ledgers")
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--stage", choices=STAGES)
    ap.add_argument("--until", choices=STAGES)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--keep-smoke", action="store_true")
    ap.add_argument("--only")
    ap.add_argument("--threads", type=int, default=1, help="CPU threads per training run (0 = PyTorch default)")
    args = ap.parse_args()
    set_cpu_threads(args.threads)
    if args.only and not args.smoke:
        ap.error("--only is for smoke diagnostics; registered production uses the complete stage graph")
    if args.parallel < 1:
        ap.error("--parallel must be positive")
    if os.environ.get("SEAL_TEST", "1") not in ("1", "true", "True"):
        ap.error("Every Round10 training run requires SEAL_TEST=1")
    os.environ["SEAL_TEST"] = "1"
    if args.dry_run:
        dry_run(args)
    elif args.restart_v4:
        receipt = restart_v4(HERE)
        print("V4 restart audit PASS; old runs and test ledgers preserved; namespace runs/_round10_v4_results")
    elif args.smoke:
        smoke(args)
    elif args.preflight:
        preflight(os.path.join(HERE, "runs", "_round10_v4_results", "S0"))
    else:
        pipe = Pipeline(args)
        lock = os.path.join(pipe.out, "pipeline.lock")
        if os.path.isfile(lock):
            with open(lock) as f:
                prior_pid = int(f.read().strip())
            if pid_alive(prior_pid):
                raise RuntimeError("A pipeline is already running")
            os.unlink(lock)
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            f.write(str(os.getpid()))
        try:
            pipe.run()
        finally:
            os.unlink(lock)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError) as exc:
        print("Round10 stopped: %s\nRerun the same command after resolving the reported cause." % exc, file=sys.stderr)
        sys.exit(1)
