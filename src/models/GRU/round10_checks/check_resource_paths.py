"""Check resource transport across cwd changes without training or test data."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import analysis_common
import round10_pipeline as pipeline
import run_round10 as runner

RESOURCE_KEYS = ("PHYSICS_CACHE_DIR", "CHAMPION_RUN_DIR", "ROUND9_RUNS_ROOT")


@contextlib.contextmanager
def working_directory(path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def set_paths(values):
    for key in RESOURCE_KEYS:
        os.environ.pop(key, None)
    os.environ.update(values)


def fixture_directory(root, name):
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def main():
    checks = []

    def passed(label):
        checks.append(label)
        print("PASS " + label, flush=True)

    scratch = HERE / "scratch"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="resource_paths_", dir=str(scratch)) as folder:
        root = Path(folder)
        caller = fixture_directory(root, "caller working directory")
        custom = {key: fixture_directory(caller, key.lower() + " with spaces") for key in RESOURCE_KEYS}
        expected_defaults = {
            "PHYSICS_CACHE_DIR": HERE / "round10_resources/physics/nominal",
            "ROUND9_RUNS_ROOT": HERE / "round10_resources/round9",
            "CHAMPION_RUN_DIR": HERE / "round10_resources/round9" /
                "2026-08-06_forward_direct_W10_Tin-anchor_l1_w1-6-3_oth-abs_h160x1dp0.2_ep2000_vs2_seed7",
        }
        modes = (
            ("unset", {}, expected_defaults),
            ("empty", dict.fromkeys(RESOURCE_KEYS, ""), expected_defaults),
            ("absolute paths with spaces", {k: str(v) for k, v in custom.items()}, custom),
            ("caller-relative paths", {k: v.name for k, v in custom.items()}, custom),
        )
        probe = root / "resource_probe.py"
        probe.write_text(
            "import json, os, sys\n"
            "keys=('PHYSICS_CACHE_DIR','CHAMPION_RUN_DIR','ROUND9_RUNS_ROOT')\n"
            "result={'cwd':os.getcwd(),'paths':{k:os.environ[k] for k in keys},"
            "'exists':{k:os.path.isdir(os.environ[k]) for k in keys}}\n"
            "with open(sys.argv[1], 'w') as stream: json.dump(result, stream)\n",
            encoding="utf-8")
        with working_directory(caller):
            for label, values, expected in modes:
                with patch.dict(os.environ):
                    set_paths(values)
                    output = root / "child.json"
                    pipeline.command(str(probe), [str(output)], str(root / "logs"))
                    observed = json.loads(output.read_text(encoding="utf-8"))
                    assert Path(observed["cwd"]) == HERE, observed
                    for key in RESOURCE_KEYS:
                        assert Path(observed["paths"][key]).is_absolute(), observed
                        assert Path(observed["paths"][key]) == expected[key], (label, key, observed)
                        assert observed["exists"][key], (label, key, observed)
                    manifest = pipeline.write_manifest(str(root / "manifest.json"), [])
                    assert manifest["physics_cache_dir"] == observed["paths"]["PHYSICS_CACHE_DIR"]
                    assert manifest["round9_root"] == observed["paths"]["ROUND9_RUNS_ROOT"]
                    training = pipeline.isolated_env(pipeline.base_env())
                    assert all(training[k] == observed["paths"][k] for k in RESOURCE_KEYS)
                    passed("actual child, training and manifest agree for " + label)

            # A synthetic repository verifies the default without touching any
            # real dataset or relying on the developer's DATA_ROOT setting.
            fake_here = fixture_directory(root, "synthetic repo/src/models/GRU")
            fake_data = fixture_directory(root, "synthetic repo/data")
            fixture_directory(fake_data, "Latest Database (Use this for training)")
            for value in (None, ""):
                with patch.dict(os.environ), patch.object(analysis_common, "HERE", fake_here):
                    os.environ.pop("DATA_ROOT", None)
                    if value is not None:
                        os.environ["DATA_ROOT"] = value
                    assert analysis_common.resolve_data_root() == fake_data
                    passed("analysis fallback for DATA_ROOT=%r from another cwd" % value)

            # Exercise verify_sealed_run's real invocation and artifact logic.
            # Only the evaluator is mocked; it writes synthetic matching metrics
            # instead of loading a model or any input CSV.
            run = fixture_directory(root, "synthetic sealed run")
            variant = fixture_directory(run, "variants/forward_direct")
            checkpoint = fixture_directory(run, "checkpoints")
            metrics = dict(zip(runner.CH, (0.25, 0.5, 0.75)))
            meta = dict(sealed=True, Scored_N_files=20, Scored_N_files_T_avg=17,
                        Val_MAE_Overall=0.5, **{"Val_MAE_" + k: v for k, v in metrics.items()})
            config = dict(sealed=True, tf32="0", tavg_mask="split", relabel_csv="",
                          latest_params={"max_epochs": 2},
                          data={"split_name": "round10_vs3_s42.json", "train_exclude_type": ""})
            (variant / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            (run / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
            (variant / "summary_errors_val.csv").write_text("synthetic fixture only\n", encoding="utf-8")
            (variant / "predictions_val.npz").write_bytes(b"synthetic fixture; never opened")
            (checkpoint / "best_gru_variant_forward_direct.constants.json").write_text("{}", encoding="utf-8")
            arm_data = fixture_directory(caller, "arm data with spaces")
            arm_physics = fixture_directory(caller, "arm physics with spaces")
            for relative in (False, True):
                arm_env = dict(TF32="0", TAVG_MASK="split", RELABEL_CSV="",
                               SPLIT_NAME="round10_vs3_s42.json", TRAIN_EXCLUDE_TYPE="", MAX_EPOCHS="2",
                               DATA_ROOT=arm_data.name if relative else str(arm_data),
                               PHYSICS_CACHE_DIR=arm_physics.name if relative else str(arm_physics))
                captured = []

                def evaluate(command, **kwargs):
                    assert command[command.index("--set") + 1] == "val", command
                    assert Path(kwargs["cwd"]) == HERE
                    assert Path(kwargs["env"]["DATA_ROOT"]) == arm_data
                    assert Path(kwargs["env"]["PHYSICS_CACHE_DIR"]) == arm_physics
                    assert kwargs["env"]["TAVG_MASK"] == "split"
                    for key in RESOURCE_KEYS:
                        assert Path(kwargs["env"][key]).is_absolute()
                    aggregate = Path(command[command.index("--out") + 1])
                    assert root in aggregate.parents, aggregate
                    aggregate.write_text(json.dumps([dict(metrics, n_files=20, n_files_T_avg=17)]), encoding="utf-8")
                    captured.append(kwargs)
                    return SimpleNamespace(returncode=0, stdout="synthetic validation evaluator")

                with patch.dict(os.environ), patch.object(tempfile, "tempdir", str(root)), \
                     patch.object(runner.subprocess, "run", side_effect=evaluate):
                    set_paths({})
                    os.environ.update(DATA_ROOT="different inherited data", PHYSICS_CACHE_DIR="different inherited cache")
                    result = runner.verify_sealed_run(str(run), arm_env)
                    assert result["ok"] and result["repro"] and result["max_delta"] == 0, result
                    assert len(captured) == 1
                    passed("smoke verification preserves arm paths over inherited paths (%s)" %
                           ("relative" if relative else "absolute"))

            # Check the public compatibility entrypoint, stopping its dry-run
            # at the config-resolution boundary before importing torch.
            for label, values, expected in modes:
                captured = []

                def inspect_plan(args):
                    assert Path(os.environ["DATA_ROOT"]) == arm_data
                    for key in RESOURCE_KEYS:
                        assert Path(os.environ[key]) == expected[key], (label, key, os.environ[key])
                        assert Path(os.environ[key]).is_absolute()
                    assert args.all and args.dry_run and args.only is None
                    captured.append(args)

                with patch.dict(os.environ), patch.dict(runner.r9.RUN_NAMES), \
                     patch.object(sys, "argv", [str(HERE / "run_round10.py"), "--dry-run"]), \
                     patch.object(pipeline, "dry_run", side_effect=inspect_plan), \
                     patch.object(runner, "launch", side_effect=AssertionError("training forbidden in path check")), \
                     contextlib.redirect_stdout(io.StringIO()):
                    set_paths(values)
                    os.environ.update(DATA_ROOT=arm_data.name, SEAL_TEST="1", SEEDS="7")
                    runner.main()
                    assert len(captured) == 1
                passed("v4 entrypoint normalizes environment before full dry-run for " + label)

    print("check_resource_paths: %d/%d PASS" % (len(checks), len(checks)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
