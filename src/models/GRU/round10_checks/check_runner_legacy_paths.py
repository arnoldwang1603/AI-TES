"""CPU fixtures exercise the retained P1-only preflight and gate functions.

No models, GPU kernels or real training/test CSVs are opened.
"""
import contextlib
import io
import os
import sys
import tempfile
import types
from unittest.mock import patch

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(HERE)
from check_h9 import block_test_csv_reads
from paper_report import atomic_json, read_json
import run_round10 as runner
import round10_validation_gate as gate

RESULTS = []


def check(label, ok):
    RESULTS.append(bool(ok))
    print("%s  %s" % ("PASS" if ok else "FAIL", label), flush=True)


def main():
    block_test_csv_reads()
    parent = os.path.join(HERE, "runs", "_round10_check_scratch")
    os.makedirs(parent, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="runner_paths_", dir=parent) as tmp:
        split = os.path.join(tmp, "split.json")
        atomic_json(split, {"train": ["t%d" % i for i in range(284)], "val": ["v%d" % i for i in range(20)]})
        frame = pd.DataFrame({"Input Temperature (C)": [50.0, 60.0, 50.0]})
        tr = [(frame, "train%d" % i) for i in range(284)]
        va = [(frame, "val%d" % i) for i in range(20)]
        calls = []
        fake_config = types.ModuleType("tes_gru.config")
        fake_config.VAL_SPLIT = "stratified"
        fake_config.SEAL_TEST = True
        fake_config.split_file_path = lambda: split
        fake_data = types.ModuleType("tes_gru.data")
        def fake_load(include_test=None):
            if include_test is not False:
                raise AssertionError("compatibility preflight attempted test loading")
            calls.append(include_test)
            return tr, va, [], object()
        fake_data.load_all_data = fake_load
        fake_data.input_case_flag_np = lambda values: True
        class FakeDataset:
            def __init__(self, *_args, **_kwargs):
                self.X = np.zeros((1, 3, 2))
        fake_data.ThermalDataset = FakeDataset
        fake_rollout = types.ModuleType("tes_gru.rollout")
        fake_rollout._input_case_gate = lambda _inputs: np.array([True])
        package = types.ModuleType("tes_gru")
        fake_modules = {"tes_gru": package, "tes_gru.config": fake_config,
                        "tes_gru.data": fake_data, "tes_gru.rollout": fake_rollout}
        captured = {}
        def synthetic_child(code, env, label):
            captured.update(code=code, env=env, label=label)
            buffer = io.StringIO()
            with patch.dict(sys.modules, fake_modules), contextlib.redirect_stdout(buffer):
                exec(compile(code, "synthetic_runner_preflight", "exec"), {})
            captured["output"] = buffer.getvalue()
        with patch.object(runner.r9, "_run_check", synthetic_child):
            runner.preflight_child()
        check("P1-only preflight explicitly excludes test loading and accepts284/20/0",
              calls == [False] and "CHECK_OK" in captured["output"] and "test excluded" in captured["output"])
        check("P1-only preflight pins SEAL_TEST=1 and checks both train/validation detectors",
              captured["env"]["SEAL_TEST"] == "1" and "('train', 284, 284)" in captured["output"]
              and "('val', 20, 20)" in captured["output"])

        # The public helper must never revive its historical --set test child.
        with patch.object(runner, "HERE", tmp), patch.object(gate, "run_gate", return_value={"ok": True, "set": "validation"}), \
                patch.object(runner.subprocess, "run", side_effect=AssertionError("unexpected scoring subprocess")):
            with contextlib.redirect_stdout(io.StringIO()):
                passed = runner.regression_gate()
            result = read_json(os.path.join(tmp, "runs", "_round10_v4_results", "S0", "regression.json"))
            check("legacy callable gate delegates to validation and records its result", passed and result["set"] == "validation")
        with patch.object(runner, "HERE", tmp), patch.object(gate, "run_gate", side_effect=FileNotFoundError("missing champion")):
            failed = False
            try:
                runner.regression_gate()
            except FileNotFoundError:
                failed = True
            check("missing champion fails instead of being treated as a skipped success", failed)
        with patch.object(runner, "HERE", tmp), patch.object(gate, "run_gate", return_value={"ok": False}):
            failed = False
            try:
                runner.regression_gate()
            except RuntimeError:
                failed = True
            check("failed numerical validation gate cannot return success", failed)
    print("\ncheck_runner_legacy_paths: %d/%d PASS" % (sum(RESULTS), len(RESULTS)))
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
