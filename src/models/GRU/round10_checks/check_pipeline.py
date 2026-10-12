"""Synthetic stage failure/resume and immutable-input pipeline checks.

All training, scoring and preflight actions are replaced with local stubs.
The actual stage journal and registered arm table are exercised without GPU.
"""
import contextlib
import io
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(HERE)
from check_h9 import block_test_csv_reads
from paper_report import atomic_json, read_json, sha256
import round10_pipeline as pipeline
import round10_decisions as decisions
import round10_migration as migration

RESULTS = []


def check(label, ok):
    RESULTS.append(bool(ok))
    print("%s  %s" % ("PASS" if ok else "FAIL", label), flush=True)


def rejects(fn, fragment):
    try:
        fn()
    except RuntimeError as exc:
        return fragment.lower() in str(exc).lower()
    return False


def check_migration(parent, prereg):
    """Exercise the real migration boundary on disposable local journals."""
    with tempfile.TemporaryDirectory(prefix="migration_", dir=parent) as tmp:
        def fixture(name, old_state=True):
            root = os.path.join(tmp, name)
            private = os.path.join(root, "prereg")
            os.makedirs(private)
            atomic_json(os.path.join(private, "round10_v4.json"), prereg)
            atomic_json(os.path.join(private, "round10_v3.json"), {"historical": True})
            old_path = os.path.join(root, "runs", "_round10_results", "pipeline_state.json")
            artifact = os.path.join(root, "runs", "r10_old_seed7", "checkpoint.pth")
            os.makedirs(os.path.dirname(artifact))
            with open(artifact, "wb") as stream:
                stream.write(b"preserved historical checkpoint; never loaded")
            if old_state:
                atomic_json(old_path, {"stages": {"S0": {"status": "done"}, "S1": {"status": "failed"}},
                    "runs": [{"arm": "B1_mask", "seed": 7, "run_dir": os.path.dirname(artifact)}]})
            return root, private, old_path, artifact

        root, private, old_path, artifact = fixture("early_v3")
        check("old v3 state requires an explicit restart audit before v4 begins",
              rejects(lambda: migration.ensure_v4_restart(root), "--restart-v4"))
        old_hashes = {path: sha256(path) for path in (old_path, artifact, os.path.join(private, "round10_v3.json"))}
        receipt = migration.restart_v4(root)
        receipt_path = os.path.join(root, "runs", "_round10_v4_results", "migration_v3_audit.json")
        receipt_hash = sha256(receipt_path)
        check("migration preserves the old state, checkpoint and prereg without reusing or resetting them",
              all(sha256(path) == digest for path, digest in old_hashes.items())
              and receipt["old_runs_reused"] is False and receipt["old_files_deleted"] is False
              and receipt["test_access_reset"] is False
              and receipt["snapshot"]["preserved_old_runs"][0]["seed"] == 7)
        check("repeated restart and ensure are idempotent and leave the receipt unchanged",
              migration.restart_v4(root) == receipt and migration.ensure_v4_restart(root) == receipt
              and sha256(receipt_path) == receipt_hash)
        with open(old_path, "rb") as stream:
            original_old = stream.read()
        with open(old_path, "ab") as stream:
            stream.write(b"\n")
        check("a changed old journal after the audit is refused",
              rejects(lambda: migration.restart_v4(root), "changed after migration audit"))
        with open(old_path, "wb") as stream:
            stream.write(original_old)
        active = os.path.join(private, "round10_v4.json")
        with open(active, "rb") as stream:
            original_active = stream.read()
        with open(active, "ab") as stream:
            stream.write(b"\n")
        check("a changed v4 prereg after the audit is refused",
              rejects(lambda: migration.restart_v4(root), "changed after migration audit"))
        with open(active, "wb") as stream:
            stream.write(original_active)

        for status in ("reserved", "completed"):
            bad_root, bad_private, _, _ = fixture("test_" + status)
            state = os.path.join(bad_private, "test_access_state.json")
            atomic_json(state, {"windows": {"B6": {"status": status}}})
            before = sha256(state)
            check("migration refuses a %s real-test window and retains its ledger" % status,
                  rejects(lambda: migration.restart_v4(bad_root), "Prior test access evidence")
                  and sha256(state) == before)
        for filename in ("test_source_seal.json", "test_access.lock", "test_set_access.log"):
            bad_root, bad_private, _, _ = fixture(filename.replace(".", "_"))
            evidence = os.path.join(bad_private, filename)
            with open(evidence, "wb") as stream:
                stream.write(b"historical test access evidence")
            before = sha256(evidence)
            check("migration refuses standalone %s evidence without deleting it" % filename,
                  rejects(lambda: migration.restart_v4(bad_root), "Prior test access evidence")
                  and sha256(evidence) == before)
        bad_root, _, bad_old, _ = fixture("old_running")
        lock = os.path.join(os.path.dirname(bad_old), "pipeline.lock")
        with open(lock, "w") as stream:
            stream.write(str(os.getpid()))
        check("migration refuses a live old pipeline PID",
              rejects(lambda: migration.restart_v4(bad_root), "still running") and os.path.isfile(lock))
        bad_root, _, bad_old, _ = fixture("b6_boundary")
        atomic_json(bad_old, {"stages": {"S1b": {"status": "failed"}}, "runs": []})
        check("an old stage at the B6 boundary is refused even without a ledger",
              rejects(lambda: migration.restart_v4(bad_root), "B6 boundary"))

        fresh_root, _, _, _ = fixture("fresh", old_state=False)
        fresh = migration.ensure_v4_restart(fresh_root)
        check("a fresh installation records an audit with no old state",
              fresh["snapshot"]["old_state_sha256"] is None)
        v4_state = os.path.join(root, "runs", "_round10_v4_results", "pipeline_state.json")
        atomic_json(v4_state, {"prereg_sha256": sha256(active), "stages": {"S1": {"status": "done"}}})
        current_ledger = os.path.join(private, "test_access_state.json")
        atomic_json(current_ledger, {"windows": {"B6": {"status": "reserved", "protocol": "v4"}}})
        before_state, before_ledger = sha256(v4_state), sha256(current_ledger)
        resumed = migration.restart_v4(root)
        check("restart after v4 initialization preserves its own state and test reservation instead of resetting",
              resumed.get("already_initialized") is True and sha256(v4_state) == before_state
              and sha256(current_ledger) == before_ledger and sha256(receipt_path) == receipt_hash)
        missing_root, missing_private, _, _ = fixture("v4_without_receipt", old_state=False)
        atomic_json(os.path.join(missing_root, "runs", "_round10_v4_results", "pipeline_state.json"),
                    {"prereg_sha256": sha256(os.path.join(missing_private, "round10_v4.json"))})
        check("an existing v4 state without an audit cannot be silently adopted",
              rejects(lambda: migration.restart_v4(missing_root), "no migration audit"))


def check_compatibility_entrypoint(parent):
    """The old filename must launch the same complete v4 protocol."""
    import run_round10 as legacy
    import round10_validation_gate as gate
    original_argv = sys.argv
    original_main = pipeline.main
    original_gate_main = gate.main
    original_gate_run = gate.run_gate
    original_transport = legacy.transport_env
    original_legacy_here = legacy.HERE
    calls = []
    try:
        legacy.transport_env = lambda: {}
        pipeline.main = lambda: calls.append(("pipeline", list(sys.argv[1:])))
        gate.main = lambda: calls.append(("gate", list(sys.argv[1:])))
        for supplied, expected in (([], ["--all"]),
                                   (["--dry-run"], ["--all", "--dry-run"]),
                                   (["--parallel", "3"], ["--all", "--parallel", "3"]),
                                   (["--all", "--parallel", "3"], ["--all", "--parallel", "3"]),
                                   (["--preflight"], ["--preflight"]),
                                   (["--smoke"], ["--smoke"]),
                                   (["--restart-v4"], ["--restart-v4"])):
            sys.argv = ["run_round10.py"] + supplied
            legacy.main()
            check("compatibility entry %s forwards to the complete v4 pipeline" % (" ".join(supplied) or "(bare)"),
                  calls[-1] == ("pipeline", expected))
        with tempfile.TemporaryDirectory(prefix="compatibility_", dir=parent) as tmp:
            legacy.HERE = tmp
            expected = os.path.join(tmp, "runs", "_round10_v4_results", "S0", "regression.json")
            sys.argv = ["run_round10.py", "--gate"]
            legacy.main()
            check("compatibility --gate targets the v4 validation-gate output path",
                  calls[-1] == ("gate", ["--out", expected]))
            gate.run_gate = lambda: {"ok": True, "synthetic": True, "test_access": False}
            with contextlib.redirect_stdout(io.StringIO()):
                ok = legacy.regression_gate()
            check("legacy regression helper also writes only the v4 gate path",
                  ok and read_json(expected)["synthetic"] is True
                  and not os.path.exists(os.path.join(tmp, "runs", "_round10_results")))
    finally:
        sys.argv = original_argv
        pipeline.main = original_main
        gate.main = original_gate_main
        gate.run_gate = original_gate_run
        legacy.transport_env = original_transport
        legacy.HERE = original_legacy_here


def main():
    block_test_csv_reads()
    p = pipeline.protocol()
    table = pipeline.full_arm_table(p, winners=[])
    unconditional = [a for a in table if not a["condition"]]
    counts = {stage: sum(len(a["seeds"]) for a in unconditional if a["stage"] == stage)
              for stage in pipeline.STAGES}
    expected_counts = dict(S1=170, S2=160, S3=120, S4=180, S5=60,
                           S6=0, S7=20, S7b=100, S8=20)
    check("v4 fixed graph has 830 runs with the registered per-stage counts",
          sum(counts.values()) == 830
          and all(counts[stage] == value for stage, value in expected_counts.items()))
    science = [a for a in table if a["stage"] in ("S1", "S2", "S3", "S4", "S5") and a["arm"] != "RETRAIN"]
    check("all 34 scientific arms use the same 20 seeds; only RETRAIN keeps ten",
          len(science) == 34 and all(a["seeds"] == p["seeds"]["S20"] for a in science)
          and next(a["seeds"] for a in table if a["arm"] == "RETRAIN") == p["seeds"]["S10"])
    check("G7c and G7d are fixed work, with no conditional base arm",
          not any(a["condition"] for a in table)
          and {a["arm"] for a in table if a["stage"] == "S4"} ==
          {"G7a", "G7b", "G7c", "G7d", "ENC_linear", "ENC_16_16", "H0_tanh", "H0_zero", "NO_TIME"})
    check("all ten F1 folds use ten seeds, h128 and fold fitting after F4",
          len([a for a in table if a["arm"].startswith("F1_")]) == 10
          and all(a["seeds"] == p["seeds"]["S10"] and a["stage"] == "S7b"
                  and a["env"]["HIDDEN_SIZE"] == "128" and not a["env"]["FROZEN_CONSTANTS"]
                  and not a["env"].get("CONSTANTS_FROM")
                  and a["env"]["SPLIT_NAME"] == "loto_v4_fold%s.json" % a["arm"].split("_")[-1]
                  for a in table if a["arm"].startswith("F1_"))
          and pipeline.STAGES.index("S7") < pipeline.STAGES.index("S7b") < pipeline.STAGES.index("S8"))
    challenged = pipeline.full_arm_table(p, candidate="R3", winners=["R3", "NO_TIME"])
    check("challenger branch adds paired F4 arms and exactly the optional fresh F10 control",
          {a["arm"] for a in challenged if a["stage"] == "S8"} == {"F9", "F10"}
          and sum(len(a["seeds"]) for a in challenged if a["arm"].startswith("F4_")) == 40
          and all(a["seeds"] == p["seeds"]["S10"] for a in challenged if a["stage"] == "S7")
          and {a["arm"] for a in challenged if a["stage"] == "S6"} == {"C17_R3", "C17_NO_TIME"}
          and all(a["seeds"] == p["seeds"]["S20"] for a in challenged if a["stage"] == "S6"))
    check("challenger F1 covers all ten families with the same seeds and capacity as direct",
          {a["arm"] for a in challenged if a["stage"] == "S7b"} ==
          {"F1_%d" % fold for fold in range(1, 11)} | {"F1_R3_%d" % fold for fold in range(1, 11)}
          and all(a["seeds"] == p["seeds"]["S10"] and a["env"]["HIDDEN_SIZE"] == "128"
                  and not a["env"]["FROZEN_CONSTANTS"]
                  for a in challenged if a["stage"] == "S7b"))
    fallback = pipeline.full_arm_table(p, candidate="R3", winners=["R3"], frozen_candidate="direct")
    check("F4 rejection returns F1 and F9 to direct without dropping the completed F4 comparison",
          {a["arm"] for a in fallback if a["stage"] == "S7b"} == {"F1_%d" % fold for fold in range(1, 11)}
          and {a["arm"] for a in fallback if a["stage"] == "S8"} == {"F9"}
          and next(a["env"] for a in fallback if a["arm"] == "F9")["OTHER_CH_MODE"] == "abs"
          and len([a for a in fallback if a["stage"] == "S7"]) == 4)
    all_followups = pipeline.full_arm_table(p, candidate="R3",
        winners=["H8_a6_l0", "R3", "C11", "ENC_linear", "G7b", "NO_TIME"])
    check("all permitted follow-ups total 1090 runs, without full arm/capacity/fold multiplication",
          sum(len(a["seeds"]) for a in all_followups) == 1090)
    check("every planned training arm seals test access and pins fresh F9 primes",
          all(a["env"]["SEAL_TEST"] == "1" and a["env"]["RUN_PREFIX"] == "r10v4" for a in challenged)
          and next(a["seeds"] for a in challenged if a["arm"] == "F9") == p["seeds"]["F9_fresh_primes"]
          and next(a["seeds"] for a in challenged if a["arm"] == "F10") == p["seeds"]["F9_fresh_primes"]
          and not set(p["seeds"]["S20"]) & set(p["seeds"]["F9_fresh_primes"]))
    prior_constants = os.environ.get("CONSTANTS_FROM")
    os.environ["CONSTANTS_FROM"] = "unexpected_research_shell_sidecar"
    try:
        isolated = pipeline.isolated_env(pipeline.base_env())
        check("legacy frozen-constants env alias cannot contaminate any pipeline arm",
              "CONSTANTS_FROM" not in isolated)
    finally:
        if prior_constants is None:
            os.environ.pop("CONSTANTS_FROM", None)
        else:
            os.environ["CONSTANTS_FROM"] = prior_constants

    parent = os.path.join(HERE, "runs", "_round10_check_scratch")
    os.makedirs(parent, exist_ok=True)
    check_migration(parent, p)
    check_compatibility_entrypoint(parent)
    original_here = pipeline.HERE
    original_table = pipeline.full_arm_table
    original_p4 = decisions.p4_adoption
    original_freeze = decisions.freeze_recipe
    original_descriptive = decisions.descriptive_tables
    original_command = pipeline.command
    original_dataset = pipeline.verify_dataset_hashes
    with tempfile.TemporaryDirectory(prefix="pipeline_", dir=parent) as tmp:
        package = os.path.join(tmp, "prereg")
        os.makedirs(package)
        prereg_path = os.path.join(package, "round10_v4.json")
        atomic_json(prereg_path, p)
        shutil.copy2(os.path.join(original_here, "prereg", "round10_execution_v4.json"), package)
        package_path = os.path.join(package, "round10_package_hashes.json")
        atomic_json(package_path, {"files": {"prereg/round10_v4.json": sha256(prereg_path)}})
        pipeline.HERE = tmp
        pipeline.full_arm_table = lambda *_args, **_kw: []
        dataset_identity = ["synthetic_training_source_A"]
        pipeline.verify_dataset_hashes = lambda: dataset_identity[0]
        args = SimpleNamespace(stage=None, until="S2", parallel=1)
        calls = []
        failed_once = [False]
        try:
            import eval_checkpoint as ec
            done_run = os.path.join(tmp, "synthetic_done_run")
            variant_dir = os.path.join(done_run, "variants", pipeline.VARIANT)
            checkpoint_dir = os.path.join(done_run, "checkpoints")
            os.makedirs(variant_dir)
            os.makedirs(checkpoint_dir)
            for filename in ("done.flag", "summary_errors_val.csv", "predictions_val.npz"):
                with open(os.path.join(variant_dir, filename), "wb") as stream:
                    stream.write(b"synthetic existence fixture")
            for filename in ("best_gru_variant_forward_direct.pth", "best_gru_variant_forward_direct.constants.json"):
                with open(os.path.join(checkpoint_dir, filename), "wb") as stream:
                    stream.write(b"synthetic existence fixture")
            atomic_json(os.path.join(done_run, "run_config.json"), {"sealed": True, "seed": 7})
            atomic_json(os.path.join(variant_dir, "meta.json"), {"sealed": True})
            original_env_from_config = ec.env_from_config
            ec.env_from_config = lambda _config: {"POS_TEMP_GATE": "0.0"}
            try:
                record = dict(arm="synthetic", seed=7, run_dir=done_run)
                check("resume accepts identical numeric env semantics 0.0 versus0",
                      pipeline.verify_done(record, {"POS_TEMP_GATE": "0"}))
                check("resume refuses a genuinely changed numeric model knob",
                      rejects(lambda: pipeline.verify_done(record, {"POS_TEMP_GATE": "1"}), "config mismatch"))
                check("resume refuses a checkpoint seed that disagrees with its manifest",
                      rejects(lambda: pipeline.verify_done(dict(record, seed=21), {"POS_TEMP_GATE": "0"}), "seed disagrees"))
            finally:
                ec.env_from_config = original_env_from_config
            def mocked_stage(instance, stage):
                calls.append(stage)
                if stage == "S1b" and not failed_once[0]:
                    failed_once[0] = True
                    raise RuntimeError("synthetic S1b failure")
                if stage == "S1":
                    instance.records.append(dict(arm="synthetic", seed=7, run_dir="synthetic"))
                    instance.save()
            first = pipeline.Pipeline(args)
            first.run_stage = lambda stage: mocked_stage(first, stage)
            with contextlib.redirect_stdout(io.StringIO()):
                stopped = rejects(first.run, "synthetic S1b failure")
            saved = read_json(first.state_path)
            check("a failing stage is durably failed and later stages never launch",
                  stopped and calls == ["S0", "S1", "S1b"]
                  and saved["stages"]["S1b"]["status"] == "failed" and "S2" not in saved["stages"])
            second = pipeline.Pipeline(args)
            second.run_stage = lambda stage: mocked_stage(second, stage)
            with contextlib.redirect_stdout(io.StringIO()):
                second.run()
            check("same-command resume skips completed stages and continues at the failed stage",
                  calls == ["S0", "S1", "S1b", "S1b", "S2"]
                  and second.state["stages"]["S2"]["status"] == "done" and len(second.records) == 1)
            later = pipeline.Pipeline(SimpleNamespace(stage="S4", until=None, parallel=1))
            check("manual later-stage entry refuses missing dependencies", rejects(later.run, "requires completed stages"))

            # P4 must finish all fixed work regardless of earlier adoption signals.
            core = [dict(arm="G7a", stage="S4", condition=None),
                    dict(arm="ENC_linear", stage="S4", condition=None)]
            repairs = [dict(arm="G7c", stage="S4", condition=None),
                       dict(arm="G7d", stage="S4", condition=None)]
            pipeline.full_arm_table = lambda *_args, **_kw: core + repairs
            launched = []
            later.add_runs = lambda arms, stage: launched.extend(a["arm"] for a in arms)
            later.decide = lambda *_args, **_kw: None
            decisions.descriptive_tables = lambda *_args, **_kw: None
            decisions.p4_adoption = lambda *_args, **_kw: (True, {"startup_pass": False, "encoder_pass": True})
            later.run_stage("S4")
            check("an equivalent small encoder does not suppress fixed startup treatments",
                  launched == ["G7a", "ENC_linear", "G7c", "G7d"])
            launched[:] = []
            decisions.p4_adoption = lambda _records, name, *_args, **_kw: (True, {"startup_pass": name == "G7a"})
            later.run_stage("S4")
            check("even a passing core startup treatment cannot skip G7c or G7d",
                  launched == ["G7a", "ENC_linear", "G7c", "G7d"])

            # Real S7b/S8 orchestration, with scoring/training replaced by stubs.
            pipeline.full_arm_table = original_table
            sequence = []
            later.state["candidate"] = "R3"
            later.state.pop("frozen_recipe", None)
            decisions.freeze_recipe = lambda *_args, **_kw: (sequence.append("freeze") or {"arm": "direct"})
            later.manifest = lambda *_args, **_kw: (os.path.join(tmp, "synthetic_manifest.json"), {})
            pipeline.command = lambda *_args, **_kw: sequence.append("analysis")
            def capture_runs(arms, stage):
                selected = [a for a in arms if a["stage"] == stage]
                sequence.append((stage, later.state.get("frozen_recipe"), [a["arm"] for a in selected]))
                return selected
            later.add_runs = capture_runs
            later.run_stage("S7b")
            check("S7b freezes from F4 before any F1 training and plans the frozen direct fallback",
                  sequence[0] == "freeze" and sequence[1][0] == "S7b"
                  and sequence[1][1] == {"arm": "direct"}
                  and set(sequence[1][2]) == {"F1_%d" % fold for fold in range(1, 11)})
            decisions.freeze_recipe = lambda *_args, **_kw: (_ for _ in ()).throw(RuntimeError("selection must not reopen"))
            later.run_stage("S7b")
            later.run_stage("S8")
            check("S7b resume and S8 consume the frozen recipe without reselecting from F1",
                  sequence.count("freeze") == 1 and sequence[-1] == ("S8", {"arm": "direct"}, ["F9"]))
            frozen = later.state.pop("frozen_recipe")
            check("S8 refuses launch without the pre-F1 frozen recipe",
                  rejects(lambda: later.run_stage("S8"), "frozen before F1"))
            later.state["frozen_recipe"] = frozen

            dataset_identity[0] = "synthetic_training_source_B"
            check("a changed dataset seal cannot mix resumed stage results",
                  rejects(lambda: pipeline.Pipeline(args), "cannot combine stages"))
            dataset_identity[0] = "synthetic_training_source_A"

            # Immutable identity checks execute even after S0 was completed.
            with open(prereg_path, "rb") as f:
                original_prereg = f.read()
            with open(prereg_path, "ab") as f:
                f.write(b"\n ")
            check("changed registered package input cannot reuse completed stage markers",
                  rejects(lambda: pipeline.Pipeline(args), "changed"))
            with open(prereg_path, "wb") as f:
                f.write(original_prereg)
            atomic_json(package_path, {"files": {"prereg/round10_v4.json": sha256(prereg_path)}, "changed": True})
            check("changed package identity cannot resume old stages even with the same prereg",
                  rejects(lambda: pipeline.Pipeline(args), "cannot resume"))
        finally:
            pipeline.HERE = original_here
            pipeline.full_arm_table = original_table
            decisions.p4_adoption = original_p4
            decisions.freeze_recipe = original_freeze
            decisions.descriptive_tables = original_descriptive
            pipeline.command = original_command
            pipeline.verify_dataset_hashes = original_dataset
    print("\ncheck_pipeline: %d/%d PASS" % (sum(RESULTS), len(RESULTS)))
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
