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


def main():
    block_test_csv_reads()
    p = pipeline.protocol()
    table = pipeline.full_arm_table(p, winners=[])
    unconditional = [a for a in table if not a["condition"]]
    counts = {stage: sum(len(a["seeds"]) for a in unconditional if a["stage"] == stage)
              for stage in pipeline.STAGES}
    check("registered unconditional graph has 420 runs with P1=90 and F4-direct=20",
          sum(counts.values()) == 420 and counts["S1"] == 90 and counts["S7"] == 70)
    check("only the two existing G7 treatments are conditional in the core table",
          {a["arm"] for a in table if a["condition"]} == {"G7c", "G7d"})
    check("all ten F1 folds use the specified five seeds and frozen full-train constants",
          len([a for a in table if a["arm"].startswith("F1_")]) == 10
          and all(a["seeds"] == [1, 2, 3, 5, 7] and a["env"]["FROZEN_CONSTANTS"]
                  for a in table if a["arm"].startswith("F1_")))
    challenged = pipeline.full_arm_table(p, candidate="R3", winners=["R3", "NO_TIME"])
    check("challenger branch adds paired F4 arms and exactly the optional fresh F10 control",
          {a["arm"] for a in challenged if a["stage"] == "S8"} == {"F9", "F10"}
          and sum(len(a["seeds"]) for a in challenged if a["arm"].startswith("F4_")) == 40
          and {a["arm"] for a in challenged if a["stage"] == "S6"} == {"C17_R3", "C17_NO_TIME"})
    check("every planned training arm seals test access and pins fresh F9 primes",
          all(a["env"]["SEAL_TEST"] == "1" for a in challenged)
          and next(a["seeds"] for a in challenged if a["arm"] == "F9") == p["seeds"]["F9_fresh_primes"])
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
    original_here = pipeline.HERE
    original_table = pipeline.full_arm_table
    original_p4 = decisions.p4_adoption
    original_dataset = pipeline.verify_dataset_hashes
    with tempfile.TemporaryDirectory(prefix="pipeline_", dir=parent) as tmp:
        package = os.path.join(tmp, "prereg")
        os.makedirs(package)
        prereg_path = os.path.join(package, "round10_v3.json")
        atomic_json(prereg_path, p)
        package_path = os.path.join(package, "round10_package_hashes.json")
        atomic_json(package_path, {"files": {"prereg/round10_v3.json": sha256(prereg_path)}})
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

            # Test S4 control flow with synthetic predicates and record capture.
            core = [dict(arm="G7a", stage="S4", condition=None),
                    dict(arm="ENC_linear", stage="S4", condition=None)]
            conditional = [dict(arm="G7c", stage="S4", condition="no core passes"),
                           dict(arm="G7d", stage="S4", condition="no core passes")]
            pipeline.full_arm_table = lambda *_args, **_kw: core + conditional
            launched = []
            later.add_runs = lambda arms, stage: launched.extend(a["arm"] for a in arms)
            later.decide = lambda *_args, **_kw: None
            decisions.p4_adoption = lambda *_args, **_kw: (True, {"startup_pass": False, "encoder_pass": True})
            later.run_stage("S4")
            check("an equivalent small encoder alone does not suppress conditional startup repair",
                  launched == ["G7a", "ENC_linear", "G7c", "G7d"])
            launched[:] = []
            decisions.p4_adoption = lambda _records, name, *_args, **_kw: (True, {"startup_pass": name == "G7a"})
            later.run_stage("S4")
            check("a core startup treatment that passes suppresses both conditional G7 arms",
                  launched == ["G7a", "ENC_linear"])

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
            atomic_json(package_path, {"files": {"prereg/round10_v3.json": sha256(prereg_path)}, "changed": True})
            check("changed package identity cannot resume old stages even with the same prereg",
                  rejects(lambda: pipeline.Pipeline(args), "cannot resume"))
        finally:
            pipeline.HERE = original_here
            pipeline.full_arm_table = original_table
            decisions.p4_adoption = original_p4
            pipeline.verify_dataset_hashes = original_dataset
    print("\ncheck_pipeline: %d/%d PASS" % (sum(RESULTS), len(RESULTS)))
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
