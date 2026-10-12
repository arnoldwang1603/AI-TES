"""Synthetic case-ID implementation migration, never training or test scoring."""
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_h9 import block_test_csv_reads
from paper_report import atomic_json, read_json, sha256
import round10_case_id_hotfix as hotfix

RESULTS = []


def check(label, passed):
    RESULTS.append(bool(passed))
    print("%s  %s" % ("PASS" if passed else "FAIL", label), flush=True)


def refuses(fn, fragment):
    try:
        fn()
    except (RuntimeError, OSError, ValueError) as exc:
        return fragment.lower() in str(exc).lower()
    return False


def snapshot(root):
    return {p.relative_to(root).as_posix(): sha256(str(p)) for p in root.rglob("*") if p.is_file()}


def fixture(root):
    root.mkdir()
    files = {name: "old\n" for name in hotfix.CHANGED}
    files.update({"training.py": "unchanged training\n", "evaluation.py": "unchanged evaluation\n",
                  "prereg/round10_v4.json": "{\"registered\": true}\n",
                  "round10_resources/training_source_hashes.json": "{\"frozen\": true}\n"})
    old_map = {}
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        old_map[name] = sha256(str(path))
    archived = root / hotfix.OLD_SEAL
    atomic_json(str(archived), dict(format="round10.package/1", files=old_map))
    old_digest = sha256(str(archived))
    current = copy.deepcopy(old_map)
    for name in hotfix.CHANGED | hotfix.ADDED:
        path = root / name
        if name != hotfix.OLD_SEAL:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("approved implementation fix\n", encoding="utf-8")
        current[name] = sha256(str(path))
    atomic_json(str(root / "prereg/round10_package_hashes.json"), dict(format="round10.package/1", files=current))
    out = root / "runs/_round10_v4_results"
    state = dict(prereg_sha256=sha256(str(root / "prereg/round10_v4.json")),
                 package_sha256=old_digest, dataset_sha256="dataset",
                 stages={"S0": dict(status="done"), "S1": dict(status="done"),
                         "S1b": dict(status="failed", reason=hotfix.FAILURE)},
                 runs=[dict(arm="synthetic", seed=7, run_dir="untouched")],
                 decisions=[], candidate=None, winners=[])
    state_path = out / "pipeline_state.json"
    atomic_json(str(state_path), state)
    (out / "S1b").mkdir()
    (out / "S1b/P1_long.csv").write_text("duplicate incomplete output\n", encoding="utf-8")
    (out / "synthetic_checkpoint.bin").write_bytes(b"unchanged checkpoint")
    return state_path, old_digest


def check_real_run_audit(parent):
    import round10_pipeline as pipeline
    root = parent / "run_audit"
    root.mkdir()
    atomic_json(str(root / "prereg/round10_v4.json"), {})
    cases = read_json(str(HERE / "splits/round10_vs3_s42.json"))["val"]
    atomic_json(str(root / "splits/round10_vs3_s42.json"), dict(val=cases))
    seeds = read_json(str(HERE / "prereg/round10_v4.json"))["seeds"]
    names = sorted(hotfix.COMPARISON_ARMS) + ["B2_relabel", "B2_drop9", "RETRAIN"]
    table = [dict(arm=name, stage="S1", env={}, run_name="r10v4_" + name,
                  seeds=seeds["S10"] if name == "RETRAIN" else seeds["S20"]) for name in names]
    records = []
    with patch.object(pipeline, "HERE", str(root)), patch.object(pipeline, "full_arm_table", return_value=table), \
            patch.object(pipeline, "resolve_names", side_effect=lambda value: value), \
            patch.object(pipeline, "verify_done", return_value=True) as done:
        for arm in table:
            for seed in arm["seeds"]:
                record = pipeline.run_record(arm, seed)
                records.append(record)
                if arm["arm"] in hotfix.COMPARISON_ARMS:
                    path = Path(record["run_dir"]) / "variants/forward_direct/summary_errors_val.csv"
                    path.parent.mkdir(parents=True)
                    with open(path, "w", newline="", encoding="utf-8") as stream:
                        writer = csv.DictWriter(stream, fieldnames=["Case", "CaseFile"])
                        writer.writeheader()
                        writer.writerows(dict(Case=case, CaseFile=Path(case).name) for case in cases)
        state = dict(runs=records)
        check("production run audit accepts 170 registered runs and 120 panels matching frozen paths", hotfix._audit_runs(root, state) == dict(s1_runs=170, comparison_runs=120, validation_cases=20))
        check("production run audit refuses a missing run", refuses(lambda: hotfix._audit_runs(root, dict(runs=records[:-1])), "incomplete"))
        check("production run audit refuses duplicated arm/seed", refuses(lambda: hotfix._audit_runs(root, dict(runs=records[:-1] + records[:1])), "duplicated"))
        moved = copy.deepcopy(records)
        moved[0]["run_dir"] = str(root.parent / "outside")
        check("production run audit refuses a substituted run directory", refuses(lambda: hotfix._audit_runs(root, dict(runs=moved)), "registered name"))
        done.return_value = False
        check("production run audit refuses incomplete checkpoint artifacts", refuses(lambda: hotfix._audit_runs(root, state), "checkpoint is missing"))
        done.return_value = True
        path = Path(records[0]["run_dir"]) / "variants/forward_direct/summary_errors_val.csv"
        original = path.read_bytes()
        text = original.decode("utf-8").replace(cases[0], "Other family/Other case.csv")
        path.write_text(text, encoding="utf-8")
        check("production run audit refuses 20 unique paths that differ from frozen split", refuses(lambda: hotfix._audit_runs(root, state), "frozen validation split"))
        path.write_text(original.decode("utf-8").replace(cases[0], cases[1]), encoding="utf-8")
        check("production run audit refuses duplicate full Case paths", refuses(lambda: hotfix._audit_runs(root, state), "missing or duplicated"))
        path.write_bytes(original)


def main():
    block_test_csv_reads()
    scratch = HERE / "scratch"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="case_id_hotfix_", dir=str(scratch)) as temp:
        check_real_run_audit(Path(temp))
        count = [0]
        def audited_runs(root, state):
            count[0] += 1
            return dict(s1_runs=170, comparison_runs=120, validation_cases=20)
        with patch.object(hotfix, "_verify_dataset", return_value="dataset"), patch.object(hotfix, "_audit_runs", side_effect=audited_runs):
            root = Path(temp) / "success"
            state_path, digest = fixture(root)
            before = snapshot(root)
            original = read_json(str(state_path))
            with patch.object(hotfix, "OLD_SEAL_SHA256", digest):
                result = hotfix.inspect(root)
                check("read-only audit changes no files and checks completed S1 runs", result["status"] == "ready" and snapshot(root) == before and count[0] == 1)
                result = hotfix.apply(root)
                after = read_json(str(state_path))
                expected = dict(original, package_sha256=sha256(str(root / "prereg/round10_package_hashes.json")))
                check("apply changes only package_sha256 and preserves all stage/run/protocol fields", result["status"] == "applied" and after == expected)
                _, archive, backup, receipt = hotfix._receipt_paths(root)
                check("original journal and failed comparison output are backed up exactly", sha256(str(backup)) == before["runs/_round10_v4_results/pipeline_state.json"] and (archive / "S1b.before/P1_long.csv").read_bytes() == (root / "runs/_round10_v4_results/S1b/P1_long.csv").read_bytes())
                check("checkpoints, old protocol, old seal and incomplete source tables remain untouched", all(snapshot(root)[name] == val for name, val in before.items() if name != "runs/_round10_v4_results/pipeline_state.json"))
                settled = snapshot(root)
                check("reapplying hotfix is idempotent and changes no files", hotfix.apply(root)["status"] == "already_applied" and snapshot(root) == settled)
                after["stages"]["S2"] = dict(status="done")
                atomic_json(str(state_path), after)
                settled = snapshot(root)
                check("idempotent call preserves progress made after recovery", hotfix.apply(root)["status"] == "already_applied" and snapshot(root) == settled)
                receipt_before = sha256(str(receipt))
                atomic_json(str(state_path), original)
                check("receipt-before-state interrupted write recovers safely", hotfix.apply(root)["status"] == "applied" and sha256(str(receipt)) == receipt_before and read_json(str(state_path)) == expected)
                import round10_pipeline as pipeline
                pipe = pipeline.Pipeline.__new__(pipeline.Pipeline)
                pipe.args = SimpleNamespace(stage=None, until="S1b")
                pipe.out = str(state_path.parent)
                pipe.state_path = str(state_path)
                pipe.state = read_json(str(state_path))
                pipe.records = pipe.state["runs"]
                calls = []
                pipe.run_stage = calls.append
                with patch.object(pipeline, "launch_arm", side_effect=AssertionError("must not train")):
                    pipe.run()
                resumed = read_json(str(state_path))
                check("production resume skips S0/S1 and enters S1b without retraining", calls == ["S1b"] and resumed["stages"]["S1b"]["status"] == "done" and all(resumed["stages"][s] == original["stages"][s] for s in ("S0", "S1")))
            def reject_case(name, mutate, expected_fragment):
                case = Path(temp) / name
                state_file, old_digest = fixture(case)
                mutate(case, state_file)
                initial = snapshot(case)
                with patch.object(hotfix, "OLD_SEAL_SHA256", old_digest):
                    rejected = refuses(lambda: hotfix.apply(case), expected_fragment)
                check(name + " is refused without changing artifacts", rejected and snapshot(case) == initial)
            def state_edit(path, **changes):
                data = read_json(str(path))
                data.update(changes)
                atomic_json(str(path), data)
            reject_case("unknown_predecessor", lambda r, p: state_edit(p, package_sha256="other"), "not the archived")
            reject_case("changed_prereg_identity", lambda r, p: state_edit(p, prereg_sha256="other"), "Preregistration changed")
            reject_case("changed_dataset_identity", lambda r, p: state_edit(p, dataset_sha256="other"), "Training source seal changed")
            reject_case("later_stage", lambda r, p: state_edit(p, stages=dict(read_json(str(p))["stages"], S2=dict(status="running"))), "later stages")
            reject_case("incomplete_s1", lambda r, p: state_edit(p, stages=dict(read_json(str(p))["stages"], S1=dict(status="failed"))), "must both be complete")
            reject_case("different_failure", lambda r, p: state_edit(p, stages=dict(read_json(str(p))["stages"], S1b=dict(status="failed", reason="other"))), "specific duplicate-index")
            reject_case("already_decided", lambda r, p: state_edit(p, decisions=[dict(stage="S1b")]), "decisions exist")
            reject_case("test_reserved", lambda r, p: atomic_json(str(r / "prereg/test_access_state.json"), {"B6": {"status": "reserved"}}), "Test access state")
            reject_case("test_ledger", lambda r, p: (r / "prereg/test_set_access.log").write_text("reserved\n"), "ledger is nonempty")
            reject_case("test_source_seal", lambda r, p: atomic_json(str(r / "prereg/test_source_seal.json"), {}), "Test access evidence")
            reject_case("b6_directory", lambda r, p: (r / "runs/_round10_v4_results/S1b/B6").mkdir(), "B6 output directory")
            reject_case("changed_training_bytes", lambda r, p: (r / "training.py").write_text("edited\n"), "Current package input changed")
            def changed_seal(root, state_path):
                path = root / "training.py"
                path.write_text("edited and resealed\n")
                seal_path = root / "prereg/round10_package_hashes.json"
                seal = read_json(str(seal_path))
                seal["files"]["training.py"] = sha256(str(path))
                atomic_json(str(seal_path), seal)
            reject_case("unapproved_reseal", changed_seal, "Package delta differs")
            reject_case("archived_seal_changed", lambda r, p: (r / hotfix.OLD_SEAL).write_text("{}"), "approved predecessor")
            reject_case("live_pipeline", lambda r, p: (r / "runs/_round10_v4_results/pipeline.lock").write_text(str(os.getpid())), "still running")
            # Linux state can identify the LF checkout of the pinned CRLF archive.
            lf_root = Path(temp) / "lf_state"
            state_file, digest = fixture(lf_root)
            lf_digest = hashlib.sha256((lf_root / hotfix.OLD_SEAL).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            state_edit(state_file, package_sha256=lf_digest)
            with patch.object(hotfix, "OLD_SEAL_SHA256", digest):
                check("approved LF/CRLF predecessor is accepted without JSON normalization", hotfix.inspect(lf_root)["status"] == "ready")
    print("check_case_id_hotfix: %d/%d PASS" % (sum(RESULTS), len(RESULTS)))
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
