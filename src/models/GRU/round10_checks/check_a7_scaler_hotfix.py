"""Synthetic post-B6 A7 recovery checks; no real training/test data are opened."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_report as report
import round10_a7_scaler_hotfix as hotfix
import round10_pipeline as pipeline
from check_h9 import block_test_csv_reads

RESULTS = []


def check(label, passed):
    RESULTS.append(bool(passed))
    print("%s  %s" % ("PASS" if passed else "FAIL", label), flush=True)


def refuses(fn, fragment):
    try:
        fn()
    except (RuntimeError, OSError, ValueError, KeyError) as exc:
        return fragment.lower() in str(exc).lower()
    return False


def snapshot(root):
    return {p.relative_to(root).as_posix(): report.sha256(str(p)) for p in root.rglob("*") if p.is_file()}


def fixture(root):
    root.mkdir()
    private = root / "prereg"
    private.mkdir()
    protocol = report.read_json(str(HERE / "prereg/round10_v4.json"))
    report.atomic_json(str(private / "round10_v4.json"), protocol)
    files = {name: "old code for " + name + "\n" for name in hotfix.CHANGED}
    files["unchanged.py"] = "unchanged\n"
    for name, content in files.items():
        (root / name).write_text(content, encoding="utf-8")
    old_map = {name: report.sha256(str(root / name)) for name in files}
    old_map["prereg/round10_v4.json"] = report.sha256(str(private / "round10_v4.json"))
    report.atomic_json(str(root / hotfix.OLD_SEAL), dict(format="round10.package/1", files=old_map))
    old_sha = report.sha256(str(root / hotfix.OLD_SEAL))
    new_map = copy.deepcopy(old_map)
    for name in hotfix.CHANGED | hotfix.ADDED:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name != hotfix.OLD_SEAL:
            path.write_text("reviewed A7 recovery\n", encoding="utf-8")
        new_map[name] = report.sha256(str(path))
    report.atomic_json(str(private / "round10_package_hashes.json"), dict(format="round10.package/1", files=new_map))
    out = root / "runs/_round10_v4_results"
    b6 = out / "S1b/B6"
    b6.mkdir(parents=True)
    records = []
    for arm in ("B1_mask", "B1_orig"):
        for seed in protocol["seeds"]["S20"]:
            rd = root / "runs" / (arm + "_seed" + str(seed))
            (rd / "checkpoints").mkdir(parents=True)
            (rd / "variants/forward_direct").mkdir(parents=True)
            (rd / "checkpoints/best_gru_variant_forward_direct.pth").write_bytes(b"synthetic checkpoint")
            (rd / "variants/forward_direct/done.flag").write_text("done", encoding="utf-8")
            report.atomic_json(str(rd / "run_config.json"), dict(seed=seed, sealed=True, latest_params=dict(max_epochs=2000)))
            records.append(dict(arm=arm, seed=seed, run_dir=str(rd)))
    resources = dict(PHYSICS_CACHE_DIR=str(root / "physics"), ROUND9_RUNS_ROOT=str(root / "round9"))
    with patch.object(pipeline, "resolved_resource_paths", return_value=resources), \
            patch.object(pipeline, "resolved_data_root", return_value=str(root / "data")):
        pipeline.write_manifest(str(b6 / "manifest.json"), records)
    (b6 / "report.json").write_bytes(b"opaque synthetic report; deliberately not JSON")
    (b6 / "restricted_pairs").mkdir()
    (b6 / "restricted_pairs/synthetic.bin").write_bytes(b"opaque synthetic cache")
    fingerprint, _ = report.fingerprint_runs(records, str(private / "round10_v4.json"), "B6")
    report_sha = report.sha256(str(b6 / "report.json"))
    report.atomic_json(str(private / "test_access_state.json"),
                       dict(B6=dict(status="done", fingerprint=fingerprint, out=str(b6), report_sha256=report_sha)))
    report.atomic_json(str(private / "test_source_seal.json"), dict(synthetic=True))
    ledger = [dict(event="begin", mode="B6", fingerprint=fingerprint),
              dict(event="complete", mode="B6", fingerprint=fingerprint, report_sha256=report_sha)]
    (private / "test_set_access.log").write_text("".join(json.dumps(e) + "\n" for e in ledger), encoding="utf-8")
    log = out / "S1b/a7_bridge.py.log"
    log.write_text("legacy_validation_scalar\n" + hotfix.FAILURE + "\n", encoding="utf-8")
    state = dict(prereg_sha256=report.sha256(str(private / "round10_v4.json")),
                 dataset_sha256="dataset", package_sha256=old_sha,
                 stages=dict(S0=dict(status="done"), S1=dict(status="done"),
                             S1b=dict(status="failed", reason="a7_bridge.py failed (exit1): " + str(log))),
                 runs=records, candidate=None, winners=[],
                 decisions=[dict(stage="S1b", key="P1 label/control interpretation (does not select a recipe)", evidence={})])
    report.atomic_json(str(out / "pipeline_state.json"), state)
    return old_sha, resources


@contextlib.contextmanager
def environment(root):
    old_sha, resources = fixture(root)
    with patch.object(hotfix, "OLD_SEAL_SHA256", old_sha), \
            patch.object(hotfix, "_verify_dataset", return_value="dataset"), \
            patch.object(hotfix, "_audit_runs", return_value=dict(s1_runs=170, comparison_runs=120, validation_cases=20)), \
            patch.object(pipeline, "resolved_resource_paths", return_value=resources), \
            patch.object(pipeline, "resolved_data_root", return_value=str(root / "data")):
        yield


def change_json(path, fn):
    value = report.read_json(str(path))
    fn(value)
    report.atomic_json(str(path), value)


def check_cached_main(root):
    private = root / "prereg"
    b6 = root / "runs/_round10_v4_results/S1b/B6"
    before = hotfix._protected_hashes(root)
    argv = ["paper_report.py", "--b6", "--manifest", str(b6 / "manifest.json"),
            "--prereg", str(private / "round10_v4.json"), "--out", str(b6)]
    with patch.object(sys, "argv", argv), \
            patch.object(report, "access_window", side_effect=AssertionError("must not reopen B6")), \
            patch.object(report, "score_run", side_effect=AssertionError("must not score")), \
            patch.object(report, "b6_report", side_effect=AssertionError("must not report again")), \
            patch.object(report, "seal_test_source", side_effect=AssertionError("must not read Test70")), \
            contextlib.redirect_stdout(io.StringIO()) as output:
        report.main()
    check("real report.main returns cached B6 before all access/scoring functions", "B6 already complete" in output.getvalue())
    check("cached B6 leaves entire output tree and access evidence byte-identical", before == hotfix._protected_hashes(root))


def main():
    block_test_csv_reads()
    with tempfile.TemporaryDirectory(prefix="round10_a7_hotfix_") as temp:
        parent = Path(temp)
        root = parent / "valid"
        with environment(root):
            before = snapshot(root)
            result = hotfix.inspect(root)
            check("default audit accepts exact post-B6 A7 failure", result["status"] == "ready")
            check("default audit is read-only", snapshot(root) == before)
            protected = hotfix._protected_hashes(root)
            state_path = root / "runs/_round10_v4_results/pipeline_state.json"
            original = report.read_json(str(state_path))
            result = hotfix.apply(root)
            after = report.read_json(str(state_path))
            expected = copy.deepcopy(original)
            expected["package_sha256"] = report.sha256(str(root / "prereg/round10_package_hashes.json"))
            check("apply changes only package_sha256", result["status"] == "applied" and after == expected)
            check("all completed B6 files and access evidence preserved", protected == hotfix._protected_hashes(root))
            _, archive, backup, receipt = hotfix._paths(root)
            check("old journal and failed A7 log backed up exactly",
                  report.read_json(str(backup)) == original and (archive / "a7_bridge.py.before.log").read_bytes()
                  == (root / "runs/_round10_v4_results/S1b/a7_bridge.py.log").read_bytes())
            check_cached_main(root)
            before = snapshot(root)
            check("repeat apply is idempotent", hotfix.apply(root)["status"] == "already_applied" and snapshot(root) == before)
            change_json(state_path, lambda x: x["stages"].update(S2=dict(status="done")))
            with open(root / "prereg/test_set_access.log", "a", encoding="utf-8") as stream:
                stream.write('{"event":"begin","mode":"F11"}\n')
            before = snapshot(root)
            check("idempotent reapply preserves later stage progress and ledger appends",
                  hotfix.apply(root)["status"] == "already_applied" and snapshot(root) == before)
            backup.write_text("changed", encoding="utf-8")
            check("changed original-state backup refused", refuses(lambda: hotfix.inspect(root), "backup"))

        mutations = [
            ("missing completed S1", "requires S0/S1", lambda r: change_json(r / "runs/_round10_v4_results/pipeline_state.json", lambda x: x["stages"]["S1"].update(status="running"))),
            ("later stages before migration", "no stages after", lambda r: change_json(r / "runs/_round10_v4_results/pipeline_state.json", lambda x: x["stages"].update(S2=dict(status="done")))),
            ("different subprocess failure", "specific A7", lambda r: change_json(r / "runs/_round10_v4_results/pipeline_state.json", lambda x: x["stages"]["S1b"].update(reason="other failure"))),
            ("wrong A7 traceback", "unnamed-MinMaxScaler", lambda r: (r / "runs/_round10_v4_results/S1b/a7_bridge.py.log").write_text("other failure", encoding="utf-8")),
            ("selected recipe", "Recipe selection", lambda r: change_json(r / "runs/_round10_v4_results/pipeline_state.json", lambda x: x.update(candidate="C1"))),
            ("wrong prereg identity", "Preregistration changed", lambda r: change_json(r / "runs/_round10_v4_results/pipeline_state.json", lambda x: x.update(prereg_sha256="changed"))),
            ("wrong dataset identity", "Training source", lambda r: change_json(r / "runs/_round10_v4_results/pipeline_state.json", lambda x: x.update(dataset_sha256="changed"))),
            ("unreviewed package code", "package input changed", lambda r: (r / "unchanged.py").write_text("changed", encoding="utf-8")),
            ("removed frozen package input", "Package delta differs", lambda r: change_json(r / "prereg/round10_package_hashes.json", lambda x: x["files"].pop("unchanged.py"))),
            ("changed checkpoint bytes", "fingerprint changed", lambda r: next((r / "runs").glob("B1_mask_seed*/checkpoints/*.pth")).write_bytes(b"changed checkpoint")),
            ("duplicate B6 arm seed", "40 registered", lambda r: change_json(r / "runs/_round10_v4_results/pipeline_state.json", lambda x: x["runs"].append(x["runs"][0]))),
            ("running B6", "completed B6 only", lambda r: change_json(r / "prereg/test_access_state.json", lambda x: x["B6"].update(status="running"))),
            ("F11 state", "no F11", lambda r: change_json(r / "prereg/test_access_state.json", lambda x: x.update(F11=dict(status="running")))),
            ("test access lock", "Test access lock", lambda r: (r / "prereg/test_access.lock").write_text("{}", encoding="utf-8")),
            ("substituted B6 directory", "output location", lambda r: change_json(r / "prereg/test_access_state.json", lambda x: x["B6"].update(out="elsewhere"))),
            ("changed B6 fingerprint", "fingerprint changed", lambda r: change_json(r / "prereg/test_access_state.json", lambda x: x["B6"].update(fingerprint="changed"))),
            ("changed report", "report is missing or changed", lambda r: (r / "runs/_round10_v4_results/S1b/B6/report.json").write_bytes(b"changed")),
            ("missing source seal", "access evidence is missing", lambda r: (r / "prereg/test_source_seal.json").unlink()),
            ("different manifest resource", "manifest differs", lambda r: change_json(r / "runs/_round10_v4_results/S1b/B6/manifest.json", lambda x: x.update(data_root="changed"))),
            ("F11 ledger event", "later test-access", lambda r: (r / "prereg/test_set_access.log").write_text('{"event":"begin","mode":"F11"}\n', encoding="utf-8")),
            ("missing complete event", "single completed access", lambda r: (r / "prereg/test_set_access.log").write_text('{}\n', encoding="utf-8")),
        ]
        for index, (label, fragment, mutate) in enumerate(mutations):
            root = parent / ("refusal_%02d" % index)
            with environment(root):
                mutate(root)
                before = snapshot(root)
                check("refuses " + label, refuses(lambda: hotfix.inspect(root), fragment))
                check("refusal preserves files: " + label, snapshot(root) == before)

        root = parent / "interrupted"
        with environment(root):
            state_path = root / "runs/_round10_v4_results/pipeline_state.json"
            original = report.atomic_json
            def interrupt_state(path, value):
                if path == str(state_path):
                    raise OSError("synthetic interruption before state replace")
                return original(path, value)
            with patch.object(hotfix, "atomic_json", side_effect=interrupt_state):
                check("simulated interruption after receipt is recoverable", refuses(lambda: hotfix.apply(root), "synthetic interruption"))
            check("interrupted receipt resumes exact unchanged state", hotfix.apply(root)["status"] == "applied")
        root = parent / "live"
        with environment(root):
            lock = root / "runs/_round10_v4_results/pipeline.lock"
            lock.write_text(str(os.getpid()), encoding="utf-8")
            check("live pipeline lock prevents concurrent migration", refuses(lambda: hotfix.apply(root), "still running"))
        root = parent / "line_endings"
        with environment(root):
            archived = root / hotfix.OLD_SEAL
            data = archived.read_bytes()
            lf = data.replace(b"\r\n", b"\n")
            archived.write_bytes(lf if b"\r\n" in data else lf.replace(b"\n", b"\r\n"))
            check("Git LF/CRLF predecessor seal representation preserves approved identity", hotfix.inspect(root)["status"] == "ready")
    print("check_a7_scaler_hotfix: %d/%d PASS" % (sum(RESULTS), len(RESULTS)))
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
