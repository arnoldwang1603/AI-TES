"""Audited recovery for the v4 A7 failure and downstream P4 sweep correction.

Default is read-only. --apply backs up the pipeline journal and failed A7 log,
then changes only package_sha256. Training, the protocol, B6 outputs and the
test-access journal are preserved. No training or test scoring is performed.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from paper_report import atomic_json, fingerprint_runs, pid_alive, read_json, sha256
from round10_integrity import matches_frozen_sha256
from round10_case_id_hotfix import _audit_runs, _under, _verify_dataset, require

HERE = Path(__file__).resolve().parent
HOTFIX = "round10.v4.a7-scaler/2"
OLD_SEAL = "prereg/round10_package_hashes_v4_before_a7_scaler_fix.json"
OLD_SEAL_SHA256 = "0bdad6d0b77239bf2ac22105ee92315a81bb6ac3217ea572ebf12f8ec81acddf"
CHANGED = frozenset(("a7_bridge.py", "round10_decisions.py"))
ADDED = frozenset((OLD_SEAL, "round10_a7_scaler_hotfix.py",
                   "round10_checks/check_a7_scaler_hotfix.py", "round10_checks/check_a7_scaler.py",
                   "round10_checks/check_p4_t0_contract.py"))
FAILURE = "AttributeError: 'MinMaxScaler' object has no attribute 'feature_names_in_'"


def _verify_package(root):
    archived = root / OLD_SEAL
    require(archived.is_file() and matches_frozen_sha256(archived, OLD_SEAL_SHA256),
            "Archived predecessor package seal differs from the reviewed A7 predecessor")
    current_path = root / "prereg/round10_package_hashes.json"
    old, current = read_json(str(archived)), read_json(str(current_path))
    require(old.get("format") == current.get("format") == "round10.package/1", "Unsupported package format")
    before, after = old["files"], current["files"]
    removed, added = set(before) - set(after), set(after) - set(before)
    changed = {key for key in set(before) & set(after) if before[key] != after[key]}
    require(not removed and added == ADDED and changed == CHANGED,
            "Package delta differs from the reviewed A7 fix: removed=%s added=%s changed=%s"
            % (sorted(removed), sorted(added), sorted(changed)))
    for name, expected in after.items():
        path = root / name
        require(_under(path, root) and path.is_file() and matches_frozen_sha256(path, expected),
                "Current package input changed: " + name)
    return sha256(str(current_path)), sorted(changed | added)


def _protected_hashes(root):
    """Opaque hashes only: never parse B6 report values or restricted scores."""
    b6 = root / "runs/_round10_v4_results/S1b/B6"
    require(b6.is_dir() and not b6.is_symlink(), "Completed B6 directory is missing or redirected")
    paths = []
    for path in b6.rglob("*"):
        require(not path.is_symlink() and _under(path, b6), "B6 contains a redirected path")
        if path.is_file():
            paths.append(path)
    for name in ("test_set_access.log", "test_access_state.json", "test_source_seal.json"):
        path = root / "prereg" / name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size > 0,
                "Completed B6 access evidence is missing: " + name)
        paths.append(path)
    return {path.relative_to(root).as_posix(): sha256(str(path)) for path in sorted(paths)}


def _verify_b6(root, state):
    import round10_pipeline as pipeline
    private = root / "prereg"
    require(not (private / "test_access.lock").exists(), "Test access lock exists; preserve it for review")
    access = read_json(str(private / "test_access_state.json"))
    require(set(access) == {"B6"} and access["B6"].get("status") == "done",
            "Recovery requires completed B6 only, with no F11 window")
    b6 = root / "runs/_round10_v4_results/S1b/B6"
    prior = access["B6"]
    require(prior.get("out") == str(b6), "B6 output location differs from its original pipeline directory")
    records = [r for r in state["runs"] if r["arm"] in {"B1_mask", "B1_orig"}]
    p = read_json(str(private / "round10_v4.json"))
    expected_keys = {(arm, int(seed)) for arm in ("B1_mask", "B1_orig") for seed in p["seeds"]["S20"]}
    keys = [(r["arm"], int(r["seed"])) for r in records]
    require(len(keys) == len(set(keys)) == 40 and set(keys) == expected_keys,
            "B6 requires the same 40 registered B1 checkpoints")
    resources = pipeline.resolved_resource_paths()
    expected_manifest = dict(runs=records, data_root=pipeline.resolved_data_root(),
                             physics_cache_dir=resources["PHYSICS_CACHE_DIR"],
                             round9_root=resources["ROUND9_RUNS_ROOT"])
    manifest_path = b6 / "manifest.json"
    require(read_json(str(manifest_path)) == expected_manifest,
            "B6 manifest differs from the current unchanged pipeline inputs")
    # write_manifest uses this exact serialization. A retry must not alter even
    # the B6 manifest's bytes while refreshing the surrounding S1b readouts.
    expected_bytes = (json.dumps(expected_manifest, indent=2, ensure_ascii=False, allow_nan=False)
                      + "\n").replace("\n", os.linesep).encode("utf-8")
    require(manifest_path.read_bytes() == expected_bytes,
            "B6 manifest serialization differs; preserve the original for review")
    digest, _ = fingerprint_runs(records, str(private / "round10_v4.json"), "B6")
    require(prior.get("fingerprint") == digest, "B6 checkpoint fingerprint changed")
    report = b6 / "report.json"
    require(report.is_file() and sha256(str(report)) == prior.get("report_sha256"),
            "Completed B6 report is missing or changed; cannot reopen the test window")
    ledger = private / "test_set_access.log"
    with open(ledger, encoding="utf-8") as stream:
        events = [json.loads(line) for line in stream if line.strip()]
    require(all(event.get("mode") in (None, "B6") for event in events),
            "A later test-access event exists; preserve it for separate review")
    begin = [e for e in events if e.get("event") == "begin"]
    complete = [e for e in events if e.get("event") == "complete"]
    require(len(begin) == len(complete) == 1 and begin[0].get("fingerprint") == digest
            and complete[0].get("fingerprint") == digest
            and complete[0].get("report_sha256") == prior["report_sha256"],
            "B6 ledger does not match its single completed access window")
    return _protected_hashes(root)


def _paths(root):
    out = root / "runs/_round10_v4_results"
    archive = out / "a7_scaler_hotfix"
    return out, archive, archive / "pipeline_state.before.json", archive / "receipt.json"


def _audit_old_state(root, state):
    require(matches_frozen_sha256(root / OLD_SEAL, state.get("package_sha256", "")),
            "State package digest does not match the reviewed predecessor")
    stages = state.get("stages", {})
    require(set(stages) == {"S0", "S1", "S1b"}
            and all(stages[s].get("status") == "done" for s in ("S0", "S1")),
            "Recovery requires S0/S1 done and no stages after S1b")
    log = root / "runs/_round10_v4_results/S1b/a7_bridge.py.log"
    reason = "a7_bridge.py failed (exit1): " + str(log)
    require(stages["S1b"].get("status") == "failed" and stages["S1b"].get("reason") == reason,
            "S1b failure is not the specific A7 subprocess failure")
    require(log.is_file() and FAILURE in log.read_text(encoding="utf-8")
            and "legacy_validation_scalar" in log.read_text(encoding="utf-8"),
            "A7 log does not identify the unnamed-MinMaxScaler failure")
    require(not state.get("candidate") and not state.get("winners") and not state.get("frozen_recipe"),
            "Recipe selection exists; preserve it for separate review")
    require(all(d.get("stage") == "S1b" and d.get("key") ==
                "P1 label/control interpretation (does not select a recipe)" for d in state.get("decisions", [])),
            "Unexpected decision journal; preserve it for separate review")
    runs = _audit_runs(root, state)
    protected = _verify_b6(root, state)
    return runs, protected, sha256(str(log))


def _inspect(root, owning_lock=False):
    out, archive, backup, receipt_path = _paths(root)
    lock = out / "pipeline.lock"
    owner = int(lock.read_text().strip()) if lock.exists() else None
    require(owner is None or not pid_alive(owner) or (owning_lock and owner == os.getpid()),
            "A pipeline is still running; stop it before applying this hotfix")
    current_sha, delta = _verify_package(root)
    state_path = out / "pipeline_state.json"
    require(state_path.is_file(), "No existing v4 state; use the ordinary pipeline for a fresh run")
    state = read_json(str(state_path))
    require(state.get("prereg_sha256") == sha256(str(root / "prereg/round10_v4.json")),
            "Preregistration changed; cannot migrate training")
    require(state.get("dataset_sha256") == _verify_dataset(root), "Training source seal changed")
    if receipt_path.exists():
        receipt = read_json(str(receipt_path))
        require(receipt.get("schema") == HOTFIX and receipt.get("new_package_sha256") == current_sha,
                "Existing receipt does not identify this A7 package")
        require(backup.is_file() and sha256(str(backup)) == receipt.get("before_state_sha256"),
                "Original state backup is missing or changed")
        before = read_json(str(backup))
        require(receipt.get("old_package_sha256") == before.get("package_sha256")
                and matches_frozen_sha256(root / OLD_SEAL, before.get("package_sha256", ""))
                and before.get("prereg_sha256") == state.get("prereg_sha256")
                and before.get("dataset_sha256") == state.get("dataset_sha256"),
                "Receipt predecessor or frozen inputs differ")
        saved_log = archive / "a7_bridge.py.before.log"
        require(saved_log.is_file() and sha256(str(saved_log)) == receipt.get("failed_log_sha256"),
                "Failed A7 log backup is missing or changed")
        if state.get("package_sha256") == current_sha:
            # Later completed stages and legitimate F11 ledger appends remain
            # valid. Never roll them back to the original stopped state.
            return dict(status="already_applied", receipt=receipt)
        require(sha256(str(state_path)) == receipt.get("before_state_sha256"),
                "State changed during interrupted recovery; preserve both journals")
        require(_protected_hashes(root) == receipt.get("protected_sha256"),
                "B6/access evidence changed during interrupted recovery")
    audit, protected, log_sha = _audit_old_state(root, state)
    return dict(status="ready", old_package_sha256=state["package_sha256"],
                new_package_sha256=current_sha, before_state_sha256=sha256(str(state_path)),
                failed_log_sha256=log_sha, protected_sha256=protected, reviewed_delta=delta, audit=audit)


def inspect(root=HERE):
    return _inspect(Path(root).resolve())


def _preserve(path, content):
    if not path.exists():
        with open(path, "xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    require(path.read_bytes() == content, "Existing backup differs: " + str(path))


def apply(root=HERE):
    root = Path(root).resolve()
    out, archive, backup, receipt_path = _paths(root)
    require(out.is_dir(), "No existing v4 state to migrate")
    lock = out / "pipeline.lock"
    if lock.exists():
        require(not pid_alive(int(lock.read_text().strip())), "A pipeline is still running")
        lock.unlink()
    fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w") as stream:
        stream.write(str(os.getpid()))
    try:
        result = _inspect(root, owning_lock=True)
        if result["status"] == "already_applied":
            return result
        state_path = out / "pipeline_state.json"
        before_bytes = state_path.read_bytes()
        require(hashlib.sha256(before_bytes).hexdigest() == result["before_state_sha256"],
                "Pipeline state changed after audit")
        archive.mkdir(exist_ok=True)
        _preserve(backup, before_bytes)
        log = out / "S1b/a7_bridge.py.log"
        require(sha256(str(log)) == result["failed_log_sha256"], "A7 log changed after audit")
        _preserve(archive / "a7_bridge.py.before.log", log.read_bytes())
        require(_protected_hashes(root) == result["protected_sha256"], "B6/access evidence changed after audit")
        receipt = dict(result, schema=HOTFIX,
                       created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       state_change="package_sha256 only", retraining=False, test_access=False)
        if not receipt_path.exists():
            atomic_json(str(receipt_path), receipt)
        updated = copy.deepcopy(read_json(str(state_path)))
        updated["package_sha256"] = result["new_package_sha256"]
        atomic_json(str(state_path), updated)
        require(_protected_hashes(root) == result["protected_sha256"], "B6/access evidence changed during recovery")
        return dict(status="applied", receipt=read_json(str(receipt_path)))
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Back up and migrate only the audited implementation package digest")
    args = parser.parse_args()
    result = apply() if args.apply else inspect()
    if result["status"] == "already_applied":
        print("A7 scaler hotfix already applied; existing stage progress preserved")
    elif args.apply:
        print("A7 scaler hotfix PASS: preserved 170 S1 runs and completed B6; state/log backed up; no retraining/test access")
    else:
        print("A7 scaler hotfix audit PASS: ready; no files changed. Run with --apply to migrate the package digest.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError, KeyError) as exc:
        print("A7 scaler hotfix refused: %s" % exc, file=sys.stderr)
        sys.exit(1)
