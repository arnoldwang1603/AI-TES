"""Audited, one-off resume for the v4 S1b basename collision.

Default: inspect only. --apply preserves the original state and failed S1b
outputs, then changes only the state journal's implementation package digest.
No training, scoring, test access, resealing or protocol amendment occurs here.
"""
import argparse
import copy
import csv
import hashlib
import os
from pathlib import Path
import shutil
import sys
import time

from paper_report import atomic_json, pid_alive, read_json, sha256
from round10_integrity import matches_frozen_sha256

HERE = Path(__file__).resolve().parent
HOTFIX = "round10.v4.case-identity/1"
OLD_SEAL = "prereg/round10_package_hashes_v4_before_case_id_fix.json"
OLD_SEAL_SHA256 = "e0690b779a5033a26dd90fc3aa11eebccd4349888e861366fcfe13d236ae2b3c"
CHANGED = frozenset(("round10_pipeline.py", "round10_decisions.py", "paper_stats.py",
                     "round10_checks/check_decisions.py"))
ADDED = frozenset(("round10_case_id_hotfix.py", "round10_checks/check_case_id_hotfix.py",
                   "round10_checks/check_case_identity.py", OLD_SEAL))
FAILURE = "Index contains duplicate entries, cannot reshape"
COMPARISON_ARMS = frozenset(("B1_mask", "B1_orig", "B2_mask", "B2_orig", "B7", "B8"))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _under(path, parent):
    path, parent = Path(path).resolve(), Path(parent).resolve()
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _verify_package(root):
    archived = root / OLD_SEAL
    require(archived.is_file() and matches_frozen_sha256(archived, OLD_SEAL_SHA256),
            "Archived original v4 package seal does not match the approved predecessor")
    old = read_json(str(archived))
    current_path = root / "prereg/round10_package_hashes.json"
    current = read_json(str(current_path))
    require(old.get("format") == current.get("format") == "round10.package/1",
            "Unsupported package manifest format")
    old_files, new_files = old["files"], current["files"]
    removed = set(old_files) - set(new_files)
    added = set(new_files) - set(old_files)
    changed = {name for name in set(old_files) & set(new_files)
               if old_files[name] != new_files[name]}
    require(not removed and added == ADDED and changed == CHANGED,
            "Package delta differs from the reviewed case-ID hotfix: removed=%s added=%s changed=%s"
            % (sorted(removed), sorted(added), sorted(changed)))
    for relative, expected in new_files.items():
        path = root / relative
        require(_under(path, root), "Package path escapes the GRU root: " + relative)
        require(path.is_file() and matches_frozen_sha256(path, expected),
                "Current package input changed: " + relative)
    return sha256(str(current_path)), sorted(changed | added)


def _verify_no_test_access(root):
    private = root / "prereg"
    state = private / "test_access_state.json"
    require(not state.exists() or read_json(str(state)) == {},
            "Test access state exists; this pre-B6 hotfix cannot migrate it")
    for name in ("test_source_seal.json", "test_access.lock"):
        require(not (private / name).exists(), "Test access evidence exists: " + name)
    ledger = private / "test_set_access.log"
    require(not ledger.exists() or ledger.stat().st_size == 0,
            "Test access ledger is nonempty; preserve it for review")
    require(not (root / "runs/_round10_v4_results/S1b/B6").exists(),
            "S1b B6 output directory exists; this recovery only covers the pre-B6 failure")


def _verify_dataset(root):
    import round10_pipeline as pipeline
    require(Path(pipeline.HERE).resolve() == root.resolve(), "Hotfix loaded a different pipeline checkout")
    return pipeline.verify_dataset_hashes()


def _audit_runs(root, state):
    import round10_pipeline as pipeline
    p = read_json(str(root / "prereg/round10_v4.json"))
    table = pipeline.resolve_names([a for a in pipeline.full_arm_table(p) if a["stage"] == "S1"])
    expected = {(a["arm"], int(seed)): (a, pipeline.run_record(a, seed))
                for a in table for seed in a["seeds"]}
    records = state["runs"]
    keys = [(r["arm"], int(r["seed"])) for r in records]
    require(len(keys) == len(set(keys)) and set(keys) == set(expected),
            "S1 run journal is incomplete, duplicated, or contains unregistered runs")
    require(len(keys) == 170, "This hotfix expects the registered 170 completed S1 runs")
    split = read_json(str(root / "splits/round10_vs3_s42.json"))
    registered_cases = split["val"]
    # Split keys are relative to train_dir, but load_all_data._case_id and
    # evaluate export Case relative to DATA_ROOT, including train_subdir.
    # Match that exact registered prefix, never basename or arbitrary suffix.
    prefix = split["train_subdir"].replace("\\", "/") + "/"
    case_set = {prefix + key.replace("\\", "/") for key in registered_cases}
    require(len(registered_cases) == len(case_set) == 20,
            "Frozen split does not contain exactly 20 distinct validation paths")
    comparison_runs = 0
    for record in records:
        a, expected_record = expected[(record["arm"], int(record["seed"]))]
        path = Path(record["run_dir"])
        require(_under(path, root / "runs") and path.resolve() == Path(expected_record["run_dir"]).resolve(),
                "S1 run directory differs from its registered name: " + str(path))
        require(pipeline.verify_done(record, dict(a["env"], SEEDS=str(record["seed"]), SEAL_TEST="1")),
                "S1 completed checkpoint is missing: " + str(path))
        if record["arm"] not in COMPARISON_ARMS:
            continue
        with open(path / "variants/forward_direct/summary_errors_val.csv", newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        identities = [row.get("Case", "").strip().replace("\\", "/") for row in rows
                      if row.get("Case") != "AVERAGE"]
        require(len(identities) == 20 and all(identities) and len(set(identities)) == 20,
                "Validation Case paths are missing or duplicated in " + str(path))
        require(set(identities) == case_set,
                "S1b validation paths differ from the frozen validation split in %s; missing=%s; unexpected=%s"
                % (str(path), sorted(case_set - set(identities))[:3], sorted(set(identities) - case_set)[:3]))
        comparison_runs += 1
    require(comparison_runs == 120, "Expected all 120 S1b comparison runs")
    return dict(s1_runs=len(keys), comparison_runs=comparison_runs, validation_cases=len(case_set))


def _audit_old_state(root, state):
    require(matches_frozen_sha256(root / OLD_SEAL, state.get("package_sha256", "")),
            "State package digest is not the archived original v4 package")
    require(state.get("prereg_sha256") == sha256(str(root / "prereg/round10_v4.json")),
            "Preregistration changed; existing training cannot be migrated")
    require(state.get("dataset_sha256") == _verify_dataset(root),
            "Training source seal changed; existing training cannot be migrated")
    stages = state.get("stages", {})
    require(set(stages) == {"S0", "S1", "S1b"}, "Recovery requires exactly S0/S1/S1b; later stages are refused")
    require(all(stages[s].get("status") == "done" for s in ("S0", "S1")),
            "S0 and S1 must both be complete")
    require(stages["S1b"].get("status") == "failed" and stages["S1b"].get("reason") == FAILURE,
            "S1b is not the specific duplicate-index failure covered by this hotfix")
    require(not state.get("decisions") and not state.get("candidate") and not state.get("winners")
            and not state.get("frozen_recipe"), "Post-comparison decisions exist; preserve them for separate review")
    _verify_no_test_access(root)
    return _audit_runs(root, state)


def _receipt_paths(root):
    out = root / "runs/_round10_v4_results"
    archive = out / "case_id_hotfix"
    return out, archive, archive / "pipeline_state.before.json", archive / "receipt.json"


def _inspect(root, owning_lock=False):
    root = Path(root).resolve()
    out, archive, backup, receipt_path = _receipt_paths(root)
    lock = out / "pipeline.lock"
    owner = int(lock.read_text().strip()) if lock.exists() else None
    require(owner is None or not pid_alive(owner) or (owning_lock and owner == os.getpid()),
            "A pipeline is still running; stop it before applying this hotfix")
    current_sha, delta = _verify_package(root)
    state_path = out / "pipeline_state.json"
    require(state_path.is_file(), "No existing v4 state to migrate; use the ordinary pipeline for a fresh run")
    state = read_json(str(state_path))
    if receipt_path.exists():
        receipt = read_json(str(receipt_path))
        require(receipt.get("schema") == HOTFIX and receipt.get("new_package_sha256") == current_sha,
                "Existing hotfix receipt does not identify this package")
        require(backup.is_file() and sha256(str(backup)) == receipt.get("before_state_sha256"),
                "Original state backup is missing or changed")
        before = read_json(str(backup))
        require(receipt.get("old_package_sha256") == before.get("package_sha256")
                and matches_frozen_sha256(root / OLD_SEAL, before.get("package_sha256", "")),
                "Receipt predecessor does not identify the approved original v4 package")
        require(state.get("prereg_sha256") == before.get("prereg_sha256")
                == sha256(str(root / "prereg/round10_v4.json")), "Preregistration changed after hotfix")
        require(state.get("dataset_sha256") == before.get("dataset_sha256") == _verify_dataset(root),
                "Training source changed after hotfix")
        if state.get("package_sha256") == current_sha:
            return dict(status="already_applied", receipt=receipt)
        require(sha256(str(state_path)) == receipt["before_state_sha256"],
                "State changed during an interrupted hotfix; preserve both journals for review")
    audit = _audit_old_state(root, state)
    if backup.exists():
        require(sha256(str(backup)) == sha256(str(state_path)), "Existing original-state backup differs from current state")
    return dict(status="ready", old_package_sha256=state["package_sha256"],
                new_package_sha256=current_sha, before_state_sha256=sha256(str(state_path)),
                reviewed_delta=delta, audit=audit)


def inspect(root=HERE):
    return _inspect(Path(root).resolve())


def apply(root=HERE):
    root = Path(root).resolve()
    out, archive, backup, receipt_path = _receipt_paths(root)
    out.mkdir(parents=True, exist_ok=True)
    lock = out / "pipeline.lock"
    # Use the production pipeline's lock so a launch cannot race this write.
    if lock.exists():
        require(not pid_alive(int(lock.read_text().strip())), "A pipeline is still running")
        lock.unlink()
    fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w") as stream:
        stream.write(str(os.getpid()))
    try:
        # The caller owns this lock; inspect still rejects every other live PID.
        result = _inspect(root, owning_lock=True)
        if result["status"] == "already_applied":
            return result
        state_path = out / "pipeline_state.json"
        before_bytes = state_path.read_bytes()
        require(hashlib.sha256(before_bytes).hexdigest() == result["before_state_sha256"],
                "State changed after audit; refusing hotfix")
        archive.mkdir(exist_ok=True)
        if not backup.exists():
            with open(backup, "xb") as stream:
                stream.write(before_bytes)
                stream.flush()
                os.fsync(stream.fileno())
        require(backup.read_bytes() == before_bytes, "Original-state backup differs from current state")
        # Preserve incomplete tables too: the resumed comparison regenerates them.
        failed = out / "S1b"
        saved = archive / "S1b.before"
        if failed.exists():
            shutil.copytree(str(failed), str(saved), dirs_exist_ok=True)
        receipt = dict(result, schema=HOTFIX, created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       state_change="package_sha256 only", retraining=False, test_access=False)
        if not receipt_path.exists():
            atomic_json(str(receipt_path), receipt)
        updated = copy.deepcopy(read_json(str(state_path)))
        updated["package_sha256"] = result["new_package_sha256"]
        atomic_json(str(state_path), updated)
        return dict(status="applied", receipt=read_json(str(receipt_path)))
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Back up and migrate the audited stopped S1b state")
    args = parser.parse_args()
    result = apply() if args.apply else inspect()
    if result["status"] == "already_applied":
        print("Case-ID hotfix already applied; existing stage progress preserved")
    elif args.apply:
        print("Case-ID hotfix PASS: preserved 170 S1 runs; state and failed S1b outputs backed up; no retraining/test access")
    else:
        print("Case-ID hotfix audit PASS: ready; no files changed. Run with --apply to migrate the package digest.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError, KeyError) as exc:
        print("Case-ID hotfix refused: %s" % exc, file=sys.stderr)
        sys.exit(1)
