"""Read-only audit of the v3 boundary; preserve every run and test ledger."""
import os
import time

from paper_report import atomic_json, pid_alive, read_json, sha256


def _snapshot(root):
    old = os.path.join(root, "runs", "_round10_results")
    lock = os.path.join(old, "pipeline.lock")
    if os.path.isfile(lock):
        with open(lock, encoding="utf-8") as stream:
            if pid_alive(int(stream.read().strip())):
                raise RuntimeError("Old v3 pipeline is still running; stop it before --restart-v4")
    private = os.path.join(root, "prereg")
    state_path = os.path.join(private, "test_access_state.json")
    test_state = read_json(state_path) if os.path.isfile(state_path) else {}
    evidence = [name for name in ("test_source_seal.json", "test_access.lock")
                if os.path.exists(os.path.join(private, name))]
    ledger = os.path.join(private, "test_set_access.log")
    if os.path.isfile(ledger) and os.path.getsize(ledger):
        evidence.append("test_set_access.log")
    if test_state or evidence:
        raise RuntimeError("Prior test access evidence exists; v4 restart refused. Preserve all ledgers and review the protocol amendment.")
    path = os.path.join(old, "pipeline_state.json")
    state = read_json(path) if os.path.isfile(path) else {}
    stages = {name: value.get("status") for name, value in state.get("stages", {}).items()}
    if any(name not in ("S0", "S1") for name in stages):
        raise RuntimeError("Old v3 reached the B6 boundary or later; v4 restart refused")
    # Inspect only the state journal, never checkpoint metrics or predictions.
    return dict(old_state_sha256=sha256(path) if os.path.isfile(path) else None,
                old_stages=stages,
                preserved_old_runs=[dict(arm=row["arm"], seed=row["seed"], run_dir=row["run_dir"])
                                    for row in state.get("runs", [])],
                active_prereg_sha256=sha256(os.path.join(private, "round10_v4.json")))


def restart_v4(root):
    """Create an idempotent audit receipt; this is never a reset operation."""
    out = os.path.join(root, "runs", "_round10_v4_results")
    state_path = os.path.join(out, "pipeline_state.json")
    receipt_path = os.path.join(out, "migration_v3_audit.json")
    if os.path.isfile(state_path):
        state = read_json(state_path)
        if state.get("prereg_sha256") != sha256(os.path.join(root, "prereg", "round10_v4.json")):
            raise RuntimeError("Prereg changed; cannot restart an existing v4 protocol")
        if not os.path.isfile(receipt_path):
            raise RuntimeError("Existing v4 state has no migration audit; preserve it for review")
        return dict(read_json(receipt_path), already_initialized=True)
    snapshot = _snapshot(root)
    if os.path.isfile(receipt_path):
        receipt = read_json(receipt_path)
        if receipt.get("snapshot") != snapshot:
            raise RuntimeError("Old state or active prereg changed after migration audit")
        return receipt
    receipt = dict(schema="round10.v4-restart/1", created=time.time(), snapshot=snapshot,
                   result_namespace="runs/_round10_v4_results", run_prefix="r10v4",
                   old_runs_reused=False, old_files_deleted=False, test_access_reset=False)
    atomic_json(receipt_path, receipt)
    return receipt


def ensure_v4_restart(root):
    old = os.path.join(root, "runs", "_round10_results", "pipeline_state.json")
    receipt = os.path.join(root, "runs", "_round10_v4_results", "migration_v3_audit.json")
    if os.path.isfile(old) and not os.path.isfile(receipt):
        raise RuntimeError("Old v3 state found: run --restart-v4 once before starting v4")
    return restart_v4(root)
