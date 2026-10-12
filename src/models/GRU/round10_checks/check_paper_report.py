"""Synthetic-only checks for the B6/F11 access ledger and B6 disclosure scope.

No real training, validation or test CSVs are opened by this check. Dummy
checkpoint bytes and 70 synthetic summary rows exercise the report contract.
"""
import contextlib
import copy
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(HERE)
import paper_report as report
from check_h9 import block_test_csv_reads

RESULTS = []


def check(label, ok):
    RESULTS.append(bool(ok))
    print("%s  %s" % ("PASS" if ok else "FAIL", label), flush=True)


def rejects(fn, fragment=None):
    try:
        fn()
    except (RuntimeError, FileExistsError) as exc:
        return fragment is None or fragment in str(exc)
    return False


def check_source_seal(tmp, prereg):
    root = os.path.join(tmp, "synthetic_source")
    source = os.path.join(root, "Test_70_cases")
    os.makedirs(source)
    for cid in range(1, 71):
        with open(os.path.join(source, "synthetic(%d).csv" % cid), "wb") as stream:
            stream.write(("synthetic raw bytes for case %d\n" % cid).encode("ascii"))
    private = os.path.join(tmp, "source_prereg")
    out = os.path.join(tmp, "source_B6")
    check("test source hashing requires the logged lease before filesystem access", rejects(
        lambda: report.seal_test_source(private, root), "sealed"))
    try:
        with report.access_window("B6", "source_b6", {}, private, out) as ledger:
            first = report.seal_test_source(private, root)
            events = [json.loads(line) for line in open(ledger, encoding="utf-8") if line.strip()]
            check("first B6 fixes all 70 source byte hashes after the begin event",
                  len(first["files"]) == 70 and first["source_root"] == os.path.abspath(root)
                  and [e["event"] for e in events] == ["begin", "test_source_sealed"])
            raise RuntimeError("synthetic source interruption")
    except RuntimeError as exc:
        if str(exc) != "synthetic source interruption":
            raise
    with report.access_window("B6", "source_b6", {}, private, out):
        original_listdir = report.os.listdir
        report.os.listdir = lambda path: list(reversed(original_listdir(path)))
        try:
            ordered = report.seal_test_source(private, root)
        finally:
            report.os.listdir = original_listdir
        check("B6 source verification is independent of filesystem listing order", ordered == first)
        relocated = os.path.join(tmp, "relocated_identical_source")
        shutil.copytree(root, relocated)
        check("identical relative names and contents can relocate without changing the seal",
              report.seal_test_source(private, relocated) == first)
        changed = os.path.join(source, "synthetic(1).csv")
        with open(changed, "rb") as stream:
            initial = stream.read()
        with open(changed, "ab") as stream:
            stream.write(b"changed")
        check("B6 resume rejects any source byte change before further scoring", rejects(
            lambda: report.seal_test_source(private, root), "mixed-source"))
        with open(changed, "wb") as stream:
            stream.write(initial)
        os.rename(changed, changed + ".missing")
        check("source verification rejects a missing case", rejects(
            lambda: report.seal_test_source(private, root), "70 case IDs"))
        os.rename(changed + ".missing", changed)
        with tempfile.TemporaryDirectory(prefix="nested_extra_", dir=source) as nested:
            with open(os.path.join(nested, "extra.csv"), "wb") as stream:
                stream.write(b"not one of the frozen70")
            check("recursive loader cannot read an unsealed nested CSV", rejects(
                lambda: report.seal_test_source(private, root), "nested CSV"))
        report.atomic_json(os.path.join(out, "report.json"), {"mode": "B6"})
    with open(changed, "ab") as stream:
        stream.write(b"changed before F11")
    check("forced F11 window still rejects changes to the first B6 source", rejects(
        lambda: _source_window("F11", "source_f11", private, os.path.join(tmp, "source_F11"), root),
        "mixed-source"))
    with open(changed, "wb") as stream:
        stream.write(initial)

    missing_private = os.path.join(tmp, "source_missing_seal")
    report.atomic_json(os.path.join(missing_private, "test_access_state.json"),
                       {"B6": {"status": "done"}})
    check("F11 refuses a missing first-window source seal", rejects(
        lambda: _source_window("F11", "missing_source_f11", missing_private,
            os.path.join(tmp, "source_missing_F11"), root), "source seal from the first B6"))
    unsigned_private = os.path.join(tmp, "source_unsigned_scores")
    unsigned_out = os.path.join(tmp, "source_unsigned_B6")
    report.atomic_json(os.path.join(unsigned_out, "restricted_pairs", "seed1.json"), {"seed": 1})
    check("B6 cache created without a source seal cannot be silently trusted", rejects(
        lambda: _source_window("B6", "unsigned_source_b6", unsigned_private, unsigned_out, root),
        "no test source seal"))

    # Exercise actual main ordering with fake reports, never a raw CSV parser.
    main_private = os.path.join(tmp, "source_main_prereg")
    prereg_path = os.path.join(main_private, "round10_v3.json")
    report.atomic_json(prereg_path, prereg)
    manifest_path = os.path.join(tmp, "source_main_manifest.json")
    report.atomic_json(manifest_path, {"runs": [], "data_root": root})
    original = {name: getattr(report, name) for name in
                ("fingerprint_runs", "f11_input_identities", "b6_report", "f11_report", "seal_test_source")}
    old_argv, old_root = sys.argv[:], os.environ.get("DATA_ROOT")
    calls, seen = [], []

    def mock_seal(*args, **kwargs):
        calls.append("source")
        return original["seal_test_source"](*args, **kwargs)

    def mock_b6(*args):
        calls.append("score")
        seen.append(report.assert_test_access()["mode"] == "B6"
                    and os.environ.get("DATA_ROOT") == os.path.abspath(root)
                    and os.path.isfile(os.path.join(main_private, "test_source_seal.json")))
        return {"mode": "B6"}

    report.fingerprint_runs = lambda runs, path, mode, *args: ("main_" + mode, {})
    report.f11_input_identities = lambda manifest: {"synthetic": True}
    report.seal_test_source = mock_seal
    report.b6_report = mock_b6
    report.f11_report = lambda *args: calls.append("F11_score")
    try:
        sys.argv = ["paper_report", "--b6", "--manifest", manifest_path,
                    "--prereg", prereg_path, "--out", os.path.join(tmp, "source_main_B6"), "--dry-run"]
        report.main()
        check("report dry-run never lists or hashes test sources", calls == [])
        sys.argv = sys.argv[:-1]
        report.main()
        check("actual report entrypoint seals before scoring and verifies before completion",
              calls == ["source", "score", "source"] and all(seen)
              and os.environ.get("DATA_ROOT") == old_root)
        with open(changed, "ab") as stream:
            stream.write(b"changed before main F11")
        calls.clear()
        sys.argv = ["paper_report", "--f11", "--manifest", manifest_path,
                    "--prereg", prereg_path, "--out", os.path.join(tmp, "source_main_F11")]
        check("F11 entrypoint rejects changed test bytes before any report scorer", rejects(
            report.main, "mixed-source") and calls == ["source"])
        with open(changed, "wb") as stream:
            stream.write(initial)
        for key in ("physics_params", "conventions"):
            report.atomic_json(manifest_path, {"runs": [], "data_root": root, key: "override"})
            calls.clear()
            check("F11 entrypoint refuses unfrozen %s overrides before access" % key, rejects(
                report.main, "packaged frozen") and calls == [])
    finally:
        sys.argv = old_argv
        for name, value in original.items():
            setattr(report, name, value)


def _source_window(mode, digest, private, out, root):
    with report.access_window(mode, digest, {}, private, out):
        report.seal_test_source(private, root)
        report.atomic_json(os.path.join(out, "report.json"), {"mode": mode})


def main():
    block_test_csv_reads()
    # Exercise POSIX semantics on every test host, including Windows.
    # Never send a real signal to a group or any unrelated process.
    probes = []

    def posix_probe(pid, signal):
        probes.append((pid, signal))
        if pid == 101:
            raise ProcessLookupError()
        if pid == 102:
            raise PermissionError()
        # Includes pid=0: POSIX treats it as our process group, not a
        # terminated process. The guard must reject it before this call.

    with patch.object(report, "os", SimpleNamespace(name="posix", kill=posix_probe)):
        check("zero and negative lock PIDs never query POSIX process groups",
              not report.pid_alive(0) and not report.pid_alive(-1) and probes == [])
        check("POSIX live and permission-denied processes stay protected; exited PID is dead",
              report.pid_alive(100) and not report.pid_alive(101) and report.pid_alive(102)
              and probes == [(100, 0), (101, 0), (102, 0)])
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait(timeout=30)
    check("a real exited child is not alive and our current PID remains alive",
          not report.pid_alive(child.pid) and report.pid_alive(os.getpid()))
    prereg = report.read_json(os.path.join(HERE, "prereg", "round10_v3.json"))
    parent = os.path.join(HERE, "runs", "_round10_check_scratch")
    os.makedirs(parent, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="paper_report_", dir=parent) as tmp:
        private = os.path.join(tmp, "prereg")
        out_b6 = os.path.join(tmp, "B6")
        out_f11 = os.path.join(tmp, "F11")
        os.makedirs(out_b6)
        os.makedirs(out_f11)
        check("test-access capability absent outside the report", rejects(report.assert_test_access, "sealed"))
        check("F11 cannot precede a completed B6", rejects(
            lambda: _enter_window("F11", "f11", private, out_f11), "completed B6"))

        original_score = report.score_run
        calls = []

        def synthetic_score(run, dest, ledger):
            lease = report.assert_test_access()
            with open(ledger, encoding="utf-8") as f:
                events = [json.loads(line) for line in f if line.strip()]
            assert any(e.get("event") == "begin" and e.get("mode") == "B6" for e in events)
            assert lease["fingerprint"] == "b6"
            calls.append((run["arm"], run["seed"]))
            selected = {70, 42, 66, 67}
            rows = []
            for cid in range(1, 71):
                base = 0.4 + 0.001 * run["seed"] if cid in selected else 99.123456
                value = base - (0.02 if run["arm"] == "B1_mask" else 0)
                rows.append(dict(test_id=cid, Case="synthetic(%d)" % cid,
                                 **{"MAE_T_avg (C)": value, "MAE_T_outer (C)": 77.7654321}))
            return {}, pd.DataFrame(rows)

        runs = [dict(arm=arm, seed=s, run_dir="synthetic")
                for arm in ("B1_mask", "B1_orig") for s in prereg["seeds"]["S10"]]
        report.score_run = synthetic_score
        try:
            check("incomplete B1 manifest refused", rejects(
                lambda: report.b6_report(runs[:-1], prereg, out_b6, "unused"), "exactly"))
            check("duplicate arm/seed B1 manifest refused before any scoring", rejects(
                lambda: report.b6_report(runs + [runs[0]], prereg, out_b6, "unused"), "exactly"))
            # Interrupt a window after intent but before report completion.
            try:
                with report.access_window("B6", "b6", {}, private, out_b6):
                    raise RuntimeError("synthetic interruption")
            except RuntimeError:
                pass
            state = report.read_json(os.path.join(private, "test_access_state.json"))
            check("interrupted window remains reserved and restores capability env",
                  state["B6"]["status"] == "running"
                  and not os.path.exists(os.path.join(private, "test_access.lock"))
                  and not os.environ.get("ROUND10_TEST_TOKEN"))
            check("reserved B6 refuses changed manifest", rejects(
                lambda: _enter_window("B6", "changed", private, out_b6), "different checkpoints"))
            check("reserved B6 refuses a different output directory", rejects(
                lambda: _enter_window("B6", "b6", private, os.path.join(tmp, "different_out")), "original report"))
            lock_path = os.path.join(private, "test_access.lock")
            report.atomic_json(lock_path, {"mode": "B6", "fingerprint": "b6", "pid": os.getpid()})
            check("active-process report lock refuses concurrent access", rejects(
                lambda: _enter_window("B6", "b6", private, out_b6), "already running")
                and os.path.isfile(lock_path))
            # Simulate a killed process only after checking active-PID refusal.
            report.atomic_json(lock_path, {"mode": "B6", "fingerprint": "b6", "pid": child.pid})
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                with report.access_window("B6", "b6", {}, private, out_b6) as ledger:
                    import eval_checkpoint as ec
                    original_eval = ec.evaluate_one

                    def noisy_synthetic_eval(run_dir, which, which_set, out_dir):
                        os.makedirs(out_dir, exist_ok=True)
                        pd.DataFrame([{"Case": "synthetic(%d)" % k, "CaseFile": "synthetic(%d)" % k,
                                       "MAE_T_avg (C)": 99.123456} for k in range(1, 71)]).to_csv(
                            os.path.join(out_dir, "summary_errors.csv"), index=False)
                        print("arbitrary level 99.123456")
                        print("arbitrary stderr level 77.7654321", file=sys.stderr)
                        return {"seed": 7, "T_avg": 99.123456}

                    ec.evaluate_one = noisy_synthetic_eval
                    try:
                        hidden_output = io.StringIO()
                        with contextlib.redirect_stdout(hidden_output), contextlib.redirect_stderr(hidden_output):
                            _, hidden_rows = original_score(dict(arm="B1_mask", seed=7, run_dir="synthetic"),
                                                            os.path.join(tmp, "noisy_synthetic_score"), ledger)
                        check("generic score output stays private inside B6",
                              hidden_output.getvalue() == "" and len(hidden_rows) == 70)
                    finally:
                        ec.evaluate_one = original_eval
                    result = report.b6_report(runs, prereg, out_b6, ledger)
                    report.atomic_json(os.path.join(out_b6, "report.json"), result)
            with open(os.path.join(out_b6, "B6.md"), encoding="utf-8") as f:
                markdown = f.read()
            saved = json.dumps(result)
            exposure = printed.getvalue() + markdown + saved
            check("B6 exposes exactly the registered four T_avg cases",
                  {r["case"] for r in result["selected"]} == {70, 42, 66, 67}
                  and set(result["remaining66"]) == {"ci_upper", "limit", "pass"}
                  and result["decision_changes_allowed"] is False)
            check("B6 remaining 66 and other channels have no arm levels in report",
                  "99.123" not in exposure and "77.765" not in exposure)
            check("B6 T_avg paired residual66 contrast has the expected upper bound",
                  abs(result["remaining66"]["ci_upper"] + 0.02) < 1e-9)
            check("all 20 synthetic score calls occur inside a logged capability", len(calls) == 20)
            check("completed B6 cannot be read again", rejects(
                lambda: _enter_window("B6", "b6", private, out_b6), "already completed"))
            # The second mode is allowed only after B6 and has its own token.
            with report.access_window("F11", "f11", {}, private, out_f11):
                check("F11 capability matches the registered second mode",
                      report.assert_test_access()["mode"] == "F11")
                report.atomic_json(os.path.join(out_f11, "report.json"), {"mode": "F11"})
            state = report.read_json(os.path.join(private, "test_access_state.json"))
            with open(os.path.join(private, "test_set_access.log"), encoding="utf-8") as f:
                events = [json.loads(line) for line in f if line.strip()]
            check("exactly two begin events, B6 and F11; identical resume is logged",
                  [(e["mode"], e["event"]) for e in events if e["event"] == "begin"]
                  == [("B6", "begin"), ("F11", "begin")]
                  and any(e["event"] == "resume" and e["mode"] == "B6" for e in events)
                  and state["B6"]["status"] == state["F11"]["status"] == "done")
            check("terminated-process lock recovers only for the same reserved window",
                  any(e["event"] == "recover_stale_lock" and e["mode"] == "B6" for e in events))
        finally:
            report.score_run = original_score

        missing_private = os.path.join(tmp, "missing_report_prereg")
        missing_out = os.path.join(tmp, "missing_report")
        missing_rejected = False
        try:
            with report.access_window("B6", "missing", {}, missing_private, missing_out):
                pass
        except FileNotFoundError:
            missing_rejected = True
        missing_state = report.read_json(os.path.join(missing_private, "test_access_state.json"))
        check("a missing output report cannot leave a completed window",
              missing_rejected and missing_state["B6"]["status"] == "running")

        # Fingerprints use actual dummy bytes, not only arm names/seed labels.
        run_dir = os.path.join(tmp, "dummy_run")
        os.makedirs(os.path.join(run_dir, "checkpoints"))
        os.makedirs(os.path.join(run_dir, "variants", report.VARIANT))
        cp = os.path.join(run_dir, "checkpoints", "best_gru_variant_%s.pth" % report.VARIANT)
        with open(cp, "wb") as f:
            f.write(b"fake checkpoint v1")
        with open(os.path.join(run_dir, "variants", report.VARIANT, "done.flag"), "w") as f:
            f.write("done")
        rc_path = os.path.join(run_dir, "run_config.json")
        report.atomic_json(rc_path, {"sealed": True, "seed": 7, "latest_params": {"max_epochs": 2000}})
        manifest = [dict(arm="B1_mask", seed=7, run_dir=run_dir)]
        prereg_path = os.path.join(HERE, "prereg", "round10_v3.json")
        fp1, _ = report.fingerprint_runs(manifest, prereg_path, "B6")
        with open(cp, "ab") as f:
            f.write(b"modified")
        fp2, _ = report.fingerprint_runs(manifest, prereg_path, "B6")
        check("checkpoint byte change changes the reserved manifest fingerprint", fp1 != fp2)
        report.atomic_json(rc_path, {"sealed": False, "seed": 7, "latest_params": {"max_epochs": 2000}})
        check("unsealed runs cannot enter a report", rejects(
            lambda: report.fingerprint_runs(manifest, prereg_path, "B6"), "Unsealed"))
        report.atomic_json(rc_path, {"sealed": True, "seed": 7, "latest_params": {"max_epochs": 2}})
        check("smoke checkpoints cannot enter a real report", rejects(
            lambda: report.fingerprint_runs(manifest, prereg_path, "B6"), "smoke"))

        import eval_checkpoint as ec
        check("generic test scorer refuses unregistered calls before run/data loading", rejects(
            lambda: ec.evaluate_one("missing_synthetic_run", "best", which_set="test"), "sealed"))
        check_source_seal(tmp, prereg)
    print("\ncheck_paper_report: %d/%d PASS" % (sum(RESULTS), len(RESULTS)))
    return 0 if all(RESULTS) else 1


def _enter_window(mode, digest, private, out):
    with report.access_window(mode, digest, {}, private, out):
        report.atomic_json(os.path.join(out, "report.json"), {"mode": mode})


if __name__ == "__main__":
    sys.exit(main())
