"""The two registered Round 10 test-access windows (B6 and F11).

Validation, smoke and synthetic probes must never request this capability.
An interrupted window resumes its identical manifest; it cannot be reopened
with different checkpoints. Intent is logged before any test CSV is read.
"""
import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import re
import secrets
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CHANNELS = ("T_inner", "T_outer", "T_avg")
VARIANT = "forward_direct"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def atomic_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temp = path + ".tmp.%s" % os.getpid()
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(json_safe(obj), f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item"):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def pid_alive(pid):
    pid = int(pid)
    # Locks name a single process. POSIX kill(0, 0) queries our process
    # group, and negative IDs query groups too; neither is a lock owner.
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        kernel = ctypes.windll.kernel32
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = kernel.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel.CloseHandle(handle)
        return bool(ok and code.value == 259)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def append_event(path, event):
    event = dict(event, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def assert_test_access():
    """Called at the data-reader and checkpoint-scoring boundaries."""
    token = os.environ.get("ROUND10_TEST_TOKEN", "")
    lock = os.environ.get("ROUND10_TEST_LOCK", "")
    mode = os.environ.get("ROUND10_TEST_MODE", "")
    if not token or mode not in ("B6", "F11") or not os.path.isfile(lock):
        raise RuntimeError("Test data are sealed. Use paper_report.py --b6 or --f11.")
    lease = read_json(lock)
    if lease.get("token") != token or lease.get("mode") != mode:
        raise RuntimeError("Invalid Round 10 test-access capability")
    ledger = lease.get("ledger", "")
    if not os.path.isfile(ledger):
        raise RuntimeError("Test access has no write-ahead ledger")
    with open(ledger, encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]
    if not any(e.get("event") == "begin" and e.get("fingerprint") == lease.get("fingerprint")
               and e.get("mode") == mode for e in events):
        raise RuntimeError("Test access was not registered before the read")
    return lease


def test_data_root(data_root=None):
    return os.path.abspath(data_root or os.environ.get("DATA_ROOT") or
                           os.path.join(HERE, "..", "..", "..", "data"))


def seal_test_source(prereg_dir, data_root=None):
    """Fix source bytes at the first B6 read, then verify every resumed window.

    The source path is recorded for provenance. The relative names and byte
    hashes define identity, so an identical dataset can move between hosts.
    This helper must never be called by preflight or dry-run code.
    """
    lease = assert_test_access()  # Check the capability before any listing/read.
    prereg_dir = os.path.abspath(prereg_dir)
    if prereg_dir != os.path.dirname(os.path.abspath(lease["ledger"])):
        raise RuntimeError("Test source seal must use the current access ledger")
    root = test_data_root(data_root)
    test = os.path.join(root, "Test_70_cases")
    if not os.path.isdir(test):
        raise RuntimeError("Frozen test source directory is missing")
    names = sorted(name for name in os.listdir(test) if name.lower().endswith(".csv"))
    # The model loader discovers CSVs recursively, whereas the frozen F11
    # source is a flat 70-file set. Refuse hidden extra files before scoring.
    for directory, _, children in os.walk(test):
        if os.path.abspath(directory) != test and any(name.lower().endswith(".csv") for name in children):
            raise RuntimeError("Frozen test source cannot contain nested CSV files")
    if any(not os.path.isfile(os.path.join(test, name)) for name in names):
        raise RuntimeError("Frozen test source CSV entry is not a file")
    ids = [re.search(r"\((\d+)\)", name) for name in names]
    if len(names) != 70 or any(match is None for match in ids) or {
            int(match.group(1)) for match in ids if match is not None} != set(range(1, 71)):
        raise RuntimeError("Test source seal requires each of the frozen 70 case IDs exactly once")
    files = {"Test_70_cases/" + name: sha256(os.path.join(test, name)) for name in names}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    path = os.path.join(prereg_dir, "test_source_seal.json")
    if os.path.isfile(path):
        saved = read_json(path)
        saved_digest = hashlib.sha256(json.dumps(saved.get("files", {}), sort_keys=True,
            separators=(",", ":")).encode("utf-8")).hexdigest()
        if saved.get("version") != 1 or saved_digest != saved.get("content_sha256"):
            raise RuntimeError("Frozen test source seal is invalid or changed")
        if saved.get("files") != files or saved_digest != digest:
            raise RuntimeError("Frozen test source files changed; refusing mixed-source resume")
        event = "test_source_verified"
    else:
        if lease["mode"] != "B6":
            raise RuntimeError("F11 requires the source seal from the first B6 window")
        state = read_json(os.path.join(prereg_dir, "test_access_state.json"))
        cache = os.path.join(state["B6"]["out"], "restricted_pairs")
        if os.path.isdir(cache) and any(name.endswith(".json") for name in os.listdir(cache)):
            raise RuntimeError("Existing B6 scores have no test source seal; refusing an unverifiable resume")
        saved = dict(version=1, source_root=root, files=files, content_sha256=digest)
        atomic_json(path, saved)
        event = "test_source_sealed"
    append_event(lease["ledger"], dict(event=event, mode=lease["mode"],
        fingerprint=lease["fingerprint"], content_sha256=digest, n_files=len(files)))
    return saved


def validate_f11_manifest(manifest):
    if any(key in manifest for key in ("physics_params", "conventions")):
        raise RuntimeError("F11 reports require the packaged frozen physics parameters and conventions")


def fingerprint_runs(runs, prereg_path, mode, recipe=None, analysis_inputs=None):
    payload = dict(mode=mode, prereg_sha256=sha256(prereg_path), recipe=recipe, runs=[])
    if mode == "F11":
        if not analysis_inputs:
            raise RuntimeError("F11 requires frozen validation-analysis input identities")
        payload["analysis_inputs"] = analysis_inputs
    for r in sorted(runs, key=lambda x: (x["arm"], int(x["seed"]))):
        path = os.path.abspath(r["run_dir"])
        cp = os.path.join(path, "checkpoints", "best_gru_variant_%s.pth" % VARIANT)
        rc = os.path.join(path, "run_config.json")
        config = read_json(rc)
        if r["arm"].startswith("R9_"):
            pass  # immutable, disclosed historical comparison; no new training
        elif (not config.get("sealed") or int(config["latest_params"]["max_epochs"]) == 2):
            raise RuntimeError("Unsealed or smoke checkpoint cannot enter %s: %s" % (mode, path))
        if int(config["seed"]) != int(r["seed"]):
            raise RuntimeError("Manifest seed disagrees with checkpoint config: " + path)
        if not os.path.isfile(os.path.join(path, "variants", VARIANT, "done.flag")):
            raise RuntimeError("Incomplete run: " + path)
        files = {"checkpoint": sha256(cp), "run_config": sha256(rc)}
        sidecar = cp[:-4] + ".constants.json"
        if os.path.isfile(sidecar):
            files["sidecar"] = sha256(sidecar)
        payload["runs"].append(dict(arm=r["arm"], seed=int(r["seed"]), files=files))
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest, payload


def f11_input_identities(manifest):
    root = manifest["analyses_root"]
    paths = {"linear_models": os.path.join(root, "linear_sysid", "linear_sysid_models.json"),
             "training_reference_stats": os.path.join(root, "c16_curves", "c16_reference_statistics.csv"),
             "physics_params": os.path.join(HERE, "round10_resources", "physics", "params.json"),
             "conventions": os.path.join(HERE, "splits", "tavg_convention_by_file.csv")}
    for name in ("paper_report.py", "round10_f11.py", "round10_refit.py", "physics_reference.py", "radial1d_reference.py",
                 "linear_sysid.py", "error_budget.py", "c16_curves.py", "analysis_common.py", "paper_stats.py"):
        paths["code/" + name] = os.path.join(HERE, name)
    return dict(files={key: sha256(path) for key, path in paths.items()},
                confirmed=sorted(manifest.get("confirmed", [])),
                data_root=test_data_root(manifest.get("data_root")))


@contextlib.contextmanager
def access_window(mode, digest, payload, prereg_dir, out):
    ledger = os.path.join(prereg_dir, "test_set_access.log")
    state_path = os.path.join(prereg_dir, "test_access_state.json")
    lock = os.path.join(prereg_dir, "test_access.lock")
    os.makedirs(prereg_dir, exist_ok=True)
    # O_EXCL prevents parallel reports; locks are never silently removed.
    if os.path.isfile(lock):
        previous = read_json(lock)
        if pid_alive(previous.get("pid", 0)):
            raise RuntimeError("A test report is already running")
        if previous.get("fingerprint") != digest or previous.get("mode") != mode:
            raise RuntimeError("Stale test lock belongs to a different manifest")
        os.unlink(lock)
        append_event(ledger, dict(event="recover_stale_lock", mode=mode, fingerprint=digest))
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    token = secrets.token_hex(32)
    lease = dict(token=token, mode=mode, pid=os.getpid(), ledger=ledger, fingerprint=digest)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(lease, f)
        f.flush()
        os.fsync(f.fileno())
    keys = ("ROUND10_TEST_TOKEN", "ROUND10_TEST_MODE", "ROUND10_TEST_LOCK")
    old = {k: os.environ.get(k) for k in keys}
    try:
        state = read_json(state_path) if os.path.isfile(state_path) else {}
        prior = state.get(mode)
        if mode == "F11" and state.get("B6", {}).get("status") != "done":
            raise RuntimeError("F11 requires a completed B6 window")
        if prior and prior.get("fingerprint") != digest:
            raise RuntimeError("%s was already reserved for different checkpoints; refusing another look" % mode)
        if prior and os.path.abspath(out) != prior.get("out"):
            raise RuntimeError("Resume must use the original report output directory")
        if prior and prior.get("status") == "done":
            raise RuntimeError("%s already completed; use the saved report at %s" % (mode, prior["out"]))
        if not prior:
            append_event(ledger, dict(event="begin", mode=mode, fingerprint=digest, payload=payload))
            state[mode] = dict(status="running", fingerprint=digest, out=os.path.abspath(out))
            atomic_json(state_path, state)
        else:
            append_event(ledger, dict(event="resume", mode=mode, fingerprint=digest))
        os.environ.update(ROUND10_TEST_TOKEN=token, ROUND10_TEST_MODE=mode, ROUND10_TEST_LOCK=lock)
        yield ledger
        report_digest = sha256(os.path.join(out, "report.json"))
        state[mode]["status"] = "done"
        state[mode]["report_sha256"] = report_digest
        atomic_json(state_path, state)
        append_event(ledger, dict(event="complete", mode=mode, fingerprint=digest,
                                  report_sha256=report_digest))
    except BaseException as e:
        append_event(ledger, dict(event="failure", mode=mode, fingerprint=digest,
                                  error=type(e).__name__ + ": " + str(e)))
        raise
    finally:
        for k in keys:
            if old[k] is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old[k]
        os.unlink(lock)


def score_run(run, dest, ledger, physics_overrides=None):
    append_event(ledger, dict(event="score_intent", mode=os.environ["ROUND10_TEST_MODE"],
                              arm=run["arm"], seed=int(run["seed"])))
    # Generic eval output contains arbitrary test levels: it stays in this
    # private buffer and is not printed in the restricted B6 report.
    import eval_checkpoint
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        kwargs = dict(which_set="test", out_dir=dest)
        if physics_overrides is not None:
            kwargs["physics_overrides"] = physics_overrides
        agg = eval_checkpoint.evaluate_one(run["run_dir"], "best", **kwargs)
    import pandas as pd
    rows = pd.read_csv(os.path.join(dest, "summary_errors.csv"))
    rows = rows[rows["Case"].astype(str) != "AVERAGE"].copy()
    rows["test_id"] = rows["CaseFile"].astype(str).map(
        lambda s: int(re.search(r"\((\d+)\)", s).group(1)))
    if len(rows) != 70 or set(rows["test_id"]) != set(range(1, 71)):
        raise RuntimeError("Expected the frozen 70-case test set")
    return agg, rows


def paired_summary(a, b, seed=0):
    import numpy as np
    from paper_stats import paired_stats
    return paired_stats(np.asarray(a, float) - np.asarray(b, float), np.random.default_rng(seed))


def b6_report(runs, prereg, out, ledger):
    import numpy as np
    import pandas as pd
    seeds = list(map(int, prereg["seeds"]["S10"]))
    lookup = {(r["arm"], int(r["seed"])): r for r in runs}
    expected = {(a, s) for a in ("B1_mask", "B1_orig") for s in seeds}
    if set(lookup) != expected or len(runs) != len(expected) or len(lookup) != len(runs):
        raise RuntimeError("B6 requires exactly B1_mask/B1_orig on every S10 seed")
    # Read numeric thresholds from the actual registered B6 rule.
    rule = prereg["test_reads"]["B6"]
    thresholds = {int(k): float(v) for k, v in re.findall(r"(?:case\s*)?(70|42)\s*<=\s*([0-9.]+)", rule)}
    m = re.search(r"66/67\s*<=\s*([0-9.]+)", rule)
    if m:
        thresholds.update({66: float(m.group(1)), 67: float(m.group(1))})
    limit = re.search(r"upper\s*<=\s*\+([0-9.]+)", rule)
    if set(thresholds) != {70, 42, 66, 67} or not limit:
        raise RuntimeError("Cannot interpret registered B6 rule")
    cache = os.path.join(out, "restricted_pairs")
    os.makedirs(cache, exist_ok=True)
    records = []
    for s in seeds:
        cached = os.path.join(cache, "seed%d.json" % s)
        if os.path.isfile(cached):
            records.append(read_json(cached))
            continue
        with tempfile.TemporaryDirectory(prefix="b6_private_") as tmp:
            _, mask = score_run(lookup[("B1_mask", s)], os.path.join(tmp, "mask"), ledger)
            _, orig = score_run(lookup[("B1_orig", s)], os.path.join(tmp, "orig"), ledger)
            mask = mask.set_index("test_id")
            orig = orig.set_index("test_id")
            col = "MAE_T_avg (C)"
            selected = sorted(thresholds)
            remaining = sorted(set(range(1, 71)) - set(selected))
            row = dict(seed=s, selected={str(k): {"mask": float(mask.loc[k, col]),
                        "orig": float(orig.loc[k, col])} for k in selected},
                       rest66_difference=float((mask.loc[remaining, col] - orig.loc[remaining, col]).mean()))
            atomic_json(cached, row)
            records.append(row)
    table = []
    for k in sorted(thresholds):
        a = [r["selected"][str(k)]["mask"] for r in records]
        b = [r["selected"][str(k)]["orig"] for r in records]
        stats = paired_summary(a, b)
        table.append(dict(case=k, mask_T_avg=float(np.mean(a)), orig_T_avg=float(np.mean(b)),
                          diff=stats["mean"], ci_lo=stats["lo"], ci_hi=stats["hi"],
                          threshold=thresholds[k], threshold_pass=bool(np.mean(a) <= thresholds[k])))
    rest = paired_summary([r["rest66_difference"] for r in records], np.zeros(len(records)))
    report = dict(mode="B6", rule=rule, selected=table,
                  remaining66={"ci_upper": rest["hi"], "limit": float(limit.group(1)),
                               "pass": rest["hi"] <= float(limit.group(1))},
                  decision_changes_allowed=False)
    pd.DataFrame(table).to_csv(os.path.join(out, "B6_selected.csv"), index=False)
    lines = ["# B6 registered first look", "", "T_avg MAE (C), mask minus orig; no recipe changes.", ""]
    for r in table:
        lines.append("Case {case}: mask {mask_T_avg:.4f}, orig {orig_T_avg:.4f}; difference {diff:.4f} "
                     "[{ci_lo:.4f}, {ci_hi:.4f}], threshold {threshold:.2f}: {threshold_pass}".format(**r))
    lines.append("\nRemaining 66: paired-difference CI upper %.5f <= +%.3f: %s" % (
        rest["hi"], float(limit.group(1)), report["remaining66"]["pass"]))
    with open(os.path.join(out, "B6.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return report


def f11_report(runs, prereg, manifest, out, ledger):
    import numpy as np
    import pandas as pd
    from paper_stats import paired_stats
    recipe = manifest.get("frozen_recipe")
    if not recipe:
        raise RuntimeError("F11 requires the written frozen_recipe from S8")
    fresh = list(map(int, prereg["seeds"]["F9_fresh_primes"]))
    actual = sorted(int(r["seed"]) for r in runs if r["arm"] == "F9")
    if actual != sorted(fresh):
        raise RuntimeError("F11 requires all 20 F9 fresh seeds")
    if recipe.get("arm", "direct") not in ("direct", "B1_mask"):
        if sorted(int(r["seed"]) for r in runs if r["arm"] == "F10") != sorted(fresh):
            raise RuntimeError("A challenger recipe requires the paired F10 direct baseline")
    # Physics inputs/residual references must be generated inside this
    # registered window before hybrid checkpoints can be scored. The
    # shipped cache deliberately has no sealed-test entries.
    frames = read_f11_frames(manifest.get("data_root"))
    from physics_reference import from_frame
    params = read_json(os.path.join(HERE, "round10_resources", "physics", "params.json"))
    physics = {key: from_frame(frame, params=params) for key, frame in frames.items()}
    scores, bundles = [], []
    for run in runs:
        dest = os.path.join(out, "scored", "%s_seed%s" % (run["arm"], run["seed"]))
        cached = os.path.join(dest, "aggregate.json")
        if os.path.isfile(cached):
            agg = read_json(cached)
        else:
            cfg = read_json(os.path.join(run["run_dir"], "run_config.json"))
            overrides = physics if cfg.get("phys_mode") else None
            agg, _ = score_run(run, dest, ledger, physics_overrides=overrides)
            atomic_json(cached, agg)
        scores.append(dict(agg, arm=run["arm"]))
        z = np.load(os.path.join(dest, "predictions.npz"), allow_pickle=True)
        bundles.append(dict(arm=run["arm"], seed=run["seed"], case_ids=list(z["case_ids"]),
                            actuals=list(z["inv_actuals"]), preds=list(z["inv_preds"])))
    table = pd.DataFrame(scores)
    table.to_csv(os.path.join(out, "F11_per_seed.csv"), index=False)
    means = table.groupby("arm")[list(CHANNELS)].mean().reset_index()
    means.to_csv(os.path.join(out, "F11_channel_means.csv"), index=False)
    uncertainty = []
    from paper_stats import N_BOOT, CI_PERCENTILES
    for arm_name, subset in table.groupby("arm"):
        for ch in CHANNELS:
            levels = subset[ch].to_numpy(float)
            seed_ci = paired_stats(levels, np.random.default_rng(0))
            case_matrix = []
            for r in runs:
                if r["arm"] != arm_name:
                    continue
                csv_path = os.path.join(out, "scored", "%s_seed%s" % (r["arm"], r["seed"]), "summary_errors.csv")
                cases = pd.read_csv(csv_path)
                cases = cases[cases.Case.astype(str) != "AVERAGE"].sort_values("CaseFile")
                case_matrix.append(cases["MAE_%s (C)" % ch].to_numpy(float))
            matrix = np.stack(case_matrix)
            rng = np.random.default_rng(0)
            boot = []
            for start in range(0, N_BOOT, 256):
                n = min(256, N_BOOT - start)
                si = rng.integers(0, matrix.shape[0], size=(n, matrix.shape[0]))
                fi = rng.integers(0, matrix.shape[1], size=(n, matrix.shape[1]))
                boot.extend(matrix[si[:, :, None], fi[:, None, :]].mean(axis=(1, 2)).tolist())
            lo, hi = np.percentile(boot, CI_PERCENTILES)
            uncertainty.append(dict(arm=arm_name, channel=ch, mean=float(levels.mean()),
                seed_sd=float(levels.std(ddof=1)), n_seeds=len(levels), n_files=matrix.shape[1],
                seed_ci_lo=seed_ci["lo"], seed_ci_hi=seed_ci["hi"],
                seed_case_ci_lo=float(lo), seed_case_ci_hi=float(hi),
                interpretation="Descriptive levels; seed+case resampling does not select a recipe"))
    pd.DataFrame(uncertainty).to_csv(os.path.join(out, "F11_uncertainty.csv"), index=False)
    comparisons = []
    for a, b in (("F9", "F10"), ("B1_mask", "B1_orig"), ("B1_orig", "RETRAIN"), ("RETRAIN", "R9_twin")):
        left = table[table.arm == a].set_index("seed")
        right = table[table.arm == b].set_index("seed")
        common = sorted(set(left.index) & set(right.index))
        if len(common) < 3:
            continue
        for ch in CHANNELS:
            stats = paired_stats(left.loc[common, ch].values - right.loc[common, ch].values,
                                 np.random.default_rng(0))
            comparisons.append(dict(armA=a, armB=b, channel=ch, **stats))
    pd.DataFrame(comparisons).to_csv(os.path.join(out, "F11_chain_differences.csv"), index=False)
    # Test-side physics, linear rows, C16, floors and ensembles are produced
    # here by the analysis adapter, never by independent analysis CLIs.
    from round10_f11 import finish_f11
    analyses = finish_f11(manifest, bundles, out, frames)
    report = dict(mode="F11", recipe=recipe, means=means.to_dict("records"), uncertainty=uncertainty,
                  comparisons=comparisons, analyses=analyses,
                  disclosures=prereg["deviations_and_disclosures"])
    with open(os.path.join(out, "F11.md"), "w", encoding="utf-8") as f:
        f.write("# F11 frozen final report\n\n")
        for r in means.to_dict("records"):
            f.write("%s: Tin %.5f, Tout %.5f, Tavg %.5f C\n\n" % (
                r["arm"], r["T_inner"], r["T_outer"], r["T_avg"]))
        f.write("## Disclosures\n\n" + "\n".join("- " + d for d in report["disclosures"]) + "\n")
    return report


def read_f11_frames(data_root=None):
    """The sole raw-test reader supplies frames to pure-array analyses."""
    assert_test_access()
    import pandas as pd
    root = test_data_root(data_root)
    test = os.path.join(root, "Test_70_cases")
    frames = {}
    for name in sorted(os.listdir(test)):
        if not name.lower().endswith(".csv"):
            continue
        frame = pd.read_csv(os.path.join(test, name))
        frame.columns = [str(c).strip() for c in frame.columns]
        frame.rename(columns={"T_ave (C)": "T_avg (C)"}, inplace=True)
        if "Input Temperature (C)" not in frame:
            frame["Input Temperature (C)"] = frame["T_inner (C)"]
        if frame[["Time (s)", "T_inner (C)", "T_outer (C)", "T_avg (C)", "Input Temperature (C)"]].isna().any().any():
            raise RuntimeError("Non-finite frozen test file: " + name)
        frames["Test_70_cases/" + name] = frame
    if len(frames) != 70:
        raise RuntimeError("F11 requires exactly70 test files")
    return frames


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--b6", action="store_true")
    mode.add_argument("--f11", action="store_true")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--prereg", default=os.path.join(HERE, "prereg", "round10_v3.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    prereg = read_json(a.prereg)
    import paper_stats
    paper_stats.configure_policy(prereg)
    manifest = read_json(a.manifest)
    runs = manifest["runs"]
    name = "B6" if a.b6 else "F11"
    if name == "F11":
        validate_f11_manifest(manifest)
    analysis_inputs = f11_input_identities(manifest) if name == "F11" else None
    digest, payload = fingerprint_runs(runs, a.prereg, name, manifest.get("frozen_recipe"), analysis_inputs)
    if a.dry_run:
        print("%s: %d immutable checkpoints, %s; no test access" % (name, len(runs), digest))
        return
    os.makedirs(a.out, exist_ok=True)
    state_path = os.path.join(os.path.dirname(a.prereg), "test_access_state.json")
    prior = read_json(state_path).get(name, {}) if os.path.isfile(state_path) else {}
    if prior.get("status") == "done" and prior.get("fingerprint") == digest:
        saved = os.path.join(prior["out"], "report.json")
        if not os.path.isfile(saved) or sha256(saved) != prior.get("report_sha256"):
            raise RuntimeError("Completed test report is missing or changed; cannot reopen the test window")
        print("%s already complete: %s" % (name, prior["out"]))
        return
    # Bind both the source seal and checkpoint loader to the manifest's root.
    # No test bytes are read until access_window has recorded its begin event.
    root = test_data_root(manifest.get("data_root"))
    old_root = os.environ.get("DATA_ROOT")
    os.environ["DATA_ROOT"] = root
    try:
        with access_window(name, digest, payload, os.path.dirname(a.prereg), a.out) as ledger:
            seal_test_source(os.path.dirname(a.prereg), root)
            report = b6_report(runs, prereg, a.out, ledger) if a.b6 else f11_report(runs, prereg, manifest, a.out, ledger)
            seal_test_source(os.path.dirname(a.prereg), root)
            atomic_json(os.path.join(a.out, "report.json"), report)
    finally:
        if old_root is None:
            os.environ.pop("DATA_ROOT", None)
        else:
            os.environ["DATA_ROOT"] = old_root
    print("%s complete: %s" % (name, os.path.abspath(a.out)))


if __name__ == "__main__":
    main()
