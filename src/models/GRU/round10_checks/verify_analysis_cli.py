"""Functional CLI integration check on real ep2 validation checkpoints.

Runs after pipeline --smoke --keep-smoke. Every output is explicitly a smoke
artifact. E3/G1 truncate validation inputs only for this functional check;
their production defaults remain the complete registered populations.
No real test CSV is opened and no analysis output enters recipe decisions.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from round10_pipeline import full_arm_table, protocol

CSV_AUDIT_BOOTSTRAP = r'''
import json, os, runpy, sys
from pathlib import Path
script, target, *arguments = sys.argv[1:]
allowed_csv = set()
blocked_csv = []
def audit(event, arguments):
    if event != "open" or not arguments or not isinstance(arguments[0], (str, bytes)):
        return
    path = os.fsdecode(arguments[0]).replace("\\", "/")
    if path.lower().endswith(".csv"):
        if "test_70_cases" in path.lower():
            blocked_csv.append(path)
            raise PermissionError("sealed test CSV rejected by functional CLI audit")
        allowed_csv.add(path)
sys.addaudithook(audit)
sys.path.insert(0, str(Path(script).parent))
sys.argv = [script] + arguments
try:
    runpy.run_path(script, run_name="__main__")
finally:
    Path(target).mkdir(parents=True, exist_ok=True)
    (Path(target) / "csv_read_audit.json").write_text(json.dumps({
        "test_csv_attempts": len(blocked_csv), "blocked_test_csv": blocked_csv,
        "allowed_csv_unique": len(allowed_csv), "allowed_csv_paths": sorted(allowed_csv)
    }, indent=2), encoding="utf-8")
    if blocked_csv:
        raise AssertionError("functional CLI attempted sealed test access")
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    value = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    runs = value["runs"]
    if not runs:
        raise ValueError("no smoke checkpoints")
    for run in runs:
        config = json.loads((Path(run["run_dir"]) / "run_config.json").read_text(encoding="utf-8"))
        if int(config["latest_params"]["max_epochs"]) != 2 or not config.get("sealed"):
            raise ValueError("CLI check accepts sealed ep2 checkpoints only")
    table = full_arm_table(protocol())
    c16_names = {arm["arm"] for arm in table if arm["stage"] in ("S2", "S3", "S4", "S5")} | {"B1_mask", "B2_mask"}
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    selectors = {
        "c16_curves": lambda run: run["arm"] in c16_names,
        "error_budget": lambda run: run["arm"] in ("B1_mask", "B2_mask"),
        "g1_probe": lambda run: run["arm"] == "B2_mask",
        "e3_g11_shift": lambda run: run["arm"] in ("B2_mask", "C11"),
        "f1_envelope": lambda run: run["arm"].startswith("F1_") or run["arm"] == "B2_mask",
    }
    required_files = {
        "c16_curves": ["c16_reference_statistics.csv", "c16_validation_points.csv", "c16_head_ranges.csv", "c16_reference_quality.png", "c16_horizons.png"],
        "error_budget": ["error_budget.csv", "error_floors.csv", "error_oracle.csv", "error_budget.png"],
        "g1_probe": ["g1_probe.csv", "g1_switch_summary.csv", "g1_shift.png"],
        "e3_g11_shift": ["e3_g11_per_file.csv", "e3_g11_summary.csv", "e3_g11_shift.png"],
        "f1_envelope": ["f1_per_file.csv", "f1_envelope.csv", "f1_envelope_statement.json", "f1_envelope.png"],
    }
    results = []
    for script, selector in selectors.items():
        selected = [run for run in runs if selector(run)]
        if not selected:
            raise ValueError("no smoke checkpoints for%s" % script)
        subset = dict(value)
        subset["runs"] = selected
        subset["purpose"] = "ep2 functional smoke only; no experimental verdict"
        manifest_path = output / (script + "_manifest.json")
        manifest_path.write_text(json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8")
        target = output / script
        command = [sys.executable, "-c", CSV_AUDIT_BOOTSTRAP,
                   str(HERE / (script + ".py")), str(target),
                   "--manifest", str(manifest_path), "--out", str(target)]
        if script == "g1_probe":
            command.extend(["--max-shapes", "1"])
        if script == "e3_g11_shift":
            command.extend(["--max-files", "1"])
        result = subprocess.run(command, cwd=str(HERE), capture_output=True, text=True, encoding="utf-8", errors="replace")
        (output / (script + ".log")).write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
        if result.returncode:
            print(result.stdout[-1500:] + result.stderr[-3500:])
            raise RuntimeError("analysis CLI failed:%s" % script)
        for name in required_files[script]:
            if not (target / name).is_file() or not (target / name).stat().st_size:
                raise RuntimeError("analysis artifact missing:%s" % (target / name))
        audit = json.loads((target / "csv_read_audit.json").read_text(encoding="utf-8"))
        if audit["test_csv_attempts"]:
            raise RuntimeError("sealed test CSV access attempted:%s" % script)
        results.append({"script": script, "status": "PASS", "n_smoke_runs": len(selected),
                        "test_csv_attempts": audit["test_csv_attempts"],
                        "allowed_csv_unique": audit["allowed_csv_unique"],
                        "outputs": required_files[script]})
        print("PASS %s realep2validation CLI (%dcheckpoints)" % (script, len(selected)), flush=True)
    (output / "analysis_cli_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("5CLIchecks,0FAIL; functional smoke only; no realtestCSVread", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
