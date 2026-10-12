"""Check generated F4/F1 split coverage, grouping and frozen constants."""
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from make_round10_splits import digest, near_groups, verify
from round10_integrity import matches_frozen_sha256


def main():
    split_dir = HERE / "splits"
    base = json.loads((split_dir / "round10_vs3_s42.json").read_text(encoding="utf-8"))
    audit = json.loads((split_dir / "round10_split_generation_audit.json").read_text(encoding="utf-8"))
    universe = sorted(base["train"] + base["val"])
    failures = []
    for name, expected_sha in audit["files"].items():
        try:
            path = split_dir / name
            assert matches_frozen_sha256(path, expected_sha), "split hash changed"
            spec = json.loads(path.read_text(encoding="utf-8"))
            verify(spec, audit["protocol"]["groups"], universe)
            assert spec["drop"] == base["drop"] and spec["tavg_mask"] == base["tavg_mask"]
            if spec.get("heldout_family"):
                assert spec["frozen_train"] == sorted(base["train"])
                assert spec["frozen_constants_split"] == "round10_vs3_s42.json"
            else:
                counts = [sum(k.startswith(family + "/") for k in spec["val"]) for family in base["types"]]
                assert counts == [2] * 10
            print("PASS " + name)
        except Exception as error:
            failures.append(name)
            print("FAIL %s: %s" % (name, error))
    synthetic = {"a": pd.DataFrame({"Input Temperature (C)": [0.0, 0.0]}),
                 "b": pd.DataFrame({"Input Temperature (C)": [0.7, 0.7]}),
                 "c": pd.DataFrame({"Input Temperature (C)": [1.4, 1.4]})}
    if near_groups(["a", "b", "c"], synthetic) == [["a", "b", "c"]]:
        print("PASS conservative near-duplicate graph transitive closure")
    else:
        failures.append("graph")
        print("FAIL near-duplicate graph transitive closure")
    data_root = Path(os.environ.get("DATA_ROOT", HERE.parents[2] / "data")) / base["train_subdir"]
    unchanged = all(matches_frozen_sha256(data_root / key, expected) for key, expected in audit["protocol"]["input_csv_sha256"].items())
    if unchanged:
        print("PASS all304source CSV hashes unchanged")
    else:
        failures.append("source-hashes")
        print("FAIL source CSV hashes")
    print("%dchecks,%dFAIL" % (14, len(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
