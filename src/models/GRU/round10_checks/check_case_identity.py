"""Synthetic regression for full-path validation identity (no training/test reads).

Use the registered 20 validation path labels and S20 seeds, but invent every
measurement. Six S1b arms make 120 synthetic CSVs with the real basename
collisions. All three summary-to-long readers must preserve case identity.
"""
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_h9 import block_test_csv_reads
import paper_stats as stats
import round10_decisions as decisions
import round10_pipeline as pipeline

ARMS = ("B1_mask", "B1_orig", "B2_mask", "B2_orig", "B7", "B8")
PAIR = [("B1_mask", "B1_orig")]
RESULTS = []


def check(label, action):
    try:
        action()
    except Exception as exc:
        RESULTS.append(False)
        print("FAIL  %s: %s: %s" % (label, type(exc).__name__, exc), flush=True)
    else:
        RESULTS.append(True)
        print("PASS  " + label, flush=True)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def reject(action, phrase):
    try:
        action()
    except (ValueError, KeyError, AttributeError) as exc:
        require(phrase in str(exc), "expected %r in %r" % (phrase, str(exc)))
    else:
        raise AssertionError("invalid case identifiers were accepted")


def canonical(long):
    return stats.normalise_long(long).sort_values(
        ["arm", "seed", "file", "channel"]).reset_index(drop=True)


def main():
    block_test_csv_reads()
    with open(HERE / "prereg/round10_v4.json", encoding="utf-8") as stream:
        prereg = json.load(stream)
    with open(HERE / "splits/round10_vs3_s42.json", encoding="utf-8") as stream:
        split = json.load(stream)
    # data._case_id emits data_root-relative identifiers, including train_subdir;
    # split and preregistration mask lists remain train_dir-relative.
    paths = [split["train_subdir"] + "/" + key for key in split["val"]]
    seeds = prereg["seeds"]["S20"]
    masked = {split["train_subdir"] + "/" + key for key in prereg["data"]["val_tavg_masked"]}
    require(len(paths) == len(set(paths)) == 20 and len(seeds) == 20,
            "fixture requires the registered 20 cases and S20")
    require(len({Path(key).name for key in paths}) < 20,
            "fixture must contain duplicate basenames")
    require(len(masked) == 3 and masked <= set(paths), "fixture mask differs")
    scratch = HERE / "runs/_round10_check_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="case_identity_", dir=str(scratch)) as temporary:
        root = Path(temporary)
        records = []
        for arm_index, arm in enumerate(ARMS):
            for seed in seeds:
                run = root / ("%s_seed%d" % (arm, seed))
                variant = run / "variants/forward_direct"
                variant.mkdir(parents=True)
                rows = []
                for index, key in enumerate(paths):
                    row = dict(Case=key, CaseFile=Path(key).name)
                    for channel_index, channel in enumerate(stats.CHANNELS):
                        value = (0.2 + channel_index * 0.1 + index * 0.001
                                 + seed * 0.00001 + arm_index * 0.01 * (channel_index + 1))
                        row["MAE_%s (C)" % channel] = (
                            np.nan if channel == "T_avg" and key in masked else value)
                    row.update({channel: 0 for channel in stats.JUMP_COLS})
                    rows.append(row)
                pd.DataFrame(rows).to_csv(variant / "summary_errors_val.csv", index=False)
                records.append(dict(arm=arm, seed=seed, run_dir=str(run)))
        require(len(records) == 120, "fixture must reproduce the 120-run S1b panel")
        saved = {}

        def validate(name, long, tidy=None):
            normal = canonical(long)
            require(len(normal) == 120 * 20 * 5, "long table lost rows")
            require(set(normal.file) == set(paths), "reader reduced Case to basename")
            require(not normal.duplicated(["arm", "seed", "file", "channel"]).any(),
                    "duplicate statistical key")
            for arm in ARMS:
                for channel, count in zip(stats.CHANNELS, (20, 20, 17)):
                    matrix = stats.seed_file_matrix(normal, arm, channel)
                    require(matrix.shape == (20, count),
                            "%s %s matrix is %r" % (arm, channel, matrix.shape))
            if tidy is None:
                tidy, _ = stats.compare_family(long, PAIR, family="case_identity", seed=0)
            rows = tidy.set_index("channel")
            for channel, count, effect in zip(stats.CHANNELS, (20, 20, 17), (-0.01, -0.02, -0.03)):
                require(int(rows.loc[channel, "n_files"]) == count, "incorrect channel count")
                require(int(rows.loc[channel, "n_seeds"]) == 20, "incorrect seed count")
                require(abs(rows.loc[channel, "diff_mean"] - effect) < 1e-12,
                        "paired effect differs from synthetic truth")
            require(abs(rows.loc["ToTa", "diff_mean"] + 0.025) < 1e-12,
                    "ToTa differs from synthetic truth")
            saved[name] = normal

        def pipeline_check():
            tidy, _, long = pipeline.comparison(records, set(ARMS), PAIR,
                "case_identity", str(root / "pipeline"))
            validate("pipeline", long, tidy)

        check("pipeline comparison: 120 runs, distinct full paths, 20/20/17, known paired effects", pipeline_check)
        check("decision comparison: 120 runs, distinct full paths, 20/20/17, known paired effects",
              lambda: validate("decisions", decisions.long_for(records, ARMS)))
        check("CLI run reader: 120 runs, distinct full paths, 20/20/17, known paired effects",
              lambda: validate("cli", stats.long_table_from_runs(
                  {arm: arm for arm in ARMS}, str(root), seeds)))

        def equal_readers():
            require(set(saved) == {"pipeline", "decisions", "cli"}, "one reader failed")
            pd.testing.assert_frame_equal(saved["pipeline"], saved["decisions"])
            pd.testing.assert_frame_equal(saved["pipeline"], saved["cli"])

        check("all three readers preserve identical per-case values and NaN masks", equal_readers)
        row = dict(arm="B1_mask", seed=seeds[0], file=paths[0], channel="T_inner", MAE=0.2)
        check("true duplicate full-path statistical keys fail explicitly",
              lambda: reject(lambda: stats.normalise_long(pd.DataFrame([row, row])), "Duplicate"))

        # Missing the full identity must not silently fall back to CaseFile.
        first = Path(records[0]["run_dir"]) / "variants/forward_direct/summary_errors_val.csv"
        frame = pd.read_csv(first)
        frame.drop(columns=["Case"]).to_csv(first, index=False)
        check("pipeline refuses a summary missing Case", lambda: reject(
            lambda: pipeline.comparison(records, set(ARMS), PAIR, "missing", str(root / "missing")), "Case"))
        check("decision reader refuses a summary missing Case", lambda: reject(
            lambda: decisions.long_for(records, ARMS), "Case"))
        check("CLI run reader refuses a summary missing Case", lambda: reject(
            lambda: stats.long_table_from_runs({arm: arm for arm in ARMS}, str(root), seeds), "Case"))
    print("\ncheck_case_identity: %d/%d PASS; synthetic measurements only; no training or test access" %
          (sum(RESULTS), len(RESULTS)))
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
