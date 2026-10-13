"""P4 startup gate must cover the production T0 grid with finite errors.

Synthetic evaluator output only; no training, checkpoint or data reads.
"""
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).parent))
from check_h9 import block_test_csv_reads
import eval_checkpoint
import round10_decisions as decisions
from paper_report import read_json

CHECKS = []


def check(name, value):
    CHECKS.append(bool(value))
    print(("PASS " if value else "FAIL ") + name, flush=True)


def main():
    block_test_csv_reads()
    p = read_json(str(HERE / "prereg/round10_v4.json"))
    seeds = p["seeds"]["S20"]
    records = [dict(arm="G7a", seed=seed, run_dir="synthetic_seed%d" % seed) for seed in seeds]
    def readouts(_records, name):
        return pd.DataFrame([dict(seed=seed, Spike_gt1C_T_outer=0, Spike_gt1C_T_avg=0,
                                 **{"MaxErr_1_5_T_outer (C)": .1 if name == "G7a" else 2.,
                                    "MaxErr_1_5_T_avg (C)": .1 if name == "G7a" else 2.}) for seed in seeds])
    comparison = ({channel: dict(ci_hi=0., band=.1) for channel in decisions.CH}, dict(vetoed=False))
    def production_shape(run, sweep, which, out_dir):
        seed = int(run.split("seed")[-1])
        start, stop, step = map(float, sweep.split(":"))
        # Actual eval_checkpoint.t0_sweep output columns; use its requested grid.
        return pd.DataFrame([dict(seed=seed, T0=float(t0), channel=channel,
                                  Err_step1=.1, MaxErr_1_5=.1, MaxOffset_60=.1,
                                  MeanOffset_60=.1, Spike_gt1C=0.)
                             for t0 in np.arange(start, stop + .5 * step, step) for channel in decisions.CH])
    def invalid_offset(value):
        def mutate(frame):
            frame.loc[0, "MaxOffset_60"] = value
            return frame
        return mutate
    cases = [
        ("finite full grid passes", lambda frame: frame, True),
        ("one finite offset above registered bound fails", invalid_offset(.50001), False),
        ("exact registered bound passes", invalid_offset(.5), True),
        ("one NaN offset fails", invalid_offset(np.nan), False),
        ("one positive infinite offset fails", invalid_offset(np.inf), False),
        ("one negative infinite offset fails", invalid_offset(-np.inf), False),
        ("missing T0/channel row fails", lambda frame: frame.iloc[1:].copy(), False),
        ("missing entire channel fails", lambda frame: frame[frame.channel != "T_avg"].copy(), False),
        ("duplicated row replacing a required observation fails", lambda frame: pd.concat([frame.iloc[1:], frame.iloc[[1]]], ignore_index=True), False),
        ("empty grid fails", lambda frame: frame.iloc[:0].copy(), False),
        ("wrong checkpoint seed fails", lambda frame: frame.assign(seed=-1), False),
        ("missing producer column fails", lambda frame: frame.drop(columns=["T0"]), False),
        ("negative absolute error fails", invalid_offset(-.1), False),
    ]
    parent = HERE / "runs/_round10_check_scratch"
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="p4_t0_contract_", dir=str(parent)) as temporary, \
            patch.object(decisions, "compare", return_value=comparison), \
            patch.object(decisions, "encoder_params", return_value=100), \
            patch.object(decisions, "rows_for", side_effect=readouts):
        for index, (label, mutate, expected) in enumerate(cases):
            dest = Path(temporary) / str(index)
            def evaluator(*args, **kwargs):
                frame = production_shape(*args, **kwargs)
                # Corrupt just one seed, so other complete seeds cannot hide it.
                return mutate(frame) if int(frame.seed.iloc[0]) == seeds[0] else frame
            with patch.object(eval_checkpoint, "t0_sweep", side_effect=evaluator) as evaluate:
                allowed, detail = decisions.p4_adoption(records, "G7a", p, str(dest))
            check(label, allowed is expected and detail["startup_pass"] is expected and evaluate.call_count == len(seeds))
            with patch.object(eval_checkpoint, "t0_sweep", side_effect=AssertionError("cached probe must not rerun")):
                cached, replay = decisions.p4_adoption(records, "G7a", p, str(dest))
            check("cached path: " + label, cached is expected and replay["startup_pass"] is expected)
            if not detail["t0_sweep_complete"]:
                check("incomplete/nonfinite readout has explicit diagnostics: " + label,
                      detail["max_t0_offset"] is None and len(detail["t0_sweep_issues"]) == 1
                      and detail["t0_sweep_issues"][0]["seed"] == seeds[0])
    print("check_p4_t0_contract: %d/%d PASS" % (sum(CHECKS), len(CHECKS)))
    return 0 if all(CHECKS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
