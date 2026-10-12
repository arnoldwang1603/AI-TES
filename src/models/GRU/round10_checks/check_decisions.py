"""Synthetic validation readouts test the registered decision predicates.

No model loading, training, test CSVs or real measurement readouts are used.
"""
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(HERE)
from check_h9 import block_test_csv_reads
import round10_decisions as decisions
import round10_pipeline as pipeline
from paper_report import atomic_json, read_json

RESULTS = []


def check(label, ok):
    RESULTS.append(bool(ok))
    print("%s  %s" % ("PASS" if ok else "FAIL", label), flush=True)


def main():
    block_test_csv_reads()
    p = read_json(os.path.join(HERE, "prereg", "round10_v4.json"))
    parent = os.path.join(HERE, "runs", "_round10_check_scratch")
    os.makedirs(parent, exist_ok=True)
    records = []
    with tempfile.TemporaryDirectory(prefix="decisions_", dir=parent) as tmp:
        def add_arm(name, metrics, pattern=None, params=100):
            for seed in p["seeds"]["S10"]:
                run = os.path.join(tmp, name + "_seed%d" % seed)
                dest = os.path.join(run, "variants", "forward_direct")
                os.makedirs(dest, exist_ok=True)
                rows = []
                for index in range(20):
                    row = dict(Case="synthetic/family%d/val%d.csv" % (index // 10, index % 10),
                               CaseFile="val%d.csv" % (index % 10),
                               JumpSteps_gt2C_ToTa=0, JumpSteps_gt1C_ToTa=0)
                    for ch, value in zip(decisions.CH, metrics):
                        row["MAE_%s (C)" % ch] = value
                        row["Spike_gt1C_%s" % ch] = 0
                        row["MaxErr_1_5_%s (C)" % ch] = 2.0
                    if pattern:
                        row.update(pattern(index))
                    if index >= 17:
                        for column in ("MAE_T_avg (C)", "Spike_gt1C_T_avg", "MaxErr_1_5_T_avg (C)"):
                            row[column] = np.nan
                    rows.append(row)
                pd.DataFrame(rows).to_csv(os.path.join(dest, "summary_errors_val.csv"), index=False)
                atomic_json(os.path.join(dest, "meta.json"), {"Encoder_Params": params})
                records.append(dict(arm=name, seed=seed, run_dir=run))

        add_arm("B2_mask", (0.1, 0.7, 0.7))
        add_arm("C11", (0.1, 0.4, 0.3))
        add_arm("C12", (0.1, 0.35, 0.25))
        rows, verdict = decisions.compare(records, "C11", "B2_mask", p)
        check("CSV MAE readouts are normalised and produce the registered BETTER verdict",
              verdict["verdict"] == "BETTER" and rows["T_avg"]["ci_hi"] < 0)
        check("C11 must beat the pure-physics E1 row on both relevant channels",
              decisions.physics_adoption(records, "C11", p))
        check("C12 qualifies only with a qualifying C11 and a ToTa improvement over it",
              decisions.physics_adoption(records, "C12", p))
        # An apparently good C12 must not bypass a C11 that fails its E1 gate.
        saved = records
        records = [r for r in saved if r["arm"] != "C11"]
        add_arm("C11", (0.1, 0.6, 0.5))
        check("C12 cannot bypass C11's E1 gate", not decisions.physics_adoption(records, "C12", p))
        records = saved

        add_arm("P4_baseline", (0.1, 0.4, 0.3),
                lambda i: {"Spike_gt1C_T_outer": float(i < 10), "Spike_gt1C_T_avg": float(i < 10)})
        add_arm("ENC_linear", (0.1, 0.4, 0.3),
                lambda i: {"Spike_gt1C_T_outer": float(i < 10), "Spike_gt1C_T_avg": float(i >= 10)}, params=20)
        allowed, detail = decisions.p4_adoption(records, "ENC_linear", p, tmp, baseline="P4_baseline")
        check("joint first-five spike share respects masked T_avg and rejects disjoint channel spikes",
              not allowed and abs(detail["spike_difference"]["mean"] - 0.35) < 1e-12)
        add_arm("NO_TIME", (0.1, 0.4, 0.3))
        allowed, detail = decisions.p4_adoption(records, "NO_TIME", p, tmp, baseline="P4_baseline")
        check("NO_TIME equivalence is an adoption predicate without requiring a BETTER verdict",
              allowed and detail["verdict"]["verdict"] == "EQUIVALENT")
        add_arm("C17_NO_TIME", (0.1, 0.4, 0.3))
        add_arm("B1_mask", (0.1, 0.4, 0.3))
        allowed, detail = decisions.p4_adoption(records, "C17_NO_TIME", p, tmp)
        check("NO_TIME full-width confirmation uses the B1_mask baseline and the same P4 rule", allowed)

        add_arm("H8_a6_l0", (0.04, 0.7, 0.7))
        tidy, verdicts, _ = pipeline.comparison(records, {"H8_a6_l0", "B2_mask"},
            [("H8_a6_l0", "B2_mask")], "anchor_grid", os.path.join(tmp, "P2"))
        allowed, detail = pipeline.adoption_flags(tidy, verdicts, "H8_a6_l0", "S2", {"runs": records}, tmp)
        check("P2 evaluates its registered overall/T_inner predicate independently of generic BETTER",
              allowed and detail["verdict"]["verdict"] != "BETTER")

        add_arm("B1_orig", (0.1, 0.41, 0.31))
        add_arm("B2_orig", (0.1, 0.72, 0.72))
        add_arm("B7", (0.1, 0.72, 0.72))
        add_arm("B8", (0.1, 0.71, 0.71))
        original_side = decisions.side_scores
        levels = {"B1_mask": 1.0, "B1_orig": 3.0, "B2_mask": 1.0, "B2_orig": 3.0, "B7": 3.0, "B8": 1.5}
        def synthetic_side(_records):
            return pd.DataFrame([dict(arm=name, seed=seed, MAE=value) for name, value in levels.items()
                                 for seed in p["seeds"]["S10"]])
        decisions.side_scores = synthetic_side
        try:
            evidence = decisions.p1_readouts(records, p, os.path.join(tmp, "P2"))
            check("P1 labels and stopping-control interpretation use paired hot-discharge scores",
                  evidence["B1_mask"]["primary_pass"] and evidence["B2_mask"]["primary_pass"]
                  and evidence["random_mask_does_not_lower"] and evidence["B8_fraction"] == 0.75
                  and evidence["interpretation"] == "label content plus stopping rule")
            levels["B7"] = 2.0
            evidence = decisions.p1_readouts(records, p, os.path.join(tmp, "P2"))
            check("a random mask that significantly lowers the side score fails the P1 control",
                  not evidence["random_mask_does_not_lower"])
        finally:
            decisions.side_scores = original_side

        # Two redraw wins are sufficient without an uncertainty-band reversal.
        add_arm("R3", (0.1, 0.6, 0.6))
        add_arm("F4_direct_43", (0.1, 0.7, 0.7))
        add_arm("F4_direct_44", (0.1, 0.7, 0.7))
        add_arm("F4_R3_43", (0.1, 0.6, 0.6))
        add_arm("F4_R3_44", (0.1, 0.71, 0.71))
        chosen = decisions.freeze_recipe(records, "R3", p, os.path.join(tmp, "F4"))
        check("F4 adopts a capacity-qualified challenger with two of three wins", chosen["arm"] == "R3" and chosen["wins"] == 2)
        # A large reversal forces the preregistered three-draw mean check.
        saved = records
        records = [r for r in saved if r["arm"] != "F4_R3_44"]
        add_arm("F4_R3_44", (0.1, 1.0, 1.0))
        chosen = decisions.freeze_recipe(records, "R3", p, os.path.join(tmp, "F4_reversal"))
        check("F4 large rank reversal cannot be rescued by win count when three-draw mean is worse",
              chosen["arm"] == "direct" and chosen["large_rank_flip"] and chosen["wins"] == 2)
        records = saved
        direct = decisions.freeze_recipe(records, None, p, os.path.join(tmp, "F4_direct"))
        check("absence of a confirmed challenger freezes direct without fabricating comparisons", direct["arm"] == "direct")
    print("\ncheck_decisions: %d/%d PASS" % (sum(RESULTS), len(RESULTS)))
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
