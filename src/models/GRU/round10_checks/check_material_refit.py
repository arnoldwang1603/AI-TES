"""CPU fixtures verify the PI material-refit column reaches saved tables.

All metrics and temperature series below are synthetic. No checkpoint,
training operation or real sealed test CSV is opened.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from analysis_common import write_table
import c16_curves as c16
import paper_stats as stats
import round10_decisions as decisions
from round10_refit import REFIT_COLUMN, channel_refit_metadata, read_refit_rules


RESULTS = []
CHANNELS = ("T_inner", "T_outer", "T_avg")
RULE_SOURCE = "batches.P3.re_fit_column"


def check(label, operation):
    try:
        assert operation()
        RESULTS.append(True)
        print("PASS  " + label, flush=True)
    except Exception as error:
        RESULTS.append(False)
        print("FAIL  %s: %s" % (label, error), flush=True)


def refused(operation):
    try:
        operation()
    except (ValueError, RuntimeError):
        return True
    return False


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config(**updates):
    value = dict(tinner_mode="anchor", other_ch_mode="abs", touter_mode="abs",
                 phys_mode="", ratio_channels="R3", ratio_ref="kelvin",
                 anchor_lead=True, seed=7)
    value.update(updates)
    return value


def main():
    prereg_path = HERE / "prereg/round10_v3.json"
    before = sha(prereg_path)
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    # This independent parsing checks the actual frozen source, not constants
    # copied into the helper or into this test.
    registered = dict(part.strip().split(":", 1)
                      for part in prereg["batches"]["P3"]["re_fit_column"].split(";"))
    registered = {key.strip(): value.strip() for key, value in registered.items()}
    check("five refit requirements are read verbatim from the frozen prereg",
          lambda: read_refit_rules() == registered and len(registered) == 5)
    check("prereg path and prereg object give the same parsed requirements",
          lambda: read_refit_rules(prereg_path) == read_refit_rules(prereg) == registered)
    modified = copy.deepcopy(prereg)
    modified["batches"]["P3"]["re_fit_column"] = "; ".join(
        "%s: fixture_%s" % (key, index) for index, key in enumerate(registered))
    modified_rules = {key: "fixture_%s" % index for index, key in enumerate(registered)}
    check("changed prereg requirements propagate instead of using hardcoded values",
          lambda: read_refit_rules(modified) == modified_rules and
          channel_refit_metadata(config(), modified)["T_inner"][REFIT_COLUMN] == modified_rules["anchor"])
    missing = copy.deepcopy(prereg)
    missing["batches"]["P3"]["re_fit_column"] = "; ".join(
        "%s: %s" % (key, value) for key, value in registered.items() if key != "EMA")
    duplicate = copy.deepcopy(prereg)
    duplicate["batches"]["P3"]["re_fit_column"] += "; direct: conflicting"
    check("missing registered output form is refused", lambda: refused(lambda: read_refit_rules(missing)))
    check("duplicate registered output form is refused", lambda: refused(lambda: read_refit_rules(duplicate)))

    specs = {
        "B1_mask": (config(), ("anchor", "direct", "direct")),
        "B1_orig": (config(), ("anchor", "direct", "direct")),
        "RETRAIN": (config(), ("anchor", "direct", "direct")),
        "C2": (config(other_ch_mode="anchor_initial"), ("anchor", "anchor", "anchor")),
        "C3": (config(touter_mode="anchor_input"), ("anchor", "anchor", "direct")),
        "R1": (config(other_ch_mode="ratio_input", ratio_channels="R1"), ("anchor", "ratio", "direct")),
        "R2": (config(other_ch_mode="ratio_input", ratio_channels="R2"), ("anchor", "ratio", "ratio")),
        "R2C": (config(other_ch_mode="ratio_input", ratio_channels="R2", ratio_ref="celsius20"), ("anchor", "ratio", "ratio")),
        "R3": (config(other_ch_mode="ratio_input", ratio_channels="R3", tinner_mode="arfed"), ("ratio", "ratio", "ratio")),
        "C11": (config(phys_mode="residual"), ("anchor", "physics residual", "physics residual")),
        "C12": (config(phys_mode="input_residual"), ("anchor", "physics residual", "physics residual")),
        "C13": (config(phys_mode="input"), ("anchor", "direct", "direct")),
        "EMA_description": (config(touter_mode="anchor_ema"), ("anchor", "EMA", "direct")),
        "C14": (config(tinner_mode="arfed"), ("direct", "direct", "direct")),
    }
    for arm, (cfg, forms) in specs.items():
        def correct_mapping(cfg=cfg, forms=forms):
            metadata = channel_refit_metadata(cfg, prereg)
            return set(metadata) == set(CHANNELS) and all(
                metadata[channel]["output_form"] == form and
                metadata[channel][REFIT_COLUMN] == registered[form] and
                metadata[channel]["refit_rule_source"] == RULE_SOURCE
                for channel, form in zip(CHANNELS, forms))
        check(arm + " maps the actual channel output forms", correct_mapping)
    check("physics-as-input direct heads still disclose the physical input update",
          lambda: all(row["physics_input_refit"] == registered["physics residual"]
                      for row in channel_refit_metadata(specs["C13"][0], prereg).values()))
    check("plain direct heads do not acquire a physical-input requirement",
          lambda: all(row["physics_input_refit"] == registered["direct"]
                      for row in channel_refit_metadata(config(), prereg).values()))
    check("residual-only heads list the physical base in the main requirement without adding physical inputs",
          lambda: all(row["physics_input_refit"] == registered["direct"]
                      for row in channel_refit_metadata(specs["C11"][0], prereg).values()) and
          channel_refit_metadata(specs["C11"][0], prereg)["T_outer"][REFIT_COLUMN] == registered["physics residual"])
    check("physical-input residual heads expose both reference and input maintenance",
          lambda: all(row["physics_input_refit"] == registered["physics residual"]
                      for row in channel_refit_metadata(specs["C12"][0], prereg).values()))

    original_reader = pd.read_csv
    def guarded_csv(path, *args, **kwargs):
        name = getattr(path, "name", path)
        if "test_70_cases" in str(name).replace("\\", "/").lower():
            raise RuntimeError("material refit checks cannot open sealed test CSVs")
        return original_reader(path, *args, **kwargs)

    parent = HERE / "runs/_round10_check_scratch"
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="material_refit_", dir=parent) as temporary, \
            patch.object(pd, "read_csv", side_effect=guarded_csv):
        root = Path(temporary)
        records = []
        for arm, (cfg, _) in specs.items():
            for seed in (1, 2, 7):
                run = root / (arm + "_seed%d" % seed)
                variant = run / "variants/forward_direct"
                variant.mkdir(parents=True)
                own_config = dict(cfg, seed=seed, data={"split_name": "synthetic_split.json"})
                (run / "run_config.json").write_text(json.dumps(own_config), encoding="utf-8")
                rows = []
                for index in range(3):
                    row = dict(Case="synthetic_val_%d" % index, CaseFile="Case 1/synthetic_val_%d.csv" % index)
                    for number, channel in enumerate(CHANNELS):
                        row["MAE_%s (C)" % channel] = .1 + number * .2 + index * .03 + seed * .001
                    if index == 2:
                        row["MAE_T_avg (C)"] = np.nan
                    rows.append(row)
                pd.DataFrame(rows).to_csv(variant / "summary_errors_val.csv", index=False)
                records.append(dict(arm=arm, seed=seed, run_dir=str(run), config=own_config))
        p1 = [row for row in records if row["arm"] in ("B1_mask", "B1_orig", "RETRAIN")]
        p1_out = root / "P1"
        p1_out.mkdir()
        decisions.descriptive_tables(p1, str(p1_out), "P1", prereg=prereg)
        table = pd.read_csv(p1_out / "P1_arm_levels.csv")
        markdown = (p1_out / "P1_arm_levels.md").read_text(encoding="utf-8")
        check("P1 actual CSV contains all three arms and their per-channel refit requirements",
              lambda: len(table) == 9 and set(table.arm) == {"B1_mask", "B1_orig", "RETRAIN"} and
              all(row[REFIT_COLUMN] == registered["anchor" if row.channel == "T_inner" else "direct"]
                  for _, row in table.iterrows()))
        check("P1 actual Markdown exports the requested column and channel requirements",
              lambda: REFIT_COLUMN in markdown and registered["anchor"] in markdown and
              registered["direct"] in markdown and all(arm in markdown for arm in ("B1_mask", "B1_orig", "RETRAIN")))
        guide = pd.read_csv(p1_out / "material_refit_guide.csv")
        guide_markdown = (p1_out / "material_refit_guide.md").read_text(encoding="utf-8")
        check("five-form guide exports EMA as a protocol note without fabricating model metrics",
              lambda: len(guide) == 5 and set(guide.output_form) == set(registered) and
              guide.set_index("output_form")[REFIT_COLUMN].to_dict() == registered and
              REFIT_COLUMN in guide_markdown and registered["EMA"] in guide_markdown and
              not any("MAE" in column for column in guide.columns) and
              guide.status.str.contains("protocol annotation", regex=False).all())
        expected = []
        for name in sorted(set(row["arm"] for row in p1)):
            data = decisions.rows_for(p1, name)
            for channel in CHANNELS:
                means = data.groupby("seed")["MAE_%s (C)" % channel].mean()
                value = stats.paired_stats(means.values, np.random.default_rng(0))
                expected.append(dict(arm=name, channel=channel, mean=value["mean"],
                                     ci_lo=value["lo"], ci_hi=value["hi"], n_seeds=len(means),
                                     n_files=int(data.groupby("seed")["MAE_%s (C)" % channel].count().min())))
        pd.testing.assert_frame_equal(table[list(expected[0])], pd.DataFrame(expected), check_exact=False,
                                      rtol=1e-14, atol=1e-14)
        check("P1 refit annotations preserve original numerical readouts", lambda: True)
        annotated_out = root / "custom_source"
        annotated_out.mkdir()
        decisions.descriptive_tables(p1, str(annotated_out), "P1", prereg=modified)
        custom = pd.read_csv(annotated_out / "P1_arm_levels.csv")
        check("saved P1 output uses the supplied prereg source values",
              lambda: set(custom[REFIT_COLUMN]) == {modified_rules["anchor"], modified_rules["direct"]})
        incompatible = copy.deepcopy(p1)
        changed_record = next(row for row in incompatible if row["arm"] == "B1_mask" and row["seed"] == 7)
        altered_cfg = config(tinner_mode="arfed", seed=7)
        changed_record["config"] = altered_cfg
        path = Path(changed_record["run_dir"]) / "run_config.json"
        preserved = path.read_bytes()
        path.write_text(json.dumps(altered_cfg), encoding="utf-8")
        try:
            check("same arm with inconsistent output forms across seeds is refused in P1",
                  lambda: refused(lambda: decisions.descriptive_tables(incompatible, str(p1_out), "bad", prereg=prereg)))
        finally:
            path.write_bytes(preserved)

        frame = pd.DataFrame({"Time (s)": np.arange(5) * 10.0,
                              "Input Temperature (C)": [100., 120., 150., 130., 180.],
                              "T_inner (C)": [99., 119., 149., 129., 179.],
                              "T_outer (C)": [80., 85., 96., 101., 120.],
                              "T_avg (C)": [90., 96., 113., 110., 143.]})
        train_key = "Case 1/synthetic_training.csv"
        val_key = "Case 2/synthetic_validation.csv"
        training = {train_key: frame}
        physical = {train_key: {"T_outer_phys": frame["T_outer (C)"].to_numpy() - .2,
                                "T_avg_line_phys": frame["T_avg (C)"].to_numpy() - .3}}
        reference = c16.reference_statistics(records, training, physical, set(), prereg=prereg)
        check("C16 training references carry actual-channel metadata for all synthetic arms",
              lambda: len(reference) == len(specs) * 3 and
              all(row[REFIT_COLUMN] == registered[row.output_form] for _, row in reference.iterrows()))
        check("C16 rejects inconsistent per-seed output forms before accepting one arm's reference",
              lambda: refused(lambda: c16.reference_statistics(incompatible, training, physical, set(), prereg=prereg)))
        truth = frame[["T_inner (C)", "T_outer (C)", "T_avg (C)"]].to_numpy()[1:]
        bundles = [dict(arm=record["arm"], seed=record["seed"], case_ids=[val_key],
                        actuals=[truth], preds=[truth + record["seed"] * .01]) for record in records]
        points = c16.points(bundles, reference)
        check("validation point join preserves the PI refit column",
              lambda: len(points) == len(specs) * 3 and
              points[REFIT_COLUMN].notna().all() and points["refit_rule_source"].eq(RULE_SOURCE).all())
        selected = reference[reference.arm == "C11"].copy()
        selected["reference_arm"] = selected["arm"]
        selected["arm"] = "F9"
        alias_bundles = [dict(bundle, arm="F9") for bundle in bundles if bundle["arm"] == "C11"]
        f11 = c16.test_points(alias_bundles, selected)
        numeric_columns = [column for column in selected.select_dtypes(include="number").columns]
        check("F11 F9 alias join inherits frozen C11 metadata without refitting",
              lambda: len(f11) == 3 and f11["refit_rule_source"].eq(RULE_SOURCE).all() and
              f11.set_index("channel").loc["T_outer", REFIT_COLUMN] == registered["physics residual"] and
              f11.set_index("channel").loc["T_inner", REFIT_COLUMN] == registered["anchor"])
        check("F11 alias annotations preserve model errors and frozen training statistics",
              lambda: np.allclose(f11.model_MAE_C, np.mean([.01, .02, .07])) and
              np.array_equal(f11.set_index("channel").sort_index()[numeric_columns].to_numpy(),
                             selected.set_index("channel").sort_index()[numeric_columns].to_numpy()))
        legacy = selected.drop(columns=[column for column in (
            REFIT_COLUMN, "output_form", "physics_input_refit", "refit_rule_source") if column in selected])
        legacy_points = c16.test_points(alias_bundles, legacy)
        check("old pure-memory F11 fixtures stay numerically compatible without metadata",
              lambda: np.array_equal(legacy_points.model_MAE_C, f11.model_MAE_C))

        spec = dict(train=[train_key], val=[val_key], tavg_mask=[])
        analysis_out = root / "C16"
        with patch.object(c16, "manifest_context", return_value=(None, None, None)), \
                patch.object(c16, "load_manifest", return_value=records), \
                patch.object(c16, "load_val_bundles", return_value=bundles), \
                patch.object(c16, "split_spec", return_value=spec), \
                patch.object(c16, "read_frames", return_value={train_key: frame, val_key: frame}), \
                patch.object(c16, "cache_for_keys", return_value=physical), \
                patch.object(c16, "setup_run", return_value=None), \
                patch.object(c16, "predict_frame", return_value=(truth, np.zeros_like(truth))):
            c16.run_analysis({"runs": records}, analysis_out)
        for name in ("c16_reference_statistics", "c16_validation_points"):
            exported = pd.read_csv(analysis_out / (name + ".csv"))
            text = (analysis_out / (name + ".md")).read_text(encoding="utf-8")
            check(name + " actual CSV and Markdown contain the requested column",
                  lambda exported=exported, text=text: REFIT_COLUMN in exported.columns and
                  exported[REFIT_COLUMN].notna().all() and REFIT_COLUMN in text and
                  all(value in text for value in registered.values()))
        write_table(f11, analysis_out, "C16_test_points", "Synthetic in-memory F11 adapter only.")
        check("F11 actual CSV and Markdown retain the inherited column",
              lambda: REFIT_COLUMN in pd.read_csv(analysis_out / "C16_test_points.csv").columns and
              REFIT_COLUMN in (analysis_out / "C16_test_points.md").read_text(encoding="utf-8"))
        check("sealed real-test CSV paths are rejected by the verification guard",
              lambda: refused(lambda: pd.read_csv("Test_70_cases/never_opened.csv")))

    check("frozen prereg bytes are unchanged after every table-generation check",
          lambda: sha(prereg_path) == before)
    print("\ncheck_material_refit: %d/%d PASS; CPU synthetic only, no real test CSV read" %
          (sum(RESULTS), len(RESULTS)), flush=True)
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
