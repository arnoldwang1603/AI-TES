"""Meaningful CPU checks for the Round10 analysis interfaces and seal."""
import ast
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from analysis_common import CHANNELS, COLS, read_frames, split_spec, convention_map
from physics_reference import from_frame, fit_prefix, reference
from linear_sysid import fit_model, predict, score_frames
from error_budget import analyze_bundles
from g1_probe import select_shapes
from a7_bridge import discover_runs
from c16_curves import test_points, horizon_statistics


def main():
    failures = []
    def check(name, operation):
        try:
            assert operation()
            print("PASS " + name)
        except Exception as error:
            failures.append(name)
            print("FAIL %s:%s" % (name, error))
    modules = ("analysis_common", "linear_sysid", "c16_curves", "error_budget", "g1_probe",
               "a7_bridge", "e3_g11_shift", "f1_envelope", "physics_reference", "pack_round9_resources")
    check("analysis modules parse asPython3.11", lambda: all(
        ast.parse((HERE / (name + ".py")).read_text(encoding="utf-8"), feature_version=(3, 11)) is not None
        for name in modules))
    inlet = np.linspace(100, 300, 181)
    time = np.arange(181) * 10.0
    frame = pd.DataFrame({"Time (s)": time, "Input Temperature (C)": inlet})
    physical = from_frame(frame)
    original = reference.simulate(inlet, time)
    check("nominal solver exactlymatchesvendoredsource", lambda: np.array_equal(
        physical["T_outer_phys"], original["T_outer"]) and np.array_equal(
        physical["T_avg_line_phys"], original["T_avg_line"]))
    shifted = from_frame(frame, 0.8)
    estimate = fit_prefix(frame, shifted, 10)
    check("prefix fit recoverssynthetic0.8diffusivity", lambda: abs(estimate["k_scale"] - 0.8) < 1e-5)
    altered = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in shifted.items()}
    altered["T_outer_phys"][estimate["prefix_rows"]:] += 1000
    altered["T_avg_line_phys"][estimate["prefix_rows"]:] -= 1000
    check("G11 never usesobservedtailforfit", lambda: fit_prefix(frame, altered, 10) == estimate)
    frame["T_inner (C)"] = inlet
    frame["T_outer (C)"] = physical["T_outer_phys"]
    frame["T_avg (C)"] = physical["T_avg_line_phys"]
    second = frame.copy()
    second["Input Temperature (C)"] += 200
    second[list(COLS)] += 200
    training = {"Case 1/a.csv": frame, "Case 2/b.csv": second}
    model = fit_model(training, {"Case 2/b.csv"}, 8, 1e-4, True)
    changed = second.copy()
    changed["T_avg (C)"] += 1e9
    altered_model = fit_model({"Case 1/a.csv": frame, "Case 2/b.csv": changed}, {"Case 2/b.csv"}, 8, 1e-4, True)
    check("masked Tavg ignoredbyglobalandbinnedfits", lambda: model["global"]["T_avg"] == altered_model["global"]["T_avg"] and
          model["bins"]["3"]["T_avg"] == altered_model["bins"]["3"]["T_avg"])
    check("linear predict shapeandfinite", lambda: predict(model, frame).shape == (180, 3) and np.isfinite(predict(model, frame)).all())
    check("linear score excludesmaskedTavg", lambda: len(score_frames(model, training, {"Case 2/b.csv"})) == 5)
    truth = frame[list(COLS)].to_numpy()[1:]
    bundles = [{"arm": "synthetic", "seed": seed, "case_ids": ["Case 1/a.csv"],
                "actuals": [truth], "preds": [truth + sign]} for seed, sign in ((1, 1), (2, -1))]
    budget = analyze_bundles(bundles, {"Case 1/a.csv": frame}, {"Case 1/a.csv": physical},
                             {"Case 1/a.csv": {"label": "line average"}})["budget"]
    check("MSE identity andensemble-removablevariance", lambda: np.max(np.abs(budget.identity_error_C2)) < 1e-12 and
          np.allclose(budget.total_MSE_C2, 1) and np.allclose(budget.seed_variance_C2, 1) and
          np.allclose(budget.ensemble_MAE_C, 0))
    spec = split_spec()
    check("G1 uses14validationderivedshapes", lambda: len(select_shapes(spec)) == 14 and set(select_shapes(spec)) <= set(spec["val"]))
    try:
        read_frames(["Test_70_cases/Case(70).csv"])
        refused = False
    except ValueError:
        refused = True
    check("validation loader rejectstest path", lambda: refused)
    root = HERE / "round10_resources/round9"
    check("portableA7 contains60distinctcheckpoints", lambda: len(discover_runs(root)) == 60)
    labels = convention_map()
    check("F11yardstickmetadatahas67strictlineaveragefiles", lambda: sum(
        key.startswith("Test_70_cases/") and value.get("label") == "line average"
        for key, value in labels.items()) == 67)
    # Seventy named in-memory fixtures exercise F11's pure adapters.
    # The names are synthetic identifiers; no test CSV or npz is opened.
    test_frames, test_physics, test_labels = {}, {}, {}
    ids, actuals, predictions = [], [], []
    from error_budget import SWITCHED
    for number in range(1, 71):
        key = "Test_70_cases/syntheticCase(%d).csv" % number
        value = frame.copy()
        if number in SWITCHED:
            value["T_avg (C)"] = value[["T_inner (C)", "T_outer (C)"]].max(axis=1) + 0.2
        test_frames[key] = value
        test_physics[key] = physical
        test_labels[key] = {"label": "borderline" if number in (14, 26, 29) else "line average"}
        ids.append(key)
        actuals.append(value[list(COLS)].to_numpy()[1:])
        predictions.append(actuals[-1] + 1)
    fixture = [{"arm": "fixture", "seed": 1, "case_ids": ids, "actuals": actuals, "preds": predictions}]
    result = analyze_bundles(fixture, test_frames, test_physics, test_labels, which_set="test")
    check("F11providedframes67yardstickand21oracle strata", lambda:
          len(result["floors"][(result["floors"].channel == "T_avg") & result["floors"].strict_line_average]) == 67 and
          len(result["oracle"][result["oracle"].stratum == "both_phase21"]) == 21)
    refstats = pd.DataFrame([{"arm": "fixture", "channel": channel, "reference_residual_std_C": 1.0} for channel in CHANNELS])
    point = test_points(fixture, refstats)
    check("F11C16providedbundlesretaintrainingreferences", lambda: len(point) == 3 and
          np.allclose(point.reference_residual_std_C, 1) and set(point["set"]) == {"test"})
    check("F11bracketoracle positiveflooronclippedfixtures", lambda:
          np.allclose(result["oracle"][result["oracle"].stratum == "both_phase21"].bracket_clip_oracle_MAE_C, 0.2))
    print("15checks,%dFAIL; no realtestCSVread" % len(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
