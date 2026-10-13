"""A7 legacy scaler metadata regression, using synthetic arrays only."""
from pathlib import Path
import copy
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import a7_bridge as bridge
from analysis_common import BASE_COLS, COLS


def main():
    passed = 0

    def report(label, condition):
        nonlocal passed
        if not condition:
            raise AssertionError(label)
        passed += 1
        print("PASS " + label)

    # Different outer/inner/average ranges make the output order observable:
    # BASE_COLS is [time, outer, inner, average, inlet], while predictions are
    # [inner, outer, average]. Different file lengths also check equal-file
    # averaging, rather than accidentally pooling all timesteps.
    training = pd.DataFrame([[0, 0, 0, 0, 0], [10, 100, 400, 50, 500]], columns=BASE_COLS)
    unnamed = MinMaxScaler().fit(training.to_numpy())
    named = MinMaxScaler().fit(training)
    # Explicit assignment also exercises the sidecar case on sklearn versions
    # that do not add this optional attribute during DataFrame fitting.
    named.feature_names_in_ = np.asarray(BASE_COLS, dtype=object)
    if hasattr(unnamed, "feature_names_in_"):
        del unnamed.feature_names_in_
    frames = {
        "synthetic/short.csv": pd.DataFrame(np.zeros((3, 3)), columns=COLS),
        "synthetic/shorter.csv": pd.DataFrame(np.zeros((2, 3)), columns=COLS),
    }
    predictions = {
        "synthetic/short.csv": np.asarray([[4, -10, 5], [-8, 20, -15]], dtype=float),
        "synthetic/shorter.csv": np.asarray([[12, -30, 25]], dtype=float),
    }
    # Analytic scaled weighted MAE: first file = 3.03 / 6; second = 3.33 / 3.
    expected = (0.505 + 1.11) / 2

    def score(scaler):
        session = {"scaler": scaler, "run_config": {"loss_weights": [1, 6, 3]}}
        with patch.object(bridge, "predict_frame", side_effect=lambda _, frame, key: predictions[key]):
            with patch.object(scaler, "fit", side_effect=AssertionError("scaler must not be refitted")):
                return bridge.legacy_validation_scalar(session, frames)

    original = copy.deepcopy(unnamed.__dict__)
    unnamed_score = score(unnamed)
    report("unnamed scaler reproduces analytic weighted validation MAE", np.isclose(unnamed_score, expected, atol=1e-14, rtol=0))
    named_score = score(named)
    report("named scaler gives the identical scalar", named_score == unnamed_score)
    report("scaler metadata and fitted arrays remain unchanged",
           set(unnamed.__dict__) == set(original)
           and all(np.array_equal(unnamed.__dict__[key], value) for key, value in original.items()))
    oldest = copy.deepcopy(unnamed)
    if hasattr(oldest, "n_features_in_"):
        del oldest.n_features_in_
    report("optional n_features_in_ may also be absent", score(oldest) == unnamed_score)

    def rejected(label, scaler):
        try:
            score(scaler)
        except ValueError as error:
            report(label, str(error).startswith("A7 scaler"))
        else:
            raise AssertionError(label + ": invalid scaler was accepted")

    wrong_order = copy.deepcopy(named)
    wrong_order.feature_names_in_[[1, 2]] = wrong_order.feature_names_in_[[2, 1]]
    rejected("unexpected named column order refused", wrong_order)
    wrong_names = copy.deepcopy(named)
    wrong_names.feature_names_in_ = np.asarray(BASE_COLS[:-1], dtype=object)
    rejected("incomplete named schema refused", wrong_names)
    wrong_count = copy.deepcopy(unnamed)
    wrong_count.n_features_in_ = 4
    rejected("conflicting feature count refused", wrong_count)
    for label, scales in (
        ("short scale vector", [1, 2, 3, 4]),
        ("two-dimensional scale vector", [[1, 2, 3, 4, 5]]),
        ("nonfinite scale", [1, 2, np.nan, 4, 5]),
        ("infinite scale", [1, 2, np.inf, 4, 5]),
        ("zero scale", [1, 2, 0, 4, 5]),
        ("negative scale", [1, 2, -3, 4, 5]),
    ):
        broken = copy.deepcopy(unnamed)
        broken.scale_ = np.asarray(scales, dtype=float)
        rejected(label + " refused", broken)
    print("check_a7_scaler: %d/%d PASS; synthetic arrays only, no training or test access" % (passed, passed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
