import numpy as np
import pandas as pd
import pytest

from src.constants import ID_COL, LABELS
from src.pseudo.evaluate import evaluate_extraction
from src.pseudo.schema import validate_pseudo_labels


def _frame(values: np.ndarray, ids: list[str] | None = None) -> pd.DataFrame:
    df = pd.DataFrame(values, columns=list(LABELS))
    df.insert(0, ID_COL, ids or [f"1.2.{i}" for i in range(len(df))])
    return df


def _truth(n: int = 20, seed: int = 0) -> pd.DataFrame:
    y = np.random.default_rng(seed).integers(0, 2, size=(n, len(LABELS))).astype(float)
    y[0], y[1] = 0.0, 1.0
    return _frame(y)


# --- schema -------------------------------------------------------------------


def test_hard_soft_and_abstain_values_are_valid() -> None:
    values = np.full((3, len(LABELS)), np.nan)
    values[0], values[1, :] = 1.0, 0.25  # 양성, soft label, 나머지는 판단 보류(NaN)
    validate_pseudo_labels(_frame(values))


def test_out_of_range_value_fails() -> None:
    values = np.zeros((2, len(LABELS)))
    values[0, 0] = 1.5
    with pytest.raises(ValueError, match="range"):
        validate_pseudo_labels(_frame(values))


def test_wrong_columns_fail() -> None:
    df = _frame(np.zeros((2, len(LABELS)))).drop(columns=[LABELS[-1]])
    with pytest.raises(ValueError, match="column"):
        validate_pseudo_labels(df)


def test_duplicate_ids_fail() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        validate_pseudo_labels(_frame(np.zeros((2, len(LABELS))), ids=["a", "a"]))


# --- evaluate -----------------------------------------------------------------


def test_perfect_extraction_scores_one() -> None:
    truth = _truth()
    report = evaluate_extraction(truth, truth.copy())
    assert list(report.index) == [*LABELS, "macro"]
    for col in ("coverage", "accuracy", "sensitivity", "specificity"):
        assert report[col].to_numpy() == pytest.approx(1.0)


def test_known_confusion_gives_exact_sensitivity_and_specificity() -> None:
    y = np.zeros((10, len(LABELS)))
    y[:4, 0] = 1.0  # 첫 라벨: 양성 4, 음성 6
    truth = _frame(y)
    pred = truth.copy()
    pred.loc[0, LABELS[0]] = 0.0  # 양성 1개 놓침 → 민감도 3/4
    pred.loc[[4, 5], LABELS[0]] = 1.0  # 음성 2개 오탐 → 특이도 4/6
    row = evaluate_extraction(truth, pred).loc[LABELS[0]]
    assert row["sensitivity"] == pytest.approx(3 / 4)
    assert row["specificity"] == pytest.approx(4 / 6)
    assert row["accuracy"] == pytest.approx(7 / 10)


def test_abstentions_lower_coverage_and_are_excluded_from_accuracy() -> None:
    truth = _truth(n=10)
    pred = truth.copy()
    pred.loc[:4, LABELS[0]] = np.nan  # 10개 중 5개 판단 보류
    row = evaluate_extraction(truth, pred).loc[LABELS[0]]
    assert row["coverage"] == pytest.approx(0.5)
    assert row["accuracy"] == pytest.approx(1.0)


def test_soft_predictions_are_thresholded_at_half_and_scored_by_auc() -> None:
    truth = _truth(n=30)
    pred = truth.copy()
    pred[list(LABELS)] = truth[list(LABELS)] * 0.4 + 0.3  # 양성 0.7, 음성 0.3
    report = evaluate_extraction(truth, pred)
    assert report.loc[LABELS[0], "accuracy"] == pytest.approx(1.0)
    assert report.loc[LABELS[0], "auc"] == pytest.approx(1.0)


def test_unlabeled_truth_rows_are_ignored() -> None:
    truth = _truth(n=10)
    unlabeled = _frame(np.full((3, len(LABELS)), np.nan), ids=["u1", "u2", "u3"])
    pred_extra = _frame(np.zeros((3, len(LABELS))), ids=["u1", "u2", "u3"])
    report = evaluate_extraction(
        pd.concat([truth, unlabeled], ignore_index=True),
        pd.concat([truth, pred_extra], ignore_index=True),
    )
    assert report.loc[LABELS[0], "n"] == 10
    assert report.loc[LABELS[0], "accuracy"] == pytest.approx(1.0)


def test_missing_study_in_prediction_raises() -> None:
    truth = _truth(n=10)
    with pytest.raises(ValueError, match="missing"):
        evaluate_extraction(truth, truth.iloc[1:])
