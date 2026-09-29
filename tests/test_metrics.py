import numpy as np
import pytest

from src.constants import LABELS
from src.metrics import macro_auc


def _random_targets(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=(n, len(LABELS))).astype(float)
    y[0], y[1] = 0.0, 1.0  # 모든 라벨에 양성·음성이 둘 다 있도록 보장
    return y


def test_perfect_predictions_score_one() -> None:
    y = _random_targets(50)
    macro, per_label = macro_auc(y, y)
    assert macro == pytest.approx(1.0)
    assert list(per_label) == list(LABELS)
    assert all(v == pytest.approx(1.0) for v in per_label.values())


def test_macro_is_simple_mean_of_label_aucs() -> None:
    y = _random_targets(80)
    pred = np.random.default_rng(1).random(y.shape)
    macro, per_label = macro_auc(y, pred)
    assert macro == pytest.approx(np.mean(list(per_label.values())))


def test_missing_labels_are_masked_not_treated_as_negative() -> None:
    y = _random_targets(40)
    pred = y.copy()
    # 결측 행에 틀린 예측을 넣는다. 결측을 0으로 취급하면 AUC가 1보다 떨어진다.
    y[2:10, 0] = np.nan
    pred[2:10, 0] = 1.0
    _, per_label = macro_auc(y, pred)
    assert per_label[LABELS[0]] == pytest.approx(1.0)


def test_single_class_label_is_nan_and_excluded_from_macro() -> None:
    y = _random_targets(40)
    y[:, -1] = 0.0  # 이 fold에는 마지막 라벨(Fracture) 양성이 없다
    macro, per_label = macro_auc(y, y)
    assert np.isnan(per_label[LABELS[-1]])
    assert macro == pytest.approx(1.0)


def test_shape_mismatch_raises() -> None:
    y = _random_targets(10)
    with pytest.raises(ValueError):
        macro_auc(y, y[:, :-1])
