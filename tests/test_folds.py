import numpy as np
import pandas as pd
import pytest

from src.constants import ID_COL, LABELS
from src.folds import make_folds


def _train(n_labeled: int = 40, n_unlabeled: int = 103, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = np.vstack(
        [
            rng.integers(0, 2, size=(n_labeled, len(LABELS))).astype(float),
            np.full((n_unlabeled, len(LABELS)), np.nan),
        ]
    )
    df = pd.DataFrame(labels, columns=list(LABELS))
    df.insert(0, ID_COL, [f"1.2.{i}" for i in range(len(df))])
    df.insert(1, "Report", "report")
    return df


def test_every_study_gets_exactly_one_valid_fold() -> None:
    train = _train()
    folds = make_folds(train, n_splits=5, seed=42)
    assert folds[ID_COL].tolist() == train[ID_COL].tolist()
    assert folds["fold"].between(0, 4).all()


@pytest.mark.parametrize(
    ("is_labeled", "max_spread"),
    # 다중 라벨 층화는 라벨 비율을 맞추느라 fold 크기를 ±1로 보장하지 않는다
    [(True, 2), (False, 1)],
)
def test_labeled_and_unlabeled_are_each_spread_evenly(is_labeled: bool, max_spread: int) -> None:
    folds = make_folds(_train(), n_splits=5, seed=42)
    sizes = folds.loc[folds["labeled"] == is_labeled, "fold"].value_counts()
    assert len(sizes) == 5
    assert sizes.max() - sizes.min() <= max_spread


def test_unlabeled_rows_do_not_change_labeled_assignment() -> None:
    # 결측을 0으로 채워 층화에 섞으면 라벨 있는 study의 배정이 바뀐다
    with_unlabeled = make_folds(_train(n_unlabeled=103), n_splits=5, seed=42)
    labeled_only = make_folds(_train(n_unlabeled=0), n_splits=5, seed=42)
    a = with_unlabeled.loc[with_unlabeled["labeled"], "fold"].to_numpy()
    b = labeled_only["fold"].to_numpy()
    np.testing.assert_array_equal(a, b)


def test_same_seed_is_deterministic() -> None:
    train = _train()
    pd.testing.assert_frame_equal(make_folds(train, seed=7), make_folds(train, seed=7))


def test_partially_labeled_study_raises() -> None:
    train = _train()
    train.loc[0, LABELS[0]] = np.nan
    with pytest.raises(ValueError, match="partial"):
        make_folds(train)
