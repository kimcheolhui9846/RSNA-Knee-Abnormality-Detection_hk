import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.constants import ID_COL, LABELS
from src.submission import validate_submission


def _valid_submission(n: int = 5) -> pd.DataFrame:
    df = pd.DataFrame(np.full((n, len(LABELS)), 0.5), columns=list(LABELS))
    df.insert(0, ID_COL, [f"1.2.3.{i}" for i in range(n)])
    return df


def test_label_order_matches_competition_spec() -> None:
    assert LABELS == (
        "ACL",
        "MCL",
        "Medial Meniscus",
        "Lateral Meniscus",
        "Medial OA",
        "Lateral OA",
        "PF OA",
        "Effusion",
        "Synovitis",
        "Baker's",
        "Contusion",
        "Fracture",
    )


def test_valid_submission_passes() -> None:
    validate_submission(_valid_submission())


def test_wrong_column_order_fails() -> None:
    df = _valid_submission()
    cols = list(df.columns)
    cols[1], cols[2] = cols[2], cols[1]
    with pytest.raises(ValueError, match="column"):
        validate_submission(df[cols])


def test_nan_fails() -> None:
    df = _valid_submission()
    df.loc[0, LABELS[3]] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate_submission(df)


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_out_of_range_probability_fails(value: float) -> None:
    df = _valid_submission()
    df.loc[0, LABELS[0]] = value
    with pytest.raises(ValueError, match="range"):
        validate_submission(df)


def test_duplicate_ids_fail() -> None:
    df = _valid_submission()
    df.loc[1, ID_COL] = df.loc[0, ID_COL]
    with pytest.raises(ValueError, match="duplicate"):
        validate_submission(df)


def test_ids_must_match_expected_set() -> None:
    df = _valid_submission()
    expected = list(df[ID_COL])[:-1] + ["9.9.9"]
    with pytest.raises(ValueError, match="ID"):
        validate_submission(df, expected_ids=expected)


def _sample_submission_path() -> Path:
    return Path(os.environ.get("RSNA_DATA_DIR", "data")) / "sample_submission.csv"


@pytest.mark.skipif(not _sample_submission_path().exists(), reason="대회 데이터 없음")
def test_labels_match_sample_submission_header() -> None:
    header = pd.read_csv(_sample_submission_path(), nrows=0).columns.tolist()
    assert header == [ID_COL, *LABELS]
