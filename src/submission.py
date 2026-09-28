from collections.abc import Iterable

import pandas as pd

from src.constants import ID_COL, LABELS


def validate_submission(df: pd.DataFrame, expected_ids: Iterable[str] | None = None) -> None:
    """제출 CSV로 쓰기 전에 형식을 검사한다. 문제가 있으면 ValueError."""
    expected_cols = [ID_COL, *LABELS]
    if list(df.columns) != expected_cols:
        raise ValueError(f"column mismatch: {list(df.columns)} != {expected_cols}")

    probs = df[list(LABELS)]
    if probs.isna().any().any():
        raise ValueError(f"NaN in predictions: {probs.isna().sum()[lambda s: s > 0].to_dict()}")
    if ((probs < 0) | (probs > 1)).any().any():
        raise ValueError("probability out of range [0, 1]")

    ids = df[ID_COL]
    if ids.duplicated().any():
        raise ValueError(f"duplicate {ID_COL}: {ids[ids.duplicated()].tolist()[:5]}")
    if expected_ids is not None:
        missing = set(expected_ids) - set(ids)
        extra = set(ids) - set(expected_ids)
        if missing or extra:
            raise ValueError(f"ID mismatch: missing {len(missing)}, extra {len(extra)}")
