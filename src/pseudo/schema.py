"""report에서 뽑은 pseudo-label 파일의 형식.

한 행이 study 하나다. 컬럼은 `[ID_COL, *LABELS]`이고 값은 다음 중 하나다.
- 0.0 ~ 1.0: 양성일 확률 (1.0 / 0.0이면 확정, 그 사이는 soft label)
- NaN: 판단 보류. report만으로 알 수 없는 경우다. 학습에서는 결측으로 취급하고 0으로 채우지 않는다.
"""

import pandas as pd

from src.constants import ID_COL, LABELS


def validate_pseudo_labels(df: pd.DataFrame) -> None:
    """형식이 틀리면 ValueError."""
    expected_cols = [ID_COL, *LABELS]
    if list(df.columns) != expected_cols:
        raise ValueError(f"column mismatch: {list(df.columns)} != {expected_cols}")

    values = df[list(LABELS)]
    if ((values < 0) | (values > 1)).any().any():
        raise ValueError("pseudo-label value out of range [0, 1]")

    ids = df[ID_COL]
    if ids.duplicated().any():
        raise ValueError(f"duplicate {ID_COL}: {ids[ids.duplicated()].tolist()[:5]}")
