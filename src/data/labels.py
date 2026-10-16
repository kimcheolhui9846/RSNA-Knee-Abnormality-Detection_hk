"""정답 라벨과 report pseudo-label을 하나의 학습 라벨 표로 합친다."""

import pandas as pd

from src.constants import ID_COL, LABELS


def merge_pseudo_labels(train: pd.DataFrame, pseudo: pd.DataFrame) -> pd.DataFrame:
    """`[ID_COL, *LABELS, source]`, 행 순서는 train과 같다.

    - 12개 라벨이 모두 있는 study(정답)는 정답을 쓰고 `source="gt"`
    - 나머지는 pseudo-label을 쓰고 `source="pseudo"`. pseudo의 NaN(판단 보류)은 NaN 그대로
    - pseudo에도 없는 study는 전부 NaN, `source="none"`
    """
    out = train[[ID_COL, *LABELS]].copy()
    is_gt = out[list(LABELS)].notna().all(axis=1)
    by_id = pseudo.set_index(ID_COL)[list(LABELS)]
    has_pseudo = out[ID_COL].isin(by_id.index) & ~is_gt

    fill = by_id.reindex(out.loc[has_pseudo, ID_COL]).to_numpy()
    out.loc[has_pseudo, list(LABELS)] = fill
    out["source"] = "none"
    out.loc[has_pseudo, "source"] = "pseudo"
    out.loc[is_gt, "source"] = "gt"
    return out.reset_index(drop=True)
