import numpy as np
import pandas as pd
import pytest

from src.constants import ID_COL, LABELS
from src.data.refine_labels import quantile_map, refine


def _table(ids, values, source=None) -> pd.DataFrame:
    df = pd.DataFrame(np.repeat(np.array(values, float)[:, None], len(LABELS), 1), columns=LABELS)
    df.insert(0, ID_COL, ids)
    if source is not None:
        df["source"] = source
    return df


def test_quantile_map_keeps_order_and_reference_values() -> None:
    out = quantile_map(pd.Series([0.9, 0.1, 0.5]), pd.Series([0.0, 0.2, 1.0]))
    np.testing.assert_allclose(out, [1.0, 0.0, 0.2])


def test_refine_mixes_report_and_image_ranks_and_keeps_gold() -> None:
    ids = ["g", "a", "b", "c"]
    base = _table(ids, [1.0, 0.1, 0.5, 0.9], ["gt", "pseudo", "pseudo", "pseudo"])
    report = _table(["a", "b", "c"], [0.0, 0.5, 1.0])  # 판독문: a < b < c
    image = _table(["a", "b", "c"], [1.0, 0.5, 0.0])  # 이미지: 반대 순서
    only_report = refine(base, [report], image, alpha=0.0)
    only_image = refine(base, [report], image, alpha=1.0)
    lab = LABELS[0]
    assert only_report[lab].tolist() == [1.0, 0.1, 0.5, 0.9]  # 기준 분포, 판독문 순서, 정답 그대로
    assert only_image[lab].tolist() == [1.0, 0.9, 0.5, 0.1]  # 같은 값들, 이미지 순서
    # 정반대 순서를 반반 섞으면 모두 동률 → 기준 분포의 중앙값
    assert refine(base, [report], image, 0.5)[lab].iloc[1:].tolist() == [0.5, 0.5, 0.5]


def test_refine_requires_image_oof_for_every_pseudo_study() -> None:
    base = _table(["a", "b"], [0.1, 0.9], ["pseudo", "pseudo"])
    with pytest.raises(ValueError):
        refine(base, [_table(["a", "b"], [0, 1])], _table(["a"], [0.5]))
