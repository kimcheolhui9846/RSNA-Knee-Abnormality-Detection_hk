import numpy as np
import pandas as pd

from src.constants import ID_COL, LABELS
from src.data.labels import merge_pseudo_labels


def _frame(rows: dict[str, list[float]], report: bool = False) -> pd.DataFrame:
    df = pd.DataFrame(list(rows.values()), columns=list(LABELS))
    df.insert(0, ID_COL, list(rows))
    if report:
        df.insert(1, "Report", "r")
    return df


def test_ground_truth_wins_and_pseudo_fills_unlabeled_studies() -> None:
    nan = [np.nan] * len(LABELS)
    gt_row = [1.0] * len(LABELS)
    train = _frame({"gt1": gt_row, "u1": nan, "u2": nan}, report=True)
    pseudo_u1 = [0.0] * len(LABELS)
    pseudo_u1[3] = np.nan  # LLM 판단 보류
    pseudo = _frame({"u1": pseudo_u1, "gt1": [0.0] * len(LABELS)})

    out = merge_pseudo_labels(train, pseudo).set_index(ID_COL)

    assert list(out.columns) == [*LABELS, "source"]
    assert out.loc["gt1", "source"] == "gt"
    assert (
        out.loc["gt1", list(LABELS)].to_numpy(dtype=float) == 1.0
    ).all()  # 정답이 pseudo보다 우선
    assert out.loc["u1", "source"] == "pseudo"
    assert np.isnan(out.loc["u1", LABELS[3]])  # 판단 보류는 NaN 그대로 (0으로 채우지 않음)
    assert out.loc["u1", LABELS[0]] == 0.0
    assert out.loc["u2", "source"] == "none"
    assert out.loc["u2", list(LABELS)].isna().all()


def test_row_order_follows_train() -> None:
    nan = [np.nan] * len(LABELS)
    train = _frame({"b": nan, "a": nan, "c": nan}, report=True)
    pseudo = _frame({"a": [1.0] * len(LABELS), "c": [0.0] * len(LABELS)})
    assert merge_pseudo_labels(train, pseudo)[ID_COL].tolist() == ["b", "a", "c"]
