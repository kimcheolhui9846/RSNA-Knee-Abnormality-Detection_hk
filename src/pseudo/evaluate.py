"""pseudo-label 추출 결과를 라벨이 달린 study(정답)와 비교한다.

실행: `python -m src.pseudo.evaluate --pred pseudo_labels/<name>.csv`
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.constants import ID_COL, LABELS
from src.paths import data_dir
from src.pseudo.schema import validate_pseudo_labels


def evaluate_extraction(
    truth: pd.DataFrame, pred: pd.DataFrame, threshold: float = 0.5
) -> pd.DataFrame:
    """라벨별 지표와 macro 평균 행을 돌려준다.

    - truth에서 12개 라벨이 모두 있는 study만 정답으로 쓴다.
    - pred의 NaN은 판단 보류다. coverage에만 반영하고 나머지 지표와 `n`에서는 뺀다.
      macro 행의 `n`은 정답 study 수다.
    - accuracy / sensitivity / specificity는 `threshold` 이상을 양성으로 본다.
    - auc는 pred 값을 점수로 쓴다 (정답이 한 클래스뿐이면 NaN).
    """
    validate_pseudo_labels(pred)
    labeled = truth[truth[list(LABELS)].notna().all(axis=1)]
    pred_by_id = pred.set_index(ID_COL)
    missing = set(labeled[ID_COL]) - set(pred_by_id.index)
    if missing:
        raise ValueError(f"{len(missing)} labeled studies missing from prediction")
    pred_by_id = pred_by_id.loc[labeled[ID_COL]]

    rows: dict[str, dict[str, float]] = {}
    for label in LABELS:
        t = labeled[label].to_numpy()
        s = pred_by_id[label].to_numpy(dtype=float)
        answered = ~np.isnan(s)
        t, s = t[answered], s[answered]
        hard = s >= threshold
        pos = t == 1
        rows[label] = {
            "n": float(len(t)),  # 판단 보류를 뺀, 지표 계산에 실제로 쓴 study 수
            "coverage": float(answered.mean()),
            "accuracy": float((hard == pos).mean()) if len(t) else np.nan,
            "sensitivity": float(hard[pos].mean()) if pos.any() else np.nan,
            "specificity": float((~hard[~pos]).mean()) if (~pos).any() else np.nan,
            "auc": float(roc_auc_score(t, s)) if np.unique(t).size == 2 else np.nan,
        }

    report = pd.DataFrame.from_dict(rows, orient="index")
    report.loc["macro"] = report.mean()
    report.loc["macro", "n"] = float(len(labeled))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    truth = pd.read_csv(data_dir() / "train.csv", dtype={ID_COL: str})
    pred = pd.read_csv(args.pred, dtype={ID_COL: str})
    report = evaluate_extraction(truth, pred, threshold=args.threshold)
    print(report.round(3).to_string())


if __name__ == "__main__":
    main()
