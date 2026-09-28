import numpy as np
from sklearn.metrics import roc_auc_score

from src.constants import LABELS


def macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, dict[str, float]]:
    """12개 라벨 AUC의 단순 평균과 라벨별 AUC를 돌려준다.

    y_true의 NaN은 결측(라벨 없음)이므로 그 라벨 계산에서만 뺀다. 음성으로 취급하지 않는다.
    한 클래스만 남은 라벨은 AUC가 정의되지 않으므로 NaN으로 두고 평균에서 제외한다.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    expected = (y_true.shape[0], len(LABELS))
    if y_true.shape != expected or y_pred.shape != expected:
        raise ValueError(f"expected {expected}, got {y_true.shape} / {y_pred.shape}")

    per_label: dict[str, float] = {}
    for j, name in enumerate(LABELS):
        mask = ~np.isnan(y_true[:, j])
        t = y_true[mask, j]
        if np.unique(t).size < 2:
            per_label[name] = float("nan")
        else:
            per_label[name] = float(roc_auc_score(t, y_pred[mask, j]))

    return float(np.nanmean(list(per_label.values()))), per_label
