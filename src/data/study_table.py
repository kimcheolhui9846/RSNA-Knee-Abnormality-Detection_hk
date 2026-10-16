"""study 단위 학습 테이블: 시리즈 선택 + fold + 라벨/마스크. torch 없이 동작한다.

study마다 (촬영 방향 × 지방 억제) 6칸에 시리즈를 하나씩 배정한다.
`Fluid_Sensitive`는 train에서 항상 `Fat_Suppression`과 같아 칸 구분에 쓰지 않는다.
"""

import numpy as np
import pandas as pd

from src.constants import ID_COL, LABELS

SLOTS: list[tuple[str, int]] = [
    (plane, fs) for plane in ("Sagittal", "Coronal", "Axial") for fs in (1, 0)
]
OK_STATUS = ("built", "cached")


def select_series(index: pd.DataFrame, target_slices: int = 30) -> pd.DataFrame:
    """캐시 인덱스 → study × SLOTS 표 (값: 캐시 상대 경로, 없으면 None).

    같은 칸에 시리즈가 여러 개면 슬라이스 수가 `target_slices`에 가장 가까운 것을 고르고,
    같으면 SeriesInstanceUID 순서로 정해 결과가 매번 같게 한다.
    """
    ok = index[index["status"].isin(OK_STATUS)].copy()
    ok["_dist"] = (ok["n_slices"] - target_slices).abs()
    slot_cols = ["Anatomical_Plane", "Fat_Suppression"]
    ok = ok.sort_values([ID_COL, *slot_cols, "_dist", "SeriesInstanceUID"])
    best = ok.drop_duplicates([ID_COL, *slot_cols])
    wide = best.pivot(index=ID_COL, columns=slot_cols, values="path")
    wide = wide.reindex(columns=pd.MultiIndex.from_tuples(SLOTS)).astype(object)
    return wide.where(wide.notna(), None)


def resample_depth(volume: np.ndarray, depth: int) -> np.ndarray:
    """슬라이스 축을 `depth`장으로 맞춘다. 처음·끝 슬라이스를 포함해 고르게 뽑고 순서를 지킨다.
    원본이 더 짧으면 가까운 슬라이스를 반복한다."""
    idx = np.round(np.linspace(0, len(volume) - 1, depth)).astype(int)
    return volume[idx]


def build_study_table(
    train: pd.DataFrame, folds: pd.DataFrame, index: pd.DataFrame, target_slices: int = 30
) -> pd.DataFrame:
    """`[ID_COL, fold, labels, label_mask, slot_paths]`.

    - labels: 12개 float, 결측은 NaN 그대로 (0으로 채우지 않는다)
    - label_mask: 라벨이 있는 위치 True — 손실 계산에서 결측을 빼는 데 쓴다
    - slot_paths: SLOTS 순서의 캐시 경로, 없는 칸은 None
    캐시된 시리즈가 하나도 없는 study는 뺀다.
    """
    chosen = select_series(index, target_slices)
    merged = train[[ID_COL, *LABELS]].merge(folds[[ID_COL, "fold"]], on=ID_COL, how="inner")
    merged = merged[merged[ID_COL].isin(chosen.index)]

    labels = merged[list(LABELS)].to_numpy(dtype=np.float32)
    return pd.DataFrame(
        {
            ID_COL: merged[ID_COL].to_numpy(),
            "fold": merged["fold"].to_numpy(),
            "labels": list(labels),
            "label_mask": list(~np.isnan(labels)),
            "slot_paths": [list(chosen.loc[sid]) for sid in merged[ID_COL]],
        }
    )
