"""study 하나 = 학습 샘플 하나. `build_study_table`의 결과와 캐시 `.npy`를 읽는다."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.constants import ID_COL
from src.data.study_table import SLOTS, resample_depth


class KneeStudyDataset(Dataset):
    """`__getitem__` → dict

    - image: float32 (len(SLOTS), depth, size, size), 0~1. 없는 칸은 0
    - slot_mask: bool (len(SLOTS),) — 칸에 시리즈가 있으면 True
    - labels: float32 (12,) — 결측 위치는 0이지만 label_mask가 False라 손실에서 빠진다
    - label_mask: bool (12,)
    - study_id: str
    """

    def __init__(self, table: pd.DataFrame, cache_root: Path, depth: int, size: int) -> None:
        self.table = table.reset_index(drop=True)
        self.cache_root = Path(cache_root)
        self.depth = depth
        self.size = size

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, i: int) -> dict:
        row = self.table.iloc[i]
        image = np.zeros((len(SLOTS), self.depth, self.size, self.size), dtype=np.float32)
        slot_mask = np.zeros(len(SLOTS), dtype=bool)
        for k, rel in enumerate(row["slot_paths"]):
            if rel is None:
                continue
            vol = np.load(self.cache_root / rel)
            image[k] = resample_depth(vol, self.depth).astype(np.float32) / 255.0
            slot_mask[k] = True

        labels = np.nan_to_num(np.asarray(row["labels"], dtype=np.float32), nan=0.0)
        return {
            "image": torch.from_numpy(image),
            "slot_mask": torch.from_numpy(slot_mask),
            "labels": torch.from_numpy(labels),
            "label_mask": torch.from_numpy(np.asarray(row["label_mask"], dtype=bool)),
            "study_id": row[ID_COL],
        }
