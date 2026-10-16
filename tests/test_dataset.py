from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.constants import ID_COL, LABELS
from src.data.study_table import SLOTS

torch = pytest.importorskip("torch")  # CI는 torch 없이 돈다 (--extra train 미설치)

from src.data.dataset import KneeStudyDataset  # noqa: E402


def _table(cache: Path) -> pd.DataFrame:
    (cache / "s1").mkdir(parents=True)
    np.save(cache / "s1" / "sag.npy", np.full((10, 8, 8), 255, dtype=np.uint8))
    np.save(cache / "s1" / "ax.npy", np.full((40, 8, 8), 51, dtype=np.uint8))
    paths = [None] * len(SLOTS)
    paths[SLOTS.index(("Sagittal", 1))] = "s1/sag.npy"
    paths[SLOTS.index(("Axial", 1))] = "s1/ax.npy"
    labels = np.full(len(LABELS), np.nan, dtype=np.float32)
    labels[0] = 1.0
    return pd.DataFrame(
        {
            ID_COL: ["s1"],
            "fold": [0],
            "labels": [labels],
            "label_mask": [~np.isnan(labels)],
            "slot_paths": [paths],
        }
    )


def test_item_shapes_and_value_range(tmp_path: Path) -> None:
    ds = KneeStudyDataset(_table(tmp_path), cache_root=tmp_path, depth=16, size=8)
    item = ds[0]
    assert len(ds) == 1
    assert item["image"].shape == (len(SLOTS), 16, 8, 8)
    assert item["image"].dtype == torch.float32
    assert float(item["image"].max()) == pytest.approx(1.0)
    assert item["study_id"] == "s1"


def test_missing_slots_are_zero_and_masked(tmp_path: Path) -> None:
    item = KneeStudyDataset(_table(tmp_path), cache_root=tmp_path, depth=16, size=8)[0]
    sag, cor = SLOTS.index(("Sagittal", 1)), SLOTS.index(("Coronal", 1))
    assert bool(item["slot_mask"][sag]) and not bool(item["slot_mask"][cor])
    assert float(item["image"][cor].abs().sum()) == 0.0
    assert float(item["image"][SLOTS.index(("Axial", 1))].mean()) == pytest.approx(0.2)


def test_missing_labels_are_masked_and_not_nan(tmp_path: Path) -> None:
    item = KneeStudyDataset(_table(tmp_path), cache_root=tmp_path, depth=16, size=8)[0]
    assert not torch.isnan(item["labels"]).any()  # 손실 계산이 NaN으로 터지지 않게
    assert item["label_mask"].tolist() == [True] + [False] * (len(LABELS) - 1)
    assert float(item["labels"][0]) == 1.0
