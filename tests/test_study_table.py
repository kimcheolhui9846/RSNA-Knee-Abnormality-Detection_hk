import numpy as np
import pandas as pd
import pytest

from src.constants import ID_COL, LABELS
from src.data.study_table import SLOTS, build_study_table, resample_depth, select_series


def _index(rows: list[tuple[str, str, str, int, int, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=[
            ID_COL,
            "SeriesInstanceUID",
            "Anatomical_Plane",
            "Fat_Suppression",
            "n_slices",
            "status",
        ],
    ).assign(path=lambda d: d[ID_COL] + "/" + d["SeriesInstanceUID"] + ".npy")


# --- select_series ------------------------------------------------------------


def test_six_slots_are_plane_by_fat_suppression() -> None:
    assert len(SLOTS) == 6
    assert set(SLOTS) == {(p, fs) for p in ("Sagittal", "Coronal", "Axial") for fs in (1, 0)}


def test_duplicate_series_in_a_slot_picks_slice_count_closest_to_target() -> None:
    index = _index(
        [
            ("s1", "far", "Sagittal", 1, 160, "built"),
            ("s1", "near", "Sagittal", 1, 28, "built"),
        ]
    )
    chosen = select_series(index, target_slices=30)
    assert chosen.loc["s1", ("Sagittal", 1)] == "s1/near.npy"


def test_tie_is_broken_deterministically_by_series_uid() -> None:
    index = _index(
        [
            ("s1", "b", "Axial", 1, 32, "built"),
            ("s1", "a", "Axial", 1, 28, "built"),
        ]
    )
    assert select_series(index, target_slices=30).loc["s1", ("Axial", 1)] == "s1/a.npy"


def test_missing_slot_and_failed_series_are_none() -> None:
    index = _index(
        [
            ("s1", "ok", "Sagittal", 1, 30, "built"),
            ("s1", "bad", "Coronal", 1, 30, "error"),
        ]
    )
    chosen = select_series(index, target_slices=30)
    assert chosen.loc["s1", ("Coronal", 1)] is None
    assert chosen.loc["s1", ("Axial", 0)] is None


# --- resample_depth -----------------------------------------------------------


@pytest.mark.parametrize("n", [1, 7, 30, 95])
def test_resample_depth_returns_requested_depth_keeping_order(n: int) -> None:
    vol = np.arange(n, dtype=np.uint8)[:, None, None] * np.ones((1, 4, 4), dtype=np.uint8)
    out = resample_depth(vol, depth=16)
    assert out.shape == (16, 4, 4)
    firsts = out[:, 0, 0].astype(int)
    assert (np.diff(firsts) >= 0).all()
    assert firsts[0] == 0 and firsts[-1] == n - 1


# --- build_study_table --------------------------------------------------------


def _train(ids: list[str], labeled: list[bool]) -> pd.DataFrame:
    y = np.where(np.array(labeled)[:, None], 1.0, np.nan) * np.ones((len(ids), len(LABELS)))
    df = pd.DataFrame(y, columns=list(LABELS))
    df.insert(0, ID_COL, ids)
    df.insert(1, "Report", "r")
    return df


def test_study_table_joins_fold_labels_mask_and_slots() -> None:
    train = _train(["s1", "s2"], labeled=[True, False])
    folds = pd.DataFrame({ID_COL: ["s1", "s2"], "fold": [3, 1], "labeled": [True, False]})
    index = _index([("s1", "x", "Sagittal", 1, 30, "built"), ("s2", "y", "Axial", 1, 30, "built")])

    table = build_study_table(train, folds, index, target_slices=30).set_index(ID_COL)

    assert table.loc["s1", "fold"] == 3
    assert table.loc["s1", "label_mask"].all()
    assert not table.loc["s2", "label_mask"].any()
    # 결측 라벨은 0으로 바꾸지 않고 NaN 그대로 둔다 (마스크가 손실에서 뺀다)
    assert np.isnan(table.loc["s2", "labels"]).all()
    assert table.loc["s1", "slot_paths"][SLOTS.index(("Sagittal", 1))] == "s1/x.npy"
    assert table.loc["s2", "slot_paths"][SLOTS.index(("Sagittal", 1))] is None


def test_study_without_any_cached_series_is_dropped() -> None:
    train = _train(["s1", "s2"], labeled=[True, True])
    folds = pd.DataFrame({ID_COL: ["s1", "s2"], "fold": [0, 1], "labeled": [True, True]})
    index = _index([("s1", "x", "Sagittal", 1, 30, "built"), ("s2", "y", "Axial", 1, 30, "error")])
    table = build_study_table(train, folds, index, target_slices=30)
    assert table[ID_COL].tolist() == ["s1"]
