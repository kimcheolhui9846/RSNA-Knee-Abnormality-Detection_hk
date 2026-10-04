from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import ID_COL
from src.data.build_cache import build_cache
from tests.test_dicom import _slice, _write


def _dataset(root: Path) -> None:
    rows = []
    for study, series, n in [("s1", "a", 3), ("s1", "b", 2), ("s2", "c", 4)]:
        slices = [_slice(k * 10, [-float(k), 0.0, 0.0], k + 1, shape=(20, 20)) for k in range(n)]
        _write(root / "train_series" / study / series, slices)
        rows.append((study, series, 1, 1, "Sagittal"))
    # 폴더는 있지만 DICOM이 없는 망가진 시리즈
    (root / "train_series" / "s2" / "broken").mkdir(parents=True)
    rows.append(("s2", "broken", 0, 0, "Axial"))
    columns = [ID_COL, "SeriesInstanceUID", "Fluid_Sensitive", "Fat_Suppression"]
    columns.append("Anatomical_Plane")
    pd.DataFrame(rows, columns=columns).to_csv(root / "train_series.csv", index=False)


def test_builds_uint8_volumes_and_index(tmp_path: Path) -> None:
    _dataset(tmp_path / "data")
    out = tmp_path / "cache"

    index = build_cache(tmp_path / "data", out, size=16, workers=1)

    ok = index[index["status"] == "built"].set_index("SeriesInstanceUID")
    assert set(ok.index) == {"a", "b", "c"}
    vol = np.load(out / ok.loc["c", "path"])
    assert vol.dtype == np.uint8
    assert vol.shape == (4, 16, 16)
    assert ok.loc["c", "n_slices"] == 4
    assert ok.loc["c", "orig_h"] == 20
    assert (out / "train_index.csv").is_file()


def test_broken_series_is_recorded_not_fatal(tmp_path: Path) -> None:
    _dataset(tmp_path / "data")

    index = build_cache(tmp_path / "data", tmp_path / "cache", size=16, workers=1)

    broken = index.set_index("SeriesInstanceUID").loc["broken"]
    assert broken["status"] == "error"
    assert isinstance(broken["error"], str) and broken["error"]


def test_rerun_skips_existing_files(tmp_path: Path) -> None:
    _dataset(tmp_path / "data")
    out = tmp_path / "cache"
    first = build_cache(tmp_path / "data", out, size=16, workers=1)
    path = out / first.set_index("SeriesInstanceUID").loc["a", "path"]
    mtime = path.stat().st_mtime_ns

    second = build_cache(tmp_path / "data", out, size=16, workers=1)

    assert second.set_index("SeriesInstanceUID").loc["a", "status"] == "cached"
    assert second.set_index("SeriesInstanceUID").loc["a", "n_slices"] == 3
    assert path.stat().st_mtime_ns == mtime


def test_rerun_with_different_size_rebuilds(tmp_path: Path) -> None:
    _dataset(tmp_path / "data")
    out = tmp_path / "cache"
    build_cache(tmp_path / "data", out, size=16, workers=1)

    second = build_cache(tmp_path / "data", out, size=8, workers=1)

    row = second.set_index("SeriesInstanceUID").loc["c"]
    assert row["status"] == "built"
    assert np.load(out / row["path"]).shape == (4, 8, 8)


def test_wrong_dtype_or_corrupt_cache_is_rebuilt(tmp_path: Path) -> None:
    _dataset(tmp_path / "data")
    out = tmp_path / "cache"
    first = build_cache(tmp_path / "data", out, size=16, workers=1).set_index("SeriesInstanceUID")
    np.save(out / first.loc["a", "path"], np.zeros((3, 16, 16), dtype=np.float32))
    (out / first.loc["b", "path"]).write_bytes(b"not a npy file")

    second = build_cache(tmp_path / "data", out, size=16, workers=1).set_index("SeriesInstanceUID")

    assert second.loc["a", "status"] == "built"
    assert second.loc["b", "status"] == "built"
    assert np.load(out / second.loc["a", "path"]).dtype == np.uint8


def test_parallel_matches_serial(tmp_path: Path) -> None:
    _dataset(tmp_path / "data")
    serial = build_cache(tmp_path / "data", tmp_path / "c1", size=16, workers=1)
    parallel = build_cache(tmp_path / "data", tmp_path / "c2", size=16, workers=2)
    pd.testing.assert_frame_equal(serial, parallel)
    np.testing.assert_array_equal(
        np.load(tmp_path / "c1" / serial.loc[0, "path"]),
        np.load(tmp_path / "c2" / parallel.loc[0, "path"]),
    )
