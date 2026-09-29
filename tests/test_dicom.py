import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, JPEG2000Lossless, generate_uid

from src.data.dicom import load_series
from src.paths import data_dir

# 시상면(sagittal): 행 방향 +y, 열 방향 -z → 법선은 -x. 슬라이스 k는 x = -k 위치에 둔다.
SAGITTAL_IOP = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]


def _slice(
    value: int,
    position: list[float] | None,
    instance_number: int,
    shape: tuple[int, int] = (8, 8),
    slope: float | None = None,
    intercept: float | None = None,
) -> Dataset:
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"  # MR Image Storage
    ds.SOPInstanceUID = generate_uid()
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
    ds.InstanceNumber = instance_number
    if position is not None:
        ds.ImagePositionPatient = position
        ds.ImageOrientationPatient = SAGITTAL_IOP
    if slope is not None:
        ds.RescaleSlope = slope
        ds.RescaleIntercept = intercept
    ds.Rows, ds.Columns = shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.PixelData = np.full(shape, value, dtype=np.uint16).tobytes()
    return ds


def _write(series_dir: Path, slices: list[Dataset]) -> None:
    series_dir.mkdir(parents=True, exist_ok=True)
    for ds in slices:
        ds.save_as(series_dir / f"{ds.SOPInstanceUID}.dcm", enforce_file_format=True)


def test_slices_sorted_by_position_along_normal_not_instance_number(tmp_path: Path) -> None:
    n = 6
    numbers = random.Random(0).sample(range(1, n + 1), n)  # InstanceNumber는 섞어 둔다
    slices = [_slice(k, [-float(k), 0.0, 0.0], numbers[k]) for k in range(n)]
    _write(tmp_path / "s", slices)

    vol = load_series(tmp_path / "s")

    assert vol.shape == (n, 8, 8)
    assert vol[:, 0, 0].tolist() == list(range(n))


def test_falls_back_to_instance_number_without_position(tmp_path: Path) -> None:
    n = 5
    slices = [_slice(k, None, instance_number=k + 1) for k in range(n)]
    _write(tmp_path / "s", slices)

    vol = load_series(tmp_path / "s")

    assert vol[:, 0, 0].tolist() == list(range(n))


def test_rescale_slope_and_intercept_applied(tmp_path: Path) -> None:
    _write(tmp_path / "s", [_slice(10, [0.0, 0.0, 0.0], 1, slope=2.0, intercept=-5.0)])

    vol = load_series(tmp_path / "s")

    assert vol.dtype == np.float32
    assert vol[0, 0, 0] == pytest.approx(15.0)


def test_jpeg2000_compressed_series_decodes(tmp_path: Path) -> None:
    # JPEG2000 인코더는 기본 해상도 단계 수 때문에 너무 작은 이미지를 거부한다
    ds = _slice(0, [0.0, 0.0, 0.0], 1, shape=(64, 64))
    pixels = (np.arange(64 * 64, dtype=np.uint16) % 4096).reshape(64, 64)
    ds.PixelData = pixels.tobytes()
    ds.compress(JPEG2000Lossless, encoding_plugin="pylibjpeg")
    _write(tmp_path / "s", [ds])

    vol = load_series(tmp_path / "s")

    np.testing.assert_array_equal(vol[0], pixels)


def test_mismatched_slice_shapes_raise(tmp_path: Path) -> None:
    slices = [_slice(0, [0.0, 0.0, 0.0], 1), _slice(1, [-1.0, 0.0, 0.0], 2, shape=(4, 4))]
    _write(tmp_path / "s", slices)

    with pytest.raises(ValueError, match="shape"):
        load_series(tmp_path / "s")


def _real_series_dir() -> Path | None:
    try:
        root = data_dir()
    except FileNotFoundError:
        return None
    # data/ 폴더만 있고 메타데이터가 없으면 수집 단계에서 터지지 않고 skip한다
    csv = root / "train_series.csv"
    if not csv.is_file():
        return None
    row = pd.read_csv(csv, nrows=1, dtype=str).iloc[0]
    series_dir = root / "train_series" / row["StudyInstanceUID"] / row["SeriesInstanceUID"]
    return series_dir if series_dir.is_dir() else None


@pytest.mark.skipif(_real_series_dir() is None, reason="대회 데이터 없음")
def test_real_series_loads() -> None:
    series_dir = _real_series_dir()
    vol = load_series(series_dir)
    assert vol.ndim == 3
    assert vol.shape[0] == len(list(series_dir.glob("*.dcm")))
    assert np.isfinite(vol).all()
