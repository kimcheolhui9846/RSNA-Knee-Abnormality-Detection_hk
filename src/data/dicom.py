from pathlib import Path

import numpy as np
import pydicom


def _slice_position(ds: pydicom.Dataset) -> float | None:
    """슬라이스 법선 방향으로 투영한 위치. 위치·방향 태그가 없으면 None."""
    if "ImagePositionPatient" not in ds or "ImageOrientationPatient" not in ds:
        return None
    iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
    normal = np.cross(iop[:3], iop[3:])
    return float(np.dot(np.asarray(ds.ImagePositionPatient, dtype=float), normal))


def load_series(series_dir: Path) -> np.ndarray:
    """시리즈 폴더의 DICOM을 읽어 (N, H, W) float32 볼륨으로 쌓는다.

    파일 이름(SOPInstanceUID)은 순서를 뜻하지 않는다. 슬라이스 법선 방향 위치의 오름차순으로
    정렬하고, 위치 정보가 하나라도 없으면 InstanceNumber로 정렬한다.
    RescaleSlope/Intercept가 있으면 적용하고, MONOCHROME1이면 부호를 뒤집는다.
    압축 DICOM은 pylibjpeg가 디코딩한다.
    """
    datasets = [pydicom.dcmread(p) for p in sorted(Path(series_dir).glob("*.dcm"))]
    if not datasets:
        raise FileNotFoundError(f"DICOM 없음: {series_dir}")

    positions = [_slice_position(ds) for ds in datasets]
    if all(p is not None for p in positions):
        order = np.argsort(positions, kind="stable")
    else:
        order = np.argsort([int(ds.InstanceNumber) for ds in datasets], kind="stable")

    slices = []
    for i in order:
        ds = datasets[i]
        arr = ds.pixel_array.astype(np.float32)
        slope = float(ds.get("RescaleSlope", 1.0))
        intercept = float(ds.get("RescaleIntercept", 0.0))
        arr = arr * slope + intercept
        if ds.get("PhotometricInterpretation") == "MONOCHROME1":
            arr = -arr  # 값이 클수록 밝도록 뒤집는다. 절대값은 이후 백분위 정규화가 맞춘다
        slices.append(arr)

    shapes = {s.shape for s in slices}
    if len(shapes) != 1:
        raise ValueError(f"slice shape mismatch in {series_dir}: {sorted(shapes)}")
    return np.stack(slices)
