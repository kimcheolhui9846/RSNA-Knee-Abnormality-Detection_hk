"""시리즈 볼륨 → 모델 입력용 uint8 볼륨.

캐시 생성(`src/data/build_cache.py`)과 Kaggle 추론이 **같은 함수**를 써야
학습·추론 전처리가 어긋나지 않는다.
"""

import cv2
import numpy as np


def preprocess_series(
    volume: np.ndarray,
    size: int = 256,
    clip_percentiles: tuple[float, float] = (0.5, 99.5),
) -> np.ndarray:
    """(N, H, W) float → (N, size, size) uint8.

    1. 시리즈 전체의 백분위수로 극단값을 자르고 0~255로 늘린다 (MRI는 절대 밝기 단위가 없다).
    2. 정사각형이 아니면 가운데 정렬로 0을 채워 정사각형으로 만든다 (비율을 늘이지 않는다).
    3. 한 변을 `size`로 맞춘다. 줄일 때는 INTER_AREA, 키울 때는 INTER_LINEAR.
    """
    if volume.ndim != 3:
        raise ValueError(f"expected 3D volume (N, H, W), got shape {volume.shape}")

    lo, hi = np.percentile(volume, clip_percentiles)
    if hi <= lo:
        scaled = np.zeros(volume.shape, dtype=np.uint8)
    else:
        scaled = np.clip(volume, lo, hi)
        scaled = np.round((scaled - lo) / (hi - lo) * 255).astype(np.uint8)

    n, h, w = scaled.shape
    side = max(h, w)
    if h != w:
        top, left = (side - h) // 2, (side - w) // 2
        padded = np.zeros((n, side, side), dtype=np.uint8)
        padded[:, top : top + h, left : left + w] = scaled
        scaled = padded

    if side == size:
        return scaled
    interp = cv2.INTER_AREA if side > size else cv2.INTER_LINEAR
    return np.stack([cv2.resize(s, (size, size), interpolation=interp) for s in scaled])
