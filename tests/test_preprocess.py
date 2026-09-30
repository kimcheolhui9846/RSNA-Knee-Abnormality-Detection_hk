import numpy as np
import pytest

from src.data.preprocess import preprocess_series


def _ramp_volume(n: int = 4, h: int = 64, w: int = 64) -> np.ndarray:
    return np.tile(np.linspace(0, 1000, w, dtype=np.float32), (n, h, 1))


def test_output_is_uint8_square_at_target_size() -> None:
    out = preprocess_series(_ramp_volume(n=5, h=80, w=80), size=32)
    assert out.dtype == np.uint8
    assert out.shape == (5, 32, 32)


def test_intensities_are_stretched_to_full_uint8_range() -> None:
    out = preprocess_series(_ramp_volume(), size=64)
    assert out.min() == 0
    assert out.max() == 255


def test_outliers_are_clipped_by_percentile() -> None:
    vol = _ramp_volume()
    vol[0, 0, 0] = 1e6  # 극단값 하나가 전체 대비를 망치면 안 된다
    out = preprocess_series(vol, size=64)
    # 램프 중간은 여전히 중간 밝기 근처에 있어야 한다
    assert 100 < int(out[1, 32, 32]) < 155


def test_constant_volume_becomes_zeros_without_nan() -> None:
    out = preprocess_series(np.full((3, 16, 16), 7.0, dtype=np.float32), size=16)
    assert out.dtype == np.uint8
    assert (out == 0).all()


def test_non_square_slices_are_padded_not_stretched() -> None:
    vol = np.ones((2, 50, 100), dtype=np.float32)
    vol[:, :, :50] = 0.0  # 왼쪽 절반 어둡게, 오른쪽 절반 밝게
    out = preprocess_series(vol, size=100)
    # 50×100을 100×100으로 늘리지 않고 위아래를 0으로 채운다
    assert (out[:, :20, :] == 0).all()
    assert out[0, 50, 90] == 255
    assert out[0, 50, 10] == 0


def test_rejects_non_3d_input() -> None:
    with pytest.raises(ValueError, match="3D"):
        preprocess_series(np.zeros((16, 16), dtype=np.float32), size=16)
