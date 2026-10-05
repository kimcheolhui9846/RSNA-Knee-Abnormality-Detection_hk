from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")  # CI는 torch 없이 돈다

from src.constants import ID_COL, LABELS  # noqa: E402
from src.data.slot_image import (  # noqa: E402
    SLOTS6,
    augment,
    band_indices,
    pick_slot_series,
    slot_image,
)
from src.models import build_model  # noqa: E402
from src.train import make_synthetic_data, run  # noqa: E402

TINY_SLOT = {
    "name": "slot_dino",
    "backbone": "vit_tiny_patch16_224",
    "pretrained": False,
    "img_size": 32,
    "unfreeze_last": 2,
    "dropout": 0.0,
}


def test_band_indices_stay_in_central_band() -> None:
    assert band_indices(11).tolist() == [2, 5, 8]
    assert band_indices(1).tolist() == [0, 0, 0]


def test_slot_image_crops_centre_by_millimetres() -> None:
    vol = np.full((5, 256, 256), 255, np.uint8)  # 바깥은 아주 밝은 테두리
    vol[:, 64:192, 64:192] = np.arange(128, dtype=np.uint8)[None, None, :]  # 가운데: 좌→우 증가
    # 원본 512px × 0.5 mm = 256 mm 시야 → 128 mm 자르기 = 캐시 가운데 128px
    img = slot_image(vol, px=0.5, side_px=512, plane="Sagittal", lat=None, crop_mm=128, out=64)
    assert img.shape == (3, 64, 64)
    row = img[1, 32].astype(float)
    assert (np.diff(row) >= 0).all()  # 왼쪽 끝에 테두리(255)가 들어왔다면 단조 증가가 깨진다
    assert np.corrcoef(row, np.arange(64))[0, 1] > 0.99  # 가운데 기울기가 그대로 남는다
    assert row[-1] - row[0] > 200  # 잘라낸 영역 기준으로 밝기를 다시 늘렸다


def test_slot_image_flips_right_knee() -> None:
    vol = np.zeros((9, 64, 64), np.uint8)
    vol[:, :, :10] = 255  # 왼쪽 가장자리
    vol[2] += 1  # 슬라이스 구분용
    left = slot_image(vol, px=0, side_px=0, plane="Coronal", lat="L", out=64)
    right = slot_image(vol, px=0, side_px=0, plane="Coronal", lat="R", out=64)
    np.testing.assert_array_equal(right, left[:, :, ::-1])
    sag_l = slot_image(vol, px=0, side_px=0, plane="Sagittal", lat="L", out=64)
    sag_r = slot_image(vol, px=0, side_px=0, plane="Sagittal", lat="R", out=64)
    np.testing.assert_array_equal(sag_r, sag_l[::-1])


def test_pick_slot_series_takes_longest_series_per_slot() -> None:
    meta = pd.DataFrame(
        {
            ID_COL: ["a", "a", "a", "b"],
            "SeriesInstanceUID": ["1", "2", "3", "4"],
            "Anatomical_Plane": ["Sagittal", "Sagittal", "Coronal", "Axial"],
            "fluid": [True, True, False, True],
            "fatsat": [True, True, False, True],
            "n_slices": [20, 30, 25, 18],
        }
    )
    wide = pick_slot_series(meta)
    assert wide.loc["a", 0] == 1  # SAG_FLUID_FS: 슬라이스 30장짜리 (행 1)
    assert wide.loc["a", 4] == 2  # COR_T1
    assert wide.loc["b", 2] == 3  # AX_FLUID_FS
    assert np.isnan(wide.loc["b", 0])


def test_augment_keeps_shape_and_range() -> None:
    x = torch.rand(4, 3, 32, 32)
    y = augment(x, np.random.default_rng(0))
    assert y.shape == x.shape
    assert float(y.min()) >= 0 and float(y.max()) <= 1


def _batch(b: int = 2, size: int = 32) -> tuple:
    image = torch.rand(b, len(SLOTS6), 3, size, size)
    mask = torch.ones(b, len(SLOTS6), dtype=torch.bool)
    mask[0, -1] = False
    return image, mask


def test_slot_dino_output_mask_and_empty_study() -> None:
    model = build_model(TINY_SLOT).eval()
    image, mask = _batch()
    with torch.no_grad():
        out = model(image, mask)
        assert out.shape == (2, len(LABELS))
        image[0, -1] = torch.rand_like(image[0, -1])  # 없는 칸
        torch.testing.assert_close(out, model(image, mask))
        assert model(image, torch.zeros_like(mask)).isfinite().all()
        assert model(torch.rand(1, len(SLOTS6), 3, 48, 48), mask[:1]).shape == (1, len(LABELS))


def test_slot_dino_trains_only_last_blocks_and_head() -> None:
    model = build_model(TINY_SLOT)
    enc = model.encoder
    n = len(enc.blocks)
    assert not any(p.requires_grad for p in enc.patch_embed.parameters())
    assert not any(p.requires_grad for p in enc.blocks[n - 3].parameters())
    assert all(p.requires_grad for p in enc.blocks[n - 1].parameters())
    assert all(p.requires_grad for p in model.head.parameters())


def test_slot_prior_raises_attention_on_listed_slots() -> None:
    model = build_model({**TINY_SLOT, "prior": True})
    mcl = LABELS.index("MCL")
    assert model.head.slot_prior[mcl].tolist() == pytest.approx([0, 0.55, 0, 0, 0.55, 0])
    assert build_model({**TINY_SLOT, "prior": False}).head.slot_prior.abs().sum() == 0


def test_run_end_to_end_with_slot_images(tmp_path: Path) -> None:
    data = tmp_path / "data"
    make_synthetic_data(data, n_studies=8, depth=6, size=32, n_pseudo=4)
    config = {
        "model": TINY_SLOT,
        "input": "slot_image",
        "series_meta": "train_series_meta.csv",
        "img_out": 32,
        "seed": 0,
        "n_folds": 2,
        "epochs": 1,
        "batch_size": 2,
        "lr": 1e-3,
        "backbone_lr": 1e-5,
        "weight_decay": 0.02,
        "depth": 4,
        "size": 32,
        "target_slices": 6,
        "num_workers": 0,
        "amp": False,
        "train_on": "any",
        "eval_pseudo": True,
    }
    result = run(config, data_dir=data, out_dir=tmp_path / "out", device="cpu")
    assert result["n_studies"] == 8
    assert "pseudo_holdout_auc" in result
