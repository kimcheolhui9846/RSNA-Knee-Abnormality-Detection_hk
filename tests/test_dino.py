from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")  # CI는 torch 없이 돈다

from src.constants import ID_COL, LABELS  # noqa: E402
from src.data.dataset import KneeStudyDataset  # noqa: E402
from src.data.study_table import SLOTS  # noqa: E402
from src.models import build_model  # noqa: E402
from src.train import make_synthetic_data, run  # noqa: E402

TINY_DINO = {
    "name": "dino_attn",
    # 테스트용 작은 ViT (실험은 vit_small_patch14_dinov2.lvd142m)
    "backbone": "vit_tiny_patch16_224",
    "pretrained": False,
    "img_size": 32,
    "freeze_blocks": 2,
    "n_heads": 2,
    "dropout": 0.0,
}


def _batch(b: int = 2, depth: int = 4, size: int = 32) -> tuple:
    image = torch.rand(b, len(SLOTS), depth, 3, size, size)
    slot_mask = torch.ones(b, len(SLOTS), dtype=torch.bool)
    slot_mask[0, -1] = False
    return image, slot_mask


def test_dino_attn_returns_one_logit_per_label() -> None:
    model = build_model(TINY_DINO).eval()
    image, mask = _batch()
    assert model(image, mask).shape == (2, len(LABELS))


def test_dino_attn_ignores_missing_slot_content() -> None:
    model = build_model(TINY_DINO).eval()
    image, mask = _batch()
    with torch.no_grad():
        before = model(image, mask)
        image[0, -1] = torch.rand_like(image[0, -1])
        after = model(image, mask)
    torch.testing.assert_close(before, after)


def test_freeze_blocks_freezes_patch_embed_and_first_blocks() -> None:
    model = build_model(TINY_DINO)
    enc = model.encoder
    assert not any(p.requires_grad for p in enc.patch_embed.parameters())
    assert not any(p.requires_grad for p in enc.blocks[1].parameters())
    assert all(p.requires_grad for p in enc.blocks[2].parameters())


def test_build_model_defaults_to_baseline() -> None:
    model = build_model({"backbone": "resnet18", "pretrained": False, "embed_dim": 16})
    image = torch.rand(1, len(SLOTS), 4, 32, 32)
    assert model(image, torch.ones(1, len(SLOTS), dtype=torch.bool)).shape == (1, len(LABELS))


def test_dataset_slab_stacks_neighbouring_slices_as_channels(tmp_path: Path) -> None:
    (tmp_path / "s1").mkdir()
    vol = np.arange(10, dtype=np.uint8)[:, None, None] * np.ones((1, 8, 8), dtype=np.uint8)
    np.save(tmp_path / "s1" / "a.npy", vol)
    paths = [None] * len(SLOTS)
    paths[0] = "s1/a.npy"
    table = pd.DataFrame(
        {
            ID_COL: ["s1"],
            "fold": [0],
            "labels": [np.zeros(len(LABELS), dtype=np.float32)],
            "label_mask": [np.ones(len(LABELS), dtype=bool)],
            "slot_paths": [paths],
        }
    )
    item = KneeStudyDataset(table, cache_root=tmp_path, depth=10, size=8, slab=3)[0]
    img = item["image"]  # (S, D, 3, H, W)
    assert img.shape == (len(SLOTS), 10, 3, 8, 8)
    centre = (img[0, :, 1, 0, 0] * 255).round().tolist()
    prev = (img[0, :, 0, 0, 0] * 255).round().tolist()
    assert centre == list(range(10))
    assert prev == [0] + list(range(9))  # 첫 슬라이스의 이전은 자기 자신으로 채운다


def test_run_end_to_end_with_dino_attn_and_slab(tmp_path: Path) -> None:
    data = tmp_path / "data"
    make_synthetic_data(data, n_studies=8, depth=6, size=32, n_pseudo=4)
    config = {
        "model": TINY_DINO,
        "seed": 0,
        "n_folds": 2,
        "epochs": 1,
        "batch_size": 2,
        "lr": 1e-3,
        "backbone_lr": 1e-4,
        "weight_decay": 0.0,
        "depth": 4,
        "slab": 3,
        "size": 32,
        "target_slices": 6,
        "num_workers": 0,
        "amp": False,
        "train_on": "any",
    }
    result = run(config, data_dir=data, out_dir=tmp_path / "out", device="cpu")
    assert result["n_studies"] == 8
    assert len(result["fold_macro_auc"]) == 2
