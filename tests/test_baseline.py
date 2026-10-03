from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")  # CI는 torch 없이 돈다 (--extra train 미설치)

from src.constants import ID_COL, LABELS  # noqa: E402
from src.data.study_table import SLOTS  # noqa: E402
from src.losses import masked_bce  # noqa: E402
from src.models.baseline import KneeBaseline  # noqa: E402
from src.train import make_synthetic_data, run  # noqa: E402

TINY = {"backbone": "resnet18", "pretrained": False, "embed_dim": 32, "dropout": 0.0}


def _batch(b: int = 2, depth: int = 4, size: int = 32) -> tuple:
    image = torch.rand(b, len(SLOTS), depth, size, size)
    slot_mask = torch.ones(b, len(SLOTS), dtype=torch.bool)
    slot_mask[0, -1] = False
    return image, slot_mask


# --- model --------------------------------------------------------------------


def test_forward_returns_one_logit_per_label() -> None:
    model = KneeBaseline(**TINY).eval()
    image, slot_mask = _batch()
    assert model(image, slot_mask).shape == (2, len(LABELS))


def test_masked_slot_content_does_not_change_output() -> None:
    model = KneeBaseline(**TINY).eval()
    image, slot_mask = _batch()
    with torch.no_grad():
        before = model(image, slot_mask)
        image[0, -1] = torch.rand_like(image[0, -1])  # 없는 칸에 아무 값이나 넣어도
        after = model(image, slot_mask)
    torch.testing.assert_close(before, after)


# --- loss ---------------------------------------------------------------------


def test_masked_bce_ignores_missing_labels() -> None:
    logits = torch.randn(3, len(LABELS))
    labels = torch.randint(0, 2, (3, len(LABELS))).float()
    mask = torch.zeros(3, len(LABELS), dtype=torch.bool)
    mask[0] = True
    loss = masked_bce(logits, labels, mask)
    labels[1:] = 1 - labels[1:]  # 마스크 밖 라벨을 뒤집어도
    torch.testing.assert_close(masked_bce(logits, labels, mask), loss)


def test_masked_bce_with_no_labels_is_zero_not_nan() -> None:
    logits = torch.randn(2, len(LABELS), requires_grad=True)
    loss = masked_bce(logits, torch.zeros(2, len(LABELS)), torch.zeros(2, len(LABELS), dtype=bool))
    assert float(loss.detach()) == 0.0
    loss.backward()  # 그래프가 끊기지 않아야 한다


# --- end-to-end CV (합성 데이터, CPU) -----------------------------------------


def test_run_cross_validation_on_synthetic_data(tmp_path: Path) -> None:
    data = tmp_path / "data"
    make_synthetic_data(data, n_studies=12, depth=6, size=32)
    config = {
        "model": TINY,
        "seed": 0,
        "n_folds": 2,
        "epochs": 1,
        "batch_size": 2,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "depth": 4,
        "size": 32,
        "target_slices": 6,
        "num_workers": 0,
        "amp": False,
    }

    result = run(config, data_dir=data, out_dir=tmp_path / "out", device="cpu")

    assert set(result) >= {"macro_auc", "per_label_auc", "fold_macro_auc"}
    assert len(result["fold_macro_auc"]) == 2
    oof = pd.read_csv(tmp_path / "out" / "oof.csv")
    assert list(oof.columns) == [ID_COL, "fold", *LABELS]
    assert len(oof) == 12
    assert oof[list(LABELS)].apply(lambda c: c.between(0, 1)).all().all()

    from safetensors.torch import load_file

    weights = load_file(tmp_path / "out" / "model.safetensors")
    assert any(k.startswith("fold0.") for k in weights)
    assert any(k.startswith("fold1.") for k in weights)
    assert np.isfinite(result["macro_auc"]) or np.isnan(result["macro_auc"])


def test_pseudo_labels_are_trained_on_but_only_ground_truth_is_evaluated(tmp_path: Path) -> None:
    data = tmp_path / "data"
    # 정답 12개 + pseudo 8개(일부 라벨 NaN), 캐시는 데이터 루트에 바로 둔다 (HF 데이터셋 구성)
    make_synthetic_data(data, n_studies=12, depth=6, size=32, n_pseudo=8, cache_subdir=".")
    config = {
        "model": TINY,
        "seed": 0,
        "n_folds": 2,
        "epochs": 1,
        "batch_size": 4,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "depth": 4,
        "size": 32,
        "target_slices": 6,
        "num_workers": 0,
        "amp": False,
        "train_on": "any",
        "cache_dir": ".",
    }

    result = run(config, data_dir=data, out_dir=tmp_path / "out", device="cpu")

    assert result["n_studies"] == 12  # 평가 대상 = 정답 study만
    assert result["n_train_studies"] == 20  # 학습 대상 = 정답 + pseudo
    oof = pd.read_csv(tmp_path / "out" / "oof.csv")
    assert len(oof) == 12
    assert not oof[ID_COL].str.startswith("pseudo").any()
