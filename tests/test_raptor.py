from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # CI는 torch 없이 돈다

from src.constants import LABELS  # noqa: E402
from src.data.raptor_stack import make_windows, window_centers  # noqa: E402
from src.models import build_model  # noqa: E402
from src.train_raptor import make_synthetic_corpus, open_corpus, run_raptor  # noqa: E402

TINY = {"name": "raptor", "backbone": "resnet18", "pretrained": False, "dropout": 0.0}


def test_window_centers_eval_spread_and_train_random() -> None:
    mask = np.array([0, 1, 1, 1, 1, 1, 1, 1, 0, 0], np.uint8)  # 유효 1..7 → 중심 2..6
    assert window_centers(mask, 5) == [2, 3, 4, 5, 6]
    rnd = window_centers(mask, 12, np.random.default_rng(0))
    assert len(rnd) == 12 and set(rnd) <= {2, 3, 4, 5, 6}


def test_make_windows_stacks_neighbours_as_rgb() -> None:
    vol = np.arange(5, dtype=np.uint8)[:, None, None] * np.ones((1, 4, 4), np.uint8) * 50
    x = make_windows(vol, [2], res=4, normalize=False)
    assert x.shape == (1, 3, 4, 4)
    assert (x[0, :, 0, 0] * 255).round().tolist() == [50, 100, 150]


def test_raptor_model_shape_and_per_label_attention() -> None:
    model = build_model(TINY).eval()
    with torch.no_grad():
        out = model(torch.rand(2, 5, 3, 32, 32))
    assert out.shape == (2, len(LABELS))
    assert model.att[-1].out_features == len(LABELS)  # 소견마다 창 가중치


def test_open_corpus_joins_two_parts(tmp_path: Path) -> None:
    make_synthetic_corpus(tmp_path)
    vols, masks, rows = open_corpus(tmp_path / "raptor_corpus")
    assert len(vols) == len(masks) == len(rows) == 16
    assert vols[rows["ps9"]].shape == (12, 32, 32)  # 둘째 부분의 마지막 study


def test_run_raptor_end_to_end(tmp_path: Path) -> None:
    make_synthetic_corpus(tmp_path / "data")
    config = {
        "model": TINY,
        "seed": 0,
        "corpus_dir": "raptor_corpus",
        "labels_file": "labels.csv",
        "holdout_fold": 1,
        "res": 32,
        "k": 4,
        "k_eval": 6,
        "epochs": 2,
        "batch_size": 2,
        "lr": 1e-3,
        "backbone_lr": 1e-4,
        "weight_decay": 0.0,
        "swa_last": 2,
        "amp": False,
        "num_workers": 0,
    }
    result = run_raptor(config, tmp_path / "data", tmp_path / "out", device="cpu")
    assert result["n_studies"] == 6 and result["n_train_studies"] == 5  # pseudo 10 중 fold 0만 학습
    assert result["swa_epochs"] == 2
    assert "pseudo_holdout_auc" in result
    assert (tmp_path / "out" / "oof.csv").is_file()
