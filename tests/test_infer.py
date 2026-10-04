from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")  # CI는 torch 없이 돈다

from safetensors.torch import save_file  # noqa: E402

from src.constants import ID_COL, LABELS  # noqa: E402
from src.infer import FALLBACK_PROB, predict  # noqa: E402
from src.models.baseline import KneeBaseline  # noqa: E402
from tests.test_dicom import _slice, _write  # noqa: E402

TINY = {"backbone": "resnet18", "pretrained": False, "embed_dim": 16, "dropout": 0.0}
CONFIG = {"model": TINY, "depth": 4, "size": 32, "target_slices": 4}


def _test_dataset(root: Path) -> None:
    """공개 test와 같은 구성: test.csv(ID만), test_series.csv, test_series/<study>/<series>/*.dcm"""
    rows = []
    for study, series, plane, fs in [
        ("t1", "a", "Sagittal", 1),
        ("t1", "b", "Axial", 1),
        ("t2", "c", "Coronal", 0),
        ("t3", "broken", "Axial", 1),  # DICOM이 없는 망가진 study
    ]:
        if series != "broken":
            slices = [
                _slice(k * 20, [-float(k), 0.0, 0.0], k + 1, shape=(40, 40)) for k in range(5)
            ]
            _write(root / "test_series" / study / series, slices)
        else:
            (root / "test_series" / study / series).mkdir(parents=True)
        rows.append((study, series, fs, fs, plane))
    cols = [ID_COL, "SeriesInstanceUID", "Fluid_Sensitive", "Fat_Suppression", "Anatomical_Plane"]
    pd.DataFrame(rows, columns=cols).to_csv(root / "test_series.csv", index=False)
    pd.DataFrame({ID_COL: ["t1", "t2", "t3"]}).to_csv(root / "test.csv", index=False)


def _weights(path: Path, n_folds: int = 2) -> None:
    torch.manual_seed(0)
    state = {}
    for k in range(n_folds):
        model = KneeBaseline(**TINY)
        state.update({f"fold{k}.{n}": t.contiguous() for n, t in model.state_dict().items()})
    save_file(state, str(path))


def test_predict_writes_valid_submission_for_every_test_study(tmp_path: Path) -> None:
    _test_dataset(tmp_path / "data")
    _weights(tmp_path / "w.safetensors")

    sub = predict(
        tmp_path / "data",
        tmp_path / "w.safetensors",
        CONFIG,
        tmp_path / "submission.csv",
        device="cpu",
        num_workers=0,
    )

    assert list(sub.columns) == [ID_COL, *LABELS]
    assert sub[ID_COL].tolist() == ["t1", "t2", "t3"]
    on_disk = pd.read_csv(tmp_path / "submission.csv", dtype={ID_COL: str})
    pd.testing.assert_frame_equal(on_disk, sub, check_dtype=False)
    assert sub[list(LABELS)].apply(lambda c: c.between(0, 1)).all().all()


def test_unreadable_study_gets_fallback_instead_of_crashing(tmp_path: Path) -> None:
    _test_dataset(tmp_path / "data")
    _weights(tmp_path / "w.safetensors")

    sub = predict(
        tmp_path / "data",
        tmp_path / "w.safetensors",
        CONFIG,
        tmp_path / "s.csv",
        device="cpu",
        num_workers=0,
    ).set_index(ID_COL)

    assert np.allclose(sub.loc["t3"].to_numpy(dtype=float), FALLBACK_PROB)
    assert not np.allclose(sub.loc["t1"].to_numpy(dtype=float), FALLBACK_PROB)


def test_fold_count_is_read_from_weights(tmp_path: Path) -> None:
    _test_dataset(tmp_path / "data")
    _weights(tmp_path / "w1.safetensors", n_folds=1)
    _weights(tmp_path / "w3.safetensors", n_folds=3)
    for w in ("w1", "w3"):
        sub = predict(
            tmp_path / "data",
            tmp_path / f"{w}.safetensors",
            CONFIG,
            tmp_path / f"{w}.csv",
            device="cpu",
            num_workers=0,
        )
        assert len(sub) == 3


def test_predict_with_dino_attn_and_slab(tmp_path: Path) -> None:
    # exp004 모델(DINOv2 계열 + 라벨별 attention, 2.5D 입력)도 같은 경로로 추론한다
    from src.models import build_model

    dino = {
        "name": "dino_attn",
        "backbone": "vit_tiny_patch16_224",
        "pretrained": False,
        "img_size": 32,
        "freeze_blocks": 0,
        "n_heads": 2,
        "dropout": 0.0,
    }
    _test_dataset(tmp_path / "data")
    torch.manual_seed(0)
    state = {}
    for k in range(2):
        state.update(
            {f"fold{k}.{n}": t.contiguous() for n, t in build_model(dino).state_dict().items()}
        )
    save_file(state, str(tmp_path / "w.safetensors"))
    config = {"model": dino, "depth": 4, "size": 32, "target_slices": 4, "slab": 3}

    sub = predict(
        tmp_path / "data",
        tmp_path / "w.safetensors",
        config,
        tmp_path / "s.csv",
        device="cpu",
        num_workers=0,
    )
    assert len(sub) == 3
    assert sub[list(LABELS)].apply(lambda c: c.between(0, 1)).all().all()
