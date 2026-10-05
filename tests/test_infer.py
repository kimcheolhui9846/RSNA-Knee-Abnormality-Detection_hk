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


def test_missing_fat_suppression_falls_back_to_fluid_sensitive_or_skips(tmp_path: Path) -> None:
    # 숨은 test에 메타데이터 결측이 있어도 추론 전체가 멈추면 안 된다
    _test_dataset(tmp_path / "data")
    _weights(tmp_path / "w.safetensors")
    kwargs = {"device": "cpu", "num_workers": 0}
    clean = predict(
        tmp_path / "data", tmp_path / "w.safetensors", CONFIG, tmp_path / "a.csv", **kwargs
    ).set_index(ID_COL)

    series = pd.read_csv(tmp_path / "data" / "test_series.csv", dtype=str)
    series.loc[series["SeriesInstanceUID"] == "a", "Fat_Suppression"] = (
        None  # Fluid_Sensitive로 대신
    )
    series.loc[series["SeriesInstanceUID"] == "c", ["Fat_Suppression", "Fluid_Sensitive"]] = None
    series.to_csv(tmp_path / "data" / "test_series.csv", index=False)
    sub = predict(
        tmp_path / "data", tmp_path / "w.safetensors", CONFIG, tmp_path / "b.csv", **kwargs
    ).set_index(ID_COL)

    np.testing.assert_allclose(sub.loc["t1"], clean.loc["t1"], rtol=1e-5)
    assert np.allclose(
        sub.loc["t2"].to_numpy(dtype=float), FALLBACK_PROB
    )  # 칸을 못 정한 유일한 시리즈


# --- 여러 모델 앙상블 (순위 평균) ------------------------------------------------


def test_rank_average_uses_ranks_not_raw_probabilities() -> None:
    from src.infer import rank_average

    a = np.array([[0.1], [0.2], [0.3]])  # 순위 1, 2, 3
    b = np.array([[0.99], [0.98], [0.97]])  # 순위 3, 2, 1 — 값 크기는 무시된다
    out = rank_average([a, b], [1.0, 1.0])
    np.testing.assert_allclose(out[:, 0], [2 / 3, 2 / 3, 2 / 3])
    np.testing.assert_allclose(rank_average([a, b], [3.0, 1.0])[:, 0], [0.5, 2 / 3, 5 / 6])


def _member_dir(root: Path, seed: int, config: dict) -> None:
    root.mkdir(parents=True)
    torch.manual_seed(seed)
    state = {}
    for k in range(2):
        model = KneeBaseline(**TINY)
        state.update({f"fold{k}.{n}": t.contiguous() for n, t in model.state_dict().items()})
    save_file(state, str(root / "model.safetensors"))
    import yaml

    (root / "config.yaml").write_text(yaml.safe_dump(config))


def test_load_members_reads_single_model_or_ensemble_spec(tmp_path: Path) -> None:
    from src.infer import load_members

    _member_dir(tmp_path / "single", 0, CONFIG)
    single = load_members(tmp_path / "single")
    assert [(m.name, m.weight) for m in single] == [("model", 1.0)]

    ens = tmp_path / "ens"
    _member_dir(ens / "a", 0, CONFIG)
    _member_dir(ens / "b", 1, CONFIG)
    (ens / "ensemble.yaml").write_text(
        "members:\n- {dir: a, weight: 0.7}\n- {dir: b, weight: 0.3}\n"
    )
    members = load_members(ens)
    assert [(m.name, m.weight) for m in members] == [("a", 0.7), ("b", 0.3)]
    assert members[1].weights_path == ens / "b" / "model.safetensors"


def test_ensemble_submission_is_rank_average_of_members(tmp_path: Path) -> None:
    from src.infer import load_members, predict_members, rank_average

    _test_dataset(tmp_path / "data")
    ens = tmp_path / "ens"
    _member_dir(ens / "a", 0, CONFIG)
    _member_dir(ens / "b", 1, CONFIG)
    (ens / "ensemble.yaml").write_text(
        "members:\n- {dir: a, weight: 0.7}\n- {dir: b, weight: 0.3}\n"
    )
    kw = {"device": "cpu", "num_workers": 0}
    data = tmp_path / "data"

    sub = predict_members(data, load_members(ens), tmp_path / "e.csv", **kw)
    a = predict(data, ens / "a" / "model.safetensors", CONFIG, tmp_path / "a.csv", **kw)
    b = predict(data, ens / "b" / "model.safetensors", CONFIG, tmp_path / "b.csv", **kw)

    expected = rank_average([a[list(LABELS)].to_numpy(), b[list(LABELS)].to_numpy()], [0.7, 0.3])
    np.testing.assert_allclose(sub[list(LABELS)].to_numpy(), expected, rtol=1e-5)
    assert sub[ID_COL].tolist() == ["t1", "t2", "t3"]


SLOT_DINO = {
    "name": "slot_dino",
    "backbone": "vit_tiny_patch16_224",
    "pretrained": False,
    "img_size": 32,
    "unfreeze_last": 1,
    "dropout": 0.0,
}
SLOT_CONFIG = {
    "model": SLOT_DINO,
    "input": "slot_image",
    "img_out": 32,
    "depth": 4,
    "size": 32,
    "target_slices": 4,
}


def _slot_member(path: Path) -> None:
    from src.models import build_model

    torch.manual_seed(2)
    state = {}
    for k in range(2):
        model = build_model(SLOT_DINO)
        state.update({f"fold{k}.{n}": t.contiguous() for n, t in model.state_dict().items()})
    save_file(state, str(path))


def test_slot_image_member_predicts_from_dicom_headers(tmp_path: Path) -> None:
    # exp006: 헤더로 칸을 고르고 칸마다 RGB 1장. 칸이 있는 study는 fallback이 아니다
    _test_dataset(tmp_path / "data")
    _slot_member(tmp_path / "s.safetensors")
    sub = predict(
        tmp_path / "data",
        tmp_path / "s.safetensors",
        SLOT_CONFIG,
        tmp_path / "s.csv",
        device="cpu",
        num_workers=0,
    ).set_index(ID_COL)
    assert not np.allclose(sub.loc["t1"].to_numpy(dtype=float), FALLBACK_PROB)
    assert np.allclose(sub.loc["t3"].to_numpy(dtype=float), FALLBACK_PROB)  # 읽을 시리즈 없음


def test_members_with_different_inputs_run_in_one_pass(tmp_path: Path) -> None:
    # 슬라이스 입력(b0)과 칸 이미지 입력(DINO)을 함께 앙상블: 입력 묶음별로 만들고 순위 평균
    from src.infer import Member, predict_members, rank_average

    _test_dataset(tmp_path / "data")
    _member_dir(tmp_path / "a", 0, CONFIG)
    _slot_member(tmp_path / "s.safetensors")
    kw = {"device": "cpu", "num_workers": 0}
    data = tmp_path / "data"
    members = [
        Member("a", tmp_path / "a" / "model.safetensors", CONFIG),
        Member("s", tmp_path / "s.safetensors", SLOT_CONFIG),
    ]
    sub = predict_members(data, members, tmp_path / "e.csv", **kw)
    a = predict(data, tmp_path / "a" / "model.safetensors", CONFIG, tmp_path / "a.csv", **kw)
    s = predict(data, tmp_path / "s.safetensors", SLOT_CONFIG, tmp_path / "s.csv", **kw)
    expected = rank_average([a[list(LABELS)].to_numpy(), s[list(LABELS)].to_numpy()], [1, 1])
    np.testing.assert_allclose(sub[list(LABELS)].to_numpy(), expected, rtol=1e-5)
