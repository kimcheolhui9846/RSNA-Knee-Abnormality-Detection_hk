from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")  # CI는 torch 없이 돈다
pytest.importorskip("cv2")

from src.constants import ID_COL  # noqa: E402
from src.data.build_corpus import build_corpus  # noqa: E402
from src.data.raptor_stack import SLOTS64, build_stack  # noqa: E402
from src.train_raptor import open_corpus  # noqa: E402
from tests.test_dicom import _slice, _write  # noqa: E402


def _train_dataset(root: Path) -> None:
    rows = []
    for study, series, plane, fs in [
        ("s1", "a", "Sagittal", 1),
        ("s1", "b", "Coronal", 0),
        ("s2", "c", "Axial", 1),
    ]:
        slices = [_slice(k * 20, [-float(k), 0.0, 0.0], k + 1, shape=(40, 40)) for k in range(6)]
        _write(root / "train_series" / study / series, slices)
        rows.append((study, series, fs, fs, plane))
    cols = [ID_COL, "SeriesInstanceUID", "Fluid_Sensitive", "Fat_Suppression", "Anatomical_Plane"]
    pd.DataFrame(rows, columns=cols).to_csv(root / "train_series.csv", index=False)
    pd.DataFrame({ID_COL: ["s2", "s1"], "Report": ["", ""]}).to_csv(root / "train.csv", index=False)


def test_build_corpus_matches_build_stack_and_opens(tmp_path: Path) -> None:
    _train_dataset(tmp_path)
    out = build_corpus(tmp_path, tmp_path / "c64", "SLOTS64", (0.06, 0.94), 32, 140.0, workers=2)
    vols, masks, rows = open_corpus(out)
    assert list(rows) == ["s2", "s1"]  # train.csv 순서
    assert vols[0].shape == (64, 32, 32)
    series = pd.read_csv(tmp_path / "train_series.csv", dtype=str)
    for sid in ("s1", "s2"):
        recs = series[series[ID_COL] == sid].to_dict("records")
        vol, mask = build_stack(recs, tmp_path / "train_series" / sid, SLOTS64, (0.06, 0.94), 32)
        np.testing.assert_array_equal(vols[rows[sid]], vol)  # 추론과 같은 스택
        np.testing.assert_array_equal(masks[rows[sid]], mask)
    m1 = masks[rows["s1"]]
    # 칸: 시상 fluid 0–17, 시상 non-fluid 18–31, 관상 fluid 32–43, 관상 44–51, 축상 52–63
    assert m1[:18].any() and m1[32:44].any()  # 관상은 fluid 칸이 non-fluid 시리즈로 대신 채워짐
    assert not m1[18:32].any() and not m1[44:].any()


def test_build_corpus_resumes_after_interruption(tmp_path: Path, monkeypatch) -> None:
    import src.data.build_corpus as bc

    _train_dataset(tmp_path)
    out = tmp_path / "c"
    full = build_corpus(tmp_path, tmp_path / "full", "SLOTS64", (0.06, 0.94), 32, workers=1)
    real_one, calls = bc._one, []

    def flaky(args):
        calls.append(args[0])
        if len(calls) == 2:
            raise RuntimeError("중단")
        return real_one(args)

    monkeypatch.setattr(bc, "_one", flaky)
    monkeypatch.setattr(bc, "Pool", _SerialPool)
    with pytest.raises(RuntimeError):  # 첫 study만 끝내고 두 번째에서 끊긴다
        build_corpus(tmp_path, out, "SLOTS64", (0.06, 0.94), 32, workers=1, save_every=1)
    assert (out / "_partial.npz").exists()
    calls.clear()
    monkeypatch.setattr(bc, "_one", lambda a: (calls.append(a[0]), real_one(a))[1])
    build_corpus(tmp_path, out, "SLOTS64", (0.06, 0.94), 32, workers=1, save_every=1)
    assert calls == [1]  # 끝난 study는 다시 만들지 않는다
    np.testing.assert_array_equal(np.load(out / "all_vols.npy"), np.load(full / "all_vols.npy"))
    np.testing.assert_array_equal(np.load(out / "all_masks.npy"), np.load(full / "all_masks.npy"))
    assert not (out / "_partial.npz").exists()


class _SerialPool:
    def __init__(self, *a, **k) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None

    def imap_unordered(self, fn, jobs):
        return map(fn, jobs)
