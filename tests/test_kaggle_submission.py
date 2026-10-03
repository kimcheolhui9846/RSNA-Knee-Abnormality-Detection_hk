import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))
import submission  # noqa: E402


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")
    return path


def test_finds_files_in_current_kaggle_mount_layout(tmp_path: Path) -> None:
    code = _touch(tmp_path / "datasets" / "me" / "rsna-knee-code" / "rsna_knee_code.marker")
    comp = _touch(tmp_path / "competitions" / "rsna-knee" / "test_series.csv")
    assert submission.find_one("rsna_knee_code.marker", tmp_path) == code
    assert submission.find_one("test_series.csv", tmp_path) == comp


def test_finds_files_in_legacy_flat_layout(tmp_path: Path) -> None:
    marker = _touch(tmp_path / "rsna-knee-weights" / "rsna_knee_weights.marker")
    assert submission.find_one("rsna_knee_weights.marker", tmp_path) == marker


def test_does_not_descend_into_dicom_trees(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 대회 DICOM 폴더(수십만 파일) 안은 열지 않는다 — 1차 제출에서 재귀 glob이 15분 넘게 걸렸다
    comp = tmp_path / "competitions" / "rsna-knee"
    _touch(comp / "test_series.csv")
    _touch(comp / "train_series" / "s1" / "series" / "rsna_knee_code.marker")  # 미끼
    opened: list[str] = []
    real_scandir = os.scandir

    def spy(path):
        opened.append(Path(path).name)
        return real_scandir(path)

    monkeypatch.setattr(submission.os, "scandir", spy)
    with pytest.raises(FileNotFoundError):
        submission.find_one("rsna_knee_code.marker", tmp_path)
    assert "train_series" not in opened and "s1" not in opened


def test_respects_max_depth(tmp_path: Path) -> None:
    _touch(tmp_path / "a" / "b" / "c" / "d" / "e" / "deep.marker")
    with pytest.raises(FileNotFoundError):
        submission.find_one("deep.marker", tmp_path, max_depth=4)
