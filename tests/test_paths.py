from pathlib import Path

import pytest

from src.paths import data_dir


def test_env_var_takes_priority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSNA_DATA_DIR", str(tmp_path))
    assert data_dir(candidates=[Path("/does/not/exist")]) == tmp_path


def test_first_existing_candidate_is_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RSNA_DATA_DIR", raising=False)
    existing = tmp_path / "local"
    existing.mkdir()
    assert data_dir(candidates=[tmp_path / "kaggle", existing]) == existing


def test_no_candidate_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RSNA_DATA_DIR", raising=False)
    with pytest.raises(FileNotFoundError):
        data_dir(candidates=[tmp_path / "missing"])
