from pathlib import Path

import pytest

pytest.importorskip("huggingface_hub")


def test_fetch_data_downloads_only_what_the_config_uses(tmp_path: Path, monkeypatch) -> None:
    import huggingface_hub

    import train

    calls, beats = [], []

    def fake(repo, **kw):
        calls.append((repo, kw))

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
    config = {"data_repo": "u/d", "corpus_dir": "raptor_corpus64", "labels_file": "labels_x.csv"}
    train.fetch_data(config, tmp_path, beat=lambda: beats.append(1))
    ((repo, kw),) = calls
    assert repo == "u/d" and kw["repo_type"] == "dataset" and kw["local_dir"] == str(tmp_path)
    assert kw["allow_patterns"] == ["raptor_corpus64/*", "labels_x.csv", "folds.csv"]
