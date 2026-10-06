"""하네스 진입점 `train.py --smoke-test`가 저장소의 모든 실험 config에서 끝까지 도는지 (CPU)."""

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("timm")

REPO = Path(__file__).resolve().parents[1]
CONFIGS = sorted((REPO / "configs").glob("exp*.yaml"))


def _entry():
    spec = importlib.util.spec_from_file_location("harness_entry", REPO / "train.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass가 자기 모듈을 찾을 수 있게
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("config", CONFIGS, ids=lambda p: p.stem)
def test_smoke_test_runs_for_every_experiment_config(config: Path, tmp_path: Path) -> None:
    entry = _entry()
    cfg = entry.Config(
        run_id="test",
        data_dir=tmp_path / "unused",
        output_dir=tmp_path / "out",
        checkpoint_dir=tmp_path / "ckpt",
        hf_repo_id=None,
        git_commit="test",
        smoke_test=True,
        config_path=str(config),
    )
    final = entry.train_model(cfg)
    if "trainer: raptor_predict" in config.read_text(encoding="utf-8"):
        # 예측 전용: 새 가중치 없이 train OOF만 남긴다
        assert (tmp_path / "out" / "oof_pseudo.csv").is_file()
    else:
        assert Path(final).is_file()
