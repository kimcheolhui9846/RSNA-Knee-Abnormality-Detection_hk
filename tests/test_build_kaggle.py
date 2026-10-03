import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))
from build_kaggle import build  # noqa: E402


def test_build_creates_code_weights_and_offline_kernel(tmp_path: Path) -> None:
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"w")
    config = tmp_path / "exp.yaml"
    config.write_text("depth: 16\n")
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    (wheels / "timm-1.0-py3-none-any.whl").write_bytes(b"x")

    dirs = build("me", weights, config, tmp_path / "out", wheels)

    code = dirs["code"]
    assert (code / "src" / "infer.py").is_file()
    assert (code / "rsna_knee_code.marker").is_file()
    assert (code / "wheels" / "timm-1.0-py3-none-any.whl").is_file()
    assert not list(code.rglob("__pycache__"))
    assert json.loads((code / "dataset-metadata.json").read_text())["id"] == "me/rsna-knee-code"

    w = dirs["weights"]
    assert (w / "model.safetensors").read_bytes() == b"w"
    assert (w / "config.yaml").read_text() == "depth: 16\n"
    assert (w / "rsna_knee_weights.marker").is_file()

    meta = json.loads((dirs["kernel"] / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False  # 대회 규칙: 인터넷 OFF
    assert meta["competition_sources"] == ["rsna-knee-abnormality-detection"]
    assert meta["dataset_sources"] == ["me/rsna-knee-code", "me/rsna-knee-weights"]
    assert (dirs["kernel"] / "submission.py").is_file()

    # 웹에서 "Import Notebook"으로 올릴 .ipynb: 제출 스크립트 전체가 코드 셀 하나에 들어 있다
    nb = json.loads((dirs["kernel"] / "rsna-knee-submit.ipynb").read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) == 1
    script = (dirs["kernel"] / "submission.py").read_text(encoding="utf-8")
    assert "".join(code_cells[0]["source"]) == script
