import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))
from build_public_blend import BLEND, build  # noqa: E402

LABELS = ["ACL", "MCL"]


def _public(tmp: Path) -> Path:
    d = tmp / "public"
    d.mkdir()
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {
                "cell_type": "code",
                "id": "x",
                "metadata": {},
                "execution_count": 3,
                "outputs": [{"output_type": "stream", "text": "old"}],
                "source": ["print(1)\n"],
            },
        ],
    }
    (d / "pub.ipynb").write_text(json.dumps(nb))
    meta = {
        "id": "someone/pub",
        "code_file": "pub.ipynb",
        "dataset_sources": ["a/weights"],
        "kernel_sources": ["b/train"],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": ["m/dinov2"],
        "docker_image": "gcr.io/pinned@sha256:abc",
        "machine_shape": "NvidiaTeslaT4",
    }
    (d / "kernel-metadata.json").write_text(json.dumps(meta))
    return d


def test_build_keeps_public_cells_and_appends_ours(tmp_path: Path) -> None:
    out = build(_public(tmp_path), "me", 0.15, tmp_path / "out")
    nb = json.loads((out / "rsna-knee-public-blend.ipynb").read_text(encoding="utf-8"))
    ids = [c["id"] for c in nb["cells"]]
    assert ids == ["ours-attribution", "x", "ours-run", "ours-blend"]
    assert nb["cells"][1]["outputs"] == []  # 공개 노트북의 저장된 출력은 버린다
    assert "OUR_WEIGHT = 0.15" in "".join(nb["cells"][-1]["source"])
    meta = json.loads((out / "kernel-metadata.json").read_text())
    assert meta["dataset_sources"] == ["a/weights", "me/rsna-knee-code", "me/rsna-knee-weights"]
    assert meta["kernel_sources"] == ["b/train"] and meta["model_sources"] == ["m/dinov2"]
    assert meta["docker_image"] == "gcr.io/pinned@sha256:abc"  # 공개 노트북과 같은 실행 환경
    assert meta["enable_internet"] is False and meta["is_private"] is True


def _run_blend(work: Path, weight: float, ours_ok: bool) -> pd.DataFrame:
    ids = [f"s{i}" for i in range(4)]
    pub = pd.DataFrame(
        {"StudyInstanceUID": ids, "ACL": [0.1, 0.2, 0.3, 0.4], "MCL": [0.4, 0.3, 0.2, 0.1]}
    )
    pub.to_csv(work / "public_submission.csv", index=False)
    ours = work / "ours_submission.csv"
    if ours_ok:  # 순서를 섞어 저장해도 study로 맞춘다
        pd.DataFrame(
            {"StudyInstanceUID": ids[::-1], "ACL": [0.9, 0.8, 0.7, 0.6], "MCL": [1, 2, 3, 4]}
        ).to_csv(ours, index=False)
    exec(BLEND.format(w=weight), {"_work": work, "_ours": ours})  # noqa: S102
    return pd.read_csv(work / "submission.csv")


def test_blend_mixes_ranks(tmp_path: Path) -> None:
    out = _run_blend(tmp_path, 0.5, ours_ok=True)
    # ACL: 공개 순위 .25 .5 .75 1, 우리(study 순서로 맞춘 값 .6 .7 .8 .9) 같은 순위 → 그대로
    np.testing.assert_allclose(out["ACL"], [0.25, 0.5, 0.75, 1.0])
    # MCL: 공개 1 .75 .5 .25, 우리(s0..s3 = 4 3 2 1) 1 .75 .5 .25 → 그대로
    np.testing.assert_allclose(out["MCL"], [1.0, 0.75, 0.5, 0.25])


def test_blend_falls_back_to_public_when_ours_missing(tmp_path: Path) -> None:
    out = _run_blend(tmp_path, 0.3, ours_ok=False)
    np.testing.assert_allclose(out["ACL"], [0.1, 0.2, 0.3, 0.4])
