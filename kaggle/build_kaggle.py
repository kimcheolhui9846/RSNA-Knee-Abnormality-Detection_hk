"""Kaggle 업로드용 폴더 3개를 만든다 (업로드는 하지 않는다).

    outputs/kaggle/code/     → 데이터셋 <user>/rsna-knee-code
                               (src/, configs/, wheels/, 표시 파일)
    outputs/kaggle/weights/  → 데이터셋 <user>/rsna-knee-weights
                               (model.safetensors, config.yaml, 표시 파일)
    outputs/kaggle/kernel/   → 제출 노트북 <user>/rsna-knee-submit
                               (submission.py, kernel-metadata.json)

실행: `python kaggle/build_kaggle.py --user <kaggle 사용자명> --weights <model.safetensors>
       --config configs/exp001_baseline.yaml [--wheels <휠 폴더>]`
앙상블: `--weights` 대신 `--member <이름>=<model.safetensors>,<config.yaml>[,<가중치>]`를 여러 번
       → weights/<이름>/ 폴더들과 `ensemble.yaml` (추론은 라벨별 순위 가중 평균)
업로드(사용자 승인 후): `kaggle datasets create -p outputs/kaggle/code` (갱신은 `datasets version`),
`kaggle kernels push -p outputs/kaggle/kernel`
"""

import argparse
import json
import shutil
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
COMPETITION = "rsna-knee-abnormality-detection"
CODE_SLUG, WEIGHTS_SLUG, KERNEL_SLUG = "rsna-knee-code", "rsna-knee-weights", "rsna-knee-submit"


def _dataset_meta(user: str, slug: str) -> dict:
    return {"title": slug, "id": f"{user}/{slug}", "licenses": [{"name": "other"}]}


def build(
    user: str,
    weights: Path | None,
    config: Path | None,
    out: Path,
    wheels: Path | None = None,
    members: list[tuple[str, Path, Path, float]] | None = None,
) -> dict[str, Path]:
    if out.exists():
        shutil.rmtree(out)
    code, wdir, kernel = out / "code", out / "weights", out / "kernel"

    # 코드: 추론에 필요한 소스만 (테스트·데이터·노트북 제외)
    shutil.copytree(
        REPO / "src", code / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    shutil.copytree(REPO / "configs", code / "configs")
    if wheels is not None:
        shutil.copytree(wheels, code / "wheels")
    (code / "rsna_knee_code.marker").write_text("rsna-knee code dataset\n", encoding="utf-8")
    (code / "dataset-metadata.json").write_text(
        json.dumps(_dataset_meta(user, CODE_SLUG), indent=2)
    )

    wdir.mkdir(parents=True)
    if members:
        spec = []
        for name, m_weights, m_config, weight in members:
            (wdir / name).mkdir()
            shutil.copy2(m_weights, wdir / name / "model.safetensors")
            shutil.copy2(m_config, wdir / name / "config.yaml")
            spec.append({"dir": name, "weight": weight})
        (wdir / "ensemble.yaml").write_text(
            yaml.safe_dump({"members": spec}, sort_keys=False), encoding="utf-8"
        )
        source = ", ".join(f"{n}={w}" for n, w, _, _ in members)
    else:
        shutil.copy2(weights, wdir / "model.safetensors")
        shutil.copy2(config, wdir / "config.yaml")
        source = str(weights)
    (wdir / "rsna_knee_weights.marker").write_text(f"weights from {source}\n", encoding="utf-8")
    (wdir / "dataset-metadata.json").write_text(
        json.dumps(_dataset_meta(user, WEIGHTS_SLUG), indent=2)
    )

    kernel.mkdir(parents=True)
    shutil.copy2(REPO / "kaggle" / "submission.py", kernel / "submission.py")
    meta = {
        "id": f"{user}/{KERNEL_SLUG}",
        "title": KERNEL_SLUG,
        "code_file": "submission.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": [f"{user}/{CODE_SLUG}", f"{user}/{WEIGHTS_SLUG}"],
        "competition_sources": [COMPETITION],
        "kernel_sources": [],
    }
    (kernel / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    _write_notebook(kernel / "submission.py", kernel / f"{KERNEL_SLUG}.ipynb")
    return {"code": code, "weights": wdir, "kernel": kernel}


def _write_notebook(script: Path, out: Path) -> None:
    """Kaggle 웹 "Import Notebook"용 .ipynb: 제출 스크립트 전체를 코드 셀 하나에 담는다."""
    source = script.read_text(encoding="utf-8")
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"},
        },
        "cells": [
            {
                "cell_type": "markdown",
                "id": "intro",
                "metadata": {},
                "source": [
                    "# RSNA Knee — 제출 노트북\n",
                    "인터넷 OFF, GPU ON. 입력: 대회 데이터 + "
                    "rsna-knee-code + rsna-knee-weights 데이터셋",
                ],
            },
            {
                "cell_type": "code",
                "id": "submit",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": source.splitlines(keepends=True),
            },
        ],
    }
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


def parse_member(text: str) -> tuple[str, Path, Path, float]:
    """`exp002=w/model.safetensors,w/config.yaml,0.5` → (이름, 가중치, config, 가중치 비율)."""
    name, rest = text.split("=", 1)
    parts = rest.split(",")
    weight = float(parts[2]) if len(parts) > 2 else 1.0
    return name, Path(parts[0]), Path(parts[1]), weight


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--user", required=True, help="Kaggle 사용자명")
    p.add_argument("--weights", type=Path, default=None)
    p.add_argument(
        "--member",
        action="append",
        default=[],
        help="앙상블 멤버 <이름>=<model.safetensors>,<config.yaml>[,<가중치>] (여러 번)",
    )
    p.add_argument("--config", type=Path, default=REPO / "configs" / "exp001_baseline.yaml")
    p.add_argument("--wheels", type=Path, default=None)
    p.add_argument("--out", type=Path, default=REPO / "outputs" / "kaggle")
    args = p.parse_args()
    members = [parse_member(m) for m in args.member]
    if bool(members) == bool(args.weights):
        p.error("--weights와 --member 중 하나만 쓴다")
    built = build(args.user, args.weights, args.config, args.out, args.wheels, members or None)
    for name, path in built.items():
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        print(f"{name:8s} {path}  ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
