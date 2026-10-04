"""Kaggle 제출 노트북 (script kernel, 인터넷 OFF).

붙여야 하는 입력:
- 대회 데이터: rsna-knee-abnormality-detection
- 코드 데이터셋: <user>/rsna-knee-code   (src/, configs/, wheels/ — kaggle/build_kaggle.py가 만든다)
- 가중치 데이터셋: <user>/rsna-knee-weights (model.safetensors, config.yaml)

Kaggle의 입력 마운트 경로가 바뀌어도 동작하도록 파일을 찾아서 경로를 정한다.
"""

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# 로컬에서 Kaggle 구조를 흉내 내 시험할 때만 환경변수로 바꾼다
INPUT = Path(os.environ.get("KAGGLE_INPUT_DIR", "/kaggle/input"))
OUT = Path(os.environ.get("KAGGLE_SUBMISSION_PATH", "/kaggle/working/submission.csv"))


# 대회 DICOM 트리(수십만 파일). 1차 제출에서 `/kaggle/input/**` 재귀 glob이
# 이 안까지 훑어 15분 넘게 걸렸다
SKIP_DIRS = {"train_series", "test_series"}


def find_one(name: str, root: Path | None = None, max_depth: int = 4) -> Path:
    """`root` 아래에서 파일 이름이 `name`인 첫 파일을 얕은 폴더부터(너비 우선) 찾는다.

    깊이 `max_depth`까지의 폴더만 열고, DICOM 트리(`SKIP_DIRS`)는 열지 않는다.
    현재 마운트(`datasets/<user>/<slug>/`, `competitions/<slug>/`)와
    예전 마운트(`<slug>/`) 모두 깊이 3 이내다.
    """
    root = INPUT if root is None else Path(root)
    level = [root]
    for _ in range(max_depth + 1):
        next_level: list[Path] = []
        for d in sorted(level):
            try:
                entries = list(os.scandir(d))
            except OSError:
                continue
            for e in sorted(entries, key=lambda x: x.name):
                if e.is_file() and e.name == name:
                    return Path(e.path)
                if e.is_dir() and e.name not in SKIP_DIRS:
                    next_level.append(Path(e.path))
        level = next_level
    raise FileNotFoundError(f"{root}에서 {name}을 찾지 못했다 (깊이 {max_depth}까지)")


def main() -> None:
    started = time.time()
    logging.basicConfig(
        level=logging.INFO, format="[submit %(asctime)s] %(message)s", datefmt="%H:%M:%S"
    )
    log = logging.getLogger("submit")

    code = find_one("rsna_knee_code.marker").parent
    weights = find_one("rsna_knee_weights.marker").parent
    competition = find_one("test_series.csv").parent
    log.info(
        "code %s | weights %s | data %s (경로 탐색 %.1fs)",
        code,
        weights,
        competition,
        time.time() - started,
    )

    log.info("python %s", sys.version.split()[0])

    # 오프라인 휠: 압축 DICOM 디코더와 학습 때와 같은 버전의 timm.
    # 패키지마다 따로 설치한다 — 한 줄로 묶으면 하나(맞는 Python 버전 휠 없음)만 실패해도
    # 전부 설치되지 않는다 (2026-10-03 Kaggle 1차 실행에서 확인).
    # 실패해도 Kaggle 이미지에 이미 있는 패키지로 진행한다
    wheels = code / "wheels"
    if wheels.is_dir():
        cmd = [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q"]
        for pkg in ("timm", "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"):
            r = subprocess.run(
                [*cmd, "--find-links", str(wheels), pkg], capture_output=True, text=True
            )
            log.info("offline wheel %s rc=%d %s", pkg, r.returncode, r.stderr.strip()[-200:])
    try:
        import libjpeg  # noqa: F401
        import openjpeg  # noqa: F401

        log.info("compressed DICOM decoders: available")
    except ImportError as e:
        log.info("compressed DICOM decoders: MISSING (%s) — 압축 DICOM은 fallback될 수 있다", e)

    sys.path.insert(0, str(code))
    os.environ["RSNA_DATA_DIR"] = str(competition)
    import yaml

    from src.infer import predict

    config = yaml.safe_load((weights / "config.yaml").read_text(encoding="utf-8"))
    predict(competition, weights / "model.safetensors", config, OUT, num_workers=4)
    log.info("done in %.1f min → %s", (time.time() - started) / 60, OUT)


if __name__ == "__main__":
    main()
