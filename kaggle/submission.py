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


def find_one(pattern: str) -> Path:
    hits = sorted(INPUT.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"/kaggle/input에서 {pattern}을 찾지 못했다")
    return hits[0]


def main() -> None:
    started = time.time()
    logging.basicConfig(
        level=logging.INFO, format="[submit %(asctime)s] %(message)s", datefmt="%H:%M:%S"
    )
    log = logging.getLogger("submit")

    code = find_one("**/rsna_knee_code.marker").parent
    weights = find_one("**/rsna_knee_weights.marker").parent
    competition = find_one("**/test_series.csv").parent
    log.info("code %s | weights %s | data %s", code, weights, competition)

    # 오프라인 휠: 압축 DICOM 디코더와 학습 때와 같은 버전의 timm.
    # 설치에 실패해도 Kaggle 이미지에 이미 있는 패키지로 진행한다
    wheels = code / "wheels"
    if wheels.is_dir():
        pkgs = ["timm", "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"]
        cmd = [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q"]
        r = subprocess.run(
            [*cmd, "--find-links", str(wheels), *pkgs], capture_output=True, text=True
        )
        log.info("offline wheels install rc=%d %s", r.returncode, r.stderr.strip()[-300:])

    sys.path.insert(0, str(code))
    os.environ["RSNA_DATA_DIR"] = str(competition)
    import yaml

    from src.infer import predict

    config = yaml.safe_load((weights / "config.yaml").read_text(encoding="utf-8"))
    predict(competition, weights / "model.safetensors", config, OUT, num_workers=4)
    log.info("done in %.1f min → %s", (time.time() - started) / 60, OUT)


if __name__ == "__main__":
    main()
