import os
from collections.abc import Sequence
from pathlib import Path

COMPETITION = "rsna-knee-abnormality-detection"
REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CANDIDATES = (
    Path("/kaggle/input") / COMPETITION,
    REPO_ROOT / COMPETITION,
    REPO_ROOT / "data",
)


def data_dir(candidates: Sequence[Path] | None = None) -> Path:
    """대회 데이터 루트. 환경변수 RSNA_DATA_DIR > Kaggle 입력 > 저장소 안 로컬 경로 순서."""
    env = os.environ.get("RSNA_DATA_DIR")
    if env:
        path = Path(env)
        if not path.is_dir():
            raise FileNotFoundError(f"RSNA_DATA_DIR가 가리키는 폴더가 없다: {path}")
        return path
    candidates = DEFAULT_CANDIDATES if candidates is None else candidates
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError(
        f"데이터 폴더를 찾지 못했다. RSNA_DATA_DIR을 지정하라. 확인한 경로: {list(candidates)}"
    )
