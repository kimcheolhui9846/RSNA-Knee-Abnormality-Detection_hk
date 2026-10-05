# RSNA Knee Abnormality Detection

Kaggle "RSNA Knee Abnormality Detection" — 무릎 MRI study 단위로 12개 이상 소견의 확률을 예측한다.
평가는 12개 라벨 AUC의 단순 평균(macro AUC), 제출은 오프라인 Notebook(9시간 이내)이다.

## 환경

```bash
uv sync                 # 데이터·테스트용 (numpy, pandas, pydicom, ...)
uv sync --extra train   # + torch / timm (CUDA 12.8)
uv run ruff check . && uv run pytest -q
```

데이터는 `RSNA_DATA_DIR` 환경변수, Kaggle의 `/kaggle/input/rsna-knee-abnormality-detection`,
저장소 안 `rsna-knee-abnormality-detection/` 순서로 찾는다 (`src/paths.py`).

Windows에서 스마트 앱 컨트롤이 `.venv`의 실행 파일·DLL을 막으면 WSL에서 실행한다.
가상환경은 WSL 홈에 따로 둔다: `UV_PROJECT_ENVIRONMENT=~/.venvs/rsna-knee uv run python -m pytest -q`

## 구조

| 경로 | 내용 |
|------|------|
| `configs/` | 실험별 YAML |
| `src/data/` | DICOM 로딩, 시리즈 선택, 전처리, Dataset |
| `src/models/` | 모델 |
| `folds/` | 5-fold 분할 파일 (커밋해서 공유) |
| `notebooks/` | EDA |
| `kaggle/` | 제출 노트북 |
| `experiments/` | 실험 기록 `expXXX.md` |
| `reviews/` | PR 리뷰 기록 |
| `handoff/` | 작업별 핸드오프 문서 |
| `tests/` | 제출 형식·metric·로더 테스트 |

## 실험 리더보드

| exp | 설명 | CV macro AUC | fold std | 라벨별 최저 AUC | LB | 추론 시간 | PR |
|-----|------|--------------|----------|-----------------|----|-----------|----|
| exp001 | efficientnet_b0 2D + 칸/study 집계, 라벨 있는 58 study만 | 0.590 | 0.097 | MCL 0.494 | - | - | #10 |
| exp002 | exp001 + LLM pseudo-label 4,349 study 추가 학습 (평가는 정답 58) | 0.778 | 0.042 | MCL 0.587 | 0.834 | 공개 test 0.9분 | #13 |
| exp003 | exp002 + 공개 LLM 라벨(soft) | (실패: Community GPU에서 CUDA 불가) | | | | | #14 |
| exp004 | 공개 라벨 + DINOv2 ViT-S + 라벨별 attention + 2.5D | 0.703 | 0.054 | MCL 0.478 | 0.741 | 공개 test 0.9분 | #15 |
| exp003r | exp002 모델 + 공개 LLM 라벨(v2), 보조 CV | 0.780 | 0.046 | Synovitis 0.673 | - | - | #17 |
| exp005 | exp004 + lr↑·freeze↓·칸 풀링·질의 초기화 (백본 표현 붕괴) | 0.511 | 0.053 | Fracture 0.421 | - | - | #17 |
| exp006 | 칸 이미지 DINOv2 (공개 상위 레시피 재구현: 130 mm·좌우·헤더 칸·3장 RGB·칸 attention·lr 8e-6) | 0.812 | 0.046 | Lateral OA 0.669 | - | - | #19 |
