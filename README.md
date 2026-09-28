# RSNA Knee Abnormality Detection

Kaggle "RSNA Knee Abnormality Detection" — 무릎 MRI study 단위로 12개 이상 소견의 확률을 예측한다.
평가는 12개 라벨 AUC의 단순 평균(macro AUC), 제출은 오프라인 Notebook(9시간 이내)이다.

## 환경

```bash
uv sync                 # 데이터·테스트용 (numpy, pandas, pydicom, ...)
uv sync --extra train   # + torch / timm (CUDA 12.8)
uv run ruff check . && uv run pytest -q
```

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
