# HANDOFF — exp006: 칸 이미지 DINOv2 (공개 상위 레시피 재구현)

작성: kim cheol hui | 날짜: 2026-10-05 | 브랜치: `exp/006-slot-dino` (base: `exp/005-dino-fix`, stacked PR)

## 1. 요약 (TL;DR)
- 공개 상위 노트북(0.94대)의 DINO 단계를 분석해 입력·헤드·학습률을 우리 코드로 다시 구현했다 (`experiments/exp006.md`의 비교표).
- 새 모듈: `src/data/headers.py`(헤더 판정), `src/data/slot_image.py`(칸 이미지·Dataset), `src/models/slot_dino.py`(모델).
- 학습 데이터에 헤더 메타 `train_series_meta.csv`가 필요하다 (`python -m src.data.headers --split train --out ...`).

## 2. 한 일과 결정 기록
| 항목 | 결정 | 이유 |
|------|------|------|
| 코드 | 공개 노트북 코드를 복사하지 않고 규칙만 가져와 새로 작성 | 우리 저장소 구조·테스트에 맞추고, 라이선스·출처 문제를 피한다 |
| 캐시 | 256px 캐시 위에서 mm 자르기 | 원본 크기·PixelSpacing이 인덱스에 있어 다시 만들 필요가 없다 (50 GiB 재생성 회피) |
| 칸 정의 | 헤더 정규식 지방억제 + 가중(fluid) | 공개 레시피. CSV `Fat_Suppression`과 210개 시리즈가 다르다 |
| 좌우 | 태그 우선, 없으면 영상 중심 x (|x|<20 mm면 미정) | 공개 레시피. 미정 72개는 뒤집지 않는다 |
| 학습률 | 백본 8e-6, 마지막 6블록 | exp005 붕괴(1e-4) 원인 분석 + 공개 레시피 |
| 스케줄 | 5% warmup + cosine, bf16, clip 1.0 | 공개 노트북에는 학습 코드가 없어(추론 전용) 우리 기존 옵션 사용 |

## 3. 결과
- `ruff check`, `ruff format --check` 통과. `pytest -q` 99 passed, 2 skipped / torch 없는 환경 53 passed, 7 skipped
- 새 테스트: 헤더 판정(지방억제·가중·px·좌우), 가운데 구간 슬라이스, mm 자르기, 좌우 뒤집기, 칸 선택, 증강, 모델(마스크·빈 study·학습 블록·사전 가산점), end-to-end, smoke(전 config)
- 로컬 CPU 사전 점검: 특징 붕괴 없음(이미지 간 std/평균 0.61)
- RunPod 결과: (실행 후 기록)

## 4. 미해결 문제
- Kaggle 추론 경로는 아직 칸 이미지 입력을 지원하지 않는다 (제출 전에 `src/infer.py`에 test 헤더 판정 + `slot_image` 추가 필요)
- 앙상블에서 b0(슬라이스 입력)와 섞으려면 `predict_members`를 입력 방식별 묶음으로 확장해야 한다

## 5. 다음 작업자가 할 일
1. RunPod 결과를 `experiments/exp006.md`, README에 기록
2. 정답 58 CV·보조 CV가 b0(0.78 / 0.80)보다 좋거나, b0와 앙상블해 오르면 추론 경로를 붙여 제출
