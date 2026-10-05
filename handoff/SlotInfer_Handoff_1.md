# HANDOFF — 칸 이미지 모델(exp006) 추론 + 입력 방식이 다른 모델 앙상블

작성: kim cheol hui | 날짜: 2026-10-05 | 브랜치: `feat/slot-infer` (base: `exp/006-slot-dino`, `feat/ensemble-submit` merge)

## 1. 요약 (TL;DR)
- Kaggle 추론이 exp006의 칸 이미지 입력(`input: slot_image`)을 지원한다: test 시리즈 헤더를 읽어 지방억제·가중·px·좌우를 판정하고 학습과 같은 `slot_image()`로 입력을 만든다.
- 앙상블 멤버의 입력 방식이 달라도 된다: 입력 설정이 같은 멤버끼리 묶고, DICOM은 study마다 한 번만 읽어 모든 묶음의 입력을 만든다.
- 학습(캐시) 경로와 추론(DICOM) 경로의 칸 이미지가 실제 train study 6개(좌 3, 우 3)에서 **비트 단위로 같다** (최대 차이 0).

## 2. 한 일과 결정 기록
| 항목 | 결정 | 이유 |
|------|------|------|
| 입력 묶음 | `DATA_KEYS`(input, depth, size, slab, target_slices, img_out, crop_mm, slice_band)가 같은 멤버끼리 | 같은 입력은 한 번만 만든다 |
| DICOM 읽기 | study마다 한 번 → 모든 묶음 | 읽기·디코딩이 병목 |
| `size` | 묶음끼리 같아야 함 (다르면 `ValueError`) | 캐시와 같은 `preprocess_series(size)` 결과를 공유 |
| 헤더 | 칸 이미지 묶음이 있을 때만 시리즈 가운데 파일 헤더를 읽음 | 학습 메타(`train_series_meta.csv`)와 같은 `read_header` |
| 칸 없는 묶음 | 그 묶음 멤버만 `FALLBACK_PROB` | 다른 묶음 예측은 살린다 |
| 이전 테스트 | "입력 설정이 다르면 거부" → "묶음별로 돌려 순위 평균"으로 교체 | 지원 기능이 바뀜 |

## 3. 결과
- `ruff check` 통과. `pytest -q` 117 passed, 2 skipped
- 새 테스트: 칸 이미지 멤버가 DICOM 헤더로 예측, 슬라이스 입력(b0)과 칸 이미지(DINO) 멤버를 한 번에 돌린 결과 = 각자 예측의 순위 평균
- 학습·추론 입력 대조 (CPU, 실제 train DICOM): 칸 마스크 동일, 이미지 최대 차이 0.0

## 4. 미해결 문제
- exp006 RunPod 결과가 나와야 앙상블 가중치·제출 여부를 정한다
- 숨은 test 1,300 study 실행 시간: 헤더 읽기 추가(시리즈당 파일 1개)로 약간 늘어남. 제출 상세에서 확인 필요

## 5. 다음 작업자가 할 일
1. exp006 결과가 좋으면 `build_kaggle.py --member`로 exp002·exp003r·exp006 꾸러미를 만들어 제출
