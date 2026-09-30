# HANDOFF — study 단위 학습 테이블과 PyTorch Dataset

작성: kim cheol hui | 날짜: 2026-09-30 | 브랜치: `feat/dataset` (base: `feat/preprocess-cache`, stacked PR)

## 1. 요약 (TL;DR)
- `src/data/study_table.py` (torch 없음, CI에서 검증):
  - `SLOTS`: (Sagittal, Coronal, Axial) × (지방 억제 1, 0) = 6칸
  - `select_series`: 캐시 인덱스 → study × 6칸 표. 같은 칸에 여러 시리즈면 슬라이스 수가 목표(30)에 가장 가까운 것
  - `resample_depth`: 슬라이스 축을 고정 장수로 (처음·끝 포함, 순서 유지)
  - `build_study_table`: fold + 라벨(NaN 유지) + 라벨 마스크 + 6칸 경로
- `src/data/dataset.py` `KneeStudyDataset`: `image (6, depth, 256, 256)` float 0~1, `slot_mask`, `labels`, `label_mask`, `study_id`
- 테스트 49 passed (WSL, torch CPU). CI에는 torch가 없어 Dataset 테스트 3개는 skip되고 나머지는 돈다.

## 2. 한 일과 결정 기록
| 항목 | 결정 | 이유 |
|------|------|------|
| 칸 구분 | `Anatomical_Plane × Fat_Suppression` | train에서 `Fluid_Sensitive == Fat_Suppression`이 항상 참 (24,371개 전부) |
| 중복 시리즈 | 슬라이스 수가 `target_slices`(30)에 가장 가까운 것, 같으면 SeriesInstanceUID 순 | 칸의 13%에 2개 이상. 320장짜리 등 긴 꼬리를 피하고 결과를 결정적으로 |
| 빈 칸 | 이미지 0 + `slot_mask=False` | 비지방억제 Axial은 study의 19%에만 있다. 모델이 "없음"과 "검은 영상"을 구분할 수 있게 |
| 슬라이스 수 | 균등 간격 인덱스로 `depth`장 (짧으면 반복) | 보간 없이 원본 슬라이스를 그대로 쓴다. depth는 config로 |
| 라벨 결측 | 테이블에서는 NaN 유지, Dataset 출력에서만 0 + `label_mask=False` | 결측 ≠ 음성. 텐서에 NaN이 있으면 손실이 NaN이 되므로 마스크로 뺀다 |
| 캐시 없는 study | 테이블에서 제외 | 시리즈가 하나도 없으면 입력이 없다 |
| torch 의존 분리 | 로직은 `study_table.py`(numpy/pandas), Dataset은 얇은 래퍼 | CI가 torch 없이도 핵심 로직을 검증한다 |

슬롯별 시리즈 보유율 (train_series.csv): Axial-FS 100%, Sagittal-nonFS 96.8%, Coronal-FS 96.4%, Sagittal-FS 94.2%, Coronal-nonFS 77.3%, Axial-nonFS 19.4%. study당 채워진 칸: 3칸 4 / 4칸 1,259 / 5칸 2,578 / 6칸 566.

## 3. 결과
- 테스트 49 passed (study_table 10, dataset 3 추가), `ruff check`·`ruff format --check` 통과 (WSL, torch CPU)
- 전체 캐시는 아직 생성 중이라 실제 캐시로 Dataset을 끝까지 돌리는 확인은 캐시 완료 후 추가한다.

## 4. 미해결 문제
- **캐시 생성 속도**: `/mnt/d` 읽기 병목으로 분당 약 50 시리즈 → 전체 약 8시간. 진행 중.
- **라벨 있는 study는 58개**: 이 Dataset만으로는 지도 학습 샘플이 58개다. pseudo-label(2단계)이 들어오면 `labels`/`label_mask`에 그대로 합친다.
- **메모리·속도**: 샘플 하나가 6 × depth × 256² float32 (depth 32면 약 50MB). DataLoader worker 수와 depth는 RunPod에서 조정.
- **증강 없음**: 좌우 반전(무릎 좌/우), 밝기 등은 exp001에서 정한다.

## 5. 다음 작업자가 할 일
1. 캐시 완료 후: 실제 캐시로 `build_study_table` → `KneeStudyDataset` 스모크, 통계를 이 문서에 추가
2. exp001 (로드맵 1단계): 2D 백본 + 슬라이스/시리즈 집계, 58개 라벨로 fold CV
3. 캐시를 RunPod으로 옮기기
