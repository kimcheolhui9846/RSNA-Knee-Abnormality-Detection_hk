# HANDOFF — exp007: Raptor 방식 CoAtNet 학습 (공개 코퍼스)

작성: kim cheol hui | 날짜: 2026-10-05 | 브랜치: `exp/007-raptor` (base: `exp/006-slot-dino`, stacked PR)

## 1. 요약 (TL;DR)
- 공개 Raptor(단일 CoAtNet LB 0.924)의 학습 절차를 우리 코드로 다시 작성했다: `src/data/raptor_stack.py`, `src/models/raptor.py`, `src/train_raptor.py`, config `exp007_raptor_coatnet.yaml`(`trainer: raptor`).
- 학습 데이터는 공개 코퍼스(44×336×336 스택). 우리 `build_stack()`이 train DICOM에서 **비트 단위로 같은 스택**을 만드는 것을 7 study로 확인 → Kaggle 추론 입력이 학습과 같다.
- 학습 데이터셋: 비공개 HF `cheolhhh9846/rsna-knee-raptor-corpus` (코퍼스 + `labels_dread.csv` + `labels_v2.csv` + `folds.csv`).

## 2. 한 일과 결정 기록
| 항목 | 결정 | 이유 |
|------|------|------|
| 코드 | 공개 노트북 코드를 복사하지 않고 같은 규칙으로 새로 작성 | 우리 구조·테스트, 출처 문제 |
| 코퍼스 | 직접 만들지 않고 공개 코퍼스 사용 | 22 GB를 DICOM에서 다시 만드는 데 몇 시간 → 공개본과 비트 단위 동일 확인으로 대체 |
| 모델 선택 | 마지막 3 epoch 가중치 평균 (원본: 정답 58 최고 epoch) | 정답 58로 고르면 그 58 점수가 부푼다 |
| 라벨 | 1차는 공개 Raptor와 같은 라벨 | 먼저 재현 여부를 확인. 라벨 변형은 다음 |
| fold | 없음 (정답 58은 학습에 안 쓰므로 그대로 OOF) | 원본과 같음. `holdout_fold`로 보조 CV 선택 가능 |
| 학습 진입 | `train.py`에서 `trainer: raptor`면 `run_raptor` | 기존 실험 경로는 그대로 |

## 3. 결과
- `ruff check`, `ruff format --check` 통과. `pytest -q` 106 passed, 2 skipped (smoke: exp007 포함 전 config)
- 새 테스트: 창 중심(평가 고르게/학습 무작위), 3장 RGB 창, 모델 출력·소견별 attention, 코퍼스 두 부분 합치기, end-to-end(보조 CV 포함)
- RunPod 결과: (실행 후 기록)

## 4. 미해결 문제
- GPU 메모리: 공개 측정 bs 8 + grad checkpointing 23.7 GB (24 GB 4090에 빠듯). OOM이면 bs 6
- 학습 시간: 공개 기준 4090에서 16 epoch 약 3시간

## 5. 다음 작업자가 할 일
1. RunPod 결과(정답 58 OOF) 기록, 공개 단일 모델(0.917)과 비교
2. 좋으면 라벨 변형(v2, 평균)·시드 변형으로 2–3개 더 → 공개 0.943 파이프라인과 순위 혼합 (`kaggle/build_public_blend.py`, PR #21)
