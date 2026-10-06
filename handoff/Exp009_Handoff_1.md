# HANDOFF — exp009: 라벨 정제용 5-fold Raptor 학습

작성: kim cheol hui | 날짜: 2026-10-06 | 브랜치: `exp/009-label-refine` (base: `exp/008-raptor-variants`, stacked PR)

## 1. 요약 (TL;DR)
- 판독문 LLM 라벨은 정답 58에서 0.84–0.89, 이미지 모델 0.904, 둘을 섞으면 0.923 → 라벨 정제가 다음 지렛대.
- exp007 절차를 `holdout_fold` 0–4로 5번 학습(config `exp009_raptor_fold{k}.yaml`, seed 100+k) → train 전체 이미지 OOF. 코드 변경 없음.

## 2. 한 일과 결정 기록
| 항목 | 결정 | 이유 |
|------|------|------|
| fold 수 | 5 (기존 `folds.csv`) | 코드 변경 없이 `holdout_fold` 사용, 각 모델이 데이터 4/5로 학습해 앙상블 멤버로도 강하다 |
| 라벨 | dread (exp007과 같음) | 이미지 OOF는 판독문 라벨과 섞을 것이므로 가장 강한 단일 레시피 |
| seed | fold마다 다르게 | 앙상블 다양성 |

## 3. 결과
- `ruff check` 통과, `pytest -q` 통과 (smoke가 새 config 5개 포함)
- RunPod 1차(2026-10-06 00:24 UTC, 4090 ×5): **5대 모두 02:31 UTC에 동시에 SIGTERM**(epoch 13/16, exit 143). 상한(5h)·정체 감지가 아님 → 원인 확인: **RunPod 계정 잔액 소진** (다음 실행이 "Your account balance is too low to rent a pod"로 거절됨).
  epoch마다 저장한 `checkpoints/last.safetensors`(epoch 13)가 HF에 남음. 학습이 끝나야 만들어지는 `oof_pseudo.csv`는 없음.
- 복구: `trainer: raptor_predict`(`predict_folds`) — fold 체크포인트로 그 fold의 pseudo study와 정답 58을 예측만 한다 (`configs/exp009_raptor_predict.yaml`).
  실제 fold 0 체크포인트가 모델에 빠짐없이 로드됨을 로컬에서 확인(`All keys matched`).

- OOF 예측: Kaggle T4 34분(무료), 5-fold 평균 정답 58 0.9065, train 4,349 전체 OOF
- 정제 라벨 `labels_refined.csv` (alpha 0.5): 정답 58 기준 라벨 품질 0.923 (판독문 라벨 0.893). 상세 `experiments/exp009.md`
- `configs/exp011_raptor_refined.yaml`: exp007 + `labels_file: labels_refined.csv` (RunPod 충전 후 실행)

## 4. 미해결 문제
- 정제 비율을 정답 58로 정하면 그 58 점수가 부푼다 → 비율은 0.5 고정을 기본으로, 58은 확인용으로만 쓴다

## 5. 다음 작업자가 할 일
1. ~~정제 라벨 생성~~ 완료. HF 업로드는 요청 한도(429)로 보류 중이면 다시 올린다
2. 정제 라벨로 Raptor 재학습, 5개 fold 모델을 앙상블 후보로 평가
