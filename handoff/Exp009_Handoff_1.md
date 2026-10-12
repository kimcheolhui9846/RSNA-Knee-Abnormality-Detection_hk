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
- RunPod 1차(2026-10-06 00:24 UTC, 4090 ×5): **5대 모두 02:31 UTC에 동시에 SIGTERM**(epoch 13/16, exit 143). 상한(5h)·정체 감지가 아님 → 원인: RunPod 계정 쪽 강제 종료 (코드·하네스 문제 아님).
  epoch마다 저장한 `checkpoints/last.safetensors`(epoch 13)가 HF에 남음. 학습이 끝나야 만들어지는 `oof_pseudo.csv`는 없음.
- 복구: `trainer: raptor_predict`(`predict_folds`) — fold 체크포인트로 그 fold의 pseudo study와 정답 58을 예측만 한다 (`configs/exp009_raptor_predict.yaml`).
  실제 fold 0 체크포인트가 모델에 빠짐없이 로드됨을 로컬에서 확인(`All keys matched`).

- OOF 예측: Kaggle T4 34분(무료), 5-fold 평균 정답 58 0.9065, train 4,349 전체 OOF
- 정제 라벨 `labels_refined.csv` (alpha 0.5): 정답 58 기준 라벨 품질 0.923 (판독문 라벨 0.893). 상세 `experiments/exp009.md`
- `configs/exp011_raptor_refined.yaml`: exp007 + `labels_file: labels_refined.csv`
- **exp011 결과 (2026-10-12, RunPod 4090 3.15시간, run `20261012-070019-fcf927`): 정답 58 0.9157** (exp007 0.9037). 기존 3모델과 같은 비중 순위 평균 0.9191 (기존 0.9125). 라벨별 표는 `experiments/exp009.md`
- 4모델 앙상블 제출 `57104106`: **Public LB 0.933** (기존 3모델 0.930)
- **exp012 (정제 라벨, seed 12, A6000 4.8시간, run `20261012-111146-8fc5f9`): 정답 58 0.9149.** 5모델 같은 비중 0.9209 → 제출(`rsna-knee-submit` v10, 가중치 `kimche12/rsna-knee-weights-r5`)
- `train.py` `fetch_data` + config `data_repo`: Pod에서 그 config가 쓰는 코퍼스·라벨만 HF에서 받는다 (`tests/test_fetch_data.py`). 데이터셋 전체 54 GB가 일부 호스트 디스크에 안 들어가 추가

## 4. 미해결 문제
- 정제 비율을 정답 58로 정하면 그 58 점수가 부푼다 → 비율은 0.5 고정을 기본으로, 58은 확인용으로만 쓴다

## 5. 다음 작업자가 할 일
1. ~~정제 라벨 생성~~ 완료. HF 업로드는 요청 한도(429)로 보류 중이면 다시 올린다
2. ~~정제 라벨로 Raptor 재학습~~ 완료 (exp011). 5개 fold 모델은 4모델 앙상블에 더해도 오르지 않아(0.9186) 넣지 않는다
3. 4모델(exp007·v2·mean·exp011) 같은 비중 앙상블 제출, 정제 라벨 변형(seed, 64장 코퍼스) 추가 학습
