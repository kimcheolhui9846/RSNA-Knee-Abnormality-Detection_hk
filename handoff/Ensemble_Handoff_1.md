# HANDOFF — 여러 모델 앙상블 제출 (순위 평균)

작성: kim cheol hui | 날짜: 2026-10-05 | 브랜치: `feat/ensemble-submit` (base: `feat/submit-v2`, stacked PR)

## 1. 요약 (TL;DR)
- 가중치 폴더에 `ensemble.yaml`이 있으면 멤버 모델을 모두 돌려 **라벨별 순위(rank)를 가중 평균**해 제출한다. 없으면 예전과 같은 단일 모델.
- 첫 앙상블: exp002(b0 + Qwen 라벨) 0.5 + exp003r(b0 + v2 라벨) 0.5. 정답 58 OOF 0.798 (exp002 단독 0.778, CI [−0.001, +0.044]).

## 2. 한 일과 결정 기록
| 항목 | 결정 | 이유 |
|------|------|------|
| 결합 방식 | 라벨별 순위 백분위의 가중 평균 | 평가는 AUC(순위만 봄). 모델마다 확률 보정이 달라도 안전 |
| 입력 공유 | 멤버의 `depth/size/slab/target_slices`가 같아야 하고, 다르면 `ValueError` | DICOM 읽기가 병목이라 study마다 한 번만 읽어 모든 멤버에 넣는다. 다른 입력 설정의 멤버는 필요할 때 확장 |
| 가중치 | 0.5 / 0.5 | 58개로 가중치를 고르면 과적합. 동일 가중 |
| 꾸러미 | `build_kaggle.py --member 이름=가중치,config[,비율]` (여러 번) | `weights/<이름>/` + `ensemble.yaml` |
| 하위 호환 | `predict()`는 그대로(멤버 1개짜리 `predict_members`) | 기존 테스트·CLI 유지. 멤버가 1개면 확률을 그대로 낸다 |

## 3. 결과
- `ruff check`, `ruff format --check` 통과. `pytest -q` 94 passed, 2 skipped / torch 없는 환경 56 passed, 7 skipped
- 새 테스트: 순위 평균 계산, `ensemble.yaml` 읽기, 앙상블 제출 = 멤버별 예측의 순위 평균, 입력 설정이 다른 멤버 거부, 꾸러미 빌드, `--member` 파싱
- 로컬 Kaggle 흉내 실행(CPU, 공개 test 3 study): 멤버 2 × 5 fold, fallback 0, 1.0분
- Kaggle 제출 `56841079` (노트북 v5, GPU 1.3분 — 공개 test 3 study): **Public LB 0.847** (exp002 단독 0.834 → +0.013). OOF 개선(+0.020)과 같은 방향

## 4. 미해결 문제
- 추론 시간: b0 두 개라 모델 계산은 약 2배, DICOM 읽기는 그대로. 숨은 test 1,300 study 기준 시간은 제출 상세에서 확인 필요
- 이전 꾸러미 보관: `kaggle_upload_exp002/`, `kaggle_upload_exp004/` (git 무시)

## 5. 다음 작업자가 할 일
1. LB를 exp002(0.834)와 비교해 기록
2. 새 모델은 `--member`로 추가. 입력 설정(2.5D 등)이 다른 모델을 섞으려면 `predict_members`를 입력 설정별 묶음으로 확장
