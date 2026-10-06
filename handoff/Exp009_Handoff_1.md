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
- RunPod 결과: (실행 후 기록)

## 4. 미해결 문제
- 정제 비율을 정답 58로 정하면 그 58 점수가 부푼다 → 비율은 0.5 고정을 기본으로, 58은 확인용으로만 쓴다

## 5. 다음 작업자가 할 일
1. 5개 `oof_pseudo.csv`를 모아 train 전체 OOF → 정제 라벨 파일(`labels_refined.csv`) 생성, HF 데이터셋에 추가
2. 정제 라벨로 Raptor 재학습, 5개 fold 모델을 앙상블 후보로 평가
