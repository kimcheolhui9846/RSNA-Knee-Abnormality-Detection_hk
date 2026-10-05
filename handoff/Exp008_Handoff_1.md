# HANDOFF — exp008: Raptor CoAtNet 라벨·seed 변형 2개

작성: kim cheol hui | 날짜: 2026-10-06 | 브랜치: `exp/008-raptor-variants` (base: `exp/007-raptor`, stacked PR)

## 1. 요약 (TL;DR)
- exp007(정답 58 0.904)과 같은 절차에 라벨·seed만 바꾼 config 2개: `exp008_raptor_v2.yaml`(v2 라벨, seed 7), `exp008_raptor_mean.yaml`(dread·v2 평균, seed 1234).
- 목적은 단일 성능이 아니라 앙상블 다양성. 코드 변경 없음.

## 2. 한 일과 결정 기록
| 항목 | 결정 | 이유 |
|------|------|------|
| 바꾸는 축 | 라벨 출처 + seed | 공개 기록상 같은 레시피 백본 앙상블은 +0.001. 라벨 계열 간 상관이 0.51–0.92로 낮은 라벨이 있다 |
| 평균 라벨 | pseudo 행만 평균, 정답 58은 원래 값 | 정답 58은 평가 전용(학습에 안 씀) |
| 모델 선택 | exp007처럼 마지막 3 epoch SWA | 정답 58로 고르지 않는다 |

## 3. 결과
- `ruff check` 통과, `pytest -q` 108 passed, 2 skipped (smoke가 새 config 2개 포함)
- RunPod 결과: exp008-v2 0.8863, exp008-mean 0.8997 (정답 58, SWA). exp007 + 두 변형 같은 비중 순위 평균 **0.9125** (exp007 단독 0.9037). 상세는 `experiments/exp008.md`
- 가중치: HF `runs/20261005-161925-7b1b15/`, `runs/20261005-162021-5ce9e8/` 의 `model.safetensors`

## 4. 미해결 문제
- 4090(24 GB)에서 bs 8 + grad checkpointing으로 OOM 없이 돌았다(각 3.2시간)
- Lateral OA는 세 모델 모두 0.75–0.76 — 라벨 변형으로는 안 오른다

## 5. 다음 작업자가 할 일
1. ~~정답 58 OOF 기록, 순위 평균 계산~~ 완료
2. 좋은 조합을 Kaggle 앙상블·공개 파이프라인 혼합에 넣는다 (`kaggle/build_kaggle.py --member`)
