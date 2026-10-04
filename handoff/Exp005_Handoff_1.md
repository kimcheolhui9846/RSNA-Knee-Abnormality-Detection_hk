# HANDOFF — exp005: DINOv2 학습 부족 수정 + 라벨 대조군 exp003r

작성: kim cheol hui | 날짜: 2026-10-04 | 브랜치: `exp/005-dino-fix` (base: `exp/004-dino`, stacked PR)

## 1. 요약 (TL;DR)
- exp004 하락(CV 0.703 / LB 0.741)의 원인을 네 갈래(파이프라인·라벨·모델·통계)로 점검 → **모델 학습 부족**이 유력, 라벨·파이프라인은 아님.
  근거 전체는 `experiments/exp005.md` 배경 절.
- 모델·학습 옵션 추가 (config 키로만 켜짐, 기본값은 exp004와 동일):
  `query_init_std`, `slot_pool`(모델) / `warmup`, `no_decay_1d`, `grad_clip`, `init_bias_prior`, `amp_dtype: bf16`(학습)
- 실험 config 2개: `exp005_dino_fix.yaml`(개선 DINOv2), `exp003r_public_labels.yaml`(exp002 모델 + labels_v2, 라벨 대조군). 둘 다 `eval_pseudo: true`.

## 2. 한 일과 결정 기록
| 항목 | 결정 | 이유 |
|------|------|------|
| 칸 풀링 경로 | attention 문맥에 칸별 mean+max 경로를 **더함** (교체 아님) | 소견별 attention의 장점은 유지하면서 exp002에서 효과가 확인된 최대 풀링을 보장 |
| 질의 초기화 1.0 | 0.02 → 1.0 | 0.02에서 최대 attention 0.0106 ≈ 1/96 (균등). 테스트로 1.0이 비균등임을 확인 |
| exp003r을 이 브랜치에서 실행 | exp003 브랜치 대신 여기 | 보조 CV(`eval_pseudo`) 코드가 exp004 이후에만 있다. baseline 모델 코드는 동일 |
| 옵션은 config 키로만 | 기본값 = 기존 동작 | exp002–004 재현성 유지 |
| 빈 배치 버그 | `view(P, d, -1)` → `view(P, d, num_features)` | 배치 전체에 칸이 없으면 reshape 오류 (테스트로 발견) |
| `tmp_path_retention_policy = "failed"` | pyproject | WSL `/tmp`가 RAM(tmpfs)이라 smoke 가중치가 쌓여 공간 부족으로 테스트 실패 |

## 3. 결과
- `ruff check`, `ruff format --check` 통과. `pytest -q` 86 passed, 2 skipped (torch 환경) / 49 passed, 6 skipped (CI와 같은 torch 없는 환경)
- 새 테스트: 칸 풀링 마스크 불변·출력 변화·빈 배치, 질의 초기화 비균등, param group 분리·no-decay, warmup 스케줄, bias 사전확률 초기화, exp005 옵션 end-to-end, smoke(모든 config)
- RunPod 결과: (실행 후 기록)

## 4. 미해결 문제
- freeze 2 + batch 2(96장 × 2)의 GPU 메모리는 로컬에서 확인 불가 (로컬 GPU 사용 금지). 추정 약 12–14 GB로 24 GB 4090에 들어감
- Kaggle 추론(`feat/submit-v2`)은 fp16 autocast. bf16으로 학습한 가중치를 fp16으로 추론해도 되는지 제출 전 OOF 재현으로 확인 필요

## 5. 다음 작업자가 할 일
1. 두 run 결과로 `experiments/exp005.md` 결과 절, README 표 갱신
2. exp005 ≥ exp003r면 DINOv2 경로 유지(제출 후보), 아니면 b0 경로 강화
