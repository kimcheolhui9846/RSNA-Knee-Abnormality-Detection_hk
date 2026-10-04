# HANDOFF — 제출 경로 v2: exp004 모델(DINOv2 + 2.5D) 추론 지원

작성: kim cheol hui | 날짜: 2026-10-04 | 브랜치: `feat/submit-v2` (base: `exp/004-dino`, stacked PR)

## 1. 요약 (TL;DR)
- `feat/kaggle-submit`(#11, exp001 위)의 제출 코드를 exp004 위로 가져와 새 모델을 지원하게 했다.
- `src/infer.py`: `build_model`로 config의 모델(baseline / dino_attn)을 만들고, config `slab`이면 학습과 같은 `slab_stack`(2.5D) 입력.
- exp004 가중치로 Kaggle 노트북 v4 실행 성공 → 대회 제출 ref `56822852`.

## 2. 한 일과 결정 기록
| 항목 | 결정 | 이유 |
|------|------|------|
| 브랜치 | exp004에서 새로 따고 `feat/kaggle-submit` merge | #11은 exp001 위라 새 모델 코드가 없다. #11 diff를 키우지 않는다 |
| 모델 생성 | `build_model({...model_cfg, pretrained: False})` | 오프라인. 학습 가중치에 백본까지 들어 있다 |
| 2.5D | 학습과 같은 `slab_stack` 재사용 | 학습·추론 전처리 일치 |

## 3. 결과
- 테스트 73 passed (dino + slab 추론 1개 추가), `ruff` 통과
- 로컬 Kaggle 흉내 실행(CPU): exp004 5 fold, 3 study, fallback 0
- Kaggle 노트북 v4 (`kimche12/rsna-knee-submit`): Python 3.13, 휠 4종 설치, 디코더 사용 가능, GPU 추론 35초, fallback 0, 전체 0.9분
- Kaggle 데이터셋 새 버전: `rsna-knee-code`(src에 dino 모델 포함), `rsna-knee-weights`(exp004 가중치 445 MB, sha `e377ad92…`)
- 제출 `56822852`: 점수는 `experiments/exp004.md`에 기록

### 추가 (2026-10-04, exp004 원인 분석 중 발견)
- `src/infer.py`: `int(row.Fat_Suppression)`이 try 밖에 있어 숨은 test에 결측이 하나만 있어도 추론 전체가 멈출 수 있었다.
  비어 있으면 `Fluid_Sensitive`(train에서 항상 같은 값)로 대신하고, 둘 다 없으면 그 시리즈만 건너뛰도록 고침. 테스트 추가 (88 passed).
- 학습·추론 입력 일치 재확인: 캐시 경로와 DICOM 경로의 입력 텐서가 train 9 / test 3 study에서 비트 단위로 같고,
  업로드 가중치로 RunPod OOF를 오차 ≤0.0012로 재현.

## 4. 미해결 문제
- 비공개 test 1,300 study 기준 실행 시간은 Kaggle 제출 상세 화면에서 확인 필요 (ViT-S 5 fold × 96장/study)
- 이전 꾸러미는 `kaggle_upload_exp001/`, `kaggle_upload_exp002/`에 보관 (git 무시)

## 5. 다음 작업자가 할 일
1. LB 확인 후 exp002 대비 판단
2. 다음 모델도 config `model.name`·`slab`만 맞으면 같은 경로로 제출 가능
