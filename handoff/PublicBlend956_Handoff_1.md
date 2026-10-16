# HANDOFF — 공개 0.956 노트북 + 우리 모델 순위 혼합

작성: kim cheol hui | 날짜: 2026-10-14 | 브랜치: `feat/public956-blend` (base: `feat/infer-slots96`)

## 1. 요약 (TL;DR)
- 공개 노트북 `varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956`을 수정 없이 복사해 제출하면 Public LB **0.956**이다 (우리 최고).
- 그 뒤에 우리 모델을 순위로 10–15% 섞은 제출 3개는 0.956 / 0.955 / 0.955 / 0.955로, 한 번도 오르지 않았다 (우리 최고 단일 모델 exp014, 공개 라벨 exp015 포함).
- 이 PR은 혼합 노트북 생성기(`kaggle/build_public_blend.py`)에 노트북 이름·가중치 데이터셋·저작자 표기 옵션을 더해, 손으로 고쳐 만들던 0.956 혼합 노트북을 명령 한 줄로 재현하게 한다.

## 2. 한 일과 결정 기록
| 항목 | 결정 | 이유 |
|------|------|------|
| 공개 노트북 | 셀을 그대로 두고 끝에 우리 셀 2개만 붙인다 (기존 생성기 방식) | 공개 파이프라인을 건드리면 0.956 재현을 보장할 수 없다 |
| 새 옵션 | `--kernel-slug`, `--weights-slug`, `--credits` | 모델마다 가중치 데이터셋(`rsna-knee-weights-e13`, `-e14`)과 노트북 이름이 다르고, 공개 노트북마다 저작자가 다르다 |
| 저작자 표기 | 첫 마크다운 셀에 공개 노트북 id와 `--credits`를 적는다 | 공개 코드·가중치 사용 출처 표기 (대회 규칙 2.6.b, 3.6.b: 공개 노트북·가중치 사용 허용) |
| 우리 단계 | 별도 프로세스 + 노트북 시작 후 8.3시간 가드 (기존 그대로) | 넘거나 실패하면 공개 결과만 제출 → 제출 실패 위험 없음 |
| 혼합 | 라벨별 `(1 - w) * rank(공개) + w * rank(우리)` (기존 그대로) | AUC는 순위만 보므로 확률 보정 차이를 무시한다 |

### 공개 0.956 노트북 구성 (참고)
- nartaa Raptor CoAtNet 384 SWA (단일 LB 0.949; 96장, 3D 좌우 거울상 TTA, OAI 보조 학습) + 같은 데이터셋의 224 crop 체크포인트(0.945)를 Raptor 몫의 35%로
- heliosli R7 ConvNeXt-small 4-fold (OrthoFoundation 초기화)
- 두 계열을 50/50 순위 혼합
- OAI 사용은 호스트가 2026-09-21 Discussion에서 "기관 심사 없이 누구나 무료로 받을 수 있으면 허용"이라고 답했다. 최종 선택 전에 다시 확인 필요 (Kaggle Discussion 탭).

## 3. 결과
- `ruff check`, `ruff format --check` 통과. `pytest -q` 131 passed, 2 skipped
- 새 테스트: 옵션으로 준 노트북 이름·가중치 데이터셋·저작자가 노트북 파일명, 메타데이터, 첫 셀에 들어가는지

### Public LB
| 제출 | 구성 | 노트북 | Public LB |
|------|------|--------|-----------|
| 57165935 | 공개 0.956 그대로 | `kimche12/rsna-knee-public956` | **0.956** |
| 57166654 | + 우리 5모델(exp007·v2·mean·exp011·exp012) 10% | `kimche12/rsna-knee-public956-blend` | 0.956 |
| 57169923 | + exp013 15% | `kimche12/rsna-knee-public956-e13` | 0.955 |
| 57182823 | + exp014 15% | `kimche12/rsna-knee-public956-e14` | 0.955 |
| 57217206 | + exp015 15% (공개 라벨, PR #32) | `kimche12/rsna-knee-public956-e15` | 0.955 |
| 57230563 | 공개 0.957 그대로 (heliosli `0-9576-latest-public-dataset-score-in-one-click`) | `kimche12/rsna-knee-public957` | **0.957** |

- 0.943 공개 파이프라인에 섞었을 때도 같은 양상이었다 (그대로 0.943, 섞으면 0.943 / 0.942 / 0.941).
- 정답 58 점수(exp014 0.923이 우리 최고)는 혼합 효과를 예측하지 못한다 → 판정은 LB로 한다.
- 0.957 노트북은 0.956과 R 단계만 다르다: R7 대신 R6 묶음(`fe055406fad6a12f`) + 좌우 뒤집기 TTA. Raptor 384/224 단계와 비율은 같다.
- 주의: `heliosli/rsna-knee-r6-frozen-fe055406`는 2026-10-14에 버전 2(R7 묶음)로 덮어써졌다. `kernels push`는 데이터셋 버전을 고정하지 못해 최신 버전이 붙고, 노트북의 묶음 id 검사에서 멈춘다.
  버전 1 파일을 하나씩 받아(`kaggle datasets download heliosli/rsna-knee-r6-frozen-fe055406/1 -f <파일>`) manifest sha256을 대조한 뒤, 비공개 데이터셋 `kimche12/rsna-knee-r6-frozen-fe055406`으로 올려 붙였다.

### 재현
```
kaggle kernels pull varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956 -p <pub> -m
python kaggle/build_public_blend.py --public <pub> --user kimche12 --weight 0.15 \
  --kernel-slug rsna-knee-public956-e14 --weights-slug rsna-knee-weights-e14 \
  --credits "varshithakolupuri, nartaa, heliosli, dreaddevelopment" --out <out>
```
- 우리 단계는 `kimche12/rsna-knee-code`(이 스택의 `src/infer.py`, 96장 스택은 PR #30)와 가중치 데이터셋을 읽는다.

## 4. 미해결 문제
- 우리 모델이 공개 혼합에 도움이 되지 않는 이유는 확인하지 못했다. 정제 라벨 가설은 exp015(공개 라벨, 0.955)로 기각했다.
- OAI 허용 여부의 후속 질문(지역 제한)에 호스트 답이 아직 없다.

## 5. 다음 작업자가 할 일
1. 공개 0.957 그대로(57230563)를 최종 후보 1로 둔다. 정제 라벨 모델의 추가 학습·혼합은 여기서 멈춘다 (exp013·exp014·exp015 모두 0.955).
2. 가중치 데이터셋은 폴더째 올려야 한다: `kaggle datasets create/version -p . -r tar` (기본값은 하위 폴더를 건너뛴다). 제출 전에 노트북 로그에서 `blended:` 줄을 확인한다.
3. 최종 2개 선택(마감 2026-10-22): LB 최고 1개 + 규칙상 가장 안전한 1개. 선택 전에 OAI 관련 Discussion을 다시 확인한다.
