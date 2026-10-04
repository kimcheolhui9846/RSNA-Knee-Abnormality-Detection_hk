# HANDOFF — report pseudo-label 평가 틀 (1단계)

작성: kim cheol hui | 날짜: 2026-09-30 | 브랜치: `feat/pseudo-eval` (base: `feat/dicom-loader`, stacked PR)

## 1. 요약 (TL;DR)
- report → 12개 라벨 추출기를 만들기 전에, 추출 결과를 **라벨이 달린 58개 study로 채점하는 틀**을 먼저 만들었다.
- `src/pseudo/schema.py`: pseudo-label 파일 형식(값 0~1, NaN = 판단 보류) 검증.
- `src/pseudo/evaluate.py`: 라벨별 coverage / accuracy / sensitivity / specificity / AUC + macro. CLI `python -m src.pseudo.evaluate --pred <csv>`.
- 실제 데이터로 확인: 정답을 넣으면 전부 1.0, 영어 키워드 매칭(라벨 이름이 report에 있으면 양성)은 **macro AUC 0.547** — 제대로 된 추출기가 필요하다는 기준선.

## 2. 한 일과 결정 기록
| 항목 | 결정 | 이유 |
|------|------|------|
| 값 형식 | 0~1 확률, NaN은 판단 보류 | LLM이 "불확실"을 낼 수 있어야 한다. 보류를 0으로 채우면 음성으로 오염된다(결측 ≠ 음성 원칙) |
| soft label 허용 | 허용 | 확신도를 학습 가중치나 soft target으로 쓸 수 있게 |
| 채점 대상 | 12개 라벨이 모두 있는 study만 | 정답이 있는 건 58개뿐 |
| 판단 보류 처리 | coverage에만 반영, 나머지 지표에서 제외 | "틀림"과 "모름"을 구분해서 본다. coverage가 낮으면 그만큼 학습 라벨이 줄어든다 |
| 이진 지표 기준 | `threshold=0.5` (CLI 옵션) | 기본값. soft label 보정은 추출기 단계에서 |
| 누락된 study | `ValueError` | 추출기가 일부 study를 빠뜨린 걸 조용히 넘기지 않는다 |
| 결과 파일 위치 | `pseudo_labels/` (git 제외) | 저장소가 공개라 대회 데이터에서 나온 라벨을 올리지 않는다 (`.gitignore`는 #6에서 추가) |

## 3. 결과
- 테스트 35 passed (pseudo 10개 추가), `ruff check`·`ruff format --check` 통과 (WSL)
- 실제 train.csv로 CLI 실행 (58개 채점, 결과 파일은 `/tmp`에만):

| 추출 방식 | macro accuracy | macro sensitivity | macro specificity | macro AUC |
|-----------|----------------|-------------------|-------------------|-----------|
| 정답 그대로 (틀 검증) | 1.000 | 1.000 | 1.000 | 1.000 |
| 영어 키워드 매칭 (기준선) | 0.641 | 0.301 | 0.793 | 0.547 |

키워드 매칭이 약한 이유: report의 약 60%가 영어가 아니고, "no effusion" 같은 부정 표현을 양성으로 잡으며, "Medial OA"처럼 약어 라벨은 본문에 그대로 나오지 않는다(Medial/Lateral/PF OA 민감도 0).

## 4. 미해결 문제
- **라벨 정의**: 추출 프롬프트에 넣을 12개 라벨의 공식 정의(예: Contusion의 범위, PF OA 기준)는 Kaggle Data/Discussion 탭 확인 필요.
- **58개로 채점하는 한계**: 라벨별 양성 9–35개라 지표 신뢰구간이 넓다. 추출기 비교 시 차이가 작으면 판단 보류.
- **58개의 언어 편중**: 라벨 있는 58개 중 영어 28개, 프랑스어 0개. 프랑스어 추출 품질은 이 틀로 측정할 수 없다.

## 5. 다음 작업자가 할 일
1. RunPod·Hugging Face 세팅 후 2단계: 오픈 가중치 다국어 LLM으로 report → JSON(라벨별 양성/음성/보류) 추출기 작성
2. 58개로 프롬프트를 조정하고 이 틀로 채점 → 4349개 전체에 적용해 `pseudo_labels/`에 저장
3. 외부 API(Claude 등)는 대회 규칙(Competition Data 이용 조건) 확인 후 검토
