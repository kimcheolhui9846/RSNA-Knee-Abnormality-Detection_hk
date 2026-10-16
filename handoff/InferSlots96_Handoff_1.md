# HANDOFF — 추론에 96장 스택(SLOTS96) 추가

작성: kim cheol hui | 날짜: 2026-10-14 | 브랜치: `feat/infer-slots96` (base: `feat/raptor-infer`)

## 1. 요약 (TL;DR)
- exp014(96장 코퍼스 + 3D 좌우 거울상, 정답 58 SWA 0.9231)를 Kaggle에서 추론하려고 `SLOTS96`을 추가했다.
- 멤버 config의 `slots` 이름을 `SLOT_SETS`로 찾는다 (이전: SLOTS64가 아니면 모두 SLOTS44로 처리).

## 2. 한 일과 결정 기록
| 항목 | 결정 | 이유 |
|------|------|------|
| 칸 정의 | 학습 쪽(`feat/corpus96`, PR #28)과 같은 `SLOTS96` | 학습 코퍼스와 추론 스택이 같은 규칙이어야 한다 (PR #28에서 공개 nartaa 코드와 비트 단위 일치 확인) |
| 모르는 이름 | `KeyError` → 그 study는 fallback | 기존 실패 처리 경로를 그대로 쓴다. 로그의 fallback 수로 드러난다 |
| 거울상 TTA | 넣지 않음 | 첫 LB 확인은 학습 때와 같은 평가 방식(k_eval 42)으로 |

## 3. 결과
- `ruff check src tests kaggle`, `ruff format --check` 통과. `pytest -q` 130 passed, 2 skipped (Raptor 멤버 DICOM 추론 테스트를 SLOTS44·SLOTS96 둘 다로)
- Kaggle: 코드 데이터셋 `kimche12/rsna-knee-code` 새 버전, 가중치 `kimche12/rsna-knee-weights-e14`, 혼합 노트북 `kimche12/rsna-knee-public956-e14`
  (공개 0.956 노트북 0.85 + exp014 0.15 순위 혼합)

## 4. 미해결 문제
- 96장 스택은 DICOM을 더 많이 읽는다 → 전체 test 실행 시간 미측정. 혼합 노트북의 8.3시간 가드가 넘치면 공개 결과만 제출된다 (점수가 0.956 그대로면 의심)

## 5. 다음 작업자가 할 일
1. LB 결과를 기록하고, 오르면 거울상 TTA와 혼합 비중을 검토
