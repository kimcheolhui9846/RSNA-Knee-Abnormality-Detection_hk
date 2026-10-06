# HANDOFF — exp010: 64장 코퍼스 Raptor

작성: kim cheol hui | 날짜: 2026-10-06 | 브랜치: `exp/010-raptor64` (base: `feat/corpus64`, stacked PR)

## 1. 요약 (TL;DR)
- exp007 절차에 입력만 64장·구간 6–94% 코퍼스(PR #25로 직접 생성)로 바꾼 config 하나. 코드 변경 없음.
- Kaggle 추론은 `slots: SLOTS64`, `span` 설정으로 같은 스택을 만든다 (`src/infer.py`가 이미 지원).

## 2. 한 일과 결정 기록
| 항목 | 결정 | 이유 |
|------|------|------|
| 바꾸는 것 | 입력(장수·구간)만 | 효과를 분리해서 본다 |
| 라벨 | dread (exp007과 같음) | 위와 같음. 정제 라벨(exp009)은 준비되면 따로 |

## 3. 결과
- `ruff check`, `pytest -q` 통과 (smoke가 새 config 포함)
- RunPod 결과: (실행 후 기록) — RunPod 잔액 충전 후 실행

## 4. 미해결 문제
- 코퍼스 31.8 GB → Pod 다운로드 시간 증가

## 5. 다음 작업자가 할 일
1. 정답 58 OOF를 exp007과 비교, 라벨별(특히 Lateral) 확인
2. 좋으면 exp007·008과 앙상블, 정제 라벨로 재학습
