# PR #7 리뷰 — report pseudo-label 평가 틀 (1단계)

- PR: https://github.com/kimcheolhui9846/RSNA-Knee-Abnormality-Detection_hk/pull/7
- 브랜치: `feat/pseudo-eval` (8945a6f)
- 리뷰어: Codex (GitHub 자동 리뷰) | 날짜: 2026-09-30

## 지적 사항
- [P2] `src/pseudo/evaluate.py:45` — 판단 보류(NaN)가 있는 라벨에서 `n`이 `len(answered)`(정답 study 전체 수)로 기록된다.
  10개 중 5개만 답해도 `coverage=0.5`, `n=10`이 되어 accuracy·sensitivity·specificity·AUC의 표본 수를 부풀린다.
  `answered.sum()` 또는 필터 후 `len(t)`를 써야 한다.

## 구현자 응답
작성: kim cheol hui (Claude Code 구현 담당) | 날짜: 2026-10-04

| 지적 | 처리 | 내용 |
|------|------|------|
| [P2] `n` 과대 집계 | **수정** | 라벨별 `n`을 필터 후 `len(t)`로 바꿈. macro 행의 `n`은 정답 study 수(코호트 크기) 그대로 둠 — 라벨마다 `n`이 다르므로 평균이 의미 없기 때문. 테스트: 기존 보류 테스트에 `n == 5` 단언 추가, `test_macro_n_is_number_of_labeled_studies` 추가 |

검증: `ruff check .` 통과, `pytest -q` 34 passed, 2 skipped. 지표 값 자체(AUC 등)는 원래도 보류를 뺀 표본으로 계산했으므로 이전 평가 결과의 AUC는 바뀌지 않는다.
