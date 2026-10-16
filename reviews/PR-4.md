# PR #4 리뷰 — 프로젝트 뼈대, macro AUC metric, 제출 형식 검증

- PR: https://github.com/kimcheolhui9846/RSNA-Knee-Abnormality-Detection_hk/pull/4
- 브랜치: `feat/project-setup` (33e2160) → `main`
- 리뷰어: 정해윤 (Claude 보조) | 날짜: 2026-09-29
- 결론: **Approve** — 머지해도 됨. 아래 제안은 후속 PR에서 처리해도 무방.

## 로컬 검증

| 항목 | 결과 |
|------|------|
| `uv sync --locked` | 성공 (lock 일치) |
| `uv run ruff check .` | All checks passed |
| `uv run ruff format --check .` | 9 files already formatted |
| `uv run pytest -q` | 14 passed, 1 skipped (`sample_submission.csv` 없음 — 의도된 skip) |
| main과 충돌 | 없음 (main 기준 fast-forward 가능, 기존 `handoff/ACL_CascadedCNN_Handoff_1.md` 유지) |

## 좋은 점

- 결측 라벨을 0으로 채우지 않고 라벨별로 마스킹한 것, 그리고 이를 `test_missing_labels_are_masked_not_treated_as_negative`로 고정한 것.
- `.gitignore`에서 `/data/`로 루트 고정해 `src/data/`가 무시되는 문제를 막은 것.
- torch를 `--extra train`으로 분리해 CI와 테스트를 가볍게 유지한 것.
- 대회 데이터가 들어오면 라벨 순서를 자동 대조하는 skip 테스트.

## 제안 (차단 아님)

1. **`src/submission.py` — 숫자형 검사 누락.** 예측 컬럼이 문자열이면 `probs < 0`에서 `TypeError`가 나서 "문제가 있으면 ValueError" 계약이 깨진다.
   ```python
   non_numeric = [c for c in LABELS if not pd.api.types.is_numeric_dtype(df[c])]
   if non_numeric:
       raise ValueError(f"non-numeric prediction columns: {non_numeric}")
   ```
2. **`src/metrics.py` — 모든 라벨이 NaN일 때.** `np.nanmean`이 `RuntimeWarning: Mean of empty slice`를 내고 `nan`을 돌려준다. 작은 fold나 디버그 subset에서 조용히 넘어갈 수 있으니 명시적으로 처리하는 편이 낫다.
   ```python
   valid = [v for v in per_label.values() if not np.isnan(v)]
   if not valid:
       raise ValueError("no label has both classes; macro AUC undefined")
   return float(np.mean(valid)), per_label
   ```
   (CV 로그에 몇 개 라벨이 평균에 들어갔는지 같이 찍어두면 fold 간 비교할 때도 편하다.)
3. **PR 템플릿 재현 명령.** `python src/train.py`로 실행하면 `sys.path[0]`이 `src/`가 되어 `from src.… import`가 실패한다. `python -m src.train --config configs/expXXX.yaml`로 바꾸자.
4. **라벨명 확인.** `"Baker's"`의 아포스트로피(`'` vs `’`)와 공백 표기는 데이터가 들어오면 skip된 테스트로 꼭 한 번 대조. 제출 컬럼이 한 글자만 달라도 0점 처리될 수 있다.
5. **오프라인 제출 준비 (다음 작업 메모).** `pylibjpeg*`, `iterative-stratification`은 Kaggle 기본 이미지에 없을 수 있으니, 제출 노트북용 wheel 데이터셋을 따로 만들어두자.

## 미해결 질문 (핸드오프 §4와 동일, 확인 담당 정하기)

- 공식 metric의 단일 클래스 라벨 처리 방식
- 환자 ID 제공 여부 → fold 분할 단위
