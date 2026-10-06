"""정제 라벨 (exp009~): 판독문 LLM 라벨과 이미지 모델 OOF 예측을 라벨별 순위로 섞는다.

정답 58에서 판독문 라벨(공개 세트 평균) 0.89, 이미지 모델 0.90, 반반 섞으면 0.92 — 서로 보완한다.
test에는 판독문이 없으므로 이 보완은 학습 라벨을 고치는 것으로만 쓸 수 있다.

- 판독문 점수 R: 판독문 라벨 세트마다 라벨별 백분위 순위 → 평균
- 이미지 점수 I: 이미지 OOF 예측(학습에서 그 study를 빼고 학습한 모델)의 라벨별 백분위 순위
- 섞은 순위 M = (1 - alpha) R + alpha I
- 확률로 되돌리기: M의 순서를 유지한 채 `ref`(기준 soft 라벨)의 라벨별 분포에 맞춘다(분위수 매핑)
  → 유병률·양성 가중치는 기준 라벨과 같고 순서만 바뀐다
- 정답 행(`source == gt`)은 그대로 둔다
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import ID_COL, LABELS


def _pct_rank(df: pd.DataFrame) -> pd.DataFrame:
    """라벨별 백분위 순위. 결측은 그 라벨의 중앙(0.5)."""
    return df.rank(pct=True).fillna(0.5)


def report_score(sets: list[pd.DataFrame], ids: pd.Index) -> pd.DataFrame:
    ranks = []
    for s in sets:
        s = s.drop_duplicates(ID_COL).set_index(ID_COL).reindex(ids)[list(LABELS)]
        ranks.append(_pct_rank(s.apply(pd.to_numeric, errors="coerce")))
    return sum(ranks) / len(ranks)


def quantile_map(score: pd.Series, ref: pd.Series) -> np.ndarray:
    """`score`의 순서를 유지하면서 값의 분포를 `ref`와 같게."""
    ref_sorted = np.sort(ref.to_numpy(dtype=float))
    q = (score.rank(method="average").to_numpy() - 0.5) / len(score)  # 각 값의 분위 위치
    pos = np.clip(q * len(ref_sorted) - 0.5, 0, len(ref_sorted) - 1)
    return np.interp(pos, np.arange(len(ref_sorted)), ref_sorted)


def refine(
    base: pd.DataFrame,
    report_sets: list[pd.DataFrame],
    image_oof: pd.DataFrame,
    alpha: float = 0.5,
) -> pd.DataFrame:
    """`base`(ID, 12 라벨, source) 중 pseudo 행의 라벨을 정제한 새 표."""
    out = base.copy()
    pseudo = out["source"] != "gt"
    ids = pd.Index(out.loc[pseudo, ID_COL].astype(str))
    img = image_oof.drop_duplicates(ID_COL).set_index(ID_COL).reindex(ids)[list(LABELS)]
    missing = img.isna().any(axis=1)
    if missing.any():
        raise ValueError(f"이미지 OOF가 없는 pseudo study {int(missing.sum())}개")
    mix = (1 - alpha) * report_score(report_sets, ids) + alpha * _pct_rank(img)
    ref = out.loc[pseudo, list(LABELS)].astype(float)
    for lab in LABELS:
        out.loc[pseudo, lab] = quantile_map(mix[lab], ref[lab].fillna(ref[lab].median()))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True, help="기준 라벨 (예: labels_dread.csv)")
    p.add_argument("--report", type=Path, nargs="+", required=True, help="판독문 라벨 세트들")
    p.add_argument("--oof", type=Path, nargs="+", required=True, help="fold별 oof_pseudo.csv")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    read = lambda f: pd.read_csv(f, dtype={ID_COL: str})  # noqa: E731
    oof = pd.concat([read(f) for f in a.oof], ignore_index=True)
    refined = refine(read(a.base), [read(f) for f in a.report], oof, a.alpha)
    refined.to_csv(a.out, index=False)
    print(f"{a.out}: {len(refined)} rows, alpha {a.alpha}")


if __name__ == "__main__":
    main()
