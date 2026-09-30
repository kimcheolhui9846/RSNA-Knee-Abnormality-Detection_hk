"""StudyInstanceUID 단위 5-fold 분할.

환자 ID가 제공되지 않으므로 study 단위로 나눈다. 12개 라벨이 모두 달린 study는 라벨로 층화하고,
라벨이 비어 있는 study(report만 있음)는 따로 섞어 fold마다 고르게 배분한다.
결측을 0으로 채워 층화에 섞지 않는다.

    python -m src.folds --out folds/folds_v1.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import KFold

from src.constants import ID_COL, LABELS
from src.paths import data_dir


def make_folds(train: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    """`[ID_COL, "fold", "labeled"]` 데이터프레임을 train과 같은 행 순서로 돌려준다."""
    n_present = train[list(LABELS)].notna().sum(axis=1)
    labeled = n_present == len(LABELS)
    partial = (n_present > 0) & ~labeled
    if partial.any():
        raise ValueError(f"partial labels in {int(partial.sum())} studies — 처리 규칙부터 정한다")

    fold = np.full(len(train), -1)
    labeled_pos = np.flatnonzero(labeled.to_numpy())
    unlabeled_pos = np.flatnonzero(~labeled.to_numpy())

    if len(labeled_pos):
        y = train[list(LABELS)].to_numpy()[labeled_pos]
        mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for k, (_, va) in enumerate(mskf.split(labeled_pos, y)):
            fold[labeled_pos[va]] = k
    if len(unlabeled_pos):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for k, (_, va) in enumerate(kf.split(unlabeled_pos)):
            fold[unlabeled_pos[va]] = k

    return pd.DataFrame(
        {ID_COL: train[ID_COL].to_numpy(), "fold": fold, "labeled": labeled.to_numpy()}
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("folds/folds_v1.csv"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train = pd.read_csv(data_dir() / "train.csv")
    folds = make_folds(train, n_splits=args.n_splits, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # OS와 무관하게 같은 바이트가 나오도록 줄바꿈을 고정한다 (팀원이 해시로 대조)
    folds.to_csv(args.out, index=False, lineterminator="\n")
    print(folds.groupby(["fold", "labeled"]).size().unstack())


if __name__ == "__main__":
    main()
