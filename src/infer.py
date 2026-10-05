"""test DICOM → submission.csv. Kaggle 제출 노트북에서 그대로 import해서 쓴다 (인터넷 불필요).

학습과 같은 함수(`load_series` → `preprocess_series` → `select_series` → `resample_depth`)로
입력을 만든다.
캐시를 쓰지 않고 study마다 DICOM을 바로 읽는다. 읽을 수 있는 시리즈가 하나도 없는 study는
`FALLBACK_PROB`로 채워 제출 파일이 항상 완성되게 한다.

여러 모델 앙상블: 가중치 폴더에 `ensemble.yaml`이 있으면 그 안의 멤버(각자 `model.safetensors`
+ `config.yaml`)를 모두 돌리고 라벨별 순위(rank)를 가중 평균한다 (`load_members`).
AUC는 순위만 보므로 모델마다 확률 보정이 달라도 순위 평균이 안전하다.

실행: `python -m src.infer --weights model.safetensors --config configs/exp001_baseline.yaml`
"""

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset

from src.constants import ID_COL, LABELS
from src.data.dataset import slab_stack
from src.data.dicom import load_series
from src.data.preprocess import preprocess_series
from src.data.study_table import SLOTS, resample_depth, select_series
from src.models import build_model
from src.paths import data_dir
from src.submission import validate_submission

FALLBACK_PROB = 0.5
# 멤버끼리 같아야 하는 입력 설정 — 같으면 DICOM을 한 번만 읽어 모든 멤버에 넣는다
DATA_KEYS = ("depth", "size", "slab", "target_slices")
log = logging.getLogger("infer")


@dataclass
class Member:
    name: str
    weights_path: Path
    config: dict
    weight: float = 1.0


def load_members(weights_dir: Path) -> list[Member]:
    """가중치 폴더 → 앙상블 멤버.

    - `ensemble.yaml` 없음: 폴더의 `model.safetensors` + `config.yaml` 하나 (단일 모델, 예전 구성)
    - `ensemble.yaml`: `members: [{dir: exp002, weight: 0.5}, ...]`
      — 각 `dir`에 `model.safetensors` + `config.yaml`
    """
    weights_dir = Path(weights_dir)
    spec_path = weights_dir / "ensemble.yaml"
    if not spec_path.is_file():
        config = yaml.safe_load((weights_dir / "config.yaml").read_text(encoding="utf-8"))
        return [Member("model", weights_dir / "model.safetensors", config)]
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    members = []
    for m in spec["members"]:
        d = weights_dir / m["dir"]
        config = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))
        members.append(Member(m["dir"], d / "model.safetensors", config, float(m.get("weight", 1))))
    return members


def rank_average(preds: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """멤버별 `(N, 12)` 확률 → 라벨별 순위(0–1 백분위)의 가중 평균."""
    total = float(sum(weights))
    ranks = [
        pd.DataFrame(p).rank(pct=True).to_numpy() * w for p, w in zip(preds, weights, strict=True)
    ]
    return (np.sum(ranks, axis=0) / total).astype(np.float32)


def load_ensemble(weights_path: Path, model_cfg: dict, device: str) -> list[torch.nn.Module]:
    """`fold{k}.` 접두사로 저장된 가중치 → fold별 모델. 사전학습 가중치는 받지 않는다."""
    state = load_file(str(weights_path))
    folds = sorted({k.split(".", 1)[0] for k in state}, key=lambda f: int(f.removeprefix("fold")))
    models = []
    for fold in folds:
        model = build_model({**model_cfg, "pretrained": False})
        prefix = f"{fold}."
        model.load_state_dict(
            {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
        )
        models.append(model.to(device).eval())
    return models


def _fat_suppression(row) -> int | None:
    """칸 구분용 `Fat_Suppression` (0/1).

    비어 있으면 train에서 항상 같은 값인 `Fluid_Sensitive`로 대신하고,
    둘 다 없거나 숫자가 아니면 None (그 시리즈는 건너뛴다)."""
    for name in ("Fat_Suppression", "Fluid_Sensitive"):
        try:
            value = float(getattr(row, name))
        except (AttributeError, TypeError, ValueError):
            continue
        if not np.isnan(value):
            return int(value)
    return None


class TestStudyDataset(Dataset):
    """study 하나 → (study_id, image (S, D, H, W) float 0~1, slot_mask (S,), 실패 여부)."""

    def __init__(self, study_ids: list[str], series: pd.DataFrame, series_root: Path, cfg: dict):
        self.study_ids = study_ids
        self.by_study = {sid: g for sid, g in series.groupby(ID_COL)}
        self.series_root = series_root
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.study_ids)

    def __getitem__(self, i: int) -> dict:
        sid = self.study_ids[i]
        depth, size, slab = self.cfg["depth"], self.cfg["size"], self.cfg.get("slab", 1)
        shape = (len(SLOTS), depth, size, size)
        if slab > 1:  # 2.5D: 학습과 같은 slab_stack
            shape = (len(SLOTS), depth, slab, size, size)
        image = np.zeros(shape, dtype=np.float32)
        slot_mask = np.zeros(len(SLOTS), dtype=bool)

        volumes, rows = {}, []
        for row in self.by_study.get(sid, pd.DataFrame()).itertuples():
            key = str(row.SeriesInstanceUID)
            fat_sup = _fat_suppression(row)
            if (
                fat_sup is None
            ):  # 칸을 정할 수 없으면 읽지 않고 건너뛴다 (예외로 추론 전체가 멈추지 않게)
                log.warning(
                    "series 메타데이터 결측 %s/%s: Fat_Suppression·Fluid_Sensitive", sid, key
                )
                continue
            try:
                vol = preprocess_series(load_series(self.series_root / sid / key), size=size)
            except Exception as e:  # noqa: BLE001 — 시리즈 하나가 망가져도 나머지로 예측한다
                log.warning("series 읽기 실패 %s/%s: %r", sid, key, e)
                continue
            volumes[key] = vol
            rows.append((sid, key, row.Anatomical_Plane, fat_sup, len(vol), "built", key))

        if rows:
            index = pd.DataFrame(
                rows,
                columns=[
                    ID_COL,
                    "SeriesInstanceUID",
                    "Anatomical_Plane",
                    "Fat_Suppression",
                    "n_slices",
                    "status",
                    "path",
                ],
            )
            chosen = select_series(index, self.cfg["target_slices"]).loc[sid]
            for k, key in enumerate(chosen):
                if key is not None:
                    vol = volumes[key]
                    stacked = (
                        slab_stack(vol, depth, slab) if slab > 1 else resample_depth(vol, depth)
                    )
                    image[k] = stacked.astype(np.float32) / 255.0
                    slot_mask[k] = True
        return {
            "study_id": sid,
            "image": torch.from_numpy(image),
            "slot_mask": torch.from_numpy(slot_mask),
            "failed": not slot_mask.any(),
        }


def predict(
    data_root: Path,
    weights_path: Path,
    config: dict,
    out_csv: Path,
    device: str | None = None,
    num_workers: int = 2,
) -> pd.DataFrame:
    """단일 모델 추론 (멤버 하나짜리 `predict_members`)."""
    member = Member("model", Path(weights_path), config)
    return predict_members(data_root, [member], out_csv, device, num_workers)


def predict_members(
    data_root: Path,
    members: list[Member],
    out_csv: Path,
    device: str | None = None,
    num_workers: int = 2,
) -> pd.DataFrame:
    """멤버마다 fold 평균 확률을 내고, 멤버가 둘 이상이면 라벨별 순위를 가중 평균한다."""
    data_root = Path(data_root)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    data_cfg = {k: members[0].config.get(k) for k in DATA_KEYS}
    for m in members[1:]:
        other = {k: m.config.get(k) for k in DATA_KEYS}
        if other != data_cfg:
            raise ValueError(
                f"멤버 입력 설정이 다르다: {members[0].name} {data_cfg} vs {m.name} {other}"
            )

    test = pd.read_csv(data_root / "test.csv", dtype={ID_COL: str})
    series = pd.read_csv(
        data_root / "test_series.csv", dtype={ID_COL: str, "SeriesInstanceUID": str}
    )
    study_ids = test[ID_COL].tolist()
    models = [load_ensemble(m.weights_path, m.config["model"], device) for m in members]
    log.info(
        "test studies %d, members %s, device %s",
        len(study_ids),
        [(m.name, len(f), m.weight) for m, f in zip(members, models, strict=True)],
        device,
    )

    ds = TestStudyDataset(study_ids, series, data_root / "test_series", members[0].config)
    loader = DataLoader(ds, batch_size=1, num_workers=num_workers)
    amp = device == "cuda"
    probs = np.full((len(members), len(study_ids), len(LABELS)), FALLBACK_PROB, np.float32)
    row = {sid: i for i, sid in enumerate(study_ids)}
    n_failed = 0
    with torch.no_grad(), torch.autocast(device_type=device, enabled=amp):
        for i, batch in enumerate(loader):
            sid = batch["study_id"][0]
            if bool(batch["failed"][0]):
                n_failed += 1
                continue
            image, mask = batch["image"].to(device), batch["slot_mask"].to(device)
            for j, folds in enumerate(models):
                fold_probs = [torch.sigmoid(f(image, mask).float())[0].cpu().numpy() for f in folds]
                probs[j, row[sid]] = np.mean(fold_probs, axis=0)
            if (i + 1) % 100 == 0:
                log.info("%d / %d studies, %.1fs", i + 1, len(study_ids), time.time() - started)

    if len(members) == 1:
        final = probs[0]
    else:
        final = rank_average(list(probs), [m.weight for m in members])
    sub = pd.DataFrame(final, columns=list(LABELS)).astype(float)
    sub.insert(0, ID_COL, study_ids)
    validate_submission(sub, expected_ids=study_ids)
    sub.to_csv(out_csv, index=False)
    log.info(
        "submission %s: %d studies (fallback %d), %.1fs",
        out_csv,
        len(sub),
        n_failed,
        time.time() - started,
    )
    return sub


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/exp001_baseline.yaml"))
    parser.add_argument("--out", type=Path, default=Path("submission.csv"))
    parser.add_argument("--data-dir", type=Path, default=None, help="기본값: src.paths.data_dir()")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="[infer %(asctime)s] %(message)s", datefmt="%H:%M:%S"
    )

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    predict(args.data_dir or data_dir(), args.weights, config, args.out, num_workers=args.workers)


if __name__ == "__main__":
    main()
