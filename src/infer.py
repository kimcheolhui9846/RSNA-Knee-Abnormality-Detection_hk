"""test DICOM → submission.csv. Kaggle 제출 노트북에서 그대로 import해서 쓴다 (인터넷 불필요).

학습과 같은 함수(`load_series` → `preprocess_series` → `select_series` → `resample_depth`)로
입력을 만든다.
캐시를 쓰지 않고 study마다 DICOM을 바로 읽는다. 읽을 수 있는 시리즈가 하나도 없는 study는
`FALLBACK_PROB`로 채워 제출 파일이 항상 완성되게 한다.

실행: `python -m src.infer --weights model.safetensors --config configs/exp001_baseline.yaml`
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset

from src.constants import ID_COL, LABELS
from src.data.dicom import load_series
from src.data.preprocess import preprocess_series
from src.data.study_table import SLOTS, resample_depth, select_series
from src.models.baseline import KneeBaseline
from src.paths import data_dir
from src.submission import validate_submission

FALLBACK_PROB = 0.5
log = logging.getLogger("infer")


def load_ensemble(weights_path: Path, model_cfg: dict, device: str) -> list[torch.nn.Module]:
    """`fold{k}.` 접두사로 저장된 가중치 → fold별 모델. 사전학습 가중치는 받지 않는다."""
    state = load_file(str(weights_path))
    folds = sorted({k.split(".", 1)[0] for k in state}, key=lambda f: int(f.removeprefix("fold")))
    models = []
    for fold in folds:
        model = KneeBaseline(**{**model_cfg, "pretrained": False})
        prefix = f"{fold}."
        model.load_state_dict(
            {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
        )
        models.append(model.to(device).eval())
    return models


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
        depth, size = self.cfg["depth"], self.cfg["size"]
        image = np.zeros((len(SLOTS), depth, size, size), dtype=np.float32)
        slot_mask = np.zeros(len(SLOTS), dtype=bool)

        volumes, rows = {}, []
        for row in self.by_study.get(sid, pd.DataFrame()).itertuples():
            key = str(row.SeriesInstanceUID)
            try:
                vol = preprocess_series(load_series(self.series_root / sid / key), size=size)
            except Exception as e:  # noqa: BLE001 — 시리즈 하나가 망가져도 나머지로 예측한다
                log.warning("series 읽기 실패 %s/%s: %r", sid, key, e)
                continue
            volumes[key] = vol
            rows.append(
                (sid, key, row.Anatomical_Plane, int(row.Fat_Suppression), len(vol), "built", key)
            )

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
                    image[k] = resample_depth(volumes[key], depth).astype(np.float32) / 255.0
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
    data_root = Path(data_root)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()

    test = pd.read_csv(data_root / "test.csv", dtype={ID_COL: str})
    series = pd.read_csv(
        data_root / "test_series.csv", dtype={ID_COL: str, "SeriesInstanceUID": str}
    )
    study_ids = test[ID_COL].tolist()
    models = load_ensemble(Path(weights_path), config["model"], device)
    log.info("test studies %d, folds %d, device %s", len(study_ids), len(models), device)

    ds = TestStudyDataset(study_ids, series, data_root / "test_series", config)
    loader = DataLoader(ds, batch_size=1, num_workers=num_workers)
    amp = device == "cuda"
    probs: dict[str, np.ndarray] = {}
    n_failed = 0
    with torch.no_grad(), torch.autocast(device_type=device, enabled=amp):
        for i, batch in enumerate(loader):
            sid = batch["study_id"][0]
            if bool(batch["failed"][0]):
                probs[sid] = np.full(len(LABELS), FALLBACK_PROB, dtype=np.float32)
                n_failed += 1
                continue
            image, mask = batch["image"].to(device), batch["slot_mask"].to(device)
            fold_probs = [torch.sigmoid(m(image, mask).float())[0].cpu().numpy() for m in models]
            probs[sid] = np.mean(fold_probs, axis=0)
            if (i + 1) % 100 == 0:
                log.info("%d / %d studies, %.1fs", i + 1, len(study_ids), time.time() - started)

    sub = pd.DataFrame([probs[s] for s in study_ids], columns=list(LABELS)).astype(float)
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
