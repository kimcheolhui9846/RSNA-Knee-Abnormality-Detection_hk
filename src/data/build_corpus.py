"""train DICOM → Raptor 코퍼스(`all_vols.npy`, `all_masks.npy`, `all_ids.npy`)를 직접 만든다.

공개 코퍼스(44장, 구간 15–85%)와 달리 칸 장수·구간을 바꾼 코퍼스(예: 64장, 6–94%)가 필요할 때 쓴다.
스택은 Kaggle 추론과 같은 `build_stack()`으로 만든다 → 학습·추론 입력이 같다.

실행 (CPU):
    python -m src.data.build_corpus --data-dir <대회 데이터> --out <폴더>/raptor_corpus64 \
        --slots SLOTS64 --span 0.06 0.94 --workers 16
"""

import argparse
import json
import logging
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import ID_COL
from src.data.raptor_stack import SLOTS44, SLOTS64, build_stack

log = logging.getLogger(__name__)
SLOT_SETS = {"SLOTS44": SLOTS44, "SLOTS64": SLOTS64}


def study_series(series_csv: Path) -> dict[str, list[dict]]:
    """`*_series.csv` → study별 시리즈 행 목록 (CSV 순서 유지, `build_stack`이 이 순서로 고른다)."""
    df = pd.read_csv(series_csv, dtype={ID_COL: str, "SeriesInstanceUID": str})
    return {sid: g.to_dict("records") for sid, g in df.groupby(ID_COL, sort=False)}


def _one(args: tuple) -> tuple[int, np.ndarray, np.ndarray]:
    i, rows, study_dir, kw = args
    vol, mask = build_stack(rows, study_dir, **kw)
    return i, vol, mask


def build_corpus(
    data_dir: Path,
    out: Path,
    slots: str = "SLOTS64",
    span: tuple[float, float] = (0.06, 0.94),
    img: int = 336,
    crop_mm: float = 140.0,
    workers: int = 8,
    split: str = "train",
    limit: int | None = None,
    save_every: int = 100,
) -> Path:
    data_dir, out = Path(data_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    series = study_series(data_dir / f"{split}_series.csv")
    ids = pd.read_csv(data_dir / f"{split}.csv", dtype={ID_COL: str})[ID_COL].tolist()
    ids = [u for u in ids if u in series][:limit]
    depth = sum(k for _, _, k in SLOT_SETS[slots])
    shape = (len(ids), depth, img, img)
    meta = {"slots": slots, "span": list(span), "img": img, "crop_mm": crop_mm, "n": len(ids)}
    vols_path, partial = out / "all_vols.npy", out / "_partial.npz"
    vols = masks = done = None
    if vols_path.is_file() and partial.is_file():  # 끊긴 작업 이어서
        p = np.load(partial, allow_pickle=True)
        if json.loads(str(p["meta"])) == meta and list(p["ids"]) == ids:
            vols = np.lib.format.open_memmap(vols_path, mode="r+")
            masks, done = p["masks"].copy(), p["done"].copy()
            log.info("이어서: %d/%d 완료", int(done.sum()), len(ids))
    if vols is None or vols.shape != shape:
        vols = np.lib.format.open_memmap(vols_path, mode="w+", dtype=np.uint8, shape=shape)
        masks, done = np.zeros((len(ids), depth), np.uint8), np.zeros(len(ids), bool)

    def save_partial() -> None:
        vols.flush()  # 스택이 디스크에 쓰인 뒤에 완료 표시를 남긴다
        tmp = out / "_partial.tmp.npz"
        np.savez(
            tmp, masks=masks, done=done, ids=np.array(ids, dtype=object), meta=json.dumps(meta)
        )
        tmp.replace(partial)

    kw = {"slots": SLOT_SETS[slots], "span": tuple(span), "img": img, "crop_mm": crop_mm}
    jobs = [
        (i, series[u], data_dir / f"{split}_series" / u, kw)
        for i, u in enumerate(ids)
        if not done[i]
    ]
    t0 = time.time()
    # maxtasksperchild: 작업자 메모리가 쌓이지 않게 주기적으로 새로 띄운다
    with Pool(workers, maxtasksperchild=50) as pool:
        for n, (i, vol, mask) in enumerate(pool.imap_unordered(_one, jobs), 1):
            vols[i], masks[i], done[i] = vol, mask, True
            if n % save_every == 0 or n == len(jobs):
                save_partial()
                log.info("%d/%d studies, %.0fs", int(done.sum()), len(ids), time.time() - t0)
    save_partial()
    np.save(out / "all_masks.npy", masks)
    np.save(out / "all_ids.npy", np.array(ids, dtype=object))
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    partial.unlink()
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[corpus %(asctime)s] %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--slots", default="SLOTS64", choices=sorted(SLOT_SETS))
    p.add_argument("--span", type=float, nargs=2, default=(0.06, 0.94))
    p.add_argument("--img", type=int, default=336)
    p.add_argument("--crop-mm", type=float, default=140.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--split", default="train")
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()
    build_corpus(
        a.data_dir, a.out, a.slots, tuple(a.span), a.img, a.crop_mm, a.workers, a.split, a.limit
    )


if __name__ == "__main__":
    main()
