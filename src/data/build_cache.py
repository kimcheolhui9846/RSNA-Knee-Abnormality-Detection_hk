"""시리즈별 전처리 결과를 `.npy`로 캐시하고 인덱스 CSV를 만든다 (CPU 전용).

실행: `python -m src.data.build_cache --workers 16`
- 출력: `<out>/<StudyInstanceUID>/<SeriesInstanceUID>.npy` (uint8, (N, size, size))
  와 인덱스 `<out>/<split>_index.csv`
- 이미 있는 파일은 건너뛰므로 중간에 끊겨도 다시 실행하면 이어서 만든다.
- 읽기에 실패한 시리즈는 전체를 멈추지 않고 인덱스에 `status=error`로 남긴다.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

from src.constants import ID_COL
from src.data.dicom import load_series
from src.data.preprocess import preprocess_series
from src.paths import REPO_ROOT, data_dir


def _process(task: tuple[dict, Path, Path, int, str]) -> dict:
    row, data_root, out_dir, size, split = task
    study, series = row[ID_COL], row["SeriesInstanceUID"]
    series_dir = data_root / f"{split}_series" / study / series
    out = out_dir / study / f"{series}.npy"
    record = {**row, "path": f"{study}/{series}.npy", "status": "", "error": ""}
    try:
        first = next(series_dir.glob("*.dcm"), None)
        if first is None:
            raise FileNotFoundError(f"DICOM 없음: {series_dir}")
        header = pydicom.dcmread(first, stop_before_pixels=True)
        if out.exists():
            n_slices = np.load(out, mmap_mode="r").shape[0]
            record["status"] = "cached"
        else:
            arr = preprocess_series(load_series(series_dir), size=size)
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".tmp.npy")
            np.save(tmp, arr)
            os.replace(tmp, out)  # 쓰다 끊긴 파일이 완성본으로 남지 않게
            n_slices = arr.shape[0]
            record["status"] = "built"
        spacing = header.get("PixelSpacing", [np.nan, np.nan])
        record.update(
            n_slices=int(n_slices),
            orig_h=int(header.Rows),
            orig_w=int(header.Columns),
            pixel_spacing_row=float(spacing[0]),
            pixel_spacing_col=float(spacing[1]),
        )
    except Exception as e:  # noqa: BLE001 — 한 시리즈의 실패로 전체 캐시를 멈추지 않는다
        record["status"] = "error"
        record["error"] = repr(e)[:500]
    return record


def build_cache(
    data_root: Path,
    out_dir: Path,
    size: int = 256,
    workers: int = 1,
    split: str = "train",
    limit: int | None = None,
) -> pd.DataFrame:
    series = pd.read_csv(Path(data_root) / f"{split}_series.csv", dtype={ID_COL: str})
    series["SeriesInstanceUID"] = series["SeriesInstanceUID"].astype(str)
    if limit is not None:
        series = series.head(limit)
    tasks = [
        (row, Path(data_root), Path(out_dir), size, split) for row in series.to_dict("records")
    ]

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(_process, tasks, chunksize=8))
    else:
        records = [_process(t) for t in tasks]

    index = pd.DataFrame(records)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    index.to_csv(Path(out_dir) / f"{split}_index.csv", index=False, lineterminator="\n")
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--out", type=Path, default=None, help="기본값: cache/<size>")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--limit", type=int, default=None, help="앞에서 N개 시리즈만 (스모크용)")
    args = parser.parse_args()

    out = args.out or REPO_ROOT / "cache" / str(args.size)
    index = build_cache(data_dir(), out, args.size, args.workers, args.split, args.limit)
    print(index["status"].value_counts().to_string())
    errors = index[index["status"] == "error"]
    if len(errors):
        print(errors[[ID_COL, "SeriesInstanceUID", "error"]].head(10).to_string())


if __name__ == "__main__":
    main()
