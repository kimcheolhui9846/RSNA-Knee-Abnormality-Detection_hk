"""시리즈 DICOM 헤더 → 지방억제·영상 가중·픽셀 간격·좌우 판정 (exp006~).

공개 상위 노트북(Speedy Raptors 계열)의 판정 규칙을 우리 코드로 다시 작성했다.
- 지방억제: `Fat_Suppression` CSV 값만으로는 놓치는 시리즈가 있다(GE Dixon water 등).
  SeriesDescription/SequenceName 정규식 + ScanOptions로 판정한다.
- 가중(T1/T2/PD/GRE): 설명 문자열, 없으면 TR/TE로. PD·T2를 fluid-sensitive로 본다.
- 좌우: `Laterality` 태그, 없으면 영상 중심의 환자 x 좌표(오른쪽 무릎은 x < 0).

학습(캐시)과 Kaggle 추론이 같은 함수를 쓴다. 헤더는 시리즈 가운데 파일 하나만 읽는다.
실행(학습 데이터 전체): `python -m src.data.headers --split train --out <csv> --workers 16`
"""

import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

from src.constants import ID_COL

HDR_TAGS = [
    "SeriesDescription",
    "SequenceName",
    "ScanOptions",
    "ScanningSequence",
    "RepetitionTime",
    "EchoTime",
    "Laterality",
    "PixelSpacing",
    "Rows",
    "Columns",
    "ImagePositionPatient",
    "ImageOrientationPatient",
]
FATSAT_OPTS = {"FS", "FATSAT", "FAT_SAT", "FSAT"}
_SEP = re.compile(r"[_\-.]")
_FATSAT_RX = re.compile(
    r"\bfs\b|fatsat|fat sat|\bstir\b|\bspair\b|\bspir\b|\bwe\b|water excit|\btirm\b|\bsting\b"
    r"|\bfatsup\b|smart fat|\bwater\b"
)
_T1_RX = re.compile(r"\bt1\b|\bt1w\b")
_T2_RX = re.compile(r"\bt2\b|\bt2w\b")
_PD_RX = re.compile(r"\bpd\b|\bpdw\b|proton|\bdp\b|dens")
LAT_MIN_OFFSET_MM = 20.0  # 영상 중심 x가 이보다 0에 가까우면 좌우를 정하지 않는다


def read_header(series_dir: Path) -> dict:
    """시리즈 폴더의 가운데 DICOM 헤더 → `HDR_TAGS` 문자열 (다중 값은 `|`로 잇는다)."""
    files = sorted(e.name for e in os.scandir(series_dir) if e.name.endswith(".dcm"))
    row: dict = {"n_files": len(files)}
    if not files:
        return row
    ds = pydicom.dcmread(
        Path(series_dir, files[len(files) // 2]), stop_before_pixels=True, force=True
    )
    for tag in HDR_TAGS:
        value = getattr(ds, tag, None)
        if value is None:
            row[tag] = None
        elif isinstance(value, (list, tuple)) or type(value).__name__ == "MultiValue":
            row[tag] = "|".join(str(x) for x in value)
        else:
            row[tag] = str(value)
    return row


def scan_headers(series: pd.DataFrame, series_root: Path, workers: int = 16) -> pd.DataFrame:
    """`series`(ID_COL, SeriesInstanceUID, ...) 각 행에 헤더 컬럼을 붙인다.

    읽기 실패는 `header_error`에 남긴다."""
    series_root = Path(series_root)

    def one(item: tuple[str, str]) -> dict:
        study, uid = item
        try:
            return read_header(series_root / study / uid)
        except Exception as e:  # noqa: BLE001 — 한 시리즈 실패가 전체를 멈추지 않게
            return {"header_error": f"{type(e).__name__}: {e}"[:200]}

    items = list(
        zip(series[ID_COL].astype(str), series["SeriesInstanceUID"].astype(str), strict=True)
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(one, items))
    return pd.concat([series.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    """헤더 → `fatsat`(bool), `weight`(T1/T2/PD/GRE/UNK), `fluid`(PD·T2), `px`(mm/픽셀)."""
    df = df.copy()
    for col in HDR_TAGS:
        if col not in df.columns:
            df[col] = None
    desc = df["SeriesDescription"].fillna("") + " " + df["SequenceName"].fillna("")
    desc = desc.str.lower().str.replace(_SEP, " ", regex=True)
    opts = df["ScanOptions"].fillna("").str.upper().str.split("|")
    opts_fs = opts.apply(lambda ts: any(t.strip() in FATSAT_OPTS for t in ts))
    df["fatsat"] = desc.str.contains(_FATSAT_RX) | opts_fs
    tr = pd.to_numeric(df["RepetitionTime"], errors="coerce")
    te = pd.to_numeric(df["EchoTime"], errors="coerce")
    gre = df["ScanningSequence"].fillna("").str.upper().str.contains("GR")
    t1, t2, pdw = desc.str.contains(_T1_RX), desc.str.contains(_T2_RX), desc.str.contains(_PD_RX)
    df["weight"] = np.select(
        [t1 & ~t2 & ~pdw, t2 & ~pdw, pdw, gre, tr < 800, te > 60, tr >= 800],
        ["T1", "T2", "PD", "GRE", "T1", "T2", "PD"],
        default="UNK",
    )
    df["fluid"] = df["weight"].isin(["PD", "T2"])
    first = df["PixelSpacing"].fillna("").astype(str).str.split("|").str[0]
    df["px"] = pd.to_numeric(first.replace("", np.nan), errors="coerce")
    return df


def _vec(text, n: int) -> np.ndarray | None:
    if not isinstance(text, str):
        return None
    try:
        v = [float(x) for x in text.split("|")]
    except ValueError:
        return None
    return np.array(v) if len(v) >= n else None


def laterality(df: pd.DataFrame) -> dict[str, str | None]:
    """study → 'L' / 'R' / None. `Laterality` 태그가 우선, 없으면 영상 중심의 환자 x 좌표."""
    centre_x: dict[str, list[float]] = {}
    for r in df.itertuples(index=False):
        ipp = _vec(getattr(r, "ImagePositionPatient", None), 3)
        iop = _vec(getattr(r, "ImageOrientationPatient", None), 6)
        ps = _vec(getattr(r, "PixelSpacing", None), 2)
        rows, cols = getattr(r, "Rows", None), getattr(r, "Columns", None)
        if ipp is None or iop is None or ps is None or not rows or not cols:
            continue
        try:
            rows_f, cols_f = float(rows), float(cols)
        except (TypeError, ValueError):
            continue
        c = ipp[:3] + iop[:3] * ps[1] * cols_f / 2 + iop[3:6] * ps[0] * rows_f / 2
        centre_x.setdefault(getattr(r, ID_COL), []).append(float(c[0]))

    out: dict[str, str | None] = {}
    for study, g in df.groupby(ID_COL):
        tags = [str(x).strip().upper() for x in g["Laterality"].dropna()]
        tags = [t[0] for t in tags if t and t[0] in ("L", "R")]
        if tags:
            out[study] = tags[0]
            continue
        xs = centre_x.get(study)
        if not xs:
            out[study] = None
            continue
        m = float(np.median(xs))
        out[study] = None if abs(m) < LAT_MIN_OFFSET_MM else ("R" if m < 0 else "L")
    return out


def main() -> None:
    from src.paths import data_dir

    p = argparse.ArgumentParser()
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()
    root = args.data_dir or data_dir()
    series = pd.read_csv(
        root / f"{args.split}_series.csv", dtype={ID_COL: str, "SeriesInstanceUID": str}
    )
    out = annotate(scan_headers(series, root / f"{args.split}_series", workers=args.workers))
    lat = laterality(out)
    out["laterality"] = out[ID_COL].map(lat)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    n_err = int(out["header_error"].notna().sum()) if "header_error" in out else 0
    print(f"{len(out)} series, header errors {n_err}, → {args.out}")
    print("laterality:", pd.Series(lat).value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()
