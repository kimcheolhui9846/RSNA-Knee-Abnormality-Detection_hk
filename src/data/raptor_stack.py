"""Raptor 방식 입력 (exp007~): study → 고정 크기 슬라이스 스택 → 2.5D 창(window) 묶음.

공개 Raptor(dreaddevelopment, 단일 CoAtNet LB 0.924)의 규칙을 우리 코드로 다시 작성했다.
- 스택: 칸 5개(시상 fluid / 시상 non-fluid / 관상 fluid / 관상 아무거나 / 축상)에서 정해진 장수,
  시리즈의 `span`(기본 15–85%, 공개 코퍼스 v2와 비트 단위 일치 확인) 구간에서 고르게,
  시리즈 단위 2–98 백분위 밝기,
  슬라이스마다 영상 중심 `crop_mm`(140 mm) 정사각형 → `img`(336) INTER_AREA. 없는 칸은 0.
- 창: 스택에서 연속한 슬라이스 3장을 RGB 1장으로. 학습은 무작위 `k`개, 평가는 고르게 `k_eval`개.
  (창이 칸 경계를 넘을 수 있는 것도 원본과 같다)

학습은 공개 코퍼스(`all_vols.npy` 등, 같은 규칙으로 만든 44×336×336)를 그대로 쓰고,
Kaggle 추론은 `build_stack()`으로 test DICOM에서 같은 스택을 만든다.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.constants import ID_COL, LABELS

# (단면, Fluid_Sensitive 선호값(-1: 상관없음), 장수)
SLOTS44 = [
    ("Sagittal", 1, 12),
    ("Sagittal", 0, 10),
    ("Coronal", 1, 8),
    ("Coronal", 0, 6),
    ("Axial", -1, 8),
]
SLOTS64 = [
    ("Sagittal", 1, 18),
    ("Sagittal", 0, 14),
    ("Coronal", 1, 12),
    ("Coronal", 0, 8),
    ("Axial", -1, 12),
]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# --- 스택 만들기 (Kaggle 추론, 검증) -------------------------------------------------


def _ordered_files(series_dir: Path) -> tuple[list[tuple[Path, float]], float]:
    """시리즈 파일을 슬라이스 법선 방향 위치로 정렬 → [(파일, PixelSpacing)], 중앙값 spacing."""
    import pydicom  # DICOM 경로(추론·검증)에서만 쓴다 — 코퍼스 학습 Pod에는 없어도 된다

    recs, spacings = [], []
    for f in Path(series_dir).glob("*.dcm"):
        try:
            h = pydicom.dcmread(f, stop_before_pixels=True)
            iop = getattr(h, "ImageOrientationPatient", None)
            ipp = getattr(h, "ImagePositionPatient", None)
            if iop is not None and ipp is not None and len(iop) == 6:
                n = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
                pos = float(np.dot(np.array(ipp, float), n))
            else:
                pos = float(getattr(h, "InstanceNumber", 0) or 0)
            ps = getattr(h, "PixelSpacing", None)
            ps = float(ps[0]) if ps is not None else 0.5
            spacings.append(ps)
            recs.append((pos, f, ps))
        except Exception:  # noqa: BLE001 — 헤더가 깨진 파일도 순서만 뒤로 두고 남긴다
            recs.append((0.0, f, 0.5))
    recs.sort(key=lambda r: r[0])
    med = float(np.median(spacings)) if spacings else 0.5
    return [(f, ps) for _, f, ps in recs], med


def _read_pixels(path: Path) -> np.ndarray:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_modality_lut

    d = pydicom.dcmread(path)
    a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
    if str(getattr(d, "PhotometricInterpretation", "")) == "MONOCHROME1":
        a = a.max() - a
    return a


def _crop_resize(a: np.ndarray, ps: float, crop_mm: float, img: int) -> np.ndarray:
    import cv2

    h, w = a.shape
    cpx = min(int(round(crop_mm / max(ps, 1e-3))), min(h, w))
    y0, x0 = (h - cpx) // 2, (w - cpx) // 2
    return cv2.resize(a[y0 : y0 + cpx, x0 : x0 + cpx], (img, img), interpolation=cv2.INTER_AREA)


def _pick_series(rows: list[dict], plane: str, fluid: int, used: set) -> dict | None:
    cands = [
        r for r in rows if r["Anatomical_Plane"] == plane and r["SeriesInstanceUID"] not in used
    ]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def build_stack(
    series_rows: list[dict],
    study_dir: Path,
    slots=SLOTS44,
    span: tuple[float, float] = (0.15, 0.85),
    img: int = 336,
    crop_mm: float = 140.0,
    pct: tuple[float, float] = (2.0, 98.0),
) -> tuple[np.ndarray, np.ndarray]:
    """study의 시리즈 행(`*_series.csv` 순서) → `(vol (D, img, img) uint8, mask (D,) uint8)`."""
    depth = sum(k for _, _, k in slots)
    vol = np.zeros((depth, img, img), np.uint8)
    idx, used = 0, set()
    for plane, fluid, k in slots:
        r = _pick_series(series_rows, plane, fluid, used)
        if r is None:
            idx += k
            continue
        used.add(r["SeriesInstanceUID"])
        files, med_ps = _ordered_files(Path(study_dir) / str(r["SeriesInstanceUID"]))
        if not files:
            idx += k
            continue
        n = len(files)
        lo, hi = int(n * span[0]), int(n * span[1]) - 1
        hi = max(hi, lo)
        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k
        arrs, pss = [], []
        for p in picks:
            fp, ps = files[min(p, n - 1)]
            try:
                arrs.append(_read_pixels(fp))
                pss.append(ps)
            except Exception:  # noqa: BLE001
                arrs.append(None)
                pss.append(med_ps)
        valid = [a for a in arrs if a is not None]
        if valid:
            loq, hiq = np.percentile(np.concatenate([a.ravel() for a in valid]), list(pct))
        else:
            loq, hiq = 0.0, 1.0
        for a, ps in zip(arrs, pss, strict=True):
            if idx >= depth:
                break
            if a is None:
                idx += 1
                continue
            aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
            aw = _crop_resize(aw, ps if ps > 0 else med_ps, crop_mm, img)
            vol[idx] = (aw * 255).astype(np.uint8)
            idx += 1
        if idx >= depth:
            break
    mask = (vol.reshape(depth, -1).sum(1) > 0).astype(np.uint8)
    return vol, mask


# --- 창 (학습·평가 공통) ---------------------------------------------------------------


def window_centers(mask: np.ndarray, k: int, rng: np.random.Generator | None = None) -> list[int]:
    """창 중심 k개. `rng`가 있으면 무작위(학습), 없으면 고르게(평가). 양 이웃이 유효 구간 안."""
    depth = len(mask)
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, depth))
    lo, hi = int(valid.min()), int(valid.max())
    cs = list(range(lo + 1, hi))
    if not cs:
        cs = [max(1, min((lo + hi) // 2, depth - 2))]
    if rng is not None:
        pool = cs * (k // len(cs) + 1)
        rng.shuffle(pool)
        return pool[:k]
    idx = np.linspace(0, len(cs) - 1, k).round().astype(int)
    return [cs[i] for i in idx]


def make_windows(
    vol: np.ndarray, centers: list[int], res: int, normalize: bool = True
) -> torch.Tensor:
    """`(D, H, W)` uint8 → `(K, 3, res, res)` float (`normalize`면 ImageNet 정규화, 아니면 0–1)."""
    depth = vol.shape[0]
    tiles = []
    for c in centers:
        c = max(1, min(c, depth - 2))
        tri = torch.from_numpy(
            np.stack([vol[c - 1], vol[c], vol[c + 1]]).astype(np.float32) / 255.0
        )
        if tri.shape[-1] != res:
            tri = F.interpolate(tri[None], size=(res, res), mode="bilinear", align_corners=False)[0]
        tiles.append(tri)
    x = torch.stack(tiles)
    return (x - IMAGENET_MEAN) / IMAGENET_STD if normalize else x


class CorpusWindows(torch.utils.data.Dataset):
    """공개 코퍼스(memmap) → `(windows (K, 3, res, res), labels, label_mask, study_id)`.

    `vols`: `(N, D, H, W)` uint8 (memmap 가능), `masks`: `(N, D)`, `rows`: study → 행 번호.
    학습이면 무작위 창 `k`개와 밝기 ±10% 변형(뒤집기 없음: 좌우는 신호다).
    """

    def __init__(
        self,
        table: pd.DataFrame,
        vols,
        masks,
        rows: dict,
        res: int,
        k: int,
        train: bool,
        seed: int = 0,
    ):
        self.table = table.reset_index(drop=True)
        self.vols, self.masks, self.rows = vols, masks, rows
        self.res, self.k, self.train, self.seed = res, k, train, seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, i: int) -> dict:
        row = self.table.iloc[i]
        r = self.rows[row[ID_COL]]
        rng = np.random.default_rng([self.seed, self.epoch, i]) if self.train else None
        vol = np.asarray(self.vols[r])
        x = make_windows(
            vol, window_centers(np.asarray(self.masks[r]), self.k, rng), self.res, False
        )
        if self.train:
            x = (x * (1.0 + (rng.random() - 0.5) * 0.2)).clamp(0, 1)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        labels = np.nan_to_num(np.asarray(row["labels"], dtype=np.float32), nan=0.0)
        return {
            "image": x,
            "labels": torch.from_numpy(labels),
            "label_mask": torch.from_numpy(np.asarray(row["label_mask"], dtype=bool)),
            "study_id": row[ID_COL],
        }


def label_table(labels: pd.DataFrame) -> pd.DataFrame:
    """라벨 표(ID, 12 라벨, source) → `ID, labels, label_mask, source`."""
    y = labels[list(LABELS)].to_numpy(dtype=np.float32)
    return pd.DataFrame(
        {
            ID_COL: labels[ID_COL].astype(str).to_numpy(),
            "labels": list(y),
            "label_mask": list(~np.isnan(y)),
            "source": labels["source"].to_numpy(),
        }
    )
