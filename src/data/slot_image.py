"""칸(slot)마다 이미지 1장: 슬라이스 3장을 RGB로 (exp006~).

공개 상위 노트북의 DINO 입력 방식을 우리 캐시(256px uint8, 가운데 정렬 패딩)에서 다시 만든다.
1. 칸 6개 = (단면, fluid-sensitive, 지방억제) 조합 (`SLOTS6`). 칸마다 슬라이스가 가장 많은 시리즈.
2. 시리즈 가운데 `band`(기본 20–80%) 안에서 슬라이스 `group`(3)장을 고르게 → 채널.
3. 무릎 중심 `crop_mm`(130 mm) 정사각형으로 자른다 (PixelSpacing 기준, 촬영 시야 차이를 없앤다).
4. 자른 영역 기준 1–99 백분위로 밝기를 다시 늘리고 `out` 크기로 바꾼다.
5. 오른 무릎은 왼 무릎 방향으로 뒤집는다 (관상·축상면은 좌우, 시상면은 슬라이스 순서).

학습(`SlotImageDataset`)과 Kaggle 추론이 `slot_image()` 하나를 같이 쓴다.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.constants import ID_COL

# (이름, 단면, fluid-sensitive, 지방억제) — 순서가 모델의 칸 순서다
SLOTS6 = [
    ("SAG_FLUID_FS", "Sagittal", True, True),
    ("COR_FLUID_FS", "Coronal", True, True),
    ("AX_FLUID_FS", "Axial", True, True),
    ("SAG_FLUID_NOFS", "Sagittal", True, False),
    ("COR_T1", "Coronal", False, False),
    ("SAG_T1", "Sagittal", False, False),
]


def pick_slot_series(meta: pd.DataFrame) -> pd.DataFrame:
    """시리즈 메타(`annotate` 결과 + `n_slices`) → study × SLOTS6 표.

    값은 시리즈 행 index, 없으면 NaN.

    같은 칸 후보가 여럿이면 슬라이스가 가장 많은 것, 같으면 SeriesInstanceUID 순.
    """
    rows = []
    for k, (_, plane, fluid, fs) in enumerate(SLOTS6):
        sel = meta[
            (meta["Anatomical_Plane"] == plane)
            & (meta["fluid"].astype(bool) == fluid)
            & (meta["fatsat"].astype(bool) == fs)
        ]
        best = sel.sort_values(
            [ID_COL, "n_slices", "SeriesInstanceUID"], ascending=[True, False, True]
        )
        best = best.drop_duplicates(ID_COL)
        rows.append(pd.DataFrame({ID_COL: best[ID_COL], "slot": k, "row": best.index}))
    long = pd.concat(rows, ignore_index=True)
    wide = long.pivot(index=ID_COL, columns="slot", values="row")
    return wide.reindex(columns=range(len(SLOTS6)))


def band_indices(n: int, group: int = 3, band: tuple[float, float] = (0.2, 0.8)) -> np.ndarray:
    """슬라이스 n장 중 `band` 구간에서 `group`장 (중복 없이 고르게, 모자라면 마지막을 반복)."""
    lo, hi = int(band[0] * (n - 1)), int(band[1] * (n - 1))
    idx = np.unique(np.linspace(lo, hi, group).astype(int)) if hi > lo else np.array([n // 2])
    while len(idx) < group:
        idx = np.append(idx, idx[-1])
    return idx[:group]


def slot_image(
    volume: np.ndarray,
    px: float,
    side_px: int,
    plane: str,
    lat: str | None,
    group: int = 3,
    band: tuple[float, float] = (0.2, 0.8),
    crop_mm: float = 130.0,
    out: int = 224,
) -> np.ndarray:
    """캐시 볼륨 `(N, S, S)` uint8 → `(group, out, out)` uint8.

    `px`: 원본 PixelSpacing(mm), `side_px`: 원본 max(행, 열) — 캐시는 이 정사각형을 S로 줄인 것이다.
    """
    n, size = volume.shape[0], volume.shape[-1]
    vol = volume[band_indices(n, group, band)].astype(np.float32)
    if px and np.isfinite(px) and px > 0 and side_px:
        want = int(round(crop_mm / px * size / side_px))
        if 16 < want < size:
            c, half = size // 2, want // 2
            vol = vol[:, c - half : c + half, c - half : c + half]
    lo, hi = np.percentile(vol, [1, 99])
    vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)
    t = F.interpolate(
        torch.from_numpy(vol)[None], size=(out, out), mode="bilinear", align_corners=False
    )[0]
    img = (t * 255).round().clamp(0, 255).to(torch.uint8).numpy()
    if lat == "R":
        img = img[:, :, ::-1] if plane in ("Coronal", "Axial") else img[::-1]
    return np.ascontiguousarray(img)


def augment(
    images: torch.Tensor,
    rng: np.random.Generator,
    rot_deg: float = 8.0,
    scale: float = 0.08,
    shift: float = 0.05,
    intensity: float = 0.1,
) -> torch.Tensor:
    """`(S, C, H, W)` float [0, 1] → 칸마다 독립적인 무작위 회전·확대·이동·밝기 (학습 전용)."""
    s = images.shape[0]
    ang = np.deg2rad(rng.uniform(-rot_deg, rot_deg, s))
    sc = 1 + rng.uniform(-scale, scale, s)
    tx, ty = rng.uniform(-shift, shift, (2, s)) * 2
    theta = np.zeros((s, 2, 3), np.float32)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = np.cos(ang) / sc, -np.sin(ang) / sc, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = np.sin(ang) / sc, np.cos(ang) / sc, ty
    grid = F.affine_grid(torch.from_numpy(theta), list(images.shape), align_corners=False)
    out = F.grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    gain = torch.from_numpy(1 + rng.uniform(-intensity, intensity, s).astype(np.float32))
    return (out * gain[:, None, None, None]).clamp(0, 1)


OK_STATUS = ("built", "cached")


def slot_inputs(meta: pd.DataFrame) -> pd.DataFrame:
    """시리즈 메타(헤더 `annotate` + 캐시 인덱스의 path·n_slices·orig_h·orig_w·status, `laterality`)
    → study마다 `slot6`(칸별 `(캐시 경로, px, 원본 변, 단면)` 또는 None)와 `lat`."""
    ok = meta[meta["status"].isin(OK_STATUS)].reset_index(drop=True)
    wide = pick_slot_series(ok)
    side = ok[["orig_h", "orig_w"]].max(axis=1)
    slot6 = []
    for _, picks in wide.iterrows():
        cells = []
        for k, r in enumerate(picks):
            if pd.isna(r):
                cells.append(None)
                continue
            r = int(r)
            cells.append((ok.at[r, "path"], float(ok.at[r, "px"]), int(side[r]), SLOTS6[k][1]))
        slot6.append(cells)
    lat = ok.drop_duplicates(ID_COL).set_index(ID_COL)["laterality"]
    out = pd.DataFrame({ID_COL: wide.index, "slot6": slot6})
    out["lat"] = out[ID_COL].map(lat).where(lambda s: s.isin(["L", "R"]), None)
    return out


class SlotImageDataset(torch.utils.data.Dataset):
    """`__getitem__` → dict: image float32 `(6, 3, out, out)` 0–1, slot_mask, labels,
    label_mask, study_id.

    table: `build_study_table` 결과에 `slot_inputs`의 `slot6`·`lat`을 붙인 것.
    """

    def __init__(
        self, table: pd.DataFrame, cache_root, cfg: dict, train: bool = False, seed: int = 0
    ):
        from pathlib import Path

        self.table = table.reset_index(drop=True)
        self.cache_root = Path(cache_root)
        self.cfg = cfg
        self.train = train
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, i: int) -> dict:
        row = self.table.iloc[i]
        out = self.cfg.get("img_out", 224)
        image = np.zeros((len(SLOTS6), 3, out, out), dtype=np.float32)
        mask = np.zeros(len(SLOTS6), dtype=bool)
        cells = row["slot6"] if isinstance(row["slot6"], list) else [None] * len(SLOTS6)
        for k, cell in enumerate(cells):
            if cell is None:
                continue
            rel, px, side, plane = cell
            vol = np.load(self.cache_root / rel, mmap_mode="r")
            image[k] = (
                slot_image(
                    np.asarray(vol),
                    px,
                    side,
                    plane,
                    row["lat"],
                    band=tuple(self.cfg.get("slice_band", (0.2, 0.8))),
                    crop_mm=self.cfg.get("crop_mm", 130.0),
                    out=out,
                )
                / 255.0
            )
            mask[k] = True
        img = torch.from_numpy(image)
        if self.train:
            rng = np.random.default_rng([self.seed, self.epoch, i])
            img[mask] = augment(img[mask], rng)
        labels = np.nan_to_num(np.asarray(row["labels"], dtype=np.float32), nan=0.0)
        return {
            "image": img,
            "slot_mask": torch.from_numpy(mask),
            "labels": torch.from_numpy(labels),
            "label_mask": torch.from_numpy(np.asarray(row["label_mask"], dtype=bool)),
            "study_id": row[ID_COL],
        }
