"""study 단위 5-fold 교차검증 학습 (exp001~).

데이터 폴더 구성 (`rce upload-data`로 올리는 폴더와 같다):
    labels.csv        StudyInstanceUID + 12개 라벨 (결측은 비움) [+ source: gt / pseudo / none]
    folds.csv         StudyInstanceUID, fold, labeled   (`python -m src.folds`로 생성)
    train_index.csv   캐시 인덱스 (`python -m src.data.build_cache`)
    <cache_dir>/<study>/<series>.npy   (config `cache_dir`, 기본 "cache")

config `train_on`: "full"(기본) = 정답 study만 학습
                   "any" = 라벨이 하나라도 있는 study 전부(pseudo 포함)
평가(OOF·macro AUC)는 언제나 정답(gt) study만으로 한다.
pseudo-label은 노이즈가 있는 학습 신호일 뿐이다.

출력 (`out_dir`): model.safetensors (fold별 가중치, 키 접두사 `fold{k}.`), oof.csv, metrics.json
"""

import json
import logging
import math
import os
import random
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from src.constants import ID_COL, LABELS
from src.data.dataset import KneeStudyDataset
from src.data.study_table import SLOTS, build_study_table
from src.losses import masked_bce
from src.metrics import macro_auc
from src.models import build_model

log = logging.getLogger("train")


def check_device(device: str | None) -> str:
    """학습 장치. RunPod Pod(`RUNPOD_POD_ID`)인데 CUDA를 못 쓰면 바로 실패한다.

    2026-10-04 exp003: Community Pod에서 nvidia-smi는 GPU를 보였지만 torch가 CUDA를 못 잡아
    CPU로 4시간 돌다 시간 초과로 끝났다. 원인 진단을 위해 `torch.cuda.init()`의 오류를 함께 남긴다.
    """
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if os.environ.get("RUNPOD_POD_ID"):
        try:
            torch.cuda.init()
            reason = "torch.cuda.init() succeeded but is_available() is False"
        except Exception as e:  # noqa: BLE001 — 원인 메시지를 그대로 보고한다
            reason = repr(e)
        raise RuntimeError(f"GPU Pod인데 CUDA를 쓸 수 없다 — 학습 중단: {reason}")
    return "cpu"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate_folds(labels: pd.DataFrame, folds: pd.DataFrame, n_folds: int | None = None) -> None:
    """학습에 쓰는 study(source가 gt / pseudo)가 folds에 정확히 한 번씩, 정수 fold로 있는지 확인.

    inner merge는 빠진 study를 조용히 학습에서 빼고 중복 study를 두 번 넣으므로 여기서 먼저 막는다.
    """
    if labels[ID_COL].duplicated().any():
        raise ValueError(
            f"labels에 중복 {ID_COL}: {labels[ID_COL][labels[ID_COL].duplicated()].iloc[0]}"
        )
    dup = folds[ID_COL].duplicated()
    if dup.any():
        raise ValueError(
            f"folds에 중복 {ID_COL} {int(dup.sum())}개 (예: {folds[ID_COL][dup].iloc[0]})"
        )
    fold = pd.to_numeric(folds["fold"], errors="coerce")
    if fold.isna().any() or (fold % 1 != 0).any():
        raise ValueError("folds의 fold 값은 비어 있지 않은 정수여야 한다")
    if n_folds is not None and not fold.between(0, n_folds - 1).all():
        raise ValueError(f"fold 값이 0..{n_folds - 1} 범위를 벗어남: {sorted(fold.unique())}")
    used = labels.loc[labels["source"].isin(["gt", "pseudo"]), ID_COL]
    missing = used[~used.isin(folds[ID_COL])]
    if len(missing):
        raise ValueError(f"folds에 없는 학습 study {len(missing)}개 (예: {missing.iloc[0]})")


def load_table(
    data_dir: Path,
    target_slices: int,
    labels_file: str = "labels.csv",
    n_folds: int | None = None,
    series_meta: str | None = None,
) -> pd.DataFrame:
    """전체 study 학습 테이블 + `source`(gt / pseudo / none).

    `labels_file`로 같은 데이터셋 안의 다른 라벨 표를 고른다 (config `labels_file`).
    라벨 표에 source 컬럼이 없으면 12개 라벨이 모두 있는 study를 gt로 본다 (exp001 데이터).
    """
    labels = pd.read_csv(data_dir / labels_file, dtype={ID_COL: str})
    if "source" not in labels.columns:
        full = labels[list(LABELS)].notna().all(axis=1)
        labels["source"] = np.where(full, "gt", "none")
    folds = pd.read_csv(data_dir / "folds.csv", dtype={ID_COL: str})
    validate_folds(labels, folds, n_folds)
    index = pd.read_csv(data_dir / "train_index.csv", dtype={ID_COL: str, "SeriesInstanceUID": str})
    table = build_study_table(labels, folds, index, target_slices=target_slices)
    table = table.merge(labels[[ID_COL, "source"]], on=ID_COL, how="left")
    if (
        series_meta
    ):  # exp006~ 칸 이미지 입력: 헤더 메타(지방억제·가중·px·좌우)로 칸 6개를 다시 고른다
        from src.data.slot_image import slot_inputs

        meta = pd.read_csv(data_dir / series_meta, dtype={ID_COL: str, "SeriesInstanceUID": str})
        cols = ["SeriesInstanceUID", "fatsat", "fluid", "px", "laterality"]
        merged = index.merge(meta[cols], on="SeriesInstanceUID", how="inner")
        table = table.merge(slot_inputs(merged), on=ID_COL, how="left")
        table["slot6"] = [v if isinstance(v, list) else [None] * 6 for v in table["slot6"]]
        table["lat"] = table["lat"].where(table["lat"].isin(["L", "R"]), None)
    return table


def _loader(table: pd.DataFrame, data_dir: Path, cfg: dict, shuffle: bool) -> DataLoader:
    cache_root = data_dir / cfg.get("cache_dir", "cache")
    if cfg.get("input") == "slot_image":  # exp006~: 칸마다 이미지 1장 (학습 loader만 증강)
        from src.data.slot_image import SlotImageDataset

        ds = SlotImageDataset(table, cache_root, cfg, train=shuffle, seed=cfg["seed"])
        return DataLoader(
            ds, batch_size=cfg["batch_size"], shuffle=shuffle, num_workers=cfg["num_workers"]
        )
    ds = KneeStudyDataset(
        table,
        cache_root=cache_root,
        depth=cfg["depth"],
        size=cfg["size"],
        slab=cfg.get("slab", 1),
    )
    return DataLoader(
        ds, batch_size=cfg["batch_size"], shuffle=shuffle, num_workers=cfg["num_workers"]
    )


def _param_groups(model: torch.nn.Module, cfg: dict) -> list[dict]:
    """학습할 파라미터만 넘긴다.

    - config `backbone_lr`가 있으면 사전학습 백본(`encoder.`)은 그 lr로 따로 둔다 (헤드보다 작게).
    - config `no_decay_1d: true`면 bias·LayerNorm·임베딩·질의 같은 1차원 이하 파라미터와
      `*_embed`·`queries`에는 weight decay를 걸지 않는다 (exp005~).
    """
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    def _no_decay(name: str, p: torch.nn.Parameter) -> bool:
        return p.ndim <= 1 or name.endswith(("_embed", "queries", "_token"))

    groups = []
    split_lr = "backbone_lr" in cfg
    for is_backbone in (True, False) if split_lr else (None,):
        params = [
            (n, p)
            for n, p in trainable
            if is_backbone is None or n.startswith("encoder.") == is_backbone
        ]
        lr = {} if is_backbone is None else {"lr": cfg["backbone_lr"] if is_backbone else cfg["lr"]}
        if cfg.get("no_decay_1d"):
            decay = [p for n, p in params if not _no_decay(n, p)]
            no_decay = [p for n, p in params if _no_decay(n, p)]
            groups += [{"params": decay, **lr}, {"params": no_decay, "weight_decay": 0.0, **lr}]
        else:
            groups.append({"params": [p for _, p in params], **lr})
    return [g for g in groups if g["params"]]


def _schedule(
    opt: torch.optim.Optimizer, total: int, warmup: float
) -> torch.optim.lr_scheduler.LRScheduler:
    """cosine. config `warmup`(전체 step 대비 비율)이 있으면 그만큼 0에서 선형으로 올린다."""
    if not warmup:
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
    n_warm = max(1, int(total * warmup))

    def factor(step: int) -> float:
        if step < n_warm:
            return (step + 1) / n_warm
        progress = (step - n_warm) / max(1, total - n_warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(opt, factor)


def _init_bias_from_prior(model: torch.nn.Module, train_t: pd.DataFrame) -> None:
    """출력 bias를 학습 라벨의 (결측 제외) 평균 확률의 logit으로 둔다.

    첫 epoch을 사전확률 학습에 쓰지 않게 한다 (exp004는 epoch 1 loss가 사전확률 수준이었다)."""
    out_b = getattr(model, "out_b", None)
    if out_b is None:
        return
    y = np.stack(train_t["labels"])
    m = np.stack(train_t["label_mask"])
    prior = np.clip(np.where(m, y, 0).sum(0) / np.maximum(m.sum(0), 1), 1e-3, 1 - 1e-3)
    with torch.no_grad():
        out_b.copy_(torch.as_tensor(np.log(prior / (1 - prior)), dtype=out_b.dtype))


def _pseudo_holdout(
    pseudo_table: pd.DataFrame, oof_pseudo: np.ndarray, out_dir: Path, enabled: bool
) -> dict:
    """pseudo-label(0.5 기준 이진화, 결측 제외)에 대한 OOF macro AUC + `oof_pseudo.csv`."""
    if not enabled or len(pseudo_table) == 0:
        return {}
    soft = np.stack(pseudo_table["labels"]).astype(np.float32)
    target = np.where(np.isnan(soft), np.nan, (soft >= 0.5).astype(np.float32))
    macro, per_label = macro_auc(target, oof_pseudo)
    df = pd.DataFrame(oof_pseudo, columns=list(LABELS))
    df.insert(0, "fold", pseudo_table["fold"].to_numpy())
    df.insert(0, ID_COL, pseudo_table[ID_COL].to_numpy())
    df.to_csv(out_dir / "oof_pseudo.csv", index=False)
    log.info("pseudo-holdout macro AUC %.4f (n=%d)", macro, len(pseudo_table))
    return {
        "pseudo_holdout_auc": macro,
        "pseudo_holdout_per_label": per_label,
        "n_pseudo_holdout": len(pseudo_table),
    }


def _amp_dtype(cfg: dict) -> torch.dtype:
    return torch.bfloat16 if cfg.get("amp_dtype") == "bf16" else torch.float16


def _predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    amp: bool,
    dtype: torch.dtype = torch.float16,
) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype, enabled=amp):
        for batch in loader:
            logits = model(batch["image"].to(device), batch["slot_mask"].to(device))
            out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out)


def train_fold(
    fold: int,
    train_t: pd.DataFrame,
    data_dir: Path,
    cfg: dict,
    device: str,
    heartbeat: Callable[[], None],
) -> torch.nn.Module:
    amp = cfg["amp"] and device == "cuda"
    # bf16은 지수 범위가 fp32와 같아 loss scaling이 필요 없다 (exp005~). 기본은 fp16 + GradScaler
    amp_dtype = _amp_dtype(cfg)
    model = build_model(cfg["model"]).to(device)
    if cfg.get("init_bias_prior"):
        _init_bias_from_prior(model, train_t)
    loader = _loader(train_t, data_dir, cfg, shuffle=True)
    opt = torch.optim.AdamW(
        _param_groups(model, cfg), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    sched = _schedule(opt, cfg["epochs"] * len(loader), cfg.get("warmup", 0.0))
    scaler = torch.amp.GradScaler(enabled=amp and amp_dtype == torch.float16)
    grad_clip = cfg.get("grad_clip")

    for epoch in range(cfg["epochs"]):
        if hasattr(loader.dataset, "epoch"):
            loader.dataset.epoch = epoch  # 증강 난수를 epoch마다 바꾼다
        model.train()
        losses = []
        for batch in loader:
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp):
                logits = model(batch["image"].to(device), batch["slot_mask"].to(device))
            loss = masked_bce(
                logits.float(), batch["labels"].to(device), batch["label_mask"].to(device)
            )
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
            heartbeat()
        log.info("fold %d epoch %d/%d loss %.4f", fold, epoch + 1, cfg["epochs"], np.mean(losses))
    return model


def run(
    config: dict,
    data_dir: Path,
    out_dir: Path,
    device: str | None = None,
    heartbeat: Callable[[], None] | None = None,
    checkpoint_dir: Path | None = None,
) -> dict:
    """fold마다 학습 → 검증 fold 예측(OOF) → 전체 OOF로 macro AUC.

    `checkpoint_dir`가 있으면 fold가 끝날 때마다 그때까지의 가중치를 `folds_{k:02d}.safetensors`로
    저장한다. 하네스는 실패·시간 초과 때 가장 최근 체크포인트를 HF에 올린다.
    """
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = check_device(device)
    heartbeat = heartbeat or (lambda: None)
    seed_everything(config["seed"])
    started = time.time()

    full_table = load_table(
        data_dir,
        config["target_slices"],
        config.get("labels_file", "labels.csv"),
        n_folds=config["n_folds"],
        series_meta=config.get("series_meta"),
    )
    is_gt = (full_table["source"] == "gt").to_numpy()
    has_label = np.stack(full_table["label_mask"]).any(axis=1)
    train_pool = has_label if config.get("train_on", "full") == "any" else is_gt
    table = full_table[is_gt].reset_index(drop=True)  # 평가 대상
    log.info(
        "eval (gt) studies: %d, train pool: %d (train_on=%s), device: %s",
        len(table),
        int(train_pool.sum()),
        config.get("train_on", "full"),
        device,
    )

    oof = np.full((len(table), len(LABELS)), np.nan, dtype=np.float32)
    # 보조 CV: fold k 학습에 쓰지 않은 pseudo study (정답 58개보다 75배 많아 분산이 작다)
    eval_pseudo = bool(config.get("eval_pseudo", False))
    pseudo_table = full_table[(full_table["source"] == "pseudo").to_numpy() & has_label]
    pseudo_table = pseudo_table.reset_index(drop=True)
    oof_pseudo = np.full((len(pseudo_table), len(LABELS)), np.nan, dtype=np.float32)
    weights: dict[str, torch.Tensor] = {}
    fold_scores: list[float] = []
    amp = config["amp"] and device == "cuda"
    for k in range(config["n_folds"]):
        train_t = full_table[train_pool & (full_table["fold"] != k).to_numpy()]
        model = train_fold(k, train_t, data_dir, config, device, heartbeat)
        is_val = (table["fold"] == k).to_numpy()
        val_t = table[is_val]
        oof[is_val] = _predict(
            model, _loader(val_t, data_dir, config, shuffle=False), device, amp, _amp_dtype(config)
        )
        if eval_pseudo:
            p_val = (pseudo_table["fold"] == k).to_numpy()
            if p_val.any():
                loader = _loader(pseudo_table[p_val], data_dir, config, shuffle=False)
                oof_pseudo[p_val] = _predict(model, loader, device, amp, _amp_dtype(config))
        fold_macro, _ = macro_auc(np.stack(val_t["labels"]), oof[is_val])
        fold_scores.append(fold_macro)
        log.info(
            "fold %d val macro AUC %.4f (val gt n=%d, train n=%d)",
            k,
            fold_macro,
            int(is_val.sum()),
            len(train_t),
        )
        weights.update(
            {f"fold{k}.{n}": t.detach().cpu().contiguous() for n, t in model.state_dict().items()}
        )
        if checkpoint_dir is not None:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            save_file(weights, str(Path(checkpoint_dir) / f"folds_{k:02d}.safetensors"))

    macro, per_label = macro_auc(np.stack(table["labels"]), oof)
    oof_df = pd.DataFrame(oof, columns=list(LABELS))
    oof_df.insert(0, "fold", table["fold"].to_numpy())
    oof_df.insert(0, ID_COL, table[ID_COL].to_numpy())
    oof_df.to_csv(out_dir / "oof.csv", index=False)
    save_file(weights, str(out_dir / "model.safetensors"))

    result = {
        "macro_auc": macro,
        "per_label_auc": per_label,
        "fold_macro_auc": fold_scores,
        "fold_std": float(np.nanstd(fold_scores)),
        "n_studies": len(table),
        "n_train_studies": int(train_pool.sum()),
        "n_slots": len(SLOTS),
        **_pseudo_holdout(pseudo_table, oof_pseudo, out_dir, eval_pseudo),
        "elapsed_sec": round(time.time() - started, 1),
        "device": device,
        "config": config,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, default=float), encoding="utf-8"
    )
    log.info("OOF macro AUC %.4f, fold std %.4f", macro, result["fold_std"])
    return result


def make_synthetic_data(
    data_dir: Path,
    n_studies: int = 12,
    depth: int = 6,
    size: int = 32,
    n_folds: int = 2,
    n_pseudo: int = 0,
    cache_subdir: str = "cache",
) -> None:
    """실데이터 없이 파이프라인을 끝까지 돌려 보기 위한 합성 데이터 (테스트·smoke-test용).

    `n_studies`개는 정답(gt), `n_pseudo`개는 일부 라벨이 NaN인 pseudo-label study다.
    """
    rng = np.random.default_rng(0)
    data_dir = Path(data_dir)
    gt_ids = [f"study{i:03d}" for i in range(n_studies)]
    pseudo_ids = [f"pseudo{i:03d}" for i in range(n_pseudo)]
    ids = gt_ids + pseudo_ids
    y = rng.integers(0, 2, size=(len(ids), len(LABELS))).astype(float)
    y[0], y[1] = 0.0, 1.0
    y[n_studies:][rng.random((n_pseudo, len(LABELS))) < 0.2] = np.nan  # LLM 판단 보류
    labels = pd.DataFrame(y, columns=list(LABELS))
    labels.insert(0, ID_COL, ids)
    labels["source"] = ["gt"] * n_studies + ["pseudo"] * n_pseudo
    data_dir.mkdir(parents=True, exist_ok=True)
    labels.to_csv(data_dir / "labels.csv", index=False)
    pd.DataFrame(
        {ID_COL: ids, "fold": [i % n_folds for i in range(len(ids))], "labeled": True}
    ).to_csv(data_dir / "folds.csv", index=False)

    cache = data_dir / cache_subdir
    rows = []
    for i, sid in enumerate(ids):
        for j, (plane, fs) in enumerate(SLOTS):
            if j == len(SLOTS) - 1 and i % 2:  # 일부 study는 마지막 칸이 없다
                continue
            rel = f"{sid}/s{j}.npy"
            (cache / sid).mkdir(parents=True, exist_ok=True)
            np.save(cache / rel, rng.integers(0, 256, (depth, size, size), dtype=np.uint8))
            rows.append((sid, f"s{j}", plane, fs, depth, "built", rel, 2 * size, 2 * size))
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
            "orig_h",
            "orig_w",
        ],
    )
    index.to_csv(data_dir / "train_index.csv", index=False)
    # 헤더 메타 (exp006 칸 이미지 입력용): 지방억제 칸은 fluid, 아니면 시상면만 fluid로
    meta = index[[ID_COL, "SeriesInstanceUID"]].copy()
    meta["fatsat"] = index["Fat_Suppression"] == 1
    meta["fluid"] = meta["fatsat"] | (index["Anatomical_Plane"] == "Sagittal")
    meta["px"] = 0.5
    meta["laterality"] = ["R" if int(s[-1]) % 2 else "L" for s in meta[ID_COL]]
    meta.to_csv(data_dir / "train_series_meta.csv", index=False)
