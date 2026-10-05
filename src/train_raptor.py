"""exp007 학습: Raptor 방식 (공개 코퍼스 → 2.5D 창 → CoAtNet 전체 미세조정 → 소견별 attention-MIL).

공개 Raptor 학습 노트북의 절차를 우리 코드로 다시 작성했다.
- 학습: 정답 58 study를 뺀 라벨 있는 study 전부 (fold 없이 모델 하나). 정답 58은 매 epoch 평가만.
- 손실: 결측을 뺀 BCE + 양성 가중치 `(1-p)/p` (p: 학습 라벨 평균, 0.03–0.7로 자르고 가중치 1–10).
- 최적화: AdamW(백본 `backbone_lr`, 나머지 `lr`), OneCycle(`pct_start`), bf16, grad clip.
- 모델 선택: 정답 58로 고르면 그 58의 점수가 부풀므로
  **마지막 `swa_last`개 epoch 가중치 평균**을 쓴다.
- 보조 CV(선택): `holdout_fold`면 그 fold의 pseudo study를 학습에서 빼고 평가한다.

출력(`out_dir`): `oof.csv`(정답 58 예측), `metrics.json`,
`model.safetensors`(`fold0.` 접두사 — 추론의 `load_ensemble`과 호환).
"""

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from src.constants import ID_COL, LABELS
from src.data.raptor_stack import CorpusWindows, label_table
from src.metrics import macro_auc
from src.models import build_model

log = logging.getLogger("train")


def open_corpus(root: Path) -> tuple[list, np.ndarray, dict]:
    """공개 코퍼스 두 부분(`all_*`, `extra_*`)을 하나로 → (vols 행 접근 객체, masks, id → 행)."""
    root = Path(root)
    parts = [p for p in ("all", "extra") if (root / f"{p}_vols.npy").is_file()]
    if not parts:
        raise FileNotFoundError(f"코퍼스가 없다: {root}/all_vols.npy")
    vols = [np.load(root / f"{p}_vols.npy", mmap_mode="r") for p in parts]
    masks = np.concatenate([np.load(root / f"{p}_masks.npy") for p in parts])
    ids = np.concatenate(
        [np.load(root / f"{p}_ids.npy", allow_pickle=True).astype(str) for p in parts]
    )
    offsets = np.cumsum([0] + [len(v) for v in vols])

    class _Rows:
        def __len__(self) -> int:
            return int(offsets[-1])

        def __getitem__(self, r: int):
            part = int(np.searchsorted(offsets, r, side="right") - 1)
            return vols[part][r - offsets[part]]

    return _Rows(), masks, {u: i for i, u in enumerate(ids)}


def _pos_weight(train_t: pd.DataFrame) -> torch.Tensor:
    y = np.stack(train_t["labels"])
    m = np.stack(train_t["label_mask"])
    prev = np.clip(np.where(m, y, 0).sum(0) / np.maximum(m.sum(0), 1), 0.03, 0.7)
    return torch.tensor(np.clip((1 - prev) / prev, 1, 10), dtype=torch.float32)


def weighted_masked_bce(logits, labels, mask, pos_weight) -> torch.Tensor:
    per = F.binary_cross_entropy_with_logits(
        logits, labels.to(logits.dtype), pos_weight=pos_weight, reduction="none"
    )
    m = mask.to(per.dtype)
    return (per * m).sum() / m.sum().clamp(min=1.0)


def _predict(model, loader, device: str, amp: bool) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=amp):
        for batch in loader:
            out.append(torch.sigmoid(model(batch["image"].to(device)).float()).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(LABELS)), np.float32)


def run_raptor(
    config: dict,
    data_dir: Path,
    out_dir: Path,
    device: str = "cpu",
    heartbeat: Callable[[], None] | None = None,
    checkpoint_dir: Path | None = None,
) -> dict:
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    heartbeat = heartbeat or (lambda: None)
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    started = time.time()
    amp = config.get("amp", True) and device == "cuda"

    vols, masks, rows = open_corpus(data_dir / config.get("corpus_dir", "raptor_corpus"))
    labels = pd.read_csv(data_dir / config.get("labels_file", "labels.csv"), dtype={ID_COL: str})
    table = label_table(labels)
    table = table[table[ID_COL].isin(rows)].reset_index(drop=True)
    has_label = np.stack(table["label_mask"]).any(axis=1)
    gold = table[(table["source"] == "gt").to_numpy()].reset_index(drop=True)
    pool = table[(table["source"] != "gt").to_numpy() & has_label]
    holdout = pool.iloc[0:0]
    if config.get("holdout_fold") is not None:
        folds = pd.read_csv(data_dir / "folds.csv", dtype={ID_COL: str}).set_index(ID_COL)["fold"]
        held = pool[ID_COL].map(folds) == config["holdout_fold"]
        holdout, pool = pool[held.to_numpy()], pool[~held.to_numpy()]
    log.info(
        "corpus studies %d | train %d | gold %d | holdout %d | device %s",
        len(rows),
        len(pool),
        len(gold),
        len(holdout),
        device,
    )

    def loader(t: pd.DataFrame, train: bool) -> DataLoader:
        k = config["k"] if train else config["k_eval"]
        ds = CorpusWindows(t, vols, masks, rows, config["res"], k, train, config["seed"])
        bs = config["batch_size"] if train else max(1, config["batch_size"] // 2)
        return DataLoader(
            ds,
            batch_size=bs,
            shuffle=train,
            num_workers=config["num_workers"],
            drop_last=train and len(t) >= bs,
        )

    model = build_model(config["model"]).to(device)
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad]
    head = [
        p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad
    ]
    opt = torch.optim.AdamW(
        [{"params": enc, "lr": config["backbone_lr"]}, {"params": head, "lr": config["lr"]}],
        weight_decay=config["weight_decay"],
    )
    train_loader = loader(pool, train=True)
    steps = max(1, len(train_loader) * config["epochs"])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=[config["backbone_lr"], config["lr"]],
        total_steps=steps,
        pct_start=config.get("pct_start", 0.15),
    )
    pw = _pos_weight(pool).to(device)
    gold_loader = loader(gold, train=False)
    y_gold = np.stack(gold["labels"])
    swa_last = max(1, config.get("swa_last", 3))
    kept: list[dict] = []
    history = []

    for epoch in range(config["epochs"]):
        train_loader.dataset.epoch = epoch
        model.train()
        losses = []
        for batch in train_loader:
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=amp):
                logits = model(batch["image"].to(device))
            loss = weighted_masked_bce(
                logits.float(), batch["labels"].to(device), batch["label_mask"].to(device), pw
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("grad_clip", 3.0))
            opt.step()
            sched.step()
            losses.append(loss.item())
            heartbeat()
        gold_auc, _ = macro_auc(y_gold, _predict(model, gold_loader, device, amp))
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "gold_auc": gold_auc})
        log.info(
            "epoch %d/%d loss %.4f gold macro AUC %.4f (참고용, 선택에 쓰지 않음)",
            epoch + 1,
            config["epochs"],
            np.mean(losses),
            gold_auc,
        )
        if epoch >= config["epochs"] - swa_last:
            kept.append(
                {k: v.detach().float().cpu().clone() for k, v in model.state_dict().items()}
            )
        if checkpoint_dir is not None:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            state = {
                f"fold0.{k}": v.detach().cpu().contiguous() for k, v in model.state_dict().items()
            }
            save_file(state, str(Path(checkpoint_dir) / "last.safetensors"))

    ref = model.state_dict()
    avg = {k: (sum(sd[k] for sd in kept) / len(kept)).to(ref[k].dtype) for k in ref}
    model.load_state_dict(avg)
    p_gold = _predict(model, gold_loader, device, amp)
    macro, per_label = macro_auc(y_gold, p_gold)
    oof = pd.DataFrame(p_gold, columns=list(LABELS))
    oof.insert(0, "fold", 0)
    oof.insert(0, ID_COL, gold[ID_COL].to_numpy())
    oof.to_csv(out_dir / "oof.csv", index=False)
    result = {
        "macro_auc": macro,
        "per_label_auc": per_label,
        "history": history,
        "n_train_studies": len(pool),
        "n_studies": len(gold),
        "swa_epochs": len(kept),
    }
    if len(holdout):
        p_hold = _predict(model, loader(holdout, train=False), device, amp)
        y_hold = np.stack(holdout["labels"])
        m_hold = np.stack(holdout["label_mask"])
        y_bin = np.where(m_hold, (y_hold >= 0.5).astype(float), np.nan)
        result["pseudo_holdout_auc"], result["pseudo_holdout_per_label"] = macro_auc(y_bin, p_hold)
        hold = pd.DataFrame(p_hold, columns=list(LABELS))
        hold.insert(0, ID_COL, holdout[ID_COL].to_numpy())
        hold.to_csv(out_dir / "oof_pseudo.csv", index=False)
    weights = {f"fold0.{k}": v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(weights, str(out_dir / "model.safetensors"))
    result.update(elapsed_sec=round(time.time() - started, 1), device=device, config=config)
    (out_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, default=float), encoding="utf-8"
    )
    log.info("SWA(마지막 %d epoch) gold macro AUC %.4f", len(kept), macro)
    return result


def make_synthetic_corpus(
    data_dir: Path, n_gold: int = 6, n_pseudo: int = 10, depth: int = 12, size: int = 32
) -> None:
    """테스트·smoke용 작은 코퍼스 + 라벨 + folds."""
    rng = np.random.default_rng(0)
    data_dir = Path(data_dir)
    root = data_dir / "raptor_corpus"
    root.mkdir(parents=True, exist_ok=True)
    ids = [f"gold{i}" for i in range(n_gold)] + [f"ps{i}" for i in range(n_pseudo)]
    half = len(ids) // 2
    for part, sl in (("all", slice(0, half)), ("extra", slice(half, None))):
        part_ids = ids[sl]
        np.save(
            root / f"{part}_vols.npy",
            rng.integers(0, 256, (len(part_ids), depth, size, size), dtype=np.uint8),
        )
        m = np.ones((len(part_ids), depth), np.uint8)
        m[:, -2:] = 0
        np.save(root / f"{part}_masks.npy", m)
        np.save(root / f"{part}_ids.npy", np.array(part_ids, dtype=object))
    y = rng.random((len(ids), len(LABELS)))
    y[:n_gold] = (y[:n_gold] > 0.5).astype(float)
    y[0, :], y[1, :] = 0.0, 1.0
    labels = pd.DataFrame(y, columns=list(LABELS))
    labels.insert(0, ID_COL, ids)
    labels["source"] = ["gt"] * n_gold + ["pseudo"] * n_pseudo
    labels.to_csv(data_dir / "labels.csv", index=False)
    pd.DataFrame({ID_COL: ids, "fold": [i % 2 for i in range(len(ids))]}).to_csv(
        data_dir / "folds.csv", index=False
    )
