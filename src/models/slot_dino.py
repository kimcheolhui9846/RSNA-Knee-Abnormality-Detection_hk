"""exp006: 칸 이미지 DINOv2 + 칸 attention 헤드 (공개 상위 노트북 방식을 다시 작성).

입력 `image (B, 6, 3, H, W)` 0–1, `slot_mask (B, 6)` → `logits (B, 12)`.
1. 있는 칸 이미지만 ViT에 넣고 CLS + patch 평균(+ 선택: 상위 1/8 patch 평균)을 특징으로 쓴다.
2. `SlotHead`: 특징 → LayerNorm·Linear·GELU(hidden) + 칸 임베딩 → 라벨별 질의로 칸 6개에 attention.
   `prior`이면 라벨별로 볼 만한 칸에 사전 가산점(`SLOT_PRIOR`)을 준다 (예: MCL은 관상면).
3. 백본은 마지막 `unfreeze_last`개 블록과 마지막 norm만 학습한다
   (exp005에서 큰 lr로 백본이 붕괴했다 → 작은 lr).
"""

import timm
import torch
from torch import nn

from src.constants import LABELS
from src.data.slot_image import SLOTS6

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# 라벨 → 사전 가산점을 줄 칸 번호 (SLOTS6 순서: 0 SAG_FLUID_FS, 1 COR_FLUID_FS, 2 AX_FLUID_FS,
# 3 SAG_FLUID_NOFS, 4 COR_T1, 5 SAG_T1)
SLOT_PRIOR = {
    "ACL": (0, 3, 5),
    "MCL": (1, 4),
    "Medial Meniscus": (0, 1, 3, 4),
    "Lateral Meniscus": (0, 1, 3, 4),
    "Medial OA": (1, 4, 5),
    "Lateral OA": (1, 4, 5),
    "PF OA": (0, 2, 5),
    "Effusion": (0, 2),
    "Synovitis": (0, 2),
    "Baker's": (0,),
    "Contusion": (0, 1, 2),
    "Fracture": (0, 1, 2, 4, 5),
}
POOL_PARTS = {"cls_mean": 2, "cls_mean_focal": 3}


class SlotHead(nn.Module):
    def __init__(
        self,
        dim: int,
        n_slot: int = len(SLOTS6),
        n_out: int = len(LABELS),
        hidden: int = 256,
        dropout: float = 0.2,
        prior: bool = False,
        prior_strength: float = 0.55,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, n_out)
        self.hidden = hidden
        table = torch.zeros(n_out, n_slot)
        if prior:
            for name, slots in SLOT_PRIOR.items():
                table[LABELS.index(name), list(slots)] = prior_strength
        self.register_buffer("slot_prior", table)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """`x (B, S, dim)`, `mask (B, S)` bool → `(B, n_out)`."""
        h = self.proj(x) + self.slot_emb
        att = torch.einsum("bsh,oh->bos", h, self.query) / self.hidden**0.5 + self.slot_prior
        att = att.float().masked_fill(~mask[:, None, :], -1e4).softmax(-1).to(h.dtype)
        ctx = self.drop(torch.einsum("bos,bsh->boh", att, h))
        return (ctx * self.out.weight[None]).sum(-1) + self.out.bias


class KneeSlotDino(nn.Module):
    def __init__(
        self,
        backbone: str = "vit_small_patch14_dinov2.lvd142m",
        pretrained: bool = True,
        img_size: int = 224,
        unfreeze_last: int = 6,
        pool: str = "cls_mean",
        prior: bool = True,
        hidden: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, img_size=img_size
        )
        self.img_size = img_size
        for p in self.encoder.parameters():
            p.requires_grad = False
        blocks = self.encoder.blocks
        for block in blocks[max(0, len(blocks) - unfreeze_last) :]:
            for p in block.parameters():
                p.requires_grad = True
        for p in self.encoder.norm.parameters():
            p.requires_grad = True
        self.pool = pool
        dim = self.encoder.num_features * POOL_PARTS[pool]
        self.head = SlotHead(dim, hidden=hidden, dropout=dropout, prior=prior)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """`(N, 3, H, W)` 0–1 → `(N, dim)`."""
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = torch.nn.functional.interpolate(
                x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False
            )
        tokens = self.encoder.forward_features((x - self.mean) / self.std)
        patch = tokens[:, self.encoder.num_prefix_tokens :]
        parts = [tokens[:, 0], patch.mean(1)]
        if self.pool == "cls_mean_focal":
            k = max(1, patch.shape[1] // 8)
            parts.append(patch.topk(k, dim=1).values.mean(1))
        return torch.cat(parts, dim=1)

    def forward(self, image: torch.Tensor, slot_mask: torch.Tensor) -> torch.Tensor:
        b, s = slot_mask.shape
        feats = self.features(image[slot_mask])
        grid = feats.new_zeros(b, s, feats.shape[-1])
        grid[slot_mask] = feats
        mask = slot_mask.clone()
        mask[~slot_mask.any(1), 0] = True  # 칸이 하나도 없는 study의 NaN 방지 (특징 0으로 예측)
        return self.head(grid, mask)
