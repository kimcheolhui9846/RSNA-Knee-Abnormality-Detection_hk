"""exp004: DINOv2 ViT 백본 + 라벨별 attention 집계 (2.5D 입력).

입력 `image (B, S, D, C, H, W)` (C=3: 앞·현재·뒤 슬라이스), `slot_mask (B, S)`
→ `logits (B, 12)`.
1. 있는 칸의 슬라이스만 ViT에 넣어 슬라이스마다 CLS 특징을 뽑는다.
2. 칸 임베딩(어느 방향·시퀀스) + 깊이 위치 임베딩을 더해 study의 토큰 열을 만든다
   (없는 칸은 마스크).
3. 12개 소견마다 학습한 질의(query)가 토큰 전체에 attention → 소견별 문맥 벡터 → 소견별 출력.
   소견마다 보는 칸·슬라이스가 다르다 (예: MCL은 관상면, PF OA는 축상면).
4. (exp005~, `slot_pool`) 칸마다 깊이 방향 평균·최대 → MLP → 있는 칸 평균한 study 벡터를
   소견별 문맥 벡터에 더한다. exp004에서 attention이 균등 평균으로 퇴화해 국소 소견이
   희석됐기 때문에, exp002처럼 최대 풀링 경로를 함께 둔다.
"""

import timm
import torch
import torch.nn.functional as F
from torch import nn

from src.constants import LABELS
from src.data.study_table import SLOTS

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class KneeDinoAttn(nn.Module):
    def __init__(
        self,
        backbone: str = "vit_small_patch14_dinov2.lvd142m",
        pretrained: bool = True,
        img_size: int = 224,
        freeze_blocks: int = 0,
        n_heads: int = 6,
        dropout: float = 0.1,
        max_depth: int = 64,
        n_slots: int = len(SLOTS),
        n_labels: int = len(LABELS),
        query_init_std: float = 0.02,
        slot_pool: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, img_size=img_size
        )
        self.img_size = img_size
        self._freeze(freeze_blocks)
        dim = self.encoder.num_features
        self.slot_embed = nn.Parameter(torch.zeros(n_slots, dim))
        self.depth_embed = nn.Parameter(torch.zeros(max_depth, dim))
        self.norm = nn.LayerNorm(dim)
        # exp004의 0.02는 q·k가 너무 작아 attention이 균등 평균으로 머물렀다 → exp005는 1.0
        self.queries = nn.Parameter(torch.randn(n_labels, dim) * query_init_std)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out_w = nn.Parameter(torch.randn(n_labels, dim) * 0.02)
        self.out_b = nn.Parameter(torch.zeros(n_labels))
        self.slot_pool = (
            nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.LayerNorm(dim))
            if slot_pool
            else None
        )
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def _freeze(self, n_blocks: int) -> None:
        """patch embedding·위치 임베딩과 앞쪽 n개 블록을 고정한다.

        메모리·시간을 아끼고 사전학습 특징을 보존한다."""
        if n_blocks <= 0:
            return
        enc = self.encoder
        for p in enc.patch_embed.parameters():
            p.requires_grad = False
        for name in ("cls_token", "pos_embed", "reg_token"):
            t = getattr(enc, name, None)
            if isinstance(t, nn.Parameter):
                t.requires_grad = False
        for block in enc.blocks[:n_blocks]:
            for p in block.parameters():
                p.requires_grad = False

    def forward(self, image: torch.Tensor, slot_mask: torch.Tensor) -> torch.Tensor:
        if image.dim() == 5:  # (B, S, D, H, W) 흑백 → 3채널 복제
            image = image.unsqueeze(3).expand(-1, -1, -1, 3, -1, -1)
        b, s, d, c, h, w = image.shape
        present = image[slot_mask]  # (P, D, C, H, W)
        x = present.reshape(-1, c, h, w)
        if h != self.img_size or w != self.img_size:
            x = F.interpolate(
                x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False
            )
        x = (x - self.mean) / self.std
        feats = self.encoder(x).view(present.shape[0], d, self.encoder.num_features)  # (P, D, F)
        slot_idx = slot_mask.nonzero()[:, 1]
        feats = feats + self.slot_embed[slot_idx][:, None, :] + self.depth_embed[:d][None]

        tokens = feats.new_zeros(b, s, d, feats.shape[-1])
        tokens[slot_mask] = feats
        tokens = self.norm(tokens.view(b, s * d, -1))
        ignore = ~slot_mask.repeat_interleave(d, dim=1)  # True = attention에서 제외
        ignore[~slot_mask.any(dim=1), 0] = False  # 칸이 하나도 없는 study의 NaN 방지

        queries = self.queries.unsqueeze(0).expand(b, -1, -1)
        ctx, _ = self.attn(queries, tokens, tokens, key_padding_mask=ignore)
        if self.slot_pool is not None:
            ctx = ctx + self._pooled(feats, slot_mask)[:, None, :]
        return (self.drop(ctx) * self.out_w).sum(-1) + self.out_b

    def _pooled(self, feats: torch.Tensor, slot_mask: torch.Tensor) -> torch.Tensor:
        """칸마다 깊이 평균·최대 → MLP → 있는 칸 평균. `feats (P, D, F)` → `(B, F)`."""
        per_slot = self.slot_pool(torch.cat([feats.mean(1), feats.amax(1)], dim=-1))  # (P, F)
        b, s = slot_mask.shape
        grid = per_slot.new_zeros(b, s, per_slot.shape[-1])
        grid[slot_mask] = per_slot
        n = slot_mask.sum(1, keepdim=True).clamp(min=1).to(grid.dtype)
        return grid.sum(1) / n
