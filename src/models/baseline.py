"""exp001 베이스라인: 공유 2D 백본 → 슬라이스 → 칸(시리즈) → study 집계."""

import timm
import torch
from torch import nn

from src.constants import LABELS
from src.data.study_table import SLOTS


class KneeBaseline(nn.Module):
    """입력 `image (B, S, D, H, W)`, `slot_mask (B, S)` → `logits (B, 12)`.

    1. 있는 칸의 슬라이스만 2D 백본(흑백 1채널)에 넣어 슬라이스 특징을 뽑는다.
    2. 칸마다 슬라이스 축으로 평균·최대 풀링 → 투영 + 칸 임베딩(어느 방향·시퀀스인지).
    3. 있는 칸만 평균해 study 특징을 만들고 12개 라벨 로짓을 낸다.
    없는 칸은 백본에 넣지 않으므로 그 칸의 픽셀 값은 출력에 영향을 주지 않는다.
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        embed_dim: int = 256,
        dropout: float = 0.2,
        n_slots: int = len(SLOTS),
        n_labels: int = len(LABELS),
    ) -> None:
        super().__init__()
        self.encoder = timm.create_model(backbone, pretrained=pretrained, in_chans=1, num_classes=0)
        n_feat = self.encoder.num_features
        self.proj = nn.Sequential(nn.Linear(2 * n_feat, embed_dim), nn.GELU())
        self.slot_embed = nn.Parameter(torch.zeros(n_slots, embed_dim))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, n_labels))
        self.embed_dim = embed_dim

    def forward(self, image: torch.Tensor, slot_mask: torch.Tensor) -> torch.Tensor:
        b, s, d, h, w = image.shape
        present = image[slot_mask]  # (P, D, H, W) — 있는 칸만
        slot_idx = slot_mask.nonzero()[:, 1]

        feats = self.encoder(present.reshape(-1, 1, h, w)).view(present.shape[0], d, -1)
        pooled = torch.cat([feats.mean(dim=1), feats.amax(dim=1)], dim=-1)
        emb = self.proj(pooled) + self.slot_embed[slot_idx]

        slots = emb.new_zeros(b, s, self.embed_dim)
        slots[slot_mask] = emb
        m = slot_mask.unsqueeze(-1).to(slots.dtype)
        study = (slots * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return self.head(study)
