"""exp007: Raptor 방식 — 2.5D 창 묶음 → CNN/하이브리드 백본(전체 미세조정) → 소견별 attention-MIL.

입력 `image (B, K, 3, H, W)` (ImageNet 정규화된 창 K개) → `logits (B, 12)`.
창마다 백본 특징 → LayerNorm → 소견별 attention 가중치(Linear·Tanh·Linear)로 창을 가중 합
→ 소견별 출력.
공개 Raptor 노트북의 구조를 우리 코드로 다시 작성했다.
"""

import timm
import torch
from torch import nn

from src.constants import LABELS


def build_backbone(arch: str, pretrained: bool) -> nn.Module:
    hybrid = arch.startswith(("maxvit", "maxxvit", "coatnet", "coat_", "convnext"))
    is_vit = (not hybrid) and any(k in arch for k in ("vit", "deit", "dinov2", "eva", "beit"))
    kw = {"pretrained": pretrained, "num_classes": 0, "in_chans": 3}
    kw.update(
        {"global_pool": "token", "dynamic_img_size": True} if is_vit else {"global_pool": "avg"}
    )
    return timm.create_model(arch, **kw)


class KneeRaptor(nn.Module):
    def __init__(
        self,
        backbone: str = "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",
        pretrained: bool = True,
        dropout: float = 0.2,
        grad_ckpt: bool = False,
        n_labels: int = len(LABELS),
    ) -> None:
        super().__init__()
        self.encoder = build_backbone(backbone, pretrained)
        if grad_ckpt:
            self.encoder.set_grad_checkpointing(True)
        dim = self.encoder.num_features
        self.norm = nn.LayerNorm(dim)
        self.att = nn.Sequential(
            nn.Linear(dim, 256), nn.Tanh(), nn.Dropout(dropout), nn.Linear(256, n_labels)
        )
        self.cls_w = nn.Parameter(torch.zeros(n_labels, dim))
        self.cls_b = nn.Parameter(torch.zeros(n_labels))
        nn.init.trunc_normal_(self.cls_w, std=0.02)

    def forward(self, image: torch.Tensor, slot_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, k = image.shape[:2]
        h = self.norm(self.encoder(image.flatten(0, 1)).view(b, k, -1))
        a = torch.softmax(self.att(h).float(), dim=1).to(h.dtype)  # (B, K, 12): 소견마다 창 가중치
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        return (pooled * self.cls_w).sum(-1) + self.cls_b
