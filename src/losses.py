import torch
import torch.nn.functional as F


def masked_bce(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """라벨이 있는 위치(mask=True)만 평균한 BCE. 결측 라벨은 손실에 들어가지 않는다.

    라벨이 하나도 없으면 0을 돌려준다 (NaN이 아니고, 그래프도 끊기지 않는다).
    """
    per = F.binary_cross_entropy_with_logits(logits, labels.to(logits.dtype), reduction="none")
    m = mask.to(per.dtype)
    return (per * m).sum() / m.sum().clamp(min=1.0)
