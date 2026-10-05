from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # torch는 모델을 만들 때만 import한다 (CI의 기본 환경에는 torch가 없다)
    from torch import nn


def build_model(cfg: dict) -> nn.Module:
    """config `model` 절 → 모델.

    `name`: "baseline"(기본, exp001–003) | "dino_attn"(exp004~) | "slot_dino"(exp006~)
    | "raptor"(exp007~).
    """
    kwargs = {k: v for k, v in cfg.items() if k != "name"}
    name = cfg.get("name", "baseline")
    if name == "baseline":
        from src.models.baseline import KneeBaseline

        return KneeBaseline(**kwargs)
    if name == "dino_attn":
        from src.models.dino import KneeDinoAttn

        return KneeDinoAttn(**kwargs)
    if name == "slot_dino":
        from src.models.slot_dino import KneeSlotDino

        return KneeSlotDino(**kwargs)
    if name == "raptor":
        from src.models.raptor import KneeRaptor

        return KneeRaptor(**kwargs)
    raise ValueError(f"unknown model name: {name}")
