from torch import nn


def build_model(cfg: dict) -> nn.Module:
    """config `model` 절 → 모델. `name`: "baseline"(기본, exp001–003) | "dino_attn"(exp004~)."""
    kwargs = {k: v for k, v in cfg.items() if k != "name"}
    name = cfg.get("name", "baseline")
    if name == "baseline":
        from src.models.baseline import KneeBaseline

        return KneeBaseline(**kwargs)
    if name == "dino_attn":
        from src.models.dino import KneeDinoAttn

        return KneeDinoAttn(**kwargs)
    raise ValueError(f"unknown model name: {name}")
