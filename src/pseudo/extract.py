"""report 묶음 → pseudo-label 표.

LLM 호출은 `generate`로 주입한다 (테스트는 가짜 생성기, Pod에서는 vLLM).
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from src.constants import ID_COL, LABELS
from src.pseudo.parse import parse_response
from src.pseudo.prompt import build_messages
from src.pseudo.schema import validate_pseudo_labels

Generate = Callable[[list[list[dict]]], list[str]]


def extract_labels(
    reports: pd.DataFrame,
    generate: Generate,
    batch_size: int = 64,
    not_mentioned: float = 0.0,
    uncertain: float = np.nan,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(pseudo-label 표 `[ID_COL, *LABELS]`, 원문 응답 표 `[ID_COL, response]`).

    원문 응답을 따로 남겨 두면 `not_mentioned`/`uncertain` 해석을 바꿀 때
    LLM을 다시 돌리지 않아도 된다.
    """
    ids = reports[ID_COL].astype(str).tolist()
    texts = reports["Report"].fillna("").astype(str).tolist()
    responses: list[str] = []
    for start in range(0, len(texts), batch_size):
        batch = [build_messages(t) for t in texts[start : start + batch_size]]
        responses.extend(generate(batch))

    probs = np.stack([parse_response(r, not_mentioned, uncertain) for r in responses])
    labels = pd.DataFrame(probs, columns=list(LABELS))
    labels.insert(0, ID_COL, ids)
    validate_pseudo_labels(labels)
    return labels, pd.DataFrame({ID_COL: ids, "response": responses})


def vllm_generate(model: str, max_model_len: int = 8192, max_tokens: int = 512) -> Generate:
    """Pod(GPU) 전용. vLLM 채팅 생성 함수를 만든다 (temperature 0, 결정적)."""
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, max_model_len=max_model_len, dtype="bfloat16")
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    def generate(batch: list[list[dict]]) -> list[str]:
        return [o.outputs[0].text for o in llm.chat(batch, params, use_tqdm=False)]

    return generate
