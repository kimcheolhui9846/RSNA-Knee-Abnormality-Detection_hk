"""LLM 응답(JSON) → 12개 라벨 확률."""

import json
import re

import numpy as np

from src.constants import LABELS

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_response(text: str, not_mentioned: float = 0.0, uncertain: float = np.nan) -> np.ndarray:
    """라벨 순서의 float 배열. positive=1, negative=0, uncertain/not_mentioned는 인자 값.

    JSON을 찾지 못하거나 키·값이 잘못된 라벨은 NaN(판단 보류)이다. 코드 블록·앞뒤 설명은 무시한다.
    `not_mentioned` 기본값 0은 "판독문에 없으면 정상"이라는 판독 관례를 따른 것이다
    (라벨 있는 58개로 검증한다).
    """
    out = np.full(len(LABELS), np.nan, dtype=np.float32)
    match = _JSON_OBJECT.search(text or "")
    if not match:
        return out
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return out
    if not isinstance(obj, dict):
        return out

    mapping = {
        "positive": 1.0,
        "negative": 0.0,
        "uncertain": uncertain,
        "not_mentioned": not_mentioned,
    }
    lowered = {str(k).strip().lower(): v for k, v in obj.items()}
    for i, label in enumerate(LABELS):
        value = lowered.get(label.lower())
        if isinstance(value, str) and value.strip().lower() in mapping:
            out[i] = mapping[value.strip().lower()]
    return out
