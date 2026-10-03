import json
import os

import numpy as np
import pandas as pd
import pytest

from src.constants import ID_COL, LABELS
from src.pseudo.extract import extract_labels
from src.pseudo.parse import parse_response
from src.pseudo.prompt import LABEL_DEFINITIONS, build_messages
from src.pseudo.schema import validate_pseudo_labels

# --- prompt -------------------------------------------------------------------


def test_every_label_has_a_definition() -> None:
    assert set(LABEL_DEFINITIONS) == set(LABELS)


def test_messages_contain_report_and_all_label_keys() -> None:
    msgs = build_messages("Derrame articular moderado.")
    assert [m["role"] for m in msgs] == ["system", "user"]
    text = msgs[0]["content"] + msgs[1]["content"]
    assert "Derrame articular moderado." in msgs[1]["content"]
    assert all(label in text for label in LABELS)


def test_very_long_report_is_truncated() -> None:
    msgs = build_messages("x" * 50_000, max_chars=1000)
    assert len(msgs[1]["content"]) < 2000


# --- parse --------------------------------------------------------------------


def _answer(**overrides: str) -> str:
    states = {label: "not_mentioned" for label in LABELS}
    states.update(overrides)
    return json.dumps(states)


def test_states_map_to_probabilities() -> None:
    text = _answer(ACL="positive", MCL="negative", Effusion="uncertain")
    p = dict(zip(LABELS, parse_response(text), strict=True))
    assert p["ACL"] == 1.0
    assert p["MCL"] == 0.0
    assert np.isnan(p["Effusion"])  # 불확실은 판단 보류(NaN)
    assert p["Fracture"] == 0.0  # 언급 없음은 기본적으로 음성


def test_not_mentioned_mapping_is_configurable() -> None:
    p = parse_response(_answer(), not_mentioned=np.nan)
    assert np.isnan(p).all()


def test_json_inside_code_fence_and_extra_text_is_parsed() -> None:
    text = "Sure, here it is:\n```json\n" + _answer(ACL="positive") + "\n```\nDone."
    assert parse_response(text)[LABELS.index("ACL")] == 1.0


def test_invalid_or_missing_keys_become_nan() -> None:
    assert np.isnan(parse_response("I cannot answer.")).all()
    partial = json.dumps({"ACL": "positive", "MCL": "banana"})
    p = parse_response(partial)
    assert p[LABELS.index("ACL")] == 1.0
    assert np.isnan(p[LABELS.index("MCL")])  # 알 수 없는 값
    assert np.isnan(p[LABELS.index("Fracture")])  # 키 없음


# --- extract (가짜 생성기, GPU 없음) -------------------------------------------


def test_extract_labels_returns_valid_pseudo_label_frame() -> None:
    reports = pd.DataFrame({ID_COL: ["a", "b"], "Report": ["ACL tear.", "Normal knee."]})

    def fake_generate(batch: list[list[dict]]) -> list[str]:
        return [
            _answer(ACL="positive")
            if "ACL" in msgs[1]["content"].split("REPORT")[-1]
            else _answer()
            for msgs in batch
        ]

    labels, raw = extract_labels(reports, fake_generate, batch_size=1)

    validate_pseudo_labels(labels)
    assert labels[ID_COL].tolist() == ["a", "b"]
    assert labels.set_index(ID_COL).loc["a", "ACL"] == 1.0
    assert labels.set_index(ID_COL).loc["b", "ACL"] == 0.0
    assert raw[ID_COL].tolist() == ["a", "b"] and "response" in raw.columns


def test_vllm_generate_disables_flashinfer_sampler_before_engine_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Pod 이미지의 nvcc(12.4)는 flashinfer JIT 옵션(--compress-mode)을 몰라 엔진이 죽었다
    # (run 20261003-083742-68db3c). 샘플러를 PyTorch 경로로 돌려 JIT 컴파일을 피한다.
    import sys
    import types

    seen: dict[str, str | None] = {}

    class FakeLLM:
        def __init__(self, **kwargs) -> None:
            seen["env"] = os.environ.get("VLLM_USE_FLASHINFER_SAMPLER")

    fake = types.SimpleNamespace(LLM=FakeLLM, SamplingParams=lambda **kw: kw)
    monkeypatch.setitem(sys.modules, "vllm", fake)
    monkeypatch.delenv("VLLM_USE_FLASHINFER_SAMPLER", raising=False)

    from src.pseudo.extract import vllm_generate

    vllm_generate("some/model")
    assert seen["env"] == "0"


def test_extract_rejects_missing_report_column() -> None:
    with pytest.raises(KeyError):
        extract_labels(pd.DataFrame({ID_COL: ["a"]}), lambda b: [], batch_size=1)
