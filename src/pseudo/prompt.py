"""report → 12개 라벨 추출 프롬프트 (언어 무관: report 원문을 번역 없이 넣는다).

라벨 정의는 일반적인 무릎 MRI 판독 기준으로 적었다.
대회 공식 정의는 Kaggle Data/Discussion 탭 확인 필요.
"""

from src.constants import LABELS

LABEL_DEFINITIONS: dict[str, str] = {
    "ACL": "anterior cruciate ligament tear (partial or complete), including graft tear",
    "MCL": "medial collateral ligament tear or sprain (any grade)",
    "Medial Meniscus": "tear of the medial meniscus (any type, including degenerative tear)",
    "Lateral Meniscus": "tear of the lateral meniscus (any type, including degenerative tear)",
    "Medial OA": "osteoarthritis / chondral degeneration of the medial tibiofemoral compartment",
    "Lateral OA": "osteoarthritis / chondral degeneration of the lateral tibiofemoral compartment",
    "PF OA": "patellofemoral osteoarthritis / chondral degeneration (patella or trochlea)",
    "Effusion": "joint effusion (more than a physiologic amount of fluid)",
    "Synovitis": "synovitis or synovial thickening",
    "Baker's": "Baker's (popliteal) cyst",
    "Contusion": "bone contusion / bone bruise / bone marrow edema",
    "Fracture": "fracture of any knee bone (including osteochondral or insufficiency fracture)",
}

STATES = ("positive", "negative", "uncertain", "not_mentioned")

SYSTEM_PROMPT = (
    "You are a musculoskeletal radiologist. Read a knee MRI report (it may be in any language) "
    "and decide, for each finding below, whether the report states it.\n\n"
    "Findings:\n"
    + "\n".join(f'- "{name}": {desc}' for name, desc in LABEL_DEFINITIONS.items())
    + "\n\nFor each finding answer exactly one of:\n"
    '- "positive": the report says it is present\n'
    '- "negative": the report explicitly says it is absent or normal\n'
    '- "uncertain": the report says possible / suspected / cannot exclude\n'
    '- "not_mentioned": the report does not address it\n\n'
    "Reply with a single JSON object whose keys are exactly the finding names above, "
    "and nothing else."
)


def build_messages(report: str, max_chars: int = 6000) -> list[dict]:
    """채팅 형식 메시지. 너무 긴 report는 앞부분만 쓴다 (소견·결론이 대개 앞에 있다)."""
    text = str(report)[:max_chars]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Return the JSON for this report.\n\nREPORT:\n{text}"},
    ]


assert set(LABEL_DEFINITIONS) == set(LABELS)
