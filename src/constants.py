"""대회 사양 상수. LABELS 순서가 곧 제출 컬럼 순서다.

sample_submission.csv 헤더와의 일치는 tests/test_submission.py가 검사한다.
"""

ID_COL = "StudyInstanceUID"

LABELS: tuple[str, ...] = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)
