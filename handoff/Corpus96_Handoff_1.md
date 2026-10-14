# HANDOFF — 96장 코퍼스 (공개 nartaa Raptor 형식)

작성: kim cheol hui | 날짜: 2026-10-14 | 브랜치: `feat/corpus96` (base: `exp/010-raptor64`)

## 1. 요약 (TL;DR)
- 공개 nartaa Raptor(단일 LB 0.949)는 96장 스택(시상 fluid 26, 시상 22, 관상 fluid 18, 관상 12, 축상 18), 구간 2–98%, 원본 384 px로 학습됐다.
- `SLOTS96`을 추가해 `build_stack`/`build_corpus`로 같은 코퍼스를 만든다. train 6건에서 nartaa 추론 코드의 `build_study`와 **비트 단위로 같다**.
- 목적: OAI 없이 nartaa와 같은 설정으로 우리 모델을 학습해, 공개 0.956 혼합과 다른 오차를 내는 강한 멤버를 만든다 (exp014 예정).

## 2. 한 일과 결정 기록
| 항목 | 결정 | 이유 |
|------|------|------|
| 스택 규칙 | 기존 `build_stack`에 칸·구간·크기만 바꿔 넣음 | 두 코드가 같은 계열(dreaddevelopment)이라 알고리즘이 같다. 새 코드를 쓰지 않는다 |
| 검증 | nartaa 노트북의 `_make_reader`·`build_study`를 그대로 실행해 비교 | 학습 입력이 공개 추론과 다르면 그 가중치·레시피를 따라갈 수 없다 |
| 생성 위치 | 로컬 CPU (8 workers, 끊기면 이어서) | 원본 DICOM이 로컬에 있다 |

## 3. 결과
- `ruff check`, `ruff format --check` 통과. `pytest -q` 117 passed, 2 skipped (새 테스트: `SLOTS96` 칸 구성)
- 비트 단위 비교 (train 6건, `/mnt/d/kag-tmp-input/nartaa_w/eq96.py`): 6건 모두 vol·mask 일치, 최대 차이 0 (채워진 칸 96, 96, 84, 96, 96, 96)
- 전체 코퍼스: 4,407 study × 96 × 384² uint8 ≈ 62 GB — 생성 중

## 4. 미해결 문제
- HF 비공개 저장 한도 100 GB: 현재 데이터셋 54 GB(44장 21.9 + 64장 31.8) + 모델 레포 4.9 GB. 96장 코퍼스를 올리려면 쓰지 않는 64장 코퍼스를 지워야 한다 (사람이 실행)
- Kaggle 추론에서 96장 스택은 44장과 따로 만들어야 한다 → 전처리 시간 증가 (nartaa 단계 실측: 공개 test 3건 51초)

## 5. 다음 작업자가 할 일
1. 코퍼스 생성이 끝나면 HF 데이터셋 `raptor_corpus96/`로 업로드
2. exp014: 96장 코퍼스 + 정제 라벨 + 94창 평가 + 좌우 미러링으로 처음부터 학습
