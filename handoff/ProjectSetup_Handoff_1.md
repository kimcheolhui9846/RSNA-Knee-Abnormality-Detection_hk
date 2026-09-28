# HANDOFF — 프로젝트 초기 세팅 1: 저장소 뼈대 + metric + 제출 검증

작성: kim cheol hui | 날짜: 2026-09-28 | 브랜치: `feat/project-setup`

## 1. 요약 (TL;DR)
- 저장소 구조, `.gitignore`, PR 템플릿, `pyproject.toml`(uv), GitHub Actions CI를 세팅했다.
- `src/metrics.py`(macro AUC)와 `src/submission.py`(제출 형식 검증)를 TDD로 작성했다. `pytest` 14 passed, 1 skipped.
- 데이터가 필요한 EDA와 fold 분할은 다음 PR에서 한다.

## 2. 한 일과 결정 기록
| 항목 | 결정 | 이유 |
|------|------|------|
| 환경 관리 | `uv` + `uv.lock` 커밋 | 팀원 간 버전 재현. `uv sync` 한 줄로 동일 환경 |
| torch 설치 | `--extra train`으로 분리, CUDA 12.8 인덱스 | 테스트·EDA·CI는 torch 없이 가볍게 돈다 |
| DICOM 디코더 | `pylibjpeg` + libjpeg + openjpeg 기본 포함 | 데이터에 JPEG Lossless / JPEG 2000이 섞여 있음. Kaggle 오프라인 제출에도 필요 |
| lint | CI(GitHub Actions)에서 `ruff check`, `ruff format --check` | 로컬 Windows에서 `ruff.exe`가 애플리케이션 제어 정책에 차단됨 |
| 결측 라벨 | metric에서 NaN은 해당 라벨 계산에서만 제외 | train의 상당수 study는 라벨 없이 report만 있음. 0으로 채우면 음성으로 오염된다 |
| 단일 클래스 라벨 | AUC를 NaN으로 두고 macro 평균에서 제외 | fold에 Fracture 양성이 없을 수 있음. AUC 정의 불가 |
| `.gitignore` 데이터 경로 | `/data/`, `/input/`로 루트 고정 | `data/`로 쓰면 `src/data/` 패키지까지 무시됨 (실제로 발생해서 수정) |

## 3. 결과
- `uv run pytest -q` → 14 passed, 1 skipped (skip: `sample_submission.csv` 헤더 대조 테스트, 데이터 없음)
- ruff: 이 PR의 CI 결과로 확인

## 4. 미해결 문제
- **공식 metric의 단일 클래스 라벨 처리 방식**: Kaggle Data/Discussion 탭 확인 필요. 지금 구현은 CV용 가정.
- **라벨 컬럼명·순서**: 대회 설명 기준으로 넣었다. 데이터가 들어오면 skip된 테스트가 자동으로 대조한다.
- **환자 ID 제공 여부**: fold 분할 단위(환자 / study)를 정하려면 확인 필요.

## 5. 다음 작업자가 할 일
1. 데이터를 `data/`(또는 `RSNA_DATA_DIR` 환경변수 경로)에 두고 `uv run pytest -q` → 헤더 대조 테스트 통과 확인
2. EDA: 라벨 분포·결측률, study당 시리즈/시퀀스 구성, 슬라이스 수, report 언어·형태
3. `folds/`에 5-fold 분할 파일 생성 (결측 라벨 study는 별도 그룹으로 고르게 배분)
4. 학습은 로컬 GPU가 아니라 RunPod에서 한다
