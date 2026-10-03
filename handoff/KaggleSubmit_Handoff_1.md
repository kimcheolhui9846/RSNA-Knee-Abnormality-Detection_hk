# HANDOFF — Kaggle 제출 경로 (추론 + 제출 꾸러미)

작성: kim cheol hui | 날짜: 2026-10-03 | 브랜치: `feat/kaggle-submit` (base: `exp/001-baseline`, stacked PR)

## 1. 요약 (TL;DR)
- `src/infer.py` `predict()`: test DICOM → 학습과 같은 전처리 → fold 앙상블(평균) → 검증된 `submission.csv`.
  읽을 수 있는 시리즈가 없는 study는 `FALLBACK_PROB=0.5`로 채워 제출 파일이 항상 완성된다.
- `kaggle/submission.py`: 인터넷 OFF 제출 스크립트. 입력 경로를 표시 파일로 찾아서 Kaggle 마운트 경로 변화에 강하다.
- `kaggle/build_kaggle.py`: 업로드용 폴더 3개(코드 데이터셋, 가중치 데이터셋, 제출 노트북 메타데이터)를 만든다. 업로드는 하지 않는다.
- 로컬 검증: 공개 test 3 study로 `/kaggle/input` 흉내 폴더에서 끝까지 실행 성공, 오프라인 휠 설치 성공.

## 2. 한 일과 결정 기록
| 항목 | 결정 | 이유 |
|------|------|------|
| 추론 입력 | 캐시 없이 study마다 DICOM을 바로 읽음 | test는 제출 때 교체된다. 학습과 같은 `load_series`/`preprocess_series`/`select_series`/`resample_depth` |
| 사전학습 가중치 | 받지 않음 (`pretrained=False` 후 학습 가중치 로드) | 인터넷 OFF. 학습 가중치에 백본까지 다 들어 있다 |
| fold 수 | 가중치 파일의 `fold{k}.` 접두사에서 읽음 | 실험마다 fold 수가 달라도 코드 수정 없음 |
| 실패한 study | 0.5로 채우고 로그 | 제출 실패(행 누락)를 막는다. 시리즈 하나 실패는 나머지 시리즈로 예측 |
| 병렬 | DataLoader worker가 DICOM 디코딩, batch 1 | 병목은 I/O·디코딩이다 |
| 오프라인 휠 | timm 1.0.30(학습과 같은 버전) + pylibjpeg 3종(cp310/311/312) | Kaggle 이미지의 Python 버전을 몰라 세 버전 모두. 설치 실패 시 기본 패키지로 진행 |
| 경로 찾기 | `rsna_knee_*.marker`, `test_series.csv`를 `/kaggle/input` 아래에서 검색 | 데이터셋 마운트 경로 규칙이 바뀌어도 동작 |
| 노트북 형식 | script kernel, GPU ON, 인터넷 OFF, 비공개 | lint·테스트 가능한 .py |

## 3. 결과
- 테스트 56 passed (infer 3, build 1 추가), `ruff check`·`ruff format --check` 통과 (WSL, GPU 숨김)
- 실제 exp001 가중치(94.4 MB, 5 fold)로 공개 test 3 study 예측: 형식 검증 통과, fallback 0
- Kaggle 흉내 실행(`/kaggle/input` 구조, 저장소 밖 작업 폴더): 성공, 24초 (CPU, HDD 경유)
- 깨끗한 Python 3.11 + pip에서 `pip install --no-index` 오프라인 휠 설치·import 성공
- 꾸러미 크기: 코드 18.9 MB (휠 포함), 가중치 94.4 MB

## 4. 미해결 문제
- **Kaggle 실측 필요**: test 약 1,300 study 기준 시간. 로컬 CPU는 study당 약 8초(5 fold, HDD). Kaggle GPU에서는 디코딩이 병목일 것 → 첫 제출 로그로 확인
- **Kaggle 인증 없음**: 이 PC에 `kaggle.json`이 없다. 업로드(`kaggle datasets create`, `kaggle kernels push`)와 제출에 필요
- **압축 DICOM**: train 표본에는 없었다. test에 있으면 오프라인 디코더가 필요 → 휠 포함으로 대비
- FALLBACK 0.5는 AUC상 중립값이지만, 실패 study가 많으면 점수를 깎는다. 로그의 fallback 수를 확인

## 5. 다음 작업자가 할 일
1. Kaggle 인증 설정 후 `python kaggle/build_kaggle.py --user <사용자명> --weights ... --wheels ...`
2. `kaggle datasets create -p outputs/kaggle/code`, `-p outputs/kaggle/weights` → `kaggle kernels push -p outputs/kaggle/kernel`
3. 노트북 실행 로그에서 시간·fallback 수 확인 → 대회에 제출
