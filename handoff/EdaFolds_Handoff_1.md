# HANDOFF — 초기 EDA + 5-fold 분할

작성: kim cheol hui | 날짜: 2026-09-29 | 브랜치: `eda/initial` (base: `feat/dicom-loader`, stacked PR)

## 1. 요약 (TL;DR)
- **라벨이 달린 train study는 4407개 중 58개(1.3%)뿐이다.** 나머지 4349개는 report만 있다. 일부만 달린 study는 없다.
- CV는 58개로만 계산되므로 fold당 10–13개, 라벨별 fold당 양성 1–10개다. CV 분산이 매우 크다.
- `src/folds.py`로 StudyInstanceUID 단위 5-fold를 만든다. 라벨 있는 study는 12개 라벨로 층화, 없는 study는 따로 균등 배분.
- `notebooks/eda_01_overview.ipynb`: 저장소가 공개라 집계 수치만 출력한다(report 원문·UID 없음).

## 2. 한 일과 결정 기록
| 항목 | 결정 | 이유 |
|------|------|------|
| 분할 단위 | `StudyInstanceUID` | 환자 ID 컬럼이 없다. 같은 환자의 여러 study가 있는지는 확인 불가 (Kaggle Data/Discussion 탭 확인 필요) |
| 라벨 있는 study | `MultilabelStratifiedKFold(shuffle, seed=42)` | 58개뿐이라 라벨 비율 균형이 크기 균형보다 중요. 결과: 모든 라벨이 모든 fold에 양성 ≥1 |
| 라벨 없는 study | `KFold(shuffle, seed=42)` 별도 적용 | 결측을 0으로 채워 층화에 섞으면 라벨 있는 study 배정이 바뀐다 (테스트로 고정) |
| 일부만 라벨 있는 study | `ValueError` | 현재 데이터엔 없다. 생기면 규칙부터 정한다 |
| fold 크기 기준 | 라벨 있는 쪽은 ±2 허용 | 다중 라벨 층화는 크기를 ±1로 보장하지 않는다 |
| fold 파일 줄바꿈 | `\n` 고정 | OS 무관하게 같은 바이트 → 해시로 대조 가능 |
| 노트북 출력 | 집계만, 차트 글자는 영어 | 공개 저장소에 대회 데이터 원문이 올라가지 않게. 기본 글꼴에 한글이 없어 차트가 깨짐 |

## 3. 결과
**규모**: train 4407 study / 24371 시리즈, 공개 test 3 study / 15 시리즈 (제출 시 약 1300 study로 교체). test.csv에는 report 컬럼이 없다.

**라벨 있는 58개의 양성률**: Effusion 0.60, Synovitis 0.47, Medial Meniscus 0.45, ACL 0.41, Lateral Meniscus 0.40, PF OA 0.36, Contusion 0.33, Fracture 0.31, Medial OA 0.26, Baker's 0.21, Lateral OA 0.19, MCL 0.16 (최소 양성 수 MCL 9개)

**시리즈**: 모든 study가 Axial·Coronal·Sagittal을 다 가진다. study당 시리즈 3–14개(중앙값 5). 조합은 plane × (Fluid_Sensitive=Fat_Suppression=0 또는 1) 여섯 가지뿐 (두 플래그가 항상 같이 움직인다).

**슬라이스 수** (1500 시리즈 표본): 중앙값 Axial 32 / Coronal 30 / Sagittal 30, 최대 320. 80장 초과는 44개(2.9%).

**report 언어** (langdetect): en 1736, es 682, tr 546, hr 406, el 321, de 262, bg 220, nl 153, fr 81. 라벨 있는 58개 중 en 28개, fr는 0개.

**fold**: 라벨 있는 study fold별 12/10/12/11/13, 없는 study 870/870/870/870/869. `folds/folds_v1.csv` (4407행)의 sha256:
`82c80f07631d7bfaba76619b417500bbb9e276bd41c915a4706f2e94c09aaff2`

테스트 30 passed, ruff 통과 (WSL).

## 4. 미해결 문제
- **fold 파일은 커밋하지 않는다 (2026-09-30 결정).** 저장소를 공개로 유지하므로 대회 데이터의 StudyInstanceUID를 올리지 않는다. `.gitignore`에 `/folds/*.csv`, `/pseudo_labels/`를 추가했다. 각자 `python -m src.folds`로 만들고 위 해시로 대조한다.
- **CV 신뢰도**: 58개 CV는 라벨별 AUC 하나가 양성 몇 개로 크게 흔들린다. repeated k-fold나 여러 seed 평균을 검토할 것.
- **라벨 있는 58개와 전체의 분포 차이**: 58개는 양성률이 높아 선별된 표본일 수 있다. test 분포와 같다는 보장이 없다.
- **환자 단위 중복 여부**: 확인 불가 (Kaggle Data/Discussion 탭 확인 필요).

## 5. 다음 작업자가 할 일
1. report → 12개 라벨 추출 (pseudo-label). 58개 라벨로 추출 정확도를 먼저 검증한다. 9개 언어라 다국어 처리 필요.
2. 전처리 캐시 (RunPod): 시리즈별 볼륨을 고정 크기로 리샘플해서 저장.
3. 베이스라인 exp001: 2D 슬라이스 + ImageNet 백본 + study 단위 집계 (로드맵 1단계).
