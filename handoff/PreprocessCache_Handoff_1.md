# HANDOFF — 전처리 함수와 시리즈 캐시

작성: kim cheol hui | 날짜: 2026-09-30 | 브랜치: `feat/preprocess-cache` (base: `feat/dicom-loader`, stacked PR)

## 1. 요약 (TL;DR)
- `src/data/preprocess.py` `preprocess_series`: 시리즈 볼륨 → `(N, 256, 256)` uint8. **캐시 생성과 Kaggle 추론이 같은 함수를 쓴다.**
- `src/data/build_cache.py`: train 시리즈 24,371개를 `cache/256/<study>/<series>.npy`로 저장하고 `cache/256/train_index.csv`를 만든다. CPU 병렬, 이어서 만들기 가능.
- `load_series`가 MONOCHROME1을 뒤집는다 (PR #5 리뷰 이관 항목).
- 실제 시리즈 200개 스모크: 실패 0. 캐시 영상 육안 확인: 대비·슬라이스 순서·비율 정상.

## 2. 한 일과 결정 기록
| 항목 | 결정 | 이유 |
|------|------|------|
| 밝기 정규화 | 시리즈 전체의 0.5 / 99.5 백분위로 자르고 0–255 | MRI는 절대 밝기 단위가 없다. 극단값 한두 개가 대비를 망치지 않게 |
| 저장 형식 | uint8 `.npy` | float32 대비 1/4 용량. 256² × 30장 ≈ 2MB/시리즈 |
| 가로·세로 | 256×256 (`--size`로 변경 가능) | 추론 9시간·Efficiency 트랙 고려. 원본은 256–704 정사각 |
| 정사각형이 아닐 때 | 가운데 정렬 0 패딩 후 리사이즈 | 비율을 늘이면 해부 구조가 왜곡된다 |
| 리사이즈 보간 | 줄일 때 INTER_AREA, 키울 때 INTER_LINEAR | 축소 시 앨리어싱 방지 |
| 슬라이스 수 | 원본 그대로 | 몇 장으로 맞출지는 Dataset 단계(다음 PR)에서 정한다 |
| MONOCHROME1 | `load_series`에서 부호 반전 | 값이 클수록 밝게 통일. 절대값은 이후 백분위 정규화가 맞춘다 |
| 실패한 시리즈 | 전체를 멈추지 않고 인덱스에 `status=error`로 기록 | 24k개 중 하나 때문에 몇 시간 작업이 날아가지 않게 |
| 쓰기 | 임시 파일에 쓰고 `os.replace` | 중간에 끊겨도 반쪽 파일이 완성본으로 남지 않는다 |
| 의존성 | `opencv-python-headless` 추가 | 빠른 리사이즈. Kaggle 이미지에도 cv2가 있다 |
| 캐시 위치 | 저장소 안 `cache/` (git 제외) | 데이터와 같은 드라이브. RunPod에는 캐시만 올린다 |

## 3. 결과
- 테스트 36 passed (preprocess 6, build_cache 4, MONOCHROME1 1 추가), `ruff check`·`ruff format --check` 통과 (WSL)
- 스모크 (실제 train 시리즈 200개, WSL 16 workers): 전부 `built`, 2분 10초. CPU 시간은 32초라 **`/mnt/d` 읽기 속도가 병목**이다.
- **전체 캐시 완료 (2026-09-30)**: 24,371 / 24,371 시리즈, **실패 0**. 용량 50 GiB (53.7 GB). `cache/256/train_index.csv` 생성
  - 슬라이스 수: 최소 11 / 사분위 25·30·34 / 최대 320
  - 원본 크기 상위: 512² 5,941 · 640² 3,200 · 384² 2,804 · 320² 1,810 · 256² 1,056 · **1024² 1,033**
  - **정사각형이 아닌 시리즈 1,769개 (7.3%)** — 처음 400여 파일 표본에는 없었다. 패딩 처리가 실제로 필요했다
  - 픽셀 간격(행): 최소 0.073 / 중앙 0.312 / 최대 1.172 mm
  - 소요: Claude Code 백그라운드 작업이 메모리 부족·시간 제한으로 두 번 끊겨, 이어서 만들기로 별도 WSL 창에서 완료. 마지막 구간은 분당 약 80 시리즈 (8 workers)
- **D 드라이브는 HDD**(Seagate 2TB)이고 캐시 생성 중 유휴 0%로 포화 → 병목은 CPU가 아니라 HDD 임의 읽기다
- 육안 확인: Sagittal / Coronal / Axial 각 1개 시리즈의 5개 슬라이스 — 대비 정상, 슬라이스 순서가 해부학적으로 연속, 비율 왜곡 없음

## 4. 미해결 문제
- **캐시 다시 만들 때 (예: 384px)**: 출력을 SSD(C 드라이브 또는 WSL 내부)에 두면 HDD 쓰기 부담이 줄어 빨라진다. WSL 메모리는 `.wslconfig`로 12GB 상한 + `autoMemoryReclaim=gradual`을 권장 (기본값에서 Windows 여유 메모리가 5GB까지 떨어져 작업이 끊겼다).
- **test 캐시**: `--split test`로 같은 방식으로 만들 수 있다. Kaggle 제출 노트북에서는 캐시 없이 `load_series` → `preprocess_series`를 바로 호출한다.
- **Kaggle 추론 시간**: 로컬 스모크 기준 시리즈당 약 0.65초(대부분 I/O). test 약 1300 study × 5.5 시리즈 ≈ 7,000 시리즈 → 전처리만 1시간 남짓 예상. Kaggle 디스크 속도로 다시 재야 한다.

## 5. 다음 작업자가 할 일
1. `feat/dataset`: study당 (plane × 시퀀스) 6칸 시리즈 선택, 슬라이스 수 고정, fold·라벨·마스크를 붙인 PyTorch Dataset
2. 캐시를 RunPod으로 옮기는 방법 정하기 (용량 약 50GB 예상)
