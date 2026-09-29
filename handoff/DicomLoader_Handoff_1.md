# HANDOFF — DICOM 시리즈 로더 + 데이터 경로

작성: kim cheol hui | 날짜: 2026-09-29 | 브랜치: `feat/dicom-loader` (base: `feat/project-setup`, stacked PR)

## 1. 요약 (TL;DR)
- `src/paths.py`: 데이터 루트를 `RSNA_DATA_DIR` → `/kaggle/input/rsna-knee-abnormality-detection` → 저장소 안 로컬 폴더 순서로 찾는다.
- `src/data/dicom.py` `load_series`: 시리즈 폴더를 `(N, H, W)` float32 볼륨으로 읽는다. 슬라이스 법선 방향 위치로 정렬하고 rescale을 적용한다.
- 실제 train 시리즈 40개를 오류 없이 불러왔다. 테스트 24 passed (실제 데이터 테스트 2개 포함), ruff 통과.

## 2. 한 일과 결정 기록
| 항목 | 결정 | 이유 |
|------|------|------|
| 슬라이스 정렬 | `ImagePositionPatient`를 `ImageOrientationPatient`의 법선(행×열 방향 외적)에 투영한 값의 오름차순 | 파일 이름은 SOPInstanceUID라 순서 정보가 없다. InstanceNumber는 스캐너마다 신뢰도가 다르다 |
| 정렬 fallback | 위치·방향 태그가 하나라도 없으면 `InstanceNumber` | 표본에서는 두 태그가 100% 있었지만 test 교체 데이터 대비 |
| rescale | `RescaleSlope/Intercept`가 있으면 적용, 출력은 float32 | 표본 파일의 26%에만 태그가 있다. 없는 파일은 원래 값 그대로 |
| 슬라이스 크기 불일치 | `ValueError` | 조용히 잘못된 볼륨을 만드는 것보다 낫다. 실제로 생기는지는 EDA에서 확인 |
| 압축 디코딩 테스트 | JPEG2000 Lossless를 만들어 디코딩 확인 | 대회 설명상 JPEG 2000이 섞여 있다. pylibjpeg가 제대로 설치됐는지도 같이 검증 |
| 로컬 실행 환경 | WSL Ubuntu + 별도 venv(`~/.venvs/rsna-knee`) | Windows 스마트 앱 컨트롤이 `.venv`의 pytest·ruff 실행 파일과 pandas DLL을 차단 |

## 3. 결과 (실제 데이터 표본)
- 시리즈 40개 무작위 로드: 실패 0, 시리즈당 로드 시간 중앙값 0.42s / 최대 4.8s (WSL에서 `/mnt/d` 경유, CPU)
- 1500 study / 8298 시리즈의 첫 파일 헤더: transfer syntax 전부 Explicit VR Little Endian. `NumberOfFrames`는 없거나 1 (멀티프레임 없음)
- 400여 파일 표본: 전부 MONOCHROME2, 크기 256–704 정사각, 시리즈당 슬라이스 15–160장(중앙값 30)
- `sample_submission.csv` 헤더가 `LABELS` 순서와 일치함을 테스트로 확인

## 4. 미해결 문제
- **압축 DICOM 실재 여부**: train 표본에서는 비압축만 나왔다. 압축본이 test에만 있거나 시리즈 중간 파일에 있을 수 있어 디코더는 유지한다. Kaggle Data/Discussion 탭 확인 필요.
- **로드 속도**: `/mnt/d` 경유라 느리다. 학습 시에는 RunPod에서 전처리 캐시(npy 등)로 만들어 쓰는 게 좋다.
- **정렬 방향 규약**: 법선 방향 오름차순으로 고정했다. 좌/우 무릎이나 plane별로 방향을 통일할지는 모델링 단계에서 정한다.

## 5. 다음 작업자가 할 일
1. EDA: 라벨 분포·결측률, study당 plane/시퀀스 조합, 슬라이스 수 분포, report 언어
2. `folds/` 5-fold 분할 (환자 ID 컬럼이 없으므로 `StudyInstanceUID` 단위, 결측 라벨 study는 별도 그룹)
3. 전처리 캐시 스크립트 (RunPod에서 실행)
