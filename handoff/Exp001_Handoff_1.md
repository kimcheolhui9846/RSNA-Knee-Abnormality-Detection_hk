# HANDOFF — exp001 베이스라인과 RunPod 학습 연결

작성: kim cheol hui | 날짜: 2026-10-03 | 브랜치: `exp/001-baseline` (base: `feat/dataset`, stacked PR)

## 1. 요약 (TL;DR)
- 모델 `src/models/baseline.py` `KneeBaseline`: 공유 2D 백본(흑백) → 칸마다 슬라이스 평균·최대 → 칸 임베딩 → 있는 칸 평균 → 12 로짓.
- 손실 `src/losses.py` `masked_bce`: 라벨 있는 위치만 평균, 라벨이 없으면 0.
- 학습 `src/train.py` `run()`: 5-fold CV → `model.safetensors`(fold별 키 접두사 `fold{k}.`), `oof.csv`, `metrics.json`.
- RunPod 연결: 저장소 루트 `train.py`(runpod-hf-harness 템플릿)의 `train_model()`이 `src.train.run`을 부른다. Pod용 `requirements.txt`.
- 데이터: 라벨 있는 58 study의 캐시(336 시리즈, 0.69 GB)만 HF 비공개 데이터셋 `cheolhhh9846/RSNA-Knee-Abnormality-Detection_hk-data`에 올렸다.

## 2. 한 일과 결정 기록
| 항목 | 결정 | 이유 |
|------|------|------|
| 없는 칸 처리 | 백본에 넣지 않고 study 평균에서 제외 | 계산 절약 + 없는 칸의 픽셀 값이 출력에 영향 없음 (테스트로 고정) |
| 슬라이스 집계 | 평균 + 최대 | 국소 병변(최대)과 전체 상태(평균)를 둘 다 본다 |
| 칸 임베딩 | 학습되는 벡터를 칸 특징에 더함 | 같은 백본 특징이라도 어느 방향·시퀀스인지 구분 |
| 검증 시점 | 마지막 epoch | 검증 fold가 약 12 study라 best-epoch 선택은 과적합 위험 |
| 증강 | 없음 | 계획한 좌우 반전은 시상면에서 앞뒤 반전이 되어 해부학적으로 부자연스럽다. exp002 이후 plane별로 검토 |
| 결과 저장 | fold 5개 가중치를 safetensors 하나에 접두사로 | 하네스가 최종 파일 하나를 올리는 구조 |
| 하네스 smoke-test | 합성 데이터로 같은 학습 코드를 작게 실행 | 더미 가중치만 저장하는 원래 smoke보다 실제 경로를 검증 |
| 업로드 데이터 | 라벨 있는 58 study만, report 원문 제외 | exp001은 지도 학습만. 0.69 GB라 컨테이너 디스크(30 GB)로 충분 |
| `.harness.env` | 커밋하지 않음 (`.git/info/exclude`) | 에이전트·하네스 설정은 로컬 전용 규칙 |

## 3. 결과
- 로컬(WSL, `CUDA_VISIBLE_DEVICES=`로 GPU 숨김): 테스트 52 passed, `ruff check`·`ruff format --check` 통과
- 하네스 smoke-test(CPU, 합성 데이터): 2-fold 끝까지 실행, 가중치·OOF·metrics 생성, 종료 코드 0
- 업로드 데이터 검증: 58 study, fold별 12/10/12/11/13, 모든 study가 3칸 이상
- RunPod 학습 결과: `experiments/exp001.md`에 추가 예정

**사고 기록**: 처음 smoke-test는 WSL torch가 Windows GPU를 자동으로 잡아 로컬 RTX 3070 Ti에서 약 4초 돌았다(합성 데이터, 32px). 이후 모든 로컬 실행에 `CUDA_VISIBLE_DEVICES=`를 붙인다.

## 4. 미해결 문제
- 58 study CV라 수치 신뢰구간이 넓다. 라벨별 AUC는 fold당 양성 1–10개로 계산된다.
- Kaggle 제출용 추론 노트북과 오프라인 사전학습 가중치 데이터셋은 아직 없다.
- pseudo-label(4349 study)을 붙이면 전체 캐시(50 GiB) 업로드가 필요하다 → Network Volume 검토.

## 5. 다음 작업자가 할 일
1. RunPod 결과(`runs/<run_id>/metrics.json`)로 `experiments/exp001.md`, README 표 갱신
2. pseudo-label 2단계(LLM 추출) 후 라벨 없는 study를 학습에 합치는 exp
3. 제출 노트북(`kaggle/`) 뼈대
