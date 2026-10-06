# HANDOFF — 64장 Raptor 코퍼스를 train DICOM에서 직접 만들기

작성: kim cheol hui | 날짜: 2026-10-06 | 브랜치: `feat/corpus64` (base: `exp/009-label-refine`, stacked PR)

## 1. 요약 (TL;DR)
- `src/data/build_corpus.py`: train DICOM → `all_vols.npy`(N×D×336×336 uint8), `all_masks.npy`, `all_ids.npy`, `meta.json`.
  `train_raptor.open_corpus()`가 그대로 읽는다. 스택은 Kaggle 추론과 같은 `build_stack()`으로 만든다.
- 목적: 공개 원작자 보고(64장, 구간 6–94%가 Lateral Meniscus·측부인대에 유리, 제3자 재현 정답 58 0.930 / LB 0.939)를 우리 학습에 쓰기.
  공개 코퍼스는 44장·15–85%만 있어 직접 만든다.

## 2. 한 일과 결정 기록
| 항목 | 결정 | 이유 |
|------|------|------|
| 스택 함수 | `build_stack()` 재사용 | 학습·추론 입력 일치 (테스트로 확인) |
| 병렬 | `multiprocessing.Pool`, `maxtasksperchild=50` | 첫 시도(작업자 24)가 PC 메모리 부족으로 강제 종료됨 → 작업자 8 + 작업자 주기 교체 |
| 이어 하기 | `save_every`(100)마다 `vols.flush()` 후 `_partial.npz`(masks, done, ids, meta) 원자적 교체 | 중간에 끊겨도 끝난 study를 다시 만들지 않는다. 설정·ID가 다르면 처음부터 |
| 순서 | `train.csv` 순서 | 재현성 |

## 3. 결과
- `ruff check`, `ruff format --check` 통과. `pytest -q` 115 passed, 2 skipped
- 새 테스트: 코퍼스 = study별 `build_stack()` 결과(비트 단위), `open_corpus`로 열림, 칸 배치; 중단 후 이어 하기(끝난 study 재계산 없음, 결과 동일)
- 로컬 속도: 작업자 16에서 48 study 102초(입출력이 병목, WSL `/mnt/d`)
- 전체 4,407 study 생성: (진행 중)

## 4. 미해결 문제
- 크기 약 32 GB(4,407×64×336×336) → HF 업로드·Pod 다운로드 시간이 든다

## 5. 다음 작업자가 할 일
1. 생성 완료 후 HF 데이터셋에 올리고 `corpus_dir: raptor_corpus64`, `slots: SLOTS64`, `span: [0.06, 0.94]` config로 학습
