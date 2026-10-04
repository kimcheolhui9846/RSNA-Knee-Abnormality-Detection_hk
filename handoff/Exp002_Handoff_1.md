# HANDOFF — exp002 pseudo-label 추가 학습

작성: kim cheol hui | 날짜: 2026-10-03 | 브랜치: `exp/002-pseudo` (base: `exp/001-baseline`, stacked PR)

## 1. 요약 (TL;DR)
- `src/data/labels.py` `merge_pseudo_labels`: 정답(12개 라벨 모두 있음)이 우선, 나머지는 pseudo-label, `source` 컬럼(gt / pseudo / none).
- `src/train.py`: config `train_on: any`면 라벨이 하나라도 있는 study 전부로 학습, **평가는 정답 study만**. `cache_dir` 설정 추가.
  exp001 동작은 기본값(`train_on: full`, `cache_dir: cache`)으로 그대로.
- 데이터: 전체 캐시(약 54 GB)를 HF 비공개 데이터셋 `cheolhhh9846/rsna-knee-cache256`에 올리고, labels.csv·folds.csv를 같은 데이터셋 루트에 둔다.
- 학습은 RunPod에서 (실행 승인 대기).

## 2. 한 일과 결정 기록
| 항목 | 결정 | 이유 |
|------|------|------|
| 정답 vs pseudo 충돌 | 정답 우선 | 58개는 사람이 단 라벨 |
| pseudo 해석 | "언급 없음 = 음성", 판단 보류 NaN → 마스크 | 58개 채점에서 AUC 0.802로 더 높았다 (PR #12) |
| 평가 대상 | 정답 58만 | pseudo는 노이즈가 있어 평가에 섞으면 점수가 부풀거나 왜곡된다 |
| fold | `folds_v1` 그대로, pseudo study도 fold 적용 | fold k 검증 때 fold k의 study 전부를 학습에서 빼서 누수 구조를 단순하게 유지 |
| 업로드 | `D:\kag\cache\256`에서 바로 (복사본 없음), 메타는 `D:\kag-exp002\outputs\exp002_meta\` | 사용자 지시: 드라이브는 D. 54 GB 사본을 만들지 않는다 |
| 업로드 실행 | 최소화된 별도 PowerShell 창 | Claude 백그라운드 작업의 시간 제한을 피하고, 끊겨도 이어서 올라간다 |
| epoch / batch | 4 / 2 (workers 8) | 학습 study가 약 60배 → epoch을 줄였다. batch는 exp001과 같은 2: study당 96장이라 batch 8(768장)은 24 GB GPU 메모리 부족 위험 |
| 템플릿 | 80 GB 디스크(`lg5zv4d3sw`), `.harness.env`로 지정 | 데이터셋 54 GB를 컨테이너 디스크로 내려받는다 |

## 3. 결과
- 로컬(WSL, GPU 숨김): 테스트 55 passed (라벨 합치기 2, pseudo 학습·gt 평가 1 추가), `ruff` 통과
- labels.csv: gt 58, pseudo 4,349, pseudo의 판단 보류 칸 4.2%
- **정답 58개와 pseudo의 양성률이 크게 다르다** (Fracture 0.31 vs 0.05, Contusion 0.33 vs 0.13, ACL 0.41 vs 0.18) →
  58개가 이상 소견 위주로 선별된 표본일 수 있다. `experiments/exp002.md` 참고
- **RunPod 학습 (run `20261003-103402-9357a9`, RTX 4090)**: 성공. 정답 58 OOF macro AUC **0.778** (exp001 0.590),
  fold std 0.042, 최저 MCL 0.587. 학습 74분, 데이터 다운로드·설치 약 1시간, Pod 약 2시간 20분 / 약 $1.7.
  끝난 뒤 남은 Pod 0개 확인. 라벨별 비교와 해석은 `experiments/exp002.md`
- 업로드 실측: 54 GB가 HF에 약 1시간 25분(전송 약 15 MB/s + 커밋), 파일 24,375개 확인

## 4. 미해결 문제
- 업로드 소요: 약 12 MB/s → 54 GB에 약 75분 (진행 중)
- Pod가 실행마다 54 GB를 내려받는다(데이터센터 안이라 빠르지만 실측 필요). 실험이 많아지면 Network Volume 검토
- 58개 CV의 대표성 (위 양성률 차이)

## 5. 다음 작업자가 할 일
1. exp002 가중치로 Kaggle 제출 → LB로 58개 CV의 대표성 확인
2. Pod마다 54 GB 다운로드에 약 1시간 → 실험을 반복하려면 Network Volume(월 과금) 검토
3. 개선 후보: 해상도·depth(16→24/32), 백본, MCL·PF OA 약점(관상면·슬개대퇴 슬라이스 선택)
- 재실행 설정(`.harness.env`, 로컬): `DATA_HF_REPO=cheolhhh9846/rsna-knee-cache256`, `RUNPOD_TEMPLATE_ID=lg5zv4d3sw`,
  `TRAIN_ARGS=--config configs/exp002_pseudo.yaml`
