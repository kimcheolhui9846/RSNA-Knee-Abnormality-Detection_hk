# HANDOFF — exp004 DINOv2 + 라벨별 attention + 2.5D

작성: kim cheol hui | 날짜: 2026-10-04 | 브랜치: `exp/004-dino` (base: `exp/003-public-labels`, stacked PR)

## 1. 요약 (TL;DR)
- `src/models/dino.py` `KneeDinoAttn`: DINOv2 ViT-S/14(timm) + 칸·깊이 임베딩 + 12개 소견별 attention 질의. `src/models.build_model`로 선택.
- `src/data/dataset.py` `slab_stack`: 2.5D(앞·현재·뒤 슬라이스 3채널). `src/train.py`: 백본 lr 분리.
- exp003에서 merge: Pod에서 CUDA 불가 시 즉시 실패(`check_device`), fold마다 체크포인트 저장.
- 결과: 정답 58 OOF macro AUC **0.703** (exp002 0.778보다 낮음). 라벨·모델 변경이 섞여 원인 분리 불가.

## 2. 한 일과 결정 기록
| 항목 | 결정 | 이유 |
|------|------|------|
| 백본 | DINOv2 ViT-S/14, 앞 6블록 고정, 224px | 공개 상위권 공통. 메모리·시간 절약 |
| 집계 | 소견별 attention 질의 | 소견마다 중요한 칸·슬라이스가 다르다 |
| 2.5D | 이웃 슬라이스 3채널 | 슬라이스 간 연속성, ImageNet 3채널 입력과 맞음 |
| exp003 생략 | 라벨 효과 별도 측정 없이 exp004 실행 | 예산 절약 (사용자 선택 A) |
| GPU | Community 2회 실패 후 **Secure** | Community 4090(드라이버 580)에서 CUDA unknown error 재현 |

## 3. 결과
- 로컬: 테스트 64 passed, `ruff` 통과, 실제 설정 CPU forward/backward 확인
- RunPod (Secure, run `20261004-084703-44e9a3`): 학습 88분, 약 $1.8. OOF 0.703, fold std 0.054. 라벨별 표: `experiments/exp004.md`
- 실패 2회: run `20261004-023051-10bb0e`(CPU로 4시간 시간 초과, $1.37), `20261004-064039-a6c17a`(사전 검사로 즉시 중단, $0.68)

## 4. 미해결 문제
- 성능 하락 원인 (라벨 vs 모델) 미분리
- 58 CV는 표본이 작고 편향 → 4,349 pseudo study의 OOF 보조 지표 필요
- Community 템플릿 CUDA 문제는 하네스 쪽 이슈 문서 6절

## 5. 다음 작업자가 할 일
1. LB 제출로 실제 방향 확인 (`feat/submit-v2`가 이 모델을 지원)
2. 보조 CV 추가 후, 라벨만 바꾼 실험(exp003)과 학습 강화(epoch·고정 블록·해상도)를 분리 측정
