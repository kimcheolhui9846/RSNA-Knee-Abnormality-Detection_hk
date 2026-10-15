# HANDOFF — exp015: exp014와 같고 라벨만 공개 Raptor 라벨

작성: kim cheol hui | 날짜: 2026-10-15 | 브랜치: `exp/015-raptor96-dread` (base: `exp/014-raptor96`)

## 1. 요약 (TL;DR)
- 공개 0.956 노트북에 섞었을 때, 정제 라벨 모델(exp013, exp014)은 둘 다 0.955로 떨어졌다. 공개 라벨 모델이 들어 있는 5모델 혼합(exp007·v2·mean·exp011·exp012)은 0.956을 지켰다.
- 가설: 우리 모델이 혼합을 깎는 원인은 정제 라벨이다.
- exp015는 exp014(96장 코퍼스 + 3D 좌우 거울상)에서 라벨 파일만 `labels_dread.csv`(공개 Raptor 라벨, exp007과 같음)로 바꾼다. seed는 15.
- 판정은 LB로 한다: 공개 0.956 + exp015 15%가 0.956 이상이면 라벨이 원인이고, 0.955면 라벨이 원인이 아니다.

## 2. 한 일과 결정 기록
| 항목 | 결정 | 이유 |
|------|------|------|
| 바꾸는 것 | `labels_file`과 seed만 | exp014와 1:1 비교. 입력·증강·학습량은 그대로 |
| 라벨 | `labels_dread.csv` | 공개 Raptor 라벨. exp007이 이 라벨로 학습했고, exp007이 든 5모델 혼합은 0.956을 지켰다 |
| 시작점 | ImageNet (nartaa 체크포인트 아님) | exp014와 같은 조건. nartaa에서 이어 학습한 exp013도 0.955였다 |
| 혼합 비율 | 15% (exp013·exp014와 같음) | 비교를 위해 같게 둔다 |

## 3. 결과
- 설정: `configs/exp015_raptor96_dread.yaml` (exp014와의 차이는 `labels_file`, `seed`, 주석뿐)
- 학습: (실행 후 기록)

## 4. 미해결 문제
- 정답 58 점수가 혼합 효과를 예측하지 못하므로, 학습이 끝나도 LB 제출 전까지는 판단할 수 없다.

## 5. 다음 작업자가 할 일
1. 학습이 끝나면 가중치를 `kimche12/rsna-knee-weights-e15`로 올린다.
2. `kaggle/build_public_blend.py --weight 0.15 --kernel-slug rsna-knee-public956-e15 --weights-slug rsna-knee-weights-e15`(PR #31)로 혼합 노트북을 만들어 LB를 확인한다.
