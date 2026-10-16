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
- 학습: RunPod 4090, run `20261015-011530-d8299e`, 학습 3.2시간 (commit 8522e80)
- 정답 58 (k=42, 참고용): epoch 9 0.911 → epoch 13–16 0.920 부근에서 평평, **SWA(14–16) 0.9197** (exp014 0.9231, exp007 0.9037)
- 라벨별 (SWA): ACL 0.973, MCL 0.977, Medial Meniscus 0.975, Lateral Meniscus 0.853, Medial OA 0.981, Lateral OA 0.863, PF OA 0.838,
  Effusion 0.975, Synovitis 0.805, Baker's 0.966, Contusion 0.907, Fracture 0.924
- Kaggle: 가중치 `kimche12/rsna-knee-weights-e15`, 혼합 노트북 `kimche12/rsna-knee-public956-e15` (PR #31 생성기, 공개 0.85 + exp015 0.15)
- LB: 제출 57217206 (공개 0.956 노트북 0.85 + exp015 0.15) → **Public LB 0.955** (공개 그대로 0.956보다 낮음, exp013·exp014 혼합과 같음)
- 결론: 라벨을 공개 라벨로 바꿔도 혼합에서 떨어진다 → **정제 라벨은 원인이 아니다**. 우리 Raptor 계열(44장·96장, 정제·공개 라벨, nartaa 이어 학습)을 15% 섞으면 모두 0.955였다.
  5모델 10% 혼합(57166654)이 0.956을 지킨 것은 비율이 낮았기 때문일 가능성이 크다.

## 4. 미해결 문제
- 우리 모델이 혼합을 깎는 원인은 찾지 못했다. 남은 후보: 공개 두 계열(nartaa Raptor + R7)이 이미 같은 정보를 담고 있어 우리 모델은 잡음만 더한다 (OAI·fastMRI 보조 학습, 4-fold 앙상블 차이)
- 정답 58 점수가 혼합 효과를 예측하지 못하므로, 학습이 끝나도 LB 제출 전까지는 판단할 수 없다.

## 5. 다음 작업자가 할 일
1. (완료) 가중치 `kimche12/rsna-knee-weights-e15` 업로드
2. (완료) `kaggle/build_public_blend.py --weight 0.15 --kernel-slug rsna-knee-public956-e15 --weights-slug rsna-knee-weights-e15`(PR #31)로 혼합 노트북 생성 → LB 확인
3. 우리 모델 혼합은 접고 공개 0.956 그대로(57165935)를 최종 후보로 둔다. 추가 학습은 하지 않는다.
