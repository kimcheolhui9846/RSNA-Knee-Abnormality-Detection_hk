# HANDOFF — report pseudo-label 2단계: LLM 추출기

작성: kim cheol hui | 날짜: 2026-10-03 | 브랜치: `feat/pseudo-llm` (base: `feat/pseudo-eval`, stacked PR)

## 1. 요약 (TL;DR)
- `src/pseudo/prompt.py`: 12개 라벨 정의를 담은 다국어 프롬프트(번역 없이 원문 report 입력). 상태 4가지: positive / negative / uncertain / not_mentioned.
- `src/pseudo/parse.py`: LLM 응답에서 JSON을 찾아 라벨별 확률로 변환. 잘못된 응답·키는 NaN(판단 보류).
- `src/pseudo/extract.py`: report 묶음 → pseudo-label 표 + 원문 응답 표. LLM 호출은 주입(`vllm_generate`는 Pod 전용).
- `pseudo_job.py`: 하네스 템플릿 기반 Pod 작업(`TRAIN_ENTRY=pseudo_job.py`). 4,407개 전체 추출 → 58개 정답으로 두 해석 채점 → HF 업로드.
- 로컬: 테스트 42 passed, 가짜 생성기 smoke-test 통과. **실제 LLM 실행은 아직 안 했다** (승인 대기).

## 2. 한 일과 결정 기록
| 항목 | 결정 | 이유 |
|------|------|------|
| 번역 | 하지 않음, 원문 그대로 | 9개 언어. 다국어 instruct 모델은 원문 이해가 가능하고 번역 오류가 끼지 않는다 |
| 상태 4가지 | positive / negative / uncertain / not_mentioned | "명시적 음성"과 "언급 없음"을 구분해야 해석을 나중에 바꿀 수 있다 |
| 기본 해석 | not_mentioned → 0, uncertain → NaN | 판독문은 정상 소견을 생략하는 관례. 대신 NaN 해석도 함께 채점해 고른다 |
| 원문 응답 저장 | `raw_responses.csv` | 해석을 바꿀 때 LLM 재실행이 필요 없다 |
| 파싱 실패 | 해당 라벨 NaN, 전체 실패 수를 metrics에 기록 | 조용히 0으로 채우지 않는다 |
| 기본 모델 | `Qwen/Qwen2.5-7B-Instruct` (Apache-2.0, 다국어) | RTX 4090(24GB)에 bf16으로 들어가고 vLLM 지원. `--model`로 교체 가능 |
| 진입 파일 | `pseudo_job.py` (하네스 `TRAIN_ENTRY`) | exp001 브랜치의 `train.py`와 충돌하지 않게 |
| 결과 위치 | HF 비공개 결과 레포 `runs/<run_id>/` | 대회 데이터 파생물이라 공개 GitHub에 올리지 않는다 |
| 라벨 정의 | 일반 무릎 MRI 판독 기준으로 작성 | 대회 공식 정의는 Kaggle Data/Discussion 탭 확인 필요 |

## 3. 결과
- 테스트 42 passed (프롬프트 3, 파싱 4, 추출 2 추가), `ruff check`·`ruff format --check` 통과
- smoke-test(가짜 생성기, CPU): 추출 → 두 해석 채점 → 파일 5개 생성, 종료 코드 0
- **2차 실행 성공 (run `20261003-084617-8e33d4`, RTX 4090, Pod 약 9분, 약 $0.11)**: Qwen2.5-7B-Instruct,
  4,407 report 추출 290.5초, 파싱 실패 5건(0.1%). 결과물은 HF 비공개 `runs/20261003-084617-8e33d4/`
  (`pseudo_labels.csv`, `raw_responses.csv`, `metrics.json`) — 대회 데이터 파생물이라 저장소에는 올리지 않는다

  58개 정답 기준 macro (비교: 영어 키워드 매칭 AUC 0.547):

  | 해석 | coverage | accuracy | sensitivity | specificity | AUC |
  |------|----------|----------|-------------|-------------|-----|
  | **언급 없음 = 음성 (채택)** | 0.960 | 0.818 | 0.799 | 0.806 | **0.802** |
  | 언급 없음 = 판단 보류 | 0.859 | 0.804 | 0.811 | 0.776 | 0.793 |

  라벨별 AUC (채택 해석): MCL 0.94, Baker's 0.89, Medial OA 0.88, ACL 0.86, Medial Meniscus 0.86,
  Lateral Meniscus 0.81, Lateral OA 0.81, Fracture 0.80, Contusion 0.76, **PF OA 0.70, Synovitis 0.68, Effusion 0.65**
  - Effusion: 민감도 0.94 / 특이도 0.36 → 소량(생리적) 삼출까지 양성으로 잡는 경향
  - PF OA: 민감도 0.48 → 슬개대퇴 연골 변성을 놓침
  - Synovitis: 민감도·특이도 모두 0.6–0.7

**1차 실행 실패 (run `20261003-083742-68db3c`, RTX 4090, 약 4분, 약 $0.05)**
- 증상: `RuntimeError: Engine core initialization failed`. 모델 로드(14.3 GiB)와 KV 캐시 측정까지는 정상
- 원인: vLLM 0.30이 설치한 flashinfer 0.6.18이 샘플링 커널을 실행 시점에 시스템 nvcc로 JIT 컴파일하는데,
  템플릿 이미지(`runpod/pytorch:2.4.0 ... cuda12.4.1`)의 nvcc 12.4가 `--compress-mode=size` 옵션을 몰라
  `nvcc fatal: Unknown option` → ninja 빌드 실패
- 확인: vLLM 0.30 소스에서 `VLLM_USE_FLASHINFER_SAMPLER=0`이면 `flashinfer_sampler_supported()`가 False가 되어
  PyTorch 샘플링 경로(`apply_top_k_top_p` + `gumbel_sample`)를 쓴다. 어텐션은 실패 전 프로파일링에서 이미 정상 실행됨
- 수정: `vllm_generate()`가 vLLM import 전에 `VLLM_USE_FLASHINFER_SAMPLER=0` 설정 (회귀 테스트 추가)
- 대안(미적용): CUDA 12.8 이상 이미지로 템플릿 교체 — 이미지 이름·호스트 드라이버 확인이 더 필요

## 4. 미해결 문제
- **실행 승인 대기**: train.csv(report 원문 포함)를 HF 비공개 데이터셋에 올리는 것과 RunPod 실행
- **컨테이너 디스크**: vLLM + 새 torch + 7B 모델(약 15GB)은 기본 템플릿 30GB에 빠듯하다 → 디스크를 키운 템플릿을 따로 만들고 이 작업 폴더의 `.harness.env`에서만 쓴다 (전역 `.env`는 그대로)
- 58개 채점의 한계와 언어 편중(라벨 있는 58개 중 프랑스어 0개)은 PR #7 핸드오프와 같다

## 5. 다음 작업자가 할 일
1. exp002: 4,349개 pseudo-label(언급 없음 = 음성 해석) + 58개 정답으로 학습. 전체 캐시(50 GiB)를 Pod에 올리는 방법 결정 필요
   (HF 데이터셋 vs RunPod Network Volume)
2. 약한 라벨(Effusion·PF OA·Synovitis) 프롬프트 개선: 생리적 소량 삼출은 음성, 슬개대퇴 연골 변성 용어(chondromalacia 등) 명시.
   단 58개로 프롬프트를 계속 맞추면 이 58개에 과적합된다 — 고친 뒤 한 번만 재채점
3. 재실행 시 `raw_responses.csv`가 있으면 LLM을 다시 돌리지 않고 `parse_response` 해석만 바꿀 수 있다
