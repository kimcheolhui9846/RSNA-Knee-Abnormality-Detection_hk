## 목적

## 변경 사항

## 실험 결과 (해당 시)
| exp | CV macro AUC | fold std | 라벨별 최저 AUC | LB | 추론 시간 |
|-----|--------------|----------|-----------------|----|-----------|

## 재현 방법
```
python src/train.py --config configs/expXXX.yaml
```

## 리뷰어 체크 포인트
- [ ] 누수 없음 (환자/스터디 단위 fold 분리)
- [ ] 오프라인 추론 가능 (인터넷 의존 없음)
- [ ] 제출 형식 테스트 통과 (`ruff check . && pytest -q`)
- [ ] 9시간 내 추론 추정치 기재
- [ ] 핸드오프 문서 포함 (`handoff/<주제>_Handoff_<번호>.md`)

## 리뷰 이력
- Codex 리뷰: (reviews/PR-XX.md 링크)
- 사람 리뷰: @
