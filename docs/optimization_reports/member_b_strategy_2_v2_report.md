# member_b_strategy_2 (v2) 최적화 보고서
**생성 날짜**: 2026-05-12 17:39:38
**전략 ID**: `member_b_strategy_2`
**이전 버전**: `member_b`

---

## 성능 비교

### 최종 성능 (LightGBM)

| 지표 | member_b | member_b_strategy_2 | 개선량 | 개선률 |
|------|------|------|--------|--------|
| **F1-score** | 0.5541 | **0.5924** | +0.0384 | **+6.9%** |
| **Recall** | 0.4539 | **0.4908** | +0.0369 | **+8.1%** |
| **F1-macro** | 0.7598 | 0.7803 | +0.0205 | +2.7% |

## Feature 구성

**전체 컬럼 수**: 77개
**추가된 Feature**: 31개

### Feature 그룹별 개수

| 그룹 | member_b | member_b_strategy_2 | 변화 |
|------|------|------|------|
| DEV | 6개 | 6개 | - |
| BURST | 7개 | 7개 | - |
| GRAPH | 8개 | 8개 | - |
| AMT | 0개 | 8개 | +8개 (신규) |
| TIME | 0개 | 5개 | +5개 (신규) |
| MED | 0개 | 18개 | +18개 (신규) |

## 실행 방법

### 전처리만 실행
```bash
python run_experiment.py --strategy member_b_strategy_2 --prepare-only
```

### 모델 학습 및 평가
```bash
# 단일 모델 (LightGBM)
python run_experiment.py --strategy member_b_strategy_2 --model lgbm

# 전체 모델 비교
python run_experiment.py --strategy member_b_strategy_2 --model all
```
