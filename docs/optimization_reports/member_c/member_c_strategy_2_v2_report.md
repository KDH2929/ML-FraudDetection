# member_c_strategy_2 (v2) 최적화 보고서
**생성 날짜**: 2026-05-12 21:25:05
**전략 ID**: `member_c_strategy_2`
**이전 버전**: `member_c`

---

## 성능 비교

### 최종 성능 (LightGBM)

| 지표 | member_c | member_c_strategy_2 | 개선량 | 개선률 |
|------|------|------|--------|--------|
| **F1-score** | 0.6516 | **0.6410** | -0.0106 | **-1.6%** |
| **Recall** | 0.5590 | **0.5535** | -0.0055 | **-1.0%** |
| **F1-macro** | 0.8116 | 0.8058 | -0.0058 | -0.7% |

## Feature 구성

**전체 컬럼 수**: 176개
**추가된 Feature**: -72개

### Feature 그룹별 개수

| 그룹 | member_c | member_c_strategy_2 | 변화 |
|------|------|------|------|
| C_claim_agg | 124개 | 76개 | -48개 (제거) |
| C_cust_tab | 26개 | 20개 | -6개 (제거) |
| C_isna | 7개 | 9개 | +2개 (신규) |
| C_log | 23개 | 10개 | -13개 (제거) |
| C_onehot | 65개 | 59개 | -6개 (제거) |

## 실행 방법

### 전처리만 실행
```bash
python run_experiment.py --strategy member_c_strategy_2 --prepare-only
```

### 모델 학습 및 평가
```bash
# 단일 모델 (LightGBM)
python run_experiment.py --strategy member_c_strategy_2 --model lgbm

# 전체 모델 비교
python run_experiment.py --strategy member_c_strategy_2 --model all
```
