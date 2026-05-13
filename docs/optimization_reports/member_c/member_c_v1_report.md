# member_c (v1) 최적화 보고서
**생성 날짜**: 2026-05-12 21:31:17
**전략 ID**: `member_c`

---

## 성능 비교

### 최종 성능 (LightGBM)

| 지표 | member_c |
|------|------|
| **F1-score** | **0.6566** |
| **Recall** | **0.5627** |
| **F1-macro** | 0.8144 |

## Feature 구성

**전체 컬럼 수**: 246개

### Feature 그룹별 개수

| 그룹 | member_c |
|------|------|
| C_claim_agg | 124개 |
| C_cust_tab | 25개 |
| C_isna | 7개 |
| C_log | 23개 |
| C_onehot | 65개 |

## 실행 방법

### 전처리만 실행
```bash
python run_experiment.py --strategy member_c --prepare-only
```

### 모델 학습 및 평가
```bash
# 단일 모델 (LightGBM)
python run_experiment.py --strategy member_c --model lgbm

# 전체 모델 비교
python run_experiment.py --strategy member_c --model all
```
