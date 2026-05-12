# member_b (v1) 최적화 보고서
**생성 날짜**: 2026-05-12 17:39:36
**전략 ID**: `member_b`

---

## 성능 비교

### 최종 성능 (LightGBM)

| 지표 | member_b |
|------|------|
| **F1-score** | **0.5541** |
| **Recall** | **0.4539** |
| **F1-macro** | 0.7598 |

## Feature 구성

**전체 컬럼 수**: 46개

### Feature 그룹별 개수

| 그룹 | member_b |
|------|------|
| DEV | 6개 |
| BURST | 7개 |
| GRAPH | 8개 |

## 실행 방법

### 전처리만 실행
```bash
python run_experiment.py --strategy member_b --prepare-only
```

### 모델 학습 및 평가
```bash
# 단일 모델 (LightGBM)
python run_experiment.py --strategy member_b --model lgbm

# 전체 모델 비교
python run_experiment.py --strategy member_b --model all
```
