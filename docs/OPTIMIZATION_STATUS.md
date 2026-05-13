# 전략별 최적화 현황 검토

> 작성일: 2026-05-13  
> Member A, B, C의 각 전략별 최적화 완료 여부 검토

---

## 📊 요약

| Member | 전략 수 | 실험 완료 | 최적화 완료 | 상태 |
|--------|---------|----------|------------|------|
| **Member A** | 1개 (v1) | ❌ | ❌ | **미실행** |
| **Member B** | 3개 (v1~v3) | ✅ | ✅ v3만 | **v3 완료** |
| **Member C** | 4개 (v1~v4) | ⚠️ v1~v2만 | ❌ | **v3~v4 미실행** |

---

## 🔍 상세 현황

### Member A

#### 전략 구현
- ✅ **member_a** (v1): 기본 전처리
  - Claim 집계 (청구 건수, 입원일수, 거리, 금액 등)
  - GroupMeanImputer (직업·나이 그룹 평균)
  - Target Encoding
  - IQR Capping
  - Robust Scaling

#### 실험 현황
- ❌ **실험 미실행**
  - `artifacts/experiment_result_member_a*.csv` 없음
  - 성능 데이터 없음

#### 최적화 현황
- ❌ **최적화 미실행**
  - Hyperparameter Tuning: 없음
  - Threshold 최적화: 없음
  - Feature Selection: 없음
  - SMOTE/샘플링: 없음

#### 권장사항
1. **v1 기본 실험 실행** (우선순위: 높음)
   ```python
   from src.experiment.experiment_runner import run
   run('member_a', 'lgbm')
   ```
2. v1 결과 분석 후 최적화 진행 여부 결정
3. 성능이 낮으면 Member B/C의 Feature Engineering 기법 차용

---

### Member B ⭐ (최적화 완료)

#### 전략 구현
- ✅ **member_b** (v1): Deviation + Burst + Graph
  - DeviationFeature: 그룹별 z-score 편차
  - BurstFeature: 청구 집중도
  - GraphFeature: 네트워크 특성
  
- ✅ **member_b_strategy_2** (v2): v1 + Amount + Time + Medical
  - AmountFeature: 금액 패턴
  - TimeFeature: 시계열 패턴
  - MedicalFeature: 의료 행태
  
- ✅ **member_b_strategy_3** (v3): v2 개선 + v3 신규 + 최적화
  - v2 개선 Feature (6종)
  - v3 신규 Feature (5종): Interaction, Sequence, Anomaly, Ratio, Temporal
  - Feature Selection (326개 → 100개)

#### 실험 현황
- ✅ **v1, v2, v3 모두 실험 완료**
  - v1: Recall 0.4539, F1 0.5541
  - v2: Recall 0.4908, F1 0.5924
  - v3: Recall 0.5830, F1 0.6345 (threshold 0.3819)

#### 최적화 현황 (v3)
- ✅ **Feature Engineering 고도화**
  - v2 개선: 다층 그룹, 백분위, 변동계수
  - v3 신규: 상호작용, 이상 탐지, 비율

- ✅ **Feature Selection**
  - CorrelationRemover (threshold=0.95)
  - ImportanceSelector (top 100)
  - 효과: 326개 → 100개 (69% 감소)

- ✅ **Hyperparameter Tuning**
  - Optuna TPE Sampler, 100 trials
  - 5-fold Stratified CV
  - 파일: `artifacts/member_b_strategy_3_tuning_results.json`
  - 최적 params:
    ```json
    {
      "learning_rate": 0.106,
      "num_leaves": 25,
      "max_depth": 7,
      "min_child_samples": 78,
      "subsample": 0.968,
      "colsample_bytree": 0.901,
      "reg_alpha": 5.08,
      "reg_lambda": 7.15,
      "n_estimators": 447
    }
    ```

- ✅ **Threshold 최적화**
  - F1 maximization
  - 최적 threshold: 0.3819
  - 파일: `artifacts/member_b_strategy_3_threshold_analysis.json`
  - 효과: Recall +8.12%p, F1 +3.95%p

- ⚠️ **SMOTE 실험** (적용 제외)
  - SMOTE, ADASYN, BorderlineSMOTE 비교
  - 결과: Recall +1.4~2.5% (미미), Precision -2.7~4.0%p (부정적)
  - 결론: 효과 대비 부작용이 커서 제외

- ✅ **자동 적용 시스템**
  - `config.py`의 `STRATEGY_THRESHOLDS`에 threshold 등록
  - `experiment_runner.py`에서 tuning 결과 자동 로드

#### 성과
- **v3가 v1 대비 Recall +28.4%p, F1 +14.5% 향상**
- 체계적 최적화 완료
- 상세 문서: `docs/MEMBER_B_STRATEGY_COMPARISON.md` (476줄)

---

### Member C

#### 전략 구현
- ✅ **member_c** (v1): 고객-청구 집계 (safe)
  - MemberCV1FeaturePipeline
  - 고카디널리티 통계 요약
  - 결측치 indicator + 중앙값/0 보완
  - 사후 정보 제외 옵션

- ✅ **member_c_strategy_2** (v2): 고객-청구 집계, 누수/스케일 분리
  - MemberCV2FeaturePipeline
  - Target encoding 계열 누수 방지
  - KFold 기반 타겟 변환
  - StandardScaler 옵션

- ✅ **member_c_strategy_3** (v3): v2 개선, safe 버전 강화
  - MemberCV3FeaturePipeline
  - 청구 반복성, 다양성, 금액, 간격, 경제적 부담
  - 보험 유지 기간 중심
  - train에서만 fit, test는 transform만

- ✅ **member_c_strategy_4** (v4): v3 + v1 효과 변수 복원
  - MemberCV4FeaturePipeline
  - v3 safe 구조 유지
  - v1 지연/변동성/반복성 변수 선별 복원

#### 실험 현황
- ✅ **v1, v2 실험 완료**
  - v1: Recall 0.5627, F1 0.6566 ⭐ (전체 1위)
  - v2: Recall 0.5535, F1 0.6410

- ❌ **v3, v4 실험 미실행**
  - `artifacts/experiment_result_member_c_strategy_3.csv` 없음
  - `artifacts/experiment_result_member_c_strategy_4.csv` 없음

#### 최적화 현황
- ❌ **최적화 미실행**
  - Hyperparameter Tuning: 없음
  - Threshold 최적화: 없음
  - Feature Selection: 없음
  - SMOTE/샘플링: 없음

#### 권장사항
1. **v3, v4 기본 실험 실행** (우선순위: 높음)
   ```python
   from src.experiment.experiment_runner import run
   run('member_c_strategy_3', 'lgbm')
   run('member_c_strategy_4', 'lgbm')
   ```

2. **v1 최적화** (우선순위: 최고)
   - v1이 현재 전체 1위 (F1 0.6566)
   - Member B v3 방식 적용 시 성능 향상 기대
   - 최적화 순서:
     1. Threshold 최적화 (빠르고 효과 큼)
     2. Hyperparameter Tuning (100 trials)
     3. Feature Selection (필요 시)

3. **v3, v4 최적화**
   - 실험 결과 확인 후 최고 성능 전략 선택
   - 선택된 전략에 대해 최적화 진행

---

## 🎯 전체 성능 비교 (현재)

| 순위 | 전략 | Recall | F1 Score | 최적화 | 상태 |
|------|------|--------|----------|--------|------|
| 🥇 | **member_c (v1)** | 0.5627 | **0.6566** | ❌ | **최적화 여지 큼** |
| 🥈 | member_c_strategy_2 (v2) | 0.5535 | 0.6410 | ❌ | - |
| 🥉 | member_b_strategy_3 (v3) | **0.5830** | 0.6345 | ✅ | 최적화 완료 |
| 4 | member_b_strategy_2 (v2) | 0.4908 | 0.5924 | ⚠️ | 일부 tuning |
| 5 | member_b (v1) | 0.4539 | 0.5541 | ❌ | - |
| - | member_a (v1) | - | - | ❌ | **미실행** |

---

## 📋 액션 아이템

### 우선순위 1: Member C v1 최적화 ⭐⭐⭐
- 현재 F1 0.6566 (전체 1위)
- Member B v3 최적화 기법 적용 시 **F1 0.70+ 달성 가능**
- 예상 작업:
  1. Threshold 최적화 (30분)
  2. Hyperparameter Tuning (2~3시간)
  3. Feature Selection (선택, 1시간)

### 우선순위 2: Member C v3, v4 실험
- 구현은 완료, 실험만 실행하면 됨
- v1, v2와 성능 비교
- 최고 성능 전략 선택

### 우선순위 3: Member A v1 실험
- 기본 실험 실행
- 성능 확인 후 최적화 여부 결정

### 우선순위 4: 전체 모델 비교
- 각 전략별로 7개 모델 비교
  - lgbm, rf, logistic, xgboost, catboost, voting, stacking
- 최적 모델 선택

---

## 🚀 최종 권장사항

### 즉시 실행 (오늘)
1. **Member C v1 Threshold 최적화** (30분)
   ```bash
   python -m src.optimization.threshold_optimizer --strategy member_c
   ```

2. **Member C v3, v4 실험** (30분)
   ```python
   from src.experiment.experiment_runner import run
   run('member_c_strategy_3', 'lgbm')
   run('member_c_strategy_4', 'lgbm')
   ```

### 단기 (1~2일)
3. **Member C v1 Hyperparameter Tuning** (2~3시간)
   ```bash
   python -m src.optimization.hyperparameter_tuner --strategy member_c --n-trials 100
   ```

4. **Member A v1 실험** (10분)
   ```python
   from src.experiment.experiment_runner import run
   run('member_a', 'lgbm')
   ```

### 중기 (1주일)
5. **Member C 최고 전략 최적화 완료**
   - Feature Selection
   - 자동 적용 시스템
   - 문서화

6. **전체 모델 비교 및 최종 선택**
   - 각 전략별 7개 모델 비교
   - 최적 조합 선택 (전략 + 모델)

---

## 📊 예상 최종 성능

**Member C v1 최적화 후 예상 성능:**
- 현재: Recall 0.5627, F1 0.6566
- Threshold 최적화: Recall +8~10%p, F1 +3~4%p
- Hyperparameter Tuning: Recall +2~3%p, F1 +1~2%p
- **예상 최종: Recall 0.65~0.67, F1 0.70~0.72** ⭐

**비교:**
- Member B v3 (최적화 완료): Recall 0.5830, F1 0.6345
- Member C v1 (최적화 후 예상): Recall 0.65~0.67, F1 0.70~0.72
- **개선 폭: Recall +7~9%p, F1 +6~8%p**

---

**작성**: 시스템 검토  
**최종 수정**: 2026-05-13  
**다음 검토**: Member C 최적화 완료 후
