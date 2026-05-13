# Member B 전략 버전별 변화 분석

> 작성일: 2026-05-13  
> Member B 전처리 전략 v1 → v3 진화 과정

---

## 📊 버전 개요

| 버전 | Feature 수 | 핵심 특징 | 최적화 | 성능 (LightGBM) |
|------|-----------|----------|--------|----------------|
| **v1** | ~43개 | Deviation + Burst + Graph | ❌ | ? |
| **v2** | ~74개 | v1 + Amount + Time + Medical | ❌ | Recall 0.4910, F1 0.5927 |
| **v3** | **100개** | v2 개선 + 신규 7개 + Feature Selection | ✅ | **Recall 0.6144, F1 0.6224** |

---

## 🔄 버전별 상세 분석

### Member B v1 (기본 3개 Feature)

#### 설계 철학
```
"B: Deviation+Burst+Graph"
```

사기 탐지의 핵심 패턴인 **그룹 편차**, **청구 집중**, **네트워크 특성**을 포착하는 3대 Feature로 시작.

#### 핵심 Feature (43개)

1. **DeviationFeature** (그룹별 평균 대비 z-score 편차)
   ```python
   # 그룹: 나이, 직업, 지역, 보험종류 등
   # 변수: 청구 금액, 청구 건수, 입원일수 등
   
   예시:
   - deviation_age_dmnd_amt: 나이 그룹 평균 대비 청구 금액 편차
   - deviation_occp_claim_cnt: 직업 그룹 평균 대비 청구 건수 편차
   ```
   
   **의미**: 같은 그룹 내에서 비정상적으로 높거나 낮은 값 포착

2. **BurstFeature** (짧은 기간 내 청구 집중도)
   ```python
   # 7일, 14일, 30일 윈도우에서 청구 집중도 계산
   
   예시:
   - burst_7d_claim_cnt: 7일 내 최대 청구 건수
   - burst_14d_dmnd_amt: 14일 내 최대 청구 금액
   - burst_concentration: 전체 청구 중 집중 기간 비율
   ```
   
   **의미**: 단기간 집중 청구는 사기 신호일 가능성

3. **GraphFeature** (고객-병원-의사 네트워크)
   ```python
   # 네트워크 중심성, 연결성 지표
   
   예시:
   - graph_customer_degree: 고객이 연결된 병원·의사 수
   - graph_hospital_centrality: 병원의 네트워크 중심성
   - graph_doctor_betweenness: 의사의 매개 중심성
   ```
   
   **의미**: 사기 네트워크는 특정 패턴을 보임 (밀집된 연결, 중심 노드)

#### 전처리 파이프라인

```python
# 1. Feature Engineering (v1 3개)
DeviationFeature → BurstFeature → GraphFeature

# 2. 범주형 인코딩
cat_cols → fillna('MISSING') → TargetEncoder

# 3. 결측치 처리
num_cols → MedianImputer

# 4. 이상치 캡핑
num_cols → QuantileCapper(lower=0.01, upper=0.99)

# 5. 스케일링
num_cols → RobustScalerWrapper
```

---

### Member B v2 (1차 + 2차 Feature)

#### v1 대비 추가 사항

**설계 철학**
```
"B2: Deviation+Burst+Graph+Amount+Time+Medical"
```

v1의 3대 Feature에 **금액 패턴**, **시계열 패턴**, **의료 행태**를 추가하여 사기 탐지의 다양한 측면을 포착.

#### 신규 Feature (v1 대비 +31개, 총 ~74개)

4. **AmountFeature** (청구/지급 금액 패턴)
   ```python
   # 금액 분포, 비율, 이상치
   
   예시:
   - amount_dmnd_mean: 평균 청구 금액
   - amount_dmnd_std: 청구 금액 표준편차
   - amount_dmnd_max_ratio: 최대/평균 비율
   - amount_paym_rate: 지급률 (지급/청구)
   - amount_outlier_cnt: 이상 금액 청구 건수
   ```

5. **TimeFeature** (시계열 청구 패턴)
   ```python
   # 시간대, 요일, 월별 패턴
   
   예시:
   - time_weekend_ratio: 주말 청구 비율
   - time_night_claim_cnt: 야간 청구 건수
   - time_claim_interval_mean: 평균 청구 간격
   - time_claim_interval_std: 청구 간격 표준편차
   - time_first_last_gap: 첫 청구~마지막 청구 기간
   ```

6. **MedicalFeature** (의료 행태 패턴)
   ```python
   # 진단, 처치, 입원 패턴
   
   예시:
   - medical_disease_nunique: 서로 다른 질병 수
   - medical_treatment_diversity: 처치 다양성
   - medical_hospitalization_rate: 입원 비율
   - medical_long_stay_cnt: 장기 입원 건수
   - medical_rare_disease_flag: 희귀 질환 청구 여부
   ```

#### 전처리 파이프라인 (v1과 동일)

```python
# 1. Feature Engineering (v1 3개 + v2 3개)
DeviationFeature → BurstFeature → GraphFeature
→ AmountFeature → TimeFeature → MedicalFeature

# 2~5. v1과 동일
```

#### 성능 (LightGBM, 최적화 전)
```
Recall:    0.4910
F1 Score:  0.5927
F1 Macro:  ?
```

---

### Member B v3 (최적화 완료 버전) ⭐

#### v2 대비 변경 사항

**설계 철학**
```
"B3: v3 최적화 (Feature Selection + Hyperparameter Tuning)"
```

v2의 6대 Feature를 **v2 버전으로 개선**하고, **7개 신규 Feature**를 추가한 뒤, **Feature Selection**으로 326개 → 100개로 축소. 최종적으로 **Hyperparameter Tuning**과 **Threshold 최적화**를 완료한 최강 버전.

#### Phase 1: v2 개선 Feature (_v2 버전)

기존 v2 Feature를 더 세밀하게 재설계:

1. **DeviationFeature_v2**
   ```python
   # 추가 개선
   - 다층 그룹: 나이×직업, 지역×보험종류 교차 그룹
   - 백분위: 그룹 내 상위 몇% 인지
   - 시간 구간별: 최근 3개월, 6개월, 1년 구간별 편차
   ```

2. **BurstFeature_v2**
   ```python
   # 추가 개선
   - 7일/14일 윈도우 추가 (기존 30일만)
   - 변동계수(CV): 집중도의 변동성
   - 급증 패턴: 청구 건수 급증 감지
   ```

3. **GraphFeature_v2**
   ```python
   # 추가 개선
   - 병원-의사 페어: (병원, 의사) 조합의 중심성
   - 희귀도: 네트워크에서 희귀한 연결 패턴
   - 클러스터 계수: 고객의 네트워크 밀집도
   ```

4. **AmountFeature_v2**
   ```python
   # 추가 개선
   - 변동계수(CV): 금액 변동성
   - MAD (Median Absolute Deviation): 이상치에 강한 편차
   - 분위수: 금액 분포의 25%, 75% 등
   ```

5. **TimeFeature_v2**
   ```python
   # 추가 개선
   - 월말/월초 청구 비율: 특정 시점 집중
   - 요일 엔트로피: 요일 분포의 무작위성
   - 계절성: 분기별 청구 패턴
   ```

6. **MedicalFeature_v2**
   ```python
   # 추가 개선
   - 희귀 질병 스코어: 전체 데이터에서 희귀한 질병
   - 질병-처치 정합성: 질병과 처치의 일치도
   - 중복 진단 비율: 동일 질병 반복 청구
   ```

#### Phase 2: v3 신규 Feature (7개)

7. **InteractionFeature** (교차 상호작용)
   ```python
   # Feature 간 곱셈·나눗셈 교차항
   
   예시:
   - inter_high_amount_burst: 고액 × 집중도
   - inter_network_amount: 네트워크 중심성 × 평균 금액
   - inter_deviation_time: 편차 × 시간 패턴
   ```

8. **SequenceFeature** (청구 순서 패턴)
   ```python
   # 시계열 추세, 가속도
   
   예시:
   - seq_claim_trend: 청구 건수 추세 (증가/감소)
   - seq_amount_acceleration: 금액 증가 가속도
   - seq_pattern_change: 청구 패턴 변화 감지
   ```

9. **AnomalyScoreFeature** (이상 탐지 점수)
   ```python
   # Isolation Forest, 왜도/첨도
   
   예시:
   - anomaly_isolation_score: Isolation Forest 점수
   - anomaly_skewness: 청구 분포 왜도
   - anomaly_kurtosis: 청구 분포 첨도
   ```

10. **RatioFeature** (또래 대비 비율)
    ```python
    # 동일 그룹 대비 상대적 위치
    
    예시:
    - ratio_peer_amount: 또래 대비 청구 금액 비율
    - ratio_concentration: 또래 대비 집중도
    - ratio_network_position: 또래 대비 네트워크 위치
    ```

11. **TemporalDecayFeature** (시간 감쇠 가중)
    ```python
    # 최근 청구에 더 높은 가중치
    
    예시:
    - temporal_weighted_amount: 시간 가중 평균 금액
    - temporal_recent_ratio: 최근 청구 비중
    - temporal_decay_score: 시간 감쇠 점수
    ```

#### Phase 3: 전처리 파이프라인 개선

**v3 전용 전처리 순서**:

```python
# 0. 결측치 지시자 생성 (Feature Engineering 전)
MissingIndicator

# 1. Feature Engineering (v2 개선 6개 + v3 신규 7개)
DeviationFeature_v2 → BurstFeature_v2 → GraphFeature_v2
→ AmountFeature_v2 → TimeFeature_v2 → MedicalFeature_v2
→ InteractionFeature → SequenceFeature → AnomalyScoreFeature
→ RatioFeature → TemporalDecayFeature

# 2. 이상치 처리
OutlierFlagger (이상치 flag) → QuantileCapper

# 3. 범주형 인코딩 강화
TargetEncoder + FrequencyEncoder + RarityEncoder (고카디널리티)

# 4. 결측치 2차 처리
MedianImputer

# 5. Feature Selection ⭐ (핵심)
CorrelationRemover (threshold=0.95) → ImportanceSelector (top_k=100)
# 326개 → 100개 (69% 감소)

# 6. 스케일링
RobustScalerWrapper
```

#### Feature Selection 상세

1. **CorrelationRemover** (threshold=0.95)
   - 상관계수 0.95 이상인 변수 쌍 중 하나 제거
   - 다중공선성 완화

2. **ImportanceSelector** (top_k=100)
   ```python
   # LightGBM Feature Importance 기반
   # train 데이터만 사용 (y가 유효한 행만)
   
   valid_mask = y.notna()
   X_train = X_features[valid_mask]
   y_train = y[valid_mask]
   
   importance_selector = ImportanceSelector(top_k=100)
   importance_selector.fit(X_train, y_train)
   ```
   
   - 상위 100개 Feature만 선택
   - 과적합 방지, 학습 속도 향상

#### 최적화 (v3만 적용)

1. **Hyperparameter Tuning**
   ```bash
   python -m src.optimization.hyperparameter_tuner --strategy member_b_strategy_3 --trials 50
   ```
   
   - Optuna TPE Sampler
   - 50 trials (later 100 trials)
   - **Recall 최적화** (사기 놓치면 안 됨)
   - 5-fold Cross-Validation

2. **Threshold 최적화**
   ```bash
   python -m src.optimization.shared.threshold_optimizer --strategy member_b_strategy_3
   ```
   
   - Precision-Recall Curve
   - **F1 maximization**
   - 최적 threshold: **0.3819**

3. **config.py 자동 적용**
   ```python
   STRATEGY_THRESHOLDS = {
       "member_b_strategy_3": 0.3819,
   }
   ```

4. **SMOTE 실험** (최종 제외)
   - 효과 미미로 제외
   - Feature Selection이 더 효과적

#### 성능 (LightGBM, 최적화 완료)

```
Recall:    0.6144 (+12.5%p vs v2)
F1 Score:  0.6224 (+3.0%p vs v2)
F1 Macro:  0.7933
Features:  100개 (326개에서 69% 감소)
Threshold: 0.3819 (최적화)
```

#### 모델별 최고 성능 (v3 전략 적용)

| 모델 | Recall | F1 Score | F1 Macro | 비고 |
|------|--------|----------|----------|------|
| **CatBoost** | **0.6919** 🏆 | 0.6351 | 0.7983 | Recall 최고 |
| XGBoost | 0.6125 | 0.6390 | 0.8030 | |
| Voting | 0.6292 | 0.6374 | 0.8015 | |
| LightGBM | 0.6144 | 0.6224 | 0.7933 | |
| Stacking | 0.7897 | 0.5940 | 0.7702 | Recall 높지만 F1 낮음 |

---

## 💡 버전별 핵심 차이 요약

### v1 → v2
- ➕ **3개 Feature 추가**: Amount, Time, Medical
- 📈 **Feature 수**: 43개 → 74개 (+72%)
- 🎯 **다차원 포착**: 그룹 편차·집중·네트워크 → 금액·시간·의료 추가

### v2 → v3
- ✨ **v2 개선**: 6대 Feature를 _v2 버전으로 업그레이드
- ➕ **7개 신규 Feature**: Interaction, Sequence, Anomaly, Ratio, TemporalDecay
- ➖ **Feature Selection**: 326개 → 100개 (69% 감소)
- ✅ **최적화 완료**: Hyperparameter + Threshold
- 📈 **성능 향상**: Recall +12.5%p, F1 +3.0%p

---

## 📊 성능 비교

| 버전 | Feature 수 | Recall | F1 Score | 최적화 |
|------|-----------|--------|----------|--------|
| v1 | ~43 | ? | ? | ❌ |
| v2 | ~74 | 0.4910 | 0.5927 | ❌ |
| **v3** | **100** | **0.6144** | **0.6224** | ✅ |

**v3 개선폭 (vs v2)**:
- Recall: +12.5%p (0.4910 → 0.6144)
- F1: +3.0%p (0.5927 → 0.6224)

---

## 🎯 권장 사용 버전

### 🥇 Member B v3 (CatBoost) - Recall 우선
```python
from src.preprocessing.strategies.member_b.member_b_strategy_3 import MemberB3Strategy

strategy = MemberB3Strategy()
# CatBoost 모델 사용
```

**성능**:
- Recall: **0.6919** (최고)
- F1: 0.6351
- 사용 시나리오: **사기 놓치면 안 될 때** (FN 최소화 우선)

### 🥈 Member B v3 (LightGBM) - 밸런스
```python
strategy = MemberB3Strategy()
# LightGBM 모델 사용 (기본)
```

**성능**:
- Recall: 0.6144
- F1: **0.6224**
- 사용 시나리오: Recall과 Precision 밸런스

### ❌ Member B v1, v2
- v3가 모든 면에서 우월
- 최적화 미완료
- Feature 수 적음

---

## 🎓 설계 철학

Member B 전략은 **다층 Feature Engineering**과 **Feature Selection**을 핵심으로 합니다.

### 핵심 원칙

1. **다차원 포착**
   - v1: 그룹 편차, 집중, 네트워크
   - v2: 금액, 시간, 의료 추가
   - v3: 교차, 순서, 이상, 비율, 감쇠 추가

2. **Feature Selection 우선**
   - 많이 만들고 (326개)
   - 잘 선택한다 (100개)
   - CorrelationRemover → ImportanceSelector

3. **최적화 체계**
   - Hyperparameter Tuning (Recall 최적화)
   - Threshold 최적화 (F1 maximization)
   - 자동 적용 (config.py, tuning_results.json)

4. **과적합 방지**
   - Feature Selection (69% 감소)
   - Cross-Validation (5-fold)
   - train 데이터만 fit

---

## 🔧 다음 작업

### 단기
1. **Member C v4 최적화 후 비교**
   - Member B v3 (최적화 완료) vs Member C v4 (최적화 예정)
   - 예상: Member C v4 최적화 후 F1 0.72+ 가능

2. **CatBoost 모델 검증**
   - v3 전략 + CatBoost = Recall 0.6919
   - 실무 적용 가능성 검토

### 중기
3. **Feature Engineering 개선**
   - v3 신규 Feature 개별 효과 분석
   - 불필요한 Feature 추가 제거

4. **앙상블 실험**
   - LightGBM + CatBoost Voting
   - Stacking 개선 (현재 F1 낮음)

---

## 📈 Feature 수 변화 추이

```
v1:  ~43개  [Deviation, Burst, Graph]
         ↓
v2:  ~74개  [v1 + Amount, Time, Medical]
         ↓
v3:  326개  [v2 개선 + 신규 7개]
         ↓
     100개  [Feature Selection]
```

**Feature Selection 효과**:
- 과적합 방지 ✅
- 학습 속도 향상 ✅
- 성능 개선 ✅ (Recall +12.5%p, F1 +3.0%p)

---

## 🏆 성공 요인

1. **체계적 Feature 확장**
   - v1 기반 → v2 확장 → v3 완성

2. **Feature Selection의 중요성**
   - 많이 만드는 것보다 잘 선택하는 것이 중요
   - 326개 → 100개로 줄이면서 성능 향상

3. **최적화 완료**
   - Hyperparameter + Threshold
   - 자동 적용 시스템

4. **모델 다양성**
   - LightGBM (밸런스)
   - CatBoost (Recall 최고)
   - 선택의 폭

---

**작성**: Member B 전략 버전 분석  
**최종 수정**: 2026-05-13  
**다음 작업**: Member C v4 최적화, CatBoost 모델 검증
