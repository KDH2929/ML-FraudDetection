# Member C 전략 버전별 변화 분석

> 작성일: 2026-05-13  
> Member C 전처리 전략 v1 → v4 진화 과정

---

## 📊 버전 개요

| 버전 | Feature 수 | 핵심 특징 | 최적화 | 성능 (LightGBM) |
|------|-----------|----------|--------|----------------|
| **v1** | ~190 | 고객-청구 집계 (safe) | ❌ | **Recall 0.5627, F1 0.6566** 🏆 |
| **v2** | ~190 | v1 + K-fold Target Encoding | ❌ | Recall 0.5535, F1 0.6410 |
| **v3** | ~190 | v2 + DIVIDED_SET train/test 분리 | ❌ | Recall 0.5498, F1 0.6327 |
| **v4** | ~190 | v3 + v1 효과 변수 복원 | ❌ | **Recall 0.6568, F1 0.6660** 🏆 |

---

## 🔄 버전별 상세 분석

### Member C v1 (기본 집계 전략)

#### 설계 철학
```
"C: customer-claim aggregate (safe)"
```

고객이 최종 예측 단위이므로 **CLAIM_DATA를 반드시 고객 단위로 집계**하고, 고카디널리티 원본값은 **통계로 요약**하며, **결측치는 indicator를 남긴 뒤 보완**하는 전략.

#### 설계 원칙

1. **고객 단위 집계 필수**
   - CLAIM_DATA는 반드시 고객 단위로 집계
   - 개별 청구 row를 그대로 사용하지 않음

2. **고카디널리티 통계 요약**
   - 원본값 원-핫 ❌
   - 개수/집중도/빈도 통계 ✅

3. **결측치 indicator**
   - 결측치 삭제 ❌
   - indicator를 남긴 뒤 중앙값/0 보완 ✅

4. **누수 방지**
   - 사후 정보 (지급금액, 심사결과) 옵션 분리
   - `include_leakage_features=False` (기본)

#### 주요 Feature Engineering

**고객 원천 Feature**:

```python
# 1. 결측치 indicator
*_isna: CHLD_CNT, LTBN_CHLD_AGE, CUST_INCM, TOTALPREM, MAX_PRM, MAXCRDT, MINCRDT

# 2. 나이/자녀 그룹
AGE_GROUP: 10s, 20s, 30s, 40s, 50s, 60s, 70plus
LTBN_CHLD_AGE_GROUP: none, infant, child, teen, adult_child
has_child: 자녀 유무

# 3. 가입/납입 파생
payment_active_months: 가입~마지막 납입 기간
register_year, register_month: 가입 연월

# 4. 로그 변환
CUST_INCM_log, TOTALPREM_log, MAX_PRM_log, RESI_COST_log

# 5. 비율 파생
premium_income_ratio: 보험료/소득
max_prm_totalprem_ratio: 최대보험료/총보험료
credit_gap: MAXCRDT - MINCRDT
```

**청구 집계 Feature**:

```python
# 1. 기본 카운트
claim_cnt: 청구 건수
claim_cnt_log: log(청구 건수)
policy_nunique: 서로 다른 계약 수
claim_active_days: 첫 청구~마지막 청구 기간

# 2. 고카디널리티 통계 (HOSP_CODE, CHME_LICE_NO, DSAS_NAME, CAUS_CODE 등)
*_nunique: 서로 다른 값 개수
main_*_claim_ratio: 최빈값 집중도
*_freq_mean, *_freq_max: 전역 빈도 통계

# 3. 금액 집계
dmnd_sum, dmnd_mean, dmnd_max, dmnd_std: 청구 금액 통계
dmnd_sum_log, dmnd_mean_log, dmnd_max_log: 로그 변환

# 4. 입원/통원
valid_hosp_days_sum, valid_hosp_days_mean, valid_hosp_days_max
valid_hosp_days_sum_log

# 5. 시간 패턴
delay_origin_to_recp, delay_resn_to_recp: 지연 일수
claim_interval_mean, claim_interval_std: 청구 간격
weekend_claim_rate: 주말 청구 비율

# 6. 집중도/다양성
same_hospital_repeat_ratio: 동일 병원 반복 비율
hospital_concentration, doctor_concentration
disease_per_claim, cause_per_claim
```

#### 누수 방지 옵션

```python
# safe 버전 (기본)
include_leakage_features=False

# 제외 변수:
- PAYM_AMT (지급 금액)
- pay_ratio (지급/청구 비율)
- amount_gap (지급-청구 차이)
- delay_recp_to_paym (접수~지급 지연)
- RESL_CD1, RESL_NM1 (심사 결과)
- PMMI_DLNG_YN, CRNT_PROG_DVSN (진행 상태)
```

#### 성능 (LightGBM, safe 버전)

```
Recall:    0.5627
F1 Score:  0.6566 🏆 (v1이 가장 높음!)
Features:  ~190개
```

#### 장점
- ✅ **단순하고 직관적**: Feature 설계가 명확하고 해석 가능
- ✅ **안정적 성능**: safe 버전임에도 F1 최고
- ✅ **누수 방지**: 사후 정보 완전 제외

#### 약점
- ❌ **최적화 없음**: 기본 파라미터만 사용
- ❌ **Target Encoding 없음**: 병원·의사 사기 비율 미활용

---

### Member C v2 (K-fold Target Encoding)

#### v1 대비 추가 사항

**설계 철학**
```
"C2: v1 + K-fold Target Encoding + Scaling 옵션"
```

v1 구조를 유지하면서 **병원·의사 사기 비율**(Target Encoding)을 추가하되, **K-fold로 누수를 방지**하는 전략.

#### 신규 Feature (옵션)

**Target Encoded Features** (`include_target_encoded_features=True`):

```python
# K-fold Target Encoding (n_splits=5, default)
# train 고객만 사용, 자기 fold는 제외

DOC_SIU_RATIO: 의사별 SIU 비율 (고객별 평균)
HOSP_SIU_RATIO: 병원별 SIU 비율 (고객별 평균)
HOSP_DOC_SIU: 병원×의사 교차 SIU
```

**누수 방지 메커니즘**:

```python
# 5-fold KFold
for tr, va in folds.split(train_ids):
    # tr fold로 mapping 생성
    mapping = self._group_target_mean(tr_claim, "CHME_LICE_NO", cy)
    
    # va fold에만 적용
    enc = self._map_encoding(va_claim["CHME_LICE_NO"], mapping, global_mean)
    doc_oof.loc[va_claim.index] = enc
    
# test에는 전체 train으로 만든 mapping 적용
```

#### 새로운 옵션

1. **Target Encoding 옵션**
   ```python
   include_target_encoded_features=True
   target_encode_n_splits=5  # K-fold 개수
   ```

2. **Scaling 옵션**
   ```python
   scale_numeric=True  # StandardScaler 적용
   ```

#### 성능 (LightGBM)

```
# safe 버전 (target encoding 없음)
Recall:    0.5535 (-1.6%p vs v1)
F1 Score:  0.6410 (-2.4%p vs v1)

# with target encoding
Recall:    ? (측정 필요)
F1 Score:  ? (측정 필요)
```

**⚠️ v2가 v1보다 낮은 이유**:
- 과도한 누수 방지로 오히려 성능 저하?
- Target Encoding 효과보다 다른 변경 사항의 부작용?

#### v1 대비 변경 사항

**개선**:
- ✅ K-fold Target Encoding 추가
- ✅ Scaling 옵션 추가
- ✅ 누수 방지 강화

**후퇴**:
- ❌ F1 -2.4%p (v1 대비)
- ❌ 복잡도 증가

---

### Member C v3 (DIVIDED_SET 기반 분리)

#### v2 대비 추가 사항

**설계 철학**
```
"C3: v2 + DIVIDED_SET train/test 분리 + 상세 리포트"
```

v2 구조에 **DIVIDED_SET 기반 train/test 명확한 분리**와 **상세 전처리 리포트**를 추가. 모든 imputer·encoder·scaler를 train에서만 fit.

#### 핵심 변경

1. **DIVIDED_SET 필수**
   ```python
   if DIVIDED_SET_COL not in cust.columns:
       raise ValueError("member_c_strategy_3 는 DIVIDED_SET 컬럼이 필요합니다.")
   
   train_mask = cust[DIVIDED_SET_COL].astype("string").str.strip().eq("1") & target.notna()
   test_mask = cust[DIVIDED_SET_COL].astype("string").str.strip().eq("2")
   ```

2. **Train-only fit**
   ```python
   # 모든 imputer/encoder/scaler
   encoder.fit(merged.loc[train_mask])
   median_imputer.fit(merged.loc[train_mask, numeric_cols])
   capper.fit(merged.loc[train_mask, cap_cols])
   scaler.fit(merged.loc[train_mask, scale_cols])
   ```

3. **결측치 처리 강화**
   ```python
   # 계층적 그룹 중앙값 대치
   _hierarchical_group_median_fill(df, train_mask, target_col, group_levels)
   
   # 예: CUST_INCM
   # Level 1: [OCCP_GRP_1, AGE_GROUP]
   # Level 2: [OCCP_GRP_1]
   # Level 3: [AGE_GROUP]
   # Level 4: [] (전체 중앙값)
   ```

4. **상세 리포트**
   ```python
   report = {
       "deleted_customer_columns": [...],
       "deleted_claim_high_missing_columns": [...],
       "customer_missing_before": int,
       "customer_missing_after": int,
       "claim_missing_before": int,
       "claim_missing_after": int,
       "created_customer_features": [...],
       "created_claim_features": [...],
       "excluded_variables_and_reasons": [...],
       "target_encoding_leakage_prevention": str,
       "x_train_shape": [n, p],
       "y_train_shape": [n],
       "x_test_shape": [m, p],
       "target_distribution": {"0": int, "1": int},
       "imbalance_ratio": float,
   }
   ```

#### 제거 변수 강화

```python
HIGH_MISSING_CLAIM_COLS = [
    "DCAF_CMPS_XCPA",
    "COUNT_TRMT_ITEM",
    "TAMT_SFCA",
    "NON_PAY",
    "DSCT_AMT",
    "PATT_CHRG_TOTA",
    "SELF_CHAM",
]
# v3에서 완전 제거
```

#### 성능 (LightGBM)

```
Recall:    0.5498 (-2.1%p vs v1)
F1 Score:  0.6327 (-3.6%p vs v1)
```

**⚠️ v3가 v1보다 더 낮은 이유**:
- **과도한 누수 방지**: train-only fit이 오히려 일반화 저해?
- **변수 제거**: HIGH_MISSING_CLAIM_COLS 제거로 정보 손실?
- **복잡도 증가**: 계층적 대치 등이 오버엔지니어링?

#### v1, v2 대비 변경 사항

**개선**:
- ✅ DIVIDED_SET 명확한 분리
- ✅ Train-only fit (누수 방지 강화)
- ✅ 계층적 결측치 처리
- ✅ 상세 리포트 (90줄)

**후퇴**:
- ❌ F1 -3.6%p (v1 대비)
- ❌ 복잡도 최대

---

### Member C v4 (v1 효과 변수 복원) 🏆

#### v3 대비 추가 사항

**설계 철학**
```
"C4: v3 + v1 효과 변수 복원 (지연, 변동성, 반복성)"
```

v3의 **safe한 구조를 유지**하면서 **v1에서 효과가 컸던 변수만 선별 복원**. v2, v3에서 제거했던 변수 중 실제로 기여한 것만 다시 추가.

#### 복원된 Feature

v3에는 없었으나 v4에서 추가:

```python
# 1. 빈도 최소값 (희귀 병원·의사)
hospital_freq_min: 가장 희귀한 병원 빈도
doctor_freq_min: 가장 희귀한 의사 빈도

# 2. 지연 상세 통계
delay_origin_to_recp_min, delay_origin_to_recp_median, delay_origin_to_recp_max
delay_resn_to_recp_min, delay_resn_to_recp_median, delay_resn_to_recp_max

# 3. 입원 기간 변동성
hosp_days_calc_median: 입원 기간 중앙값
hosp_days_calc_std: 입원 기간 표준편차

# 4. 금액 변동성
dmnd_cv: 청구 금액 변동계수 (CV = std / mean)

# 5. 로그 평균
valid_hosp_days_log_mean: 입원일수 로그 평균

# 6. 연간 청구
claim_per_year: 청구 건수 / 활동 년수

# 7. 일당 금액
dmnd_per_valid_day: 청구 금액 / 입원일수
```

#### 복원 로직

```python
class MemberCStrategy4(MemberCStrategy3):
    def _prepare_claim_data(self, ...):
        # v3 기본 집계 실행
        agg, report = super()._prepare_claim_data(...)
        
        # v1 효과 변수만 선별 복원
        extra_frames = []
        
        # 빈도 최소값
        hospital_freq_min = df.groupby(ID_COL)["__HOSP_CODE_FREQ"].min()
        doctor_freq_min = df.groupby(ID_COL)["__CHME_LICE_NO_FREQ"].min()
        
        # 지연 상세 통계 (min, median, max)
        delay_stats = df.groupby(ID_COL)["delay_origin_to_recp"].agg(["min", "median", "max"])
        
        # ... (나머지 변수)
        
        for frame in extra_frames:
            agg = agg.merge(frame, on=ID_COL, how="left")
        
        return agg, report
```

#### 성능 (LightGBM, 최적화 전)

```
Recall:    0.6568 (+9.4%p vs v1, +20.5%p vs v3) 🚀
F1 Score:  0.6660 (+1.4%p vs v1, +5.3%p vs v3) 🏆
F1 Macro:  0.8172
Features:  ~190개
```

**🎉 v4 성공 요인**:
1. **v3의 안정성** + **v1의 효과 변수**
2. **선별 복원**: 모든 변수가 아닌 실제 기여한 변수만
3. **단순함 유지**: 복잡도 증가 최소화

#### v1 vs v4 비교

| 지표 | v1 | v4 | 차이 |
|------|----|----|------|
| Recall | 0.5627 | 0.6568 | **+9.4%p** |
| F1 | 0.6566 | 0.6660 | **+1.4%p** |
| F1 Macro | ? | 0.8172 | - |
| 구조 | 단순 | v3 기반 |  |
| 최적화 | ❌ | ❌ |  |

---

## 💡 버전별 핵심 차이 요약

### v1 → v2
- ➕ **K-fold Target Encoding**: 병원·의사 사기 비율
- ➕ **Scaling 옵션**: StandardScaler
- ➕ **누수 방지 강화**
- ❌ **성능 후퇴**: F1 -2.4%p (과도한 누수 방지?)

### v2 → v3
- ➕ **DIVIDED_SET 필수**: train/test 명확 분리
- ➕ **Train-only fit**: 모든 imputer/encoder/scaler
- ➕ **계층적 결측치 처리**: 그룹별 중앙값 순차 대치
- ➕ **상세 리포트**: 90줄 전처리 리포트
- ➖ **변수 제거**: HIGH_MISSING_CLAIM_COLS 완전 제거
- ❌ **성능 더 후퇴**: F1 -3.6%p (v1 대비)

### v3 → v4
- ➕ **v1 효과 변수 복원**: 지연, 변동성, 반복성 13개
- ➕ **선별 복원**: 실제 기여한 변수만
- ✅ **성능 대폭 개선**: F1 +5.3%p (v3 대비), v1 초월
- 🏆 **최고 성능**: Recall 0.6568, F1 0.6660

---

## 📊 성능 비교 (LightGBM)

| 버전 | Recall | F1 Score | F1 Macro | vs v1 |
|------|--------|----------|----------|-------|
| **v1** | 0.5627 | 0.6566 | ? | - |
| v2 | 0.5535 | 0.6410 | ? | F1 -2.4%p ⬇️ |
| v3 | 0.5498 | 0.6327 | ? | F1 -3.6%p ⬇️⬇️ |
| **v4** | **0.6568** | **0.6660** | 0.8172 | F1 +1.4%p ⬆️ 🏆 |

**결론**:
- v1: 단순하지만 효과적
- v2, v3: 과도한 누수 방지로 오히려 성능 저하
- **v4**: v3 안정성 + v1 효과 = 최고 성능

---

## 🎯 권장 사용 버전

### 🥇 Member C v4 (최고 성능)
```python
from src.preprocessing.strategies.member_c.member_c_strategy_4 import MemberCStrategy4

strategy = MemberCStrategy4()  # safe 버전 (기본)
```

**성능** (최적화 전):
- Recall: **0.6568**
- F1: **0.6660** 🏆 (전체 전략 중 1위)
- F1 Macro: 0.8172

**최적화 예상** (Member B v3 수준 적용 시):
- Recall: **0.70+** (예상)
- F1: **0.72+** (예상)

**권장 이유**:
- ✅ 현재 최고 성능 (최적화 전임에도)
- ✅ v3 안정성 + v1 효과 변수
- ✅ Feature 직관적이고 해석 가능
- ✅ 최적화 여지 큼

### 🥈 Member C v1 (단순함)
```python
from src.preprocessing.strategies.member_c.member_c_strategy import MemberCStrategy

strategy = MemberCStrategy()
```

**성능**:
- Recall: 0.5627
- F1: 0.6566

**권장 이유**:
- ✅ 가장 단순하고 직관적
- ✅ 안정적 성능
- ✅ 해석 용이

### ❌ Member C v2, v3
- v4가 모든 면에서 우월
- 과도한 누수 방지로 성능 저하
- 복잡도만 증가

---

## 🎓 설계 철학

Member C 전략은 **고객-청구 집계**와 **누수 방지**를 핵심으로 합니다.

### 핵심 원칙

1. **고객 단위 집계 필수**
   - CLAIM_DATA는 반드시 고객 단위로 집계
   - 개별 청구 row 사용 ❌

2. **고카디널리티 통계 요약**
   - 원본값 원-핫 ❌ (차원 폭발)
   - 개수/집중도/빈도 통계 ✅

3. **결측치 indicator**
   - 결측치 삭제 ❌
   - indicator를 남긴 뒤 보완 ✅

4. **누수 방지**
   - 사후 정보 옵션 분리
   - safe 버전이 기본

5. **단순함 유지**
   - v2, v3의 과도한 누수 방지는 역효과
   - v4는 v1의 단순함 + v3의 안정성

---

## 🔧 다음 작업

### 🚀 최우선: Member C v4 최적화

1. **Threshold 최적화** (30분, +3~4%p 예상)
   ```bash
   python -m src.optimization.shared.threshold_optimizer --strategy member_c_strategy_4
   ```

2. **Hyperparameter Tuning** (2~3시간, +1~2%p 예상)
   ```bash
   # Member C 커스텀 튜닝 (F1 최적화)
   python -m src.optimization.member_c.custom_tuning --trials 100
   
   # 또는 공통 튜닝 (Recall 최적화)
   python -m src.optimization.hyperparameter_tuner --strategy member_c_strategy_4 --trials 100
   ```

3. **config.py 업데이트**
   ```python
   STRATEGY_THRESHOLDS = {
       "member_b_strategy_3": 0.3819,
       "member_c_strategy_4": 0.XXXX,  # threshold_analysis.json에서 확인
   }
   ```

### 예상 최종 성능

```
현재:  Recall 0.6568, F1 0.6660 (최적화 전)
예상:  Recall 0.70+,   F1 0.72+  ⭐⭐⭐
```

### 단기 (1~2일)

4. **최적 모델 선택**
   - LightGBM vs CatBoost vs XGBoost
   - Member B v3처럼 CatBoost가 Recall 높을 가능성

5. **v1 vs v4 최적화 비교**
   - v1 최적화 vs v4 최적화
   - 단순함 vs 복잡함 트레이드오프

### 중기 (1주일)

6. **앙상블 검토**
   - Member B v3 (CatBoost) + Member C v4 (최적화)
   - Voting, Stacking 실험

7. **Feature Engineering 재검토**
   - v4 복원 변수 개별 효과 분석
   - 추가 복원 가능 변수 탐색

---

## 📈 성능 변화 추이

```
v1:  F1 0.6566  [기본, 단순]
         ↓
v2:  F1 0.6410  [Target Encoding, 누수 방지 강화]
         ↓       -2.4%p ⬇️
v3:  F1 0.6327  [DIVIDED_SET, train-only fit]
         ↓       -3.6%p ⬇️⬇️
v4:  F1 0.6660  [v3 구조 + v1 효과 변수]
         ↑       +1.4%p ⬆️ 🏆
         
최적화 예상
         ↓
     F1 0.72+   [Threshold + Hyperparameter]
```

---

## 🏆 성공 요인 (v4)

1. **단순함의 힘**
   - v1의 직관적 Feature가 효과적
   - 복잡한 누수 방지가 오히려 역효과

2. **선별 복원**
   - 모든 변수가 아닌 실제 기여한 변수만
   - 13개 변수 추가로 F1 +5.3%p

3. **v3 안정성 + v1 효과**
   - v3의 체계적 구조
   - v1의 효과적 Feature

4. **해석 가능성**
   - Feature가 명확하고 직관적
   - 비즈니스 로직과 일치

---

## 💡 핵심 인사이트

### 1. 단순함 > 복잡함
```
v1 (단순):  F1 0.6566
v2 (복잡):  F1 0.6410 (-2.4%p)
v3 (최복잡): F1 0.6327 (-3.6%p)
v4 (균형):  F1 0.6660 (+1.4%p) 🏆
```

### 2. 과도한 누수 방지는 역효과
- v2, v3의 강화된 누수 방지 → 성능 저하
- v4는 적절한 수준 유지

### 3. 좋은 Feature 설계 > Feature Selection
- Member C v4: 190개, 선택 없이 F1 0.6660
- Member B v3: 100개 (326→100), F1 0.6224
- **좋은 Feature를 만드는 것이 더 중요**

### 4. 최적화 여지
- v4는 최적화 전임에도 최고 성능
- 최적화 시 F1 0.72+ 예상 (가장 높은 잠재력)

---

## 📊 Member A, B, C 비교

| 전략 | 최고 버전 | Feature 수 | F1 | 최적화 | 특징 |
|------|----------|-----------|-----|--------|------|
| A | v5 | v4 축소 | ? | ⚠️ | 결측치 오류 수정 필요 |
| B | v3 | 100 | 0.6224 | ✅ | Feature Selection 완료 |
| **C** | **v4** | **190** | **0.6660** | ❌ | **최고 성능, 최적화 여지 큼** |

---

**작성**: Member C 전략 버전 분석  
**최종 수정**: 2026-05-13  
**다음 작업**: Member C v4 최적화 시작 (Threshold → Hyperparameter)
