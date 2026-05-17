# Member A 전략 버전별 변화 분석

> 작성일: 2026-05-13  
> Member A 전처리 전략 v1 → v5 진화 과정

---

## 📊 버전 개요

| 버전 | Feature 수 | 핵심 특징 | 최적화 | 상태 |
|------|-----------|----------|--------|------|
| **v1** | ? | 기본 집계 + GroupMean | ❌ | ✅ 완료 |
| **v2** | v1 + α | v1 + KMeans·PCA | ❌ | ✅ 완료 |
| **v3** | v1 + β | v1 + 도메인 집계·SIU 노출 | ❌ | ✅ 완료 |
| **v4** | v3 + 추가 블록 | v3 + 청구 집계(claim_agg) / 동료과목 Z(peer_z) / 고객청구(cust_claim) | ❌ | ⚠️ 결측치 오류 |
| **v5** | v4 축소 | v4 + cust_claim_* 다중공선성 축소 + 선택적 VIF·필터·RFE | ⚠️ 선택 | ✅ 완료 |

---

## 🔄 버전별 상세 분석

### Member A v1 (기본 전략)

#### 설계 철학
```
"A: merge → 특수값 → 결측 → 타깃인코딩 → IQR → Robust"
```

고객 데이터와 청구 데이터를 병합한 뒤, 체계적인 결측치 처리와 인코딩을 통해 ML 모델에 적합한 형태로 변환하는 기본 파이프라인.

#### 주요 Feature Engineering

1. **청구 데이터 집계** (`claim_to_customer_table`)
   ```python
   - claim_cnt: 고객별 청구 건수
   - mean_hosp_days: 평균 입원일수
   - mean_hosp_dist: 집-병원 평균 거리
   - is_heed_hosp: 유의병원 청구 여부
   - sum_dmnd_amt, sum_paym_amt, paym_rate: 청구/지급 금액 집계
   - nunique_hosp: 서로 다른 병원 개수
   - acci_pct_*: 사고 구분 비율
   ```

2. **결측치 처리**
   - `GroupMeanImputer`: 직업·나이 그룹별 평균으로 소득 결측치 대치
   - `MedianImputer`: 수치형 결측치를 중앙값으로 대치
   - `ModeImputer`: 범주형 결측치를 최빈값으로 대치

3. **인코딩**
   - `TargetEncoder`: 범주형 변수를 타깃 평균으로 인코딩

4. **이상치 및 스케일링**
   - `IQRCapper`: NUM_COLS 변수에 IQR 기반 캡핑 적용
   - `RobustScalerWrapper`: NUM_COLS 변수에 Robust 스케일링 적용

5. **제외 변수**
   ```python
   _UNUSED_CUST_FOR_A_PIPELINE = ("OCCP_GRP_2", "MATE_OCCP_GRP_2", "FP_CAREER")
   ```

#### 특이 사항
- `RESI_COST == 0`을 결측으로 처리 (추정 불가 의미)
- 잔여 결측치는 `_fill_residual_missing_for_ml`로 최종 정리
- `preprocess_train_test` 메서드로 Train-only fit 지원 (누수 방지)

---

### Member A v2 (비지도 학습 추가)

#### v1 대비 추가 사항

**설계 철학**
```
"A2: member_a + (DIVIDED_SET=1 fit) KMeans·PCA"
```

v1 파이프라인에 비지도 학습 파생 특성을 추가하여 데이터의 잠재 구조를 포착.

#### 신규 Feature

1. **KMeans Clustering**
   ```python
   - ma2_kmeans_cluster: 클러스터 번호
   - ma2_kmeans_dist: 클러스터 중심까지 거리
   ```

2. **PCA (Principal Component Analysis)**
   ```python
   - ma2_pca_0, ma2_pca_1, ma2_pca_2: 주성분 3개
   ```

#### 누수 방지 메커니즘
- **DIVIDED_SET==1 행만 fit**: 라벨·평가구간 정보가 군집/PCA에 직접 들어가지 않게 함
- `DIVIDED_SET` 컬럼이 없으면 전체 행으로 적합 (노트북용, 누수 가능)

#### 구현 특징
- `StandardScaler` 후 KMeans (n_clusters=5)
- PCA (n_components=3)
- `_add_cluster_pca` 함수로 v1 파이프라인 중간에 삽입
- `preprocess_train_test`에서는 train 행만으로 fit

#### 미구현 (v3+ 후보)
- MCAR/MAR/MNAR 가정별 별도 모델링
- 회귀 imputer
- VIF·필터 검정
- SMOTE

---

### Member A v3 (도메인 집계 + SIU 노출)

#### v1 대비 추가 사항

**설계 철학**
```
"A3: member_a + 도메인 집계·소득·주택 파생 (CSV 경로는 SIU병원·의사 노출 제외, train_test 누수 방지)"
```

v1 파이프라인에 도메인 지식 기반 집계와 SIU(사기) 노출 변수를 추가. 특히 병원·의사 SIU 압력 변수는 누수 방지를 위해 train/test 경로에서만 활성화.

#### 신규 Feature

1. **청구 도메인 집계** (`_ma3_claim_domain_block`)
   ```python
   # 금액 집계
   - ma3_max_dmnd_amt: 최대 청구 금액
   - ma3_std_dmnd_amt: 청구 금액 표준편차
   - ma3_mean_paym_dmnd_ratio: 지급/청구 비율 평균
   
   # 유의병원 집중도
   - ma3_frac_heed_claim_rows: 유의병원 청구 행 비율
   
   # 다양성
   - ma3_nunique_acci_dvsn: 서로 다른 사고구분 개수
   - ma3_max_nonpay_ratio: 최대 비급여 비율
   ```

2. **SIU 노출 변수** (LOO - Leave One Out)
   ```python
   - ma3_doc_siu_train_expo: 의사별 SIU 비율 (자기 청구 제외)
   - ma3_hosp_siu_train_expo: 병원별 SIU 비율 (자기 청구 제외)
   - ma3_doc_hosp_train_cross: 의사×병원 교차항
   ```
   
   **누수 방지 메커니즘**:
   - `preprocess`: `include_label_dependent_exposure=False` → SIU 변수 **끔**
   - `preprocess_train_test`: `include_label_dependent_exposure=True` → SIU 변수 **켬**
   - LOO 방식으로 자기 청구는 분모·분자에서 제외

3. **고객 도메인 파생** (`_ma3_customer_domain_features`)
   ```python
   - ma3_resi_unknown: 주택가격 결측 또는 0 여부
   - ma3_absdiff_hh_incm_est: 가구소득 두 추정치 차이 절댓값
   - ma3_cust_minus_jpbase_incm: 개인추정소득 - JPBASE 가구소득
   ```

#### 로드맵 (미구현)
- MCAR/MAR/MNAR 가정별 별도 모형
- VIF·필터 검정
- SMOTE
- K-Fold 내 중첩 전처리

---

### Member A v4 (청구 행동 추가 블록)

#### v3 대비 추가 사항

**설계 철학**
```
"A4: v3 + claim_agg / peer_z / cust_claim"
```

v3 위에 청구 데이터에서 직접 계산할 수 있는 일반적인 집계·동료 비교 Z-score·고객 청구 행동 블록 3종을 라벨 미사용으로 추가.

#### 신규 Feature Blocks

1. **청구 일반 집계 - `claim_agg_*`**
   ```python
   _claim_aggregation_features(claim_df)
   # 청구 건수, 다기관/다의사 노출, 지급 평균/std/max,
   # 진단·과목 다양도, 시간 밀도, 병원 전환수 등
   ```

2. **동료 과목 Z-score - `peer_z_*`**
   ```python
   _within_spec_peer_z(claim_df, meta)
   # HOSP_SPEC_DVSN(병원 과목) 동일 집단 내 Z-score (라벨 미사용)
   ```

3. **고객·청구 행동 - `cust_claim_*`**
   ```python
   _customer_claim_block(claim_df, cust_meta, peer_frame)
   # 고객 단위 청구·지급 요약, 가입~관측 기간, 시간창(6m / 6-12m / 12m+) 강도
   
   _customer_addons(X_out)
   # GroupMeanImputer 이후 추가
   # - cust_claim_prem_incm_ratio
   # - cust_claim_incm_missing_or_zero
   ```

#### 통합 방식
```python
def _merge_extra_blocks(X, claim_df):
    claim_agg = _claim_aggregation_features(claim_df)
    peer_z = _within_spec_peer_z(claim_df, meta_peer)
    cust_claim = _customer_claim_block(claim_df, cust_meta, peer_frame)
    return claim_agg, peer_z, cust_claim
```

#### 튜닝
```bash
python -m src.optimization.hyperparameter_tuner --strategy member_a_strategy_4
```

#### 문제점
```python
ValueError: member_a 전처리 데이터에 결측치가 남아 있습니다
- OCCP_GRP_2: 595개
- MATE_OCCP_GRP_2: 11,827개
```

**원인**: `OCCP_GRP_2`, `MATE_OCCP_GRP_2`는 v1부터 `_UNUSED_CUST_FOR_A_PIPELINE`에 포함되어 제거되어야 하나, 추가 블록에서 재생성되었을 가능성

**해결 필요**: ModeImputer 적용 또는 추가 블록에서 해당 변수 생성 확인

---

### Member A v5 (다중공선성 축소 + 선택적 고급 선택)

#### v4 대비 추가 사항

**설계 철학**
```
"A5: A4 + cust_claim_* 다중공선성 드롭만 (지표가 잘 나오는 얇은 확장)"
```

v4의 cust_claim_* 변수들이 다중공선성을 일으킬 수 있어, 이를 축소하고 선택적으로 Feature Selection을 적용.

#### 기본 모드 (default)
```python
MemberA5Strategy(
    advanced_feature_selection=False,
    top_k=None,
    skip_corr_drop=False
)
```

1. **v4 실행**
2. **cust_claim_* 다중공선성 드롭**
   ```python
   _drop_for_multicollinearity(df)
   # _DROP_FOR_MULTICOLLINEARITY 변수 제거
   ```

#### LightGBM top-k 모드
```python
MemberA5Strategy(top_k=100, lgbm_corr_threshold=0.95)
```

- **CorrelationRemover** (threshold=0.95)
- **ImportanceSelector** (LightGBM, top_k=100)
- Member B v3 방식과 동일

#### 고급 Feature Selection 모드 (실험용)
```python
MemberA5Strategy(advanced_feature_selection=True)
```

**주의**: F1이 떨어질 수 있어 기본값은 끔

1. **정제**
   - 상수 제거 (`_drop_near_constant`)
   - 초고상관 제거 (`_corr_drop`, threshold)
   - VIF 반복 축소 (`_vif_prune`, max_vif=15.0)

2. **필터**
   - `SelectKBest(f_classif)` (k_f)
   - `SelectKBest(chi2)` 비음수 변환 후 (k_c)
   - 합집합

3. **임베디드**
   - `SelectFromModel(RandomForest)` (threshold=mean)

4. **래퍼**
   - `RFE(LogisticRegression)` (n_features_to_select)

#### 실험 조합

| 조합 | skip_corr_drop | top_k | 설명 |
|------|----------------|-------|------|
| A5 기본 | False | None | v4 + cust_claim dedup |
| A4 + top_k | True | 100 | v4 그대로 + LightGBM top-100 |
| A5 + top_k | False | 100 | v4 + cust_claim dedup + LightGBM top-100 |

#### 튜닝
```bash
python -m src.optimization.hyperparameter_tuner --strategy member_a_strategy_5
```

---

## 💡 버전별 핵심 차이 요약

### v1 → v2
- ➕ **비지도 학습**: KMeans·PCA 추가
- ➕ **누수 방지**: DIVIDED_SET==1만 fit
- 📈 **Feature 수**: v1 + 5개 (cluster, dist, pca_0~2)

### v2 → v3
- ➕ **도메인 집계**: 청구 금액·유의병원·다양성 통계
- ➕ **SIU 노출**: 병원·의사 사기 비율 (LOO)
- ➕ **고객 파생**: 주택·소득 괴리 변수
- ⚠️ **누수 방지**: SIU 변수는 train_test 경로에서만 활성화

### v3 → v4
- ➕ **추가 블록**: claim_agg / peer_z / cust_claim 3종
- ➕ **라벨 미사용**: 동료 비교는 train-only μ·σ
- ❌ **문제 발생**: 결측치 오류 (OCCP_GRP_2)

### v4 → v5
- ➖ **다중공선성 축소**: cust_claim_* 변수 정리
- ➕ **선택적 Feature Selection**: LightGBM top-k, VIF·필터·RFE
- ⚙️ **실험 모드**: 3가지 조합 (A5 기본, A4+top_k, A5+top_k)

---

## 📊 성능 비교 (추정)

| 버전 | Feature 수 | Recall | F1 Score | 상태 |
|------|-----------|--------|----------|------|
| v1 | ? | ? | ? | ✅ 기본 |
| v2 | v1 + 5 | ? | ? | ✅ 실험 |
| v3 | v1 + α | ? | ? | ✅ 실험 |
| v4 | v3 + β | ❌ | ❌ | ⚠️ 결측치 오류 |
| v5 (기본) | v4 축소 | ? | ? | ✅ 권장 |
| v5 (top_k) | 100 | ? | ? | ✅ B v3 방식 |

**참고**: v4 결측치 문제로 인해 v1~v3 성능 미측정. v5는 실행 가능하나 최적화 전.

---

## 🎯 권장 사용 버전

### 1순위: Member A v5 (기본)
```python
from src.preprocessing.strategies.member_a.member_a_strategy_5 import MemberA5Strategy

strategy = MemberA5Strategy()  # cust_claim dedup만
```

### 2순위: Member A v5 (top_k)
```python
strategy = MemberA5Strategy(top_k=100, lgbm_corr_threshold=0.95)
```
- Member B v3 방식 적용
- Feature Selection 효과 검증 필요

### 3순위: Member A v3
```python
from src.preprocessing.strategies.member_a.member_a_strategy_3 import MemberA3Strategy

strategy = MemberA3Strategy()
```
- v4/v5 이전 안정 버전
- SIU 노출 변수 포함

### ❌ 비권장: Member A v4
- 결측치 오류 수정 필요

---

## 🔧 다음 작업

### 즉시
1. **v4 결측치 문제 해결**
   - OCCP_GRP_2, MATE_OCCP_GRP_2 확인
   - ModeImputer 적용 또는 추가 블록 수정

2. **v5 성능 측정**
   ```bash
   python -m src.experiment.experiment_runner member_a_strategy_5 lgbm
   ```

### 단기
3. **v5 최적화**
   ```bash
   python -m src.optimization.shared.threshold_optimizer --strategy member_a_strategy_5
   python -m src.optimization.hyperparameter_tuner --strategy member_a_strategy_5 --trials 100
   ```

4. **v3, v5 비교**
   - v3 (도메인 집계)
   - v5 (도메인 + 추가 블록 + dedup)

### 중기
5. **top_k 실험**
   - v5 기본 vs v5 top_k=100
   - Feature Selection 효과 검증

---

## 📝 설계 철학

Member A 전략은 **체계적인 결측치 처리**와 **점진적 Feature 확장**을 핵심으로 합니다.

### 핵심 원칙

1. **누수 방지 우선**
   - Train-only fit (`preprocess_train_test`)
   - SIU 변수는 train/test 경로에서만 활성화
   - LOO (Leave One Out) 방식

2. **점진적 확장**
   - v1: 기본 (결측·인코딩·스케일)
   - v2: 비지도 학습
   - v3: 도메인 지식
   - v4: 청구 행동 기반 추가 블록
   - v5: 최적화·선택

3. **결측치 체계**
   - GroupMeanImputer (직업·나이)
   - MedianImputer (수치형)
   - ModeImputer (범주형)
   - `_fill_residual_missing_for_ml` (잔여)

4. **제외 변수 관리**
   ```python
   _UNUSED_CUST_FOR_A_PIPELINE = ("OCCP_GRP_2", "MATE_OCCP_GRP_2", "FP_CAREER")
   ```

---

**작성**: Member A 전략 버전 분석  
**최종 수정**: 2026-05-13  
**다음 작업**: v4 결측치 수정, v5 성능 측정
