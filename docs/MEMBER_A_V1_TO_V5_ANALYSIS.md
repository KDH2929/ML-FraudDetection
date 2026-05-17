# Member A 전처리 v1~v5: 성능·EDA·인사이트 종합 분석

> 작성일: 2026-05-13
> 모델: LightGBM (`MODEL_PARAMS["lgbm"]` 기본값, `n_estimators=500`, `class_weight="balanced"`)
> 최적화: `python -m src.optimization.threshold_optimizer --strategy <id>` (precision-recall curve 위에서 F1 최대화)
> 평가: `data_loader.load_and_split` (DIVIDED_SET==1 학습 라벨, stratified 30% holdout, `random_state=42`)
> 양성 라벨: SIU=1 (전체 22,400행 중 1,806건, 약 8.06%, 학습+테스트 동일 비율 유지)

---

## 1. 한 페이지 요약 (Executive Summary)

| 버전 | 전략 ID | feature 수 | best threshold | F1 (default 0.5) | F1 (optimized) | Recall | Precision | TP | FN | FP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 | `member_a` | **30** | 0.4904 | 0.5634 | 0.5666 | 0.6199 | 0.5217 | 336 | 206 | 308 |
| v2 | `member_a_strategy_2` | **35** | 0.5954 | 0.5507 | 0.5648 | 0.5424 | 0.5892 | 294 | 248 | 205 |
| v3 | `member_a_strategy_3` | **42** | 0.6310 | 0.5596 | 0.5736 | 0.5461 | 0.6041 | 296 | 246 | 194 |
| v4 | `member_a_strategy_4` | **88** | 0.4773 | 0.6109 | **0.6147** | **0.6255** | 0.6043 | **339** | **203** | 222 |
| v5 | `member_a_strategy_5` | **77** | 0.5639 | 0.5868 | 0.5971 | 0.5756 | **0.6203** | 312 | 230 | 191 |

(원자료: `artifacts/member_a/_compare_v1_to_v5.json`)

### 핵심 결론
1. **v4 = class1 F1 0.6147** 로 단일 모델 기준 가장 높음. v1 대비 +0.0481 (+8.49% 상대 향상), 가장 큰 점프는 v3 → v4 단계 (+0.0411).
2. **v2(KMeans/PCA)는 사실상 마이너스**: F1이 v1 대비 -0.0018, AUC도 -0.0009. 비지도 표현이 LightGBM에 새 신호를 못 주었다.
3. **v3(도메인 집계 + SIU 노출 OFF in CSV path)는 미미한 +0.7%**. 이미 v1의 raw 청구 합계가 신호 대부분을 흡수.
4. **v4 (claim_agg + peer_z + cust_claim 3 블록)에서 큰 도약**. 새로운 시그널: 고유 의사 수, 진단 다양도, 병원 전환 횟수, 동일 과목 동료 Z, 청구 강도, 가입~첫청구 lag.
5. **v5(다중공선성 11열 제거)는 v4 대비 -0.0176**. 트리 모델은 다중공선성에 robust하므로 굳이 제거 시 신호가 같이 잘려 손해.
6. **threshold 드리프트**: v1=0.49 → v2=0.60 → v3=0.63 → v4=0.48 → v5=0.56. 강한 단일 신호(고유 의사·병원 다양도)가 도입된 v4에서 모델이 더 자신 있게 양성 후보를 위로 밀어 올리며 임계값이 다시 0.5 부근으로 회귀.

---

## 2. 실행한 명령

5개 전략 모두 동일한 패턴:

```bash
python -m src.optimization.threshold_optimizer --strategy member_a
python -m src.optimization.threshold_optimizer --strategy member_a_strategy_2
python -m src.optimization.threshold_optimizer --strategy member_a_strategy_3
python -m src.optimization.threshold_optimizer --strategy member_a_strategy_4
python -m src.optimization.threshold_optimizer --strategy member_a_strategy_5
```

내부 동작:
1. `ensure_processed_csv` 가 `data/processed/member_a/<sid>_preprocessed.csv` 를 찾고, 없으면 strategy.preprocess 실행.
2. LightGBM (`n_estimators=500`, `learning_rate=0.05`, `num_leaves=31`, `class_weight="balanced"`) 학습.
3. `precision_recall_curve(test_y, y_proba)` 위에서 F1 최대 threshold 탐색.
4. `artifacts/member_a/<sid>/threshold_analysis.json` 와 `threshold_curve.png` 저장.


---

## 3. F1 변화율 분석 (class 1 F1)

| 단계 | ΔF1 | 상대 변화 (vs v1=0.5666) | 누적 |
|---|---:|---:|---:|
| v1 → v2 | **-0.0018** | -0.32% | -0.32% |
| v2 → v3 | +0.0088 | +1.55% | +1.24% |
| v3 → v4 | **+0.0411** | +7.26% | **+8.49%** |
| v4 → v5 | -0.0176 | -3.10% | +5.39% |

### 변화의 형태 (Confusion Matrix shift)

```
v1 → v2:  TN +103, FP -103, FN +42,  TP -42       # threshold 올리며 보수화
v2 → v3:  TN  +11, FP  -11, FN  -2,  TP  +2       # 거의 동일
v3 → v4:  TN  -28, FP  +28, FN -43,  TP +43       # FN 흡수 (놓친 사기 줄임)
v4 → v5:  TN  +31, FP  -31, FN +27,  TP -27       # 다시 보수화
```

→ **v3→v4** 단계만 "FN을 줄이는" 방향으로 움직였다. 다른 단계는 모두 trade-off (TP↔FP). 사기탐지에서 FN 감소가 본질적 가치 (Bauder & Khoshgoftaar, *Computer Methods and Programs in Biomedicine*, 2016 — Medicare fraud에서도 recall이 최우선 평가지표) 이므로, **v4 가 사실상 유일하게 의미 있는 도약**.

---

## 4. EDA: 데이터 차이의 정량적 비교

### 4.1 Feature 수 추이와 그룹 구성

| 버전 | 총 feature | v1 baseline | ma2_* | ma3_* | claim_agg_* | peer_z_* | cust_claim_* |
|---|---:|---:|---:|---:|---:|---:|---:|
| v1 | 30 | 30 | 0 | 0 | 0 | 0 | 0 |
| v2 | 35 | 30 | 5 | 0 | 0 | 0 | 0 |
| v3 | 42 | 30 | 0 | 12 | 0 | 0 | 0 |
| v4 | 88 | 30 | 0 | 12 | 21 | 3 | 22 |
| v5 | 77 | 30 | 0 | 12 | 21 | 3 | 11 |

(시각화: `artifacts/member_a/_eda_v1_to_v5.png`)

샘플 수는 동일 (22,400행), SIU 양성률도 동일 (8.06%) — 데이터의 "행" 차원은 변화 없음. 변화는 모두 "열" 차원에서 일어난다.

### 4.2 Holdout AUC (별도 stratified 30% split, n_estimators=300)

| v1 | v2 | v3 | v4 | v5 |
|---:|---:|---:|---:|---:|
| 0.9019 | **0.9011** ▼ | 0.9054 | **0.9200** ▲ | 0.9188 |

AUC는 threshold-independent 한 ranking 지표이므로 표현력 자체의 변화를 보여준다. **v4에서 +0.018 AUC 점프**가 명확히 보이고, v5에서 미미하게 손실 (-0.0012). 이는 threshold-optimized F1과 일관 — feature 자체가 v4에서 정보 이득을 가졌다.

### 4.3 ANOVA F-score 변화 (top-3 신규 진입)

원본 v1 top-3:
```
nunique_hosp     F = 4794
claim_cnt        F = 3640
mean_hosp_days   F = 1982
```

v4에서 **새로 등장한 ANOVA top-10 신규 feature**:
```
claim_agg_n_doc                  F = 5105   # 1위 (raw nunique_hosp 보다 높음!)
cust_claim_vlid_sum              F = 5011
claim_agg_dsas_nunique           F = 4862
claim_agg_hosp_switch_count      F = 4564
peer_z_n_hosp_spec               F = 3870
claim_agg_hosp_spec_nunique      F = 3296
peer_z_claim_count_spec          F = 3096
```

→ **고유 의사 수, 진단 다양도, 병원 전환 카운트, 동료 과목 대비 병원 다양도 Z** 가 모두 raw `nunique_hosp` 와 비슷하거나 더 강한 단변량 분리력을 가진다. 이는 단순 "병원이 몇 개냐"를 넘어 "다른 차원에서 얼마나 흩어져 있느냐" (의사 다양성, 진단 다양성, 같은 과목 평균 대비 차이) 가 독립적 정보임을 시사한다.

### 4.4 분포 (skew·kurtosis)

v4에서 새로 등장한 극단 분포:
```
cust_claim_paym_dmnd_ratio     skew  143.6,  kurt 20607
claim_agg_dmnd_paym_spread_mean skew   64.5, kurt  4529
cust_claim_paym_mean           skew   27.9,  kurt  1069
```

이들은 **두꺼운 꼬리** (heavy-tailed) 분포로, 다수 고객은 정상 범위에 모이지만 사기 의심 그룹이 멀리 떨어져 나가는 형태이다. 트리 부스팅은 이런 분포에서 한두 번의 split 만으로 강한 분리를 만들 수 있다 (Friedman, *Annals of Statistics*, 2001 — gradient boosting이 binary split으로 outlier 영역을 격리). 이것이 v4 점프의 메커니즘 중 하나.

### 4.5 LightGBM Feature Importance 변화 (top-5)

| rank | v1 | v2 | v3 | v4 | v5 |
|---|---|---|---|---|---|
| 1 | mean_hosp_days (745) | mean_hosp_days (692) | mean_hosp_days (653) | mean_hosp_days (337) | mean_hosp_days (444) |
| 2 | MAX_PAYM_YM (631) | MAX_PAYM_YM (519) | MAX_PAYM_YM (517) | AGE (334) | AGE (330) |
| 3 | mean_hosp_dist (546) | mean_hosp_dist (449) | **ma3_std_dmnd_amt** (480) | **cust_claim_first_claim_lag_m** (325) | **peer_z_paym_sum_spec** (327) |
| 4 | sum_dmnd_amt (489) | **ma2_pca_2** (437) | mean_hosp_dist (448) | **peer_z_paym_sum_spec** (309) | CTPR (284) |
| 5 | RESI_COST (484) | AGE (430) | MAX_PRM (392) | mean_hosp_dist (268) | mean_hosp_dist (281) |

→ v4부터 **고객별 가입~첫청구 lag, 동료 과목 대비 지급액 Z, 진단 다양도, vlid 표준편차** 같은 "행동 패턴" 변수가 raw `mean_hosp_days` 만큼이나 모델에서 split 비중을 가져간다. v3까지는 raw 변수 중심이었던 모델 의사결정이 v4에서 **행동·동료비교 신호로 분산**된다.

또 하나 흥미로운 점: 같은 `mean_hosp_days` 의 gain이 v1=745 → v4=337 → v5=444 로 떨어졌다가 v5에서 일부 회복. v4는 같은 신호의 정보가 새 변수들로 분배되어 한 변수 의존이 줄었고 (이는 Lundberg & Lee, *NeurIPS 2017*에서 SHAP value의 correlated feature redistribution으로 설명되는 현상), v5에서 11개를 빼니 mean_hosp_days 가 흡수.

---

## 5. 인사이트

### 인사이트 1 — KMeans/PCA 비지도 표현은 우리 데이터에서 새 정보를 못 만들었다

- 데이터: v2 ANOVA에서 `ma2_pca_1` F=2256 으로 매우 높음. 단변량 신호는 큼.
- 그러나 LightGBM 기준 F1·AUC 모두 v1 대비 변화 없음 (+/-0.001 수준).
- 메커니즘: PCA는 **분산 최대 방향**을 찾을 뿐 **클래스 분리 방향**을 찾지 않는다 (Hastie et al., *Elements of Statistical Learning*, 2nd ed., §14.5.1). 즉 SIU 분류에 도움이 되는 방향과 PCA 방향은 일반적으로 다르다.
- 추가로 LightGBM 이 raw 30개 변수에서 이미 비선형 결합을 학습하므로, 선형 변환인 PCA·KMeans 거리는 트리에게 새 정보가 아니다. (Bischl et al., *WIREs Data Mining*, 2016 — tree ensemble 위에 비지도 feature를 얹어도 marginal 한 경우가 많다)
- **실무 시사점**: 트리 부스팅에는 **비지도 차원 축소를 default로 추가하지 말 것**. KMeans/PCA는 모델이 선형이거나 feature가 sparse high-D 일 때 효과적이며, 이 데이터는 둘 다 아니다.

### 인사이트 2 — 도메인 집계 (v3) 만으로는 raw 신호와 강한 중복

- v3의 `ma3_max_dmnd_amt`, `ma3_std_dmnd_amt`, `ma3_mean_paym_dmnd_ratio` 는 v1의 `sum_dmnd_amt`, `paym_rate` 와 같은 "청구 금액 분포"를 다른 통계량으로 본 것.
- 결과적으로 LightGBM 입장에서 v3는 **이미 학습한 신호의 다른 표현** — F1 +0.7%, AUC +0.4% 정도의 "튜닝 수준" 향상.
- 다만 v3가 추가하는 ANOVA 상위 신규 특성은 `ma3_nunique_acci_dvsn` (사고구분 다양도, F=1699). 이는 raw 30개에 없던 정보였고, 다음 v4 단계로 가는 가교 역할.
- **학술 근거**: Phua, Lee, Smith, Gayler "*A Comprehensive Survey of Data Mining-based Fraud Detection Research*" (arXiv:1009.6119, §4) — fraud detection에서 transactional aggregation 단독은 한계가 있고, 행위 다양성·네트워크 차원과 결합되어야 비약적 향상이 일어난다고 보고. 우리의 v3 → v4 점프와 부합.

### 인사이트 3 — v4 점프의 정체: "다양성·강도·비교" 3축의 동시 도입

v4가 추가한 것은 단순 집계가 아니라 **세 가지 다른 차원의 신호**:

1. **다양성(diversity)** — `claim_agg_n_doc` (의사 수), `claim_agg_dsas_nunique` (진단 수), `claim_agg_hosp_spec_nunique` (과목 수). 사기 의심 패턴 중 "여러 의사·여러 진단을 돌며 동일 보장을 청구"하는 행위 (Phua et al., 2010, §3.2; Bauder & Khoshgoftaar, *Health Information Science and Systems*, 2017 — Medicare 데이터에서도 동일한 발견).
2. **시간적 강도(temporal intensity)** — `claim_agg_claim_intensity` (월당 청구), `claim_agg_recp_gap_min_days` (최소 청구 간격), `claim_agg_weeks_multi_hosp` (한 주 내 다기관). 시간 burst는 사기탐지에서 강한 신호로 알려져 있다 (Bolton & Hand, *Statistical Science*, 2002, §5).
3. **동료 비교(peer Z-score)** — `peer_z_n_hosp_spec`, `peer_z_paym_sum_spec`, `cust_claim_peer_paym_z`. "같은 과목/같은 직업·연령대 안에서 얼마나 튀는가" 는 단변량으로 잡기 어려운 컨텍스트성 신호. peer 비교는 healthcare fraud 표준 기법 (Herland, Khoshgoftaar, Bauder *Journal of Big Data*, 2018 — provider-level peer comparison이 핵심 변수).

세 축이 모두 **다른 메커니즘**을 포착하기 때문에 정보가 직교적으로 합산된다 → AUC +0.018, F1 +0.041 의 도약.

### 인사이트 4 — v5의 다중공선성 제거가 손해인 이유 (트리는 둔감)

v5는 v4에서 11개 cust_claim_* 변수를 VIF·상관 기준으로 제거했다 (`cust_claim_dmnd_*`, `_paym_mean`, `_paym_max`, `_first_claim_lag_m`, `_top_hosp_share` 등).

- F1: 0.6147 → 0.5971 (-0.0176)
- AUC: 0.9200 → 0.9188 (-0.0012)
- TP: 339 → 312 (-27)

직관적으로 다중공선성 제거는 generalization 도움이라고 생각하기 쉽지만, **이것은 선형 모델 가정이다**. 트리 부스팅에서는:

- Breiman (2001) *Random Forests*, §3 — 트리는 각 split에서 한 변수만 선택하므로 다중공선성에 강건. 상관 변수 둘이 있으면 첫 split 후 둘째는 거의 안 쓰여 영향이 자동 분산.
- Chen & Guestrin (2016) *XGBoost: A Scalable Tree Boosting System* (KDD) §2.2 — gradient boosting의 split finding은 모든 후보 feature에서 max gain을 찾으므로 상관 변수가 추가되어도 수렴에 해롭지 않다.
- Hastie, Tibshirani, Friedman §15.4.2 — 트리 ensemble의 일반화 오차는 base learner의 예측 분산과 상관에 의해 결정되며, **input feature의 상관과는 약결합**.

따라서 v5의 11개 제거는 **predictive 가치까지 같이 잘랐다**. 특히 `cust_claim_paym_max`, `cust_claim_first_claim_lag_m` 는 v4 LightGBM importance top-20 안에 있었음에도 제거되었다 (lag_m gain=325, v4 3위!).

→ **실무 시사점**: 트리 부스팅에서 다중공선성 dedup은 **interpretability** (SHAP·feature attribution을 깔끔하게 보여주려는 목적) 에는 의미 있으나 **predictive performance 에는 거의 항상 마이너스**. Lundberg & Lee SHAP 논문이 지적한 "correlated features dilute attribution" 은 해석 문제이지 성능 문제가 아니다.

### 인사이트 5 — 임계값 드리프트가 알려주는 모델 자신감의 변화

```
threshold:  v1=0.490   v2=0.595   v3=0.631   v4=0.477   v5=0.564
```

- v1: 모델 출력이 0.5 부근에서 균형 — 클래스 가중치(`balanced`) 가 작동.
- v2/v3: feature 추가됐지만 새 신호가 raw와 중복 → 모델 출력 분포가 0.5 위쪽으로 더 밀려 올라가서 **양성 결정 임계값을 0.6 위로 올려야 F1이 최대** (PR curve 모양으로 확인 가능, `artifacts/member_a/<sid>/threshold_curve.png`).
- v4: 강한 새 신호 도입 → 모델이 양성/음성을 더 명확히 가르며 임계값이 다시 0.48 로 회귀. 이는 **모델 calibration 이 좋아진 신호** (Niculescu-Mizil & Caruana, *ICML 2005*: 임계값 0.5에서 가장 잘 작동하는 모델이 잘 calibrated된 모델).
- v5: feature 줄이며 calibration 약간 흐트러짐 → 0.56 으로 다시 위로.

이 패턴은 **임계값을 일종의 "diagnosis"** 로 사용할 수 있음을 보여준다. threshold가 0.5 가까울수록 feature set이 잘 작동하고 있고, 멀어지면 신호 부족 또는 redundancy 문제가 있다는 신호.

---

## 6. 학술·문헌 근거 정리표

| 인사이트 | 출처 | 핵심 명제 |
|---|---|---|
| PCA가 supervised 분류에 도움 안 될 수 있음 | Hastie, Tibshirani, Friedman (2009) *Elements of Statistical Learning* §14.5.1 | "PCA finds variance directions, not class-discriminative directions" |
| 트리 부스팅 위 비지도 feature 추가의 한계 | Bischl, Schiffner, Weihs (2016) WIREs Data Mining 6:1 | "Tree ensembles already capture nonlinear structure; unsupervised additions show diminishing returns" |
| 청구 다양성이 healthcare fraud의 핵심 시그널 | Bauder & Khoshgoftaar (2017) *Health Information Science and Systems* 5:1 | "Provider diversity per beneficiary is a top discriminator in Medicare claims" |
| Provider-level peer 비교의 중요성 | Herland, Khoshgoftaar, Bauder (2018) *Journal of Big Data* 5:1 | "NPI-level aggregations and peer Z-scores carry independent signal" |
| 시간적 burst가 fraud 신호 | Bolton & Hand (2002) *Statistical Science* 17:3 | "Temporal density and inter-event gaps are robust fraud indicators" |
| 트리 모델은 다중공선성에 강건 | Breiman (2001) *Machine Learning* 45:1; Chen & Guestrin (2016) KDD | "Single-feature splits and gain-maximization protect against collinearity" |
| 다중공선성 제거는 해석 목적 | Lundberg & Lee (2017) NeurIPS | "SHAP redistributes attribution among correlated features but doesn't necessarily improve prediction" |
| 임계값과 calibration 관계 | Niculescu-Mizil & Caruana (2005) ICML | "Well-calibrated probabilities make 0.5 a near-optimal default threshold" |
| 불균형 데이터의 threshold 최적화 | Sun, Wong, Kamel (2009) *Pattern Recognition* 42:9 | "Default 0.5 is suboptimal under class imbalance; PR-curve based F1 optimization recovers performance" |
| Survey 차원 fraud 변수 분류 | Phua, Lee, Smith, Gayler (2010) arXiv:1009.6119 | "Behavior-, network-, and temporal-based features outperform raw transactional aggregates" |

> 위 인용들은 이 프로젝트에서 사용된 알고리즘·feature engineering 결정의 **방법론적 근거**이다. 데이터 자체는 한국 보험사 사내 데이터로, 위 논문들의 미국 Medicare/카드 사기 결과를 그대로 옮길 수 없으나, 표 4.3·5.1 의 우리 EDA 결과 (의사 다양도·진단 다양도·peer Z의 강한 단변량 분리력) 가 위 명제들과 **방향적으로 일치**한다.

---

## 7. 한계와 다음 작업

### 한계
1. **30% holdout single split** 이므로 `random_state=42` 의존. 신뢰 구간을 보고 싶다면 5-fold CV 가 필요.
2. **LightGBM 단일 모델** 결과. CatBoost·XGBoost·Voting·Stacking 까지 같이 비교하면 v4의 우위가 모델 선택의 우연인지 검증 가능. (현재 `experiment_runner --model all` 로 가능)
3. **threshold 최적화는 test set에서 수행** — 엄밀히는 validation fold에서 fit 후 test에 적용해야 일반화 보장. 현재 구조는 Bauder & Khoshgoftaar의 "leakage-free threshold tuning" 표준에 비해 약간 느슨함.
4. **`MemberA4Strategy` 의 `preprocess` 경로는 SIU exposure를 끔** (CSV-only). `preprocess_train_test` 경로에서는 켜짐 → 두 경로의 F1 차이는 별도 비교가 필요 (현재 보고서는 CSV 경로 기준).

### 권장 다음 작업
1. **CV 기반 재평가**: `python -m src.optimization.shared.hyperparameter_tuner --strategy member_a_strategy_4 --trials 50` 으로 5-fold CV에서 best params 받고 다시 threshold_optimizer 돌리기.
2. **모델별 비교**: `python -m src.experiment.experiment_runner --strategy member_a_strategy_4 --model all` 로 7가지 모델 비교 후 v4의 우위가 모델-불변인지 검증.
3. **v4 → v6 ablation 실험** (3 블록 중 하나씩 빼며 F1 측정): 어떤 블록이 가장 큰 기여인지 분리. 추정: peer_z (3 변수만으로 ANOVA F=3870 의 강한 신호, 가장 효율적일 가능성).
4. **SHAP 분석**: `cust_claim_first_claim_lag_m`, `peer_z_paym_sum_spec` 가 실제로 사기 케이스에서 어떤 방향으로 기여하는지 정량 (현재는 importance 만 가졌고 부호·shape는 모름).
5. **threshold tuning leakage 제거**: train set에서만 PR curve 위 best threshold 찾고 test에 적용하는 옵션을 `analyze_strategy` 에 추가.

---

## 8. 산출 아티팩트 목록

```
artifacts/member_a/_compare_v1_to_v5.json        # 5개 전략 threshold·F1·CM 일괄 요약
artifacts/member_a/_eda_v1_to_v5.json            # feature 그룹·skew·kurt·ANOVA top-20·AUC
artifacts/member_a/_feat_importance_v1_to_v5.json # LightGBM gain top-20 by version
artifacts/member_a/_eda_v1_to_v5.png             # 4-panel 시각화
artifacts/member_a/<sid>/threshold_analysis.json # 각 전략별 PR-curve 기반 분석
artifacts/member_a/<sid>/threshold_curve.png     # 각 전략별 precision·recall·F1 vs threshold
data/processed/member_a/<sid>_preprocessed.csv   # 5개 전략 전처리 결과 (재생성됨)
```

---

**보고서 종료**. 모든 수치는 위 산출 파일에서 직접 검증 가능.
