# Member A v3/v4 파생변수 근거 점검

작성일: 2026-05-15

점검 대상은 현재 코드 기준 `member_a_strategy_3`와 `member_a_strategy_4`에서 새로 붙는 파생변수다. 결론부터 말하면, v3/v4 파생변수 중 "논문에 같은 변수명/같은 수식이 그대로 있는 직접 재현"은 거의 없다. 대부분은 의료/보험 사기탐지 문헌에서 반복되는 개념을 현재 CLAIM/CUST 컬럼으로 번역한 것이다. 따라서 발표/보고서에는 "논문 직접 변수"가 아니라 "문헌 개념 차용 + 로컬 EDA 검증"으로 쓰는 편이 안전하다.

## 근거 수준 정의

| 등급 | 의미 | 보고서 표기 권장 |
| --- | --- | --- |
| A | 원문/교재 코드에 가까운 직접 구현 | 직접 근거 |
| B | 논문에서 같은 계열의 개념이 확인되지만 현재 변수는 자체 번역 | 문헌 개념 차용 |
| C | 논문 근거는 약하거나 없고, 로컬 EDA/중요도 근거가 있음 | EDA 근거 |
| D | 논문 근거도 약하고 로컬 EDA도 약함 | EDA 필요 / 독자 설계 |
| X | 누수/중복/계산 문제 가능성이 큼 | 제외 또는 별도 검증 |

## 확인한 문헌

| ID | 문헌 | 이번 점검에서 실제로 확인한 범위 |
| --- | --- | --- |
| P1 | Phua, Lee, Smith, Gayler (2010), "A Comprehensive Survey of Data Mining-based Fraud Detection Research", arXiv:1009.6119, https://arxiv.org/pdf/1009.6119 | 의료보험 사기 데이터가 환자 인구통계, 치료/서비스, 보험/청구 금액 정보를 쓴다고 정리한다. 또한 peer group의 표준화 거리/t-statistic을 사기탐지 점수로 쓰는 Bolton & Hand 계열 방법을 소개한다. |
| P2 | Thornton, Brinkhuis, Amrit, Aly (2015), "Categorizing and Describing the Types of Fraud in Healthcare", Procedia Computer Science, https://ris.utwente.nl/ws/files/28158488/categorizing.pdf | doctor shopping, too many claims, unnecessary/maximized care, peer group 대비 outlier가 의료 사기 유형/탐지 개념으로 확인된다. |
| P3 | Herland, Khoshgoftaar, Bauder (2018), "Big Data fraud detection using multiple medicare data sources", Journal of Big Data, https://link.springer.com/article/10.1186/s40537-018-0138-3 | Medicare 청구 데이터의 지급/청구 금액, 서비스 수, 전문과/specialty, provider-level aggregation이 실제 사기탐지 실험에 쓰였음을 확인했다. |
| P4 | Kumaraswamy, Markey, Barner, Rascati (2022), "Feature engineering to detect fraud using healthcare claims data", Expert Systems with Applications, https://www.sciencedirect.com/science/article/pii/S0957417422015330 | 처방/의료 claims를 provider fraud 탐지용 secondary features로 변환하는 feature engineering framework를 제안한다. 금액, 청구/거래 패턴, 관계, 고객군 기반 feature의 근거로 사용 가능하다. |
| P5 | van Capelleveen et al. (2016), "Outlier detection in healthcare fraud", International Journal of Accounting Information Systems, https://www.sciencedirect.com/science/article/pii/S1467089515300324 | Medicaid claims에서 outlier metrics를 설계하고 suspicious provider를 flagging한 사례. peer/outlier 계열 근거로 사용 가능하다. |
| P6 | du Preez et al. (2025), "Fraud detection in healthcare claims using machine learning: A systematic review", Artificial Intelligence in Medicine, https://www.sciencedirect.com/science/article/pii/S0933365724003038 | 실제 논문은 존재한다. 다만 현재 `claim_agg_*` 변수들의 직접 수식 출처라기보다는 의료청구 사기탐지 ML 문헌고찰 근거로만 쓰는 것이 안전하다. |

## 핵심 판정

| 묶음 | 판정 | 이유 |
| --- | --- | --- |
| `ma3_*` 청구 금액/다양성 | B~D | 금액, 청구상세, 치료/서비스 정보는 문헌 근거가 있으나 현재 변수 수식은 자체 설계다. |
| `ma3_*` SIU exposure | X | CSV path에서는 0으로 꺼져 있고, train/test path에서만 라벨 기반 exposure가 켜진다. LOO라도 nested CV 없이 쓰면 누수 설명이 까다롭다. |
| `claim_agg_*` | B/C | doctor shopping, too many claims, payment/utilization/outlier 개념과 잘 맞는다. 다만 exact formula는 자체 설계다. |
| `peer_z_*` | B | peer group 표준화 거리/t-statistic 문헌과 가장 가깝다. 단, `HOSP_SPEC_DVSN` 최빈값으로 고객 peer를 만드는 것은 현재 데이터 맞춤 번역이다. |
| `cust_claim_*` | B~D | 청구/지급 요약, 병원 다양도, peer paym z는 근거가 좋다. 가입~첫청구 lag, 납입/소득 비율, 소득 missing은 논문 근거보다 EDA/도메인 근거다. |

## v3 변수별 판정

로컬 EDA는 `artifacts/member_a/transition_evidence/v2_to_v3_added_feature_stats.csv` 기준이다. SMD는 SIU/비SIU 평균 차이의 표준화 크기다.

| 변수 | 현재 의미 | 근거 | 등급 | 로컬 근거 | 판단 |
| --- | --- | --- | --- | --- | --- |
| `ma3_max_dmnd_amt` | 고객별 최대 청구금액 | P1, P3, P4 | B | SMD 0.135, MI 0.044 | 금액 feature 계열 문헌 근거는 있음. 정확한 수식은 자체 설계. |
| `ma3_std_dmnd_amt` | 청구금액 표준편차 | P4 | B/C | SMD 0.066, MI 0.023, v3 importance #3 | 금액 변동성은 claims feature engineering로 설명 가능. 로컬 중요도 근거가 더 강함. |
| `ma3_mean_paym_dmnd_ratio` | 지급/청구 비율 평균 | P1, P3, P4 | B/C | SMD 0.065, MI 0.023 | 지급/청구 금액 관계의 자체 통계. 문헌 직접 수식은 아님. |
| `ma3_frac_heed_claim_rows` | 유의병원 청구 행 비율 | 데이터 정의서 | C/D | SMD 0.124, MI 0.014 | 논문 근거보다는 `HEED_HOSP_YN`라는 도메인 컬럼 자체가 근거. |
| `ma3_nunique_acci_dvsn` | 사고구분 다양도 | P1, P2 | B/C | SMD 0.878, MI 0.032 | 다양성/복잡도 계열로 설명 가능. 로컬 EDA 강함. |
| `ma3_max_nonpay_ratio` | 최대 비급여 비율 | 데이터 정의서 | C/D | SMD 0.410, MI 0.010 | 논문 직접 근거 없음. 방향성도 데이터에서 확인해 설명해야 함. |
| `ma3_resi_unknown` | 주택가격 0/결측 여부 | 데이터 정의서 | D | SMD 0.133, MI 0.003 | `RESI_COST=0`의 의미는 정의서 근거. 사기탐지 논문 근거는 약함. |
| `ma3_absdiff_hh_incm_est` | 두 가구소득 추정치 차이 | 독자 설계 | C | SMD 0.022, MI 0.001, v3/v4 importance 상위 | 논문 근거 없음. 다만 모델 importance가 있어 EDA/ablation 근거 필요. |
| `ma3_cust_minus_jpbase_incm` | 개인소득 - JPBASE 가구소득 | 독자 설계 | D | SMD 0.030, MI 0.0003 | 현 상태로는 약한 독자 설계. |
| `ma3_doc_siu_train_expo` | 의사별 SIU exposure | label-derived | X | CSV path에서는 unique 1, 값 0 | 누수 설명이 어려움. 쓰려면 fold 내부 LOO target/stat encoding으로 별도 검증. |
| `ma3_hosp_siu_train_expo` | 병원별 SIU exposure | label-derived | X | CSV path에서는 unique 1, 값 0 | 위와 동일. |
| `ma3_doc_hosp_train_cross` | 의사 exposure x 병원 exposure | label-derived | X | CSV path에서는 unique 1, 값 0 | 위 두 변수가 유효할 때만 의미. 현재 보고서 근거로 쓰지 않는 편이 안전. |

## v4 `claim_agg_*` 변수별 판정

로컬 EDA는 `artifacts/member_a/transition_evidence/v3_to_v4_added_feature_stats.csv` 기준이다.

| 변수 | 현재 의미 | 근거 | 등급 | 로컬 근거 | 판단 |
| --- | --- | --- | --- | --- | --- |
| `claim_agg_n_claims` | 고객별 청구 건수 | P1, P2, P4 | B | SMD 0.882, MI 0.052 | too many claims / service maximization 근거 있음. |
| `claim_agg_n_hosp` | 이용 병원 수 | P2 | B/C | SMD 1.008, MI 0.065 | doctor/hospital shopping 개념 차용. 로컬 강함. |
| `claim_agg_n_hosp_per_10_claims` | 10건당 병원 수 | P2 | B/C | SMD 0.309, MI 0.042 | doctor shopping을 청구량으로 정규화한 자체 지표. |
| `claim_agg_n_doc` | 담당의사 수 | P2 | B | SMD 1.008, MI 0.067 | doctor shopping과 가장 직접 연결. |
| `claim_agg_paym_sum` | 지급금액 합 | P1, P3, P4 | B/C | SMD 0.455, MI 0.048 | 지급/청구 금액 계열 문헌 근거. |
| `claim_agg_paym_cv` | 지급금액 변동계수 | P4 | C | SMD 0.111, MI 0.023, importance 상위 | 변동성 자체 수식은 자체 설계. 로컬 importance 근거. |
| `claim_agg_paym_max_ratio` | 최대 지급액 / 평균 지급액 | P4 | C | SMD 0.356, MI 0.016, importance 상위 | heavy-tail/outlier 지표로 EDA 설명 필요. |
| `claim_agg_dmnd_paym_spread_mean` | 청구-지급 차이 평균 | P1, P3, P4 | C/D | SMD 0.019, MI 0.016 | 문헌상 금액은 근거 있지만 이 수식의 로컬 근거는 약함. |
| `claim_agg_dsas_nunique` | 병명 다양도 | P1, P2 | B/C | SMD 0.939, MI 0.067 | diagnosis diversity / wrong diagnosis 계열로 설명 가능. |
| `claim_agg_dsas_top_share` | 최빈 병명 비중 | P1, P2 | C | SMD 1.048, MI 0.062 | 다양도/집중도 자체 설계. 로컬 강함. |
| `claim_agg_acci_nunique` | 사고구분 다양도 | P1, P2 | B/C | SMD 0.878, MI 0.032 | `ma3_nunique_acci_dvsn`와 중복 가능. |
| `claim_agg_hosp_spec_nunique` | 병원종별 다양도 | P3 | B/C | SMD 1.102, MI 0.053 | specialty/provider type 계열 근거. 로컬 강함. |
| `claim_agg_heed_frac` | 유의병원 비율 | 데이터 정의서 | C/D | SMD 0.124, MI 0.014 | 논문 근거보다 도메인 컬럼 근거. |
| `claim_agg_vlid_std` | 유효 입원/통원일수 표준편차 | P1, P4 | C | SMD 0.525, MI 0.042, importance 상위 | utilization variability로 설명 가능. 정확한 수식은 자체 설계. |
| `claim_agg_vlid_max` | 최대 유효 입원/통원일수 | P1, P4 | C | SMD 0.859, MI 0.064 | utilization intensity. 로컬 강함. |
| `claim_agg_house_hosp_dist_mean` | 고객-병원 거리 평균 | 독자 설계 | D | SMD 0.022, MI 0.010 | 현재 확인 문헌 근거 약함. EDA/SHAP 방향 확인 필요. |
| `claim_agg_claim_intensity` | 관측 span 대비 청구 밀도 | P1, P2, P4 | B/C | SMD 0.641, MI 0.045, importance 상위 | temporal/activity intensity 개념 차용. |
| `claim_agg_hosp_switch_count` | 시간순 병원 전환 횟수 | P2 | B/C | SMD 0.885, MI 0.064 | doctor/hospital shopping의 행동형 번역. 로컬 강함. |
| `claim_agg_recp_gap_mean_days` | 평균 청구 간격 | P1, P4 | C | SMD 0.115, MI 0.044 | temporal feature. 직접 수식 근거보다 EDA 근거. |
| `claim_agg_recp_gap_min_days` | 최소 청구 간격 | P1, P4 | C/D | SMD 0.151, MI 0.011 | temporal burst 후보이나 로컬 근거는 약함. |
| `claim_agg_weeks_multi_hosp` | 같은 주 다기관 발생 주 수 | P2 | B/C | SMD 0.539, MI 0.022 | doctor shopping을 시간창으로 번역한 자체 지표. |

## v4 `peer_z_*` 변수별 판정

| 변수 | 현재 의미 | 근거 | 등급 | 로컬 근거 | 판단 |
| --- | --- | --- | --- | --- | --- |
| `peer_z_paym_sum_spec` | 동일 병원종별 내 지급합 z-score | P1, P3, P5 | B | SMD 0.548, MI 0.051, v4 importance #4 | peer group 표준화 거리 개념과 가장 잘 맞음. |
| `peer_z_claim_count_spec` | 동일 병원종별 내 청구건수 z-score | P1, P2, P5 | B | SMD 0.863, MI 0.066 | peer/outlier + too many claims 근거. |
| `peer_z_n_hosp_spec` | 동일 병원종별 내 병원수 z-score | P1, P2, P5 | B | SMD 0.984, MI 0.076 | doctor shopping + peer outlier 근거. |

주의: 논문에서 `HOSP_SPEC_DVSN` 최빈값으로 고객 peer group을 만들라고 한 것은 아니다. "동일 집단 대비 표준화 이탈"이라는 방법론만 차용했다.

## v4 `cust_claim_*` 변수별 판정

| 변수 | 현재 의미 | 근거 | 등급 | 로컬 근거 | 판단 |
| --- | --- | --- | --- | --- | --- |
| `cust_claim_claim_rows` | 고객별 청구 행 수 | P1, P2, P4 | B | SMD 0.882, MI 0.055 | `claim_agg_n_claims`와 완전 중복. |
| `cust_claim_paym_sum` | 지급 합 | P1, P3, P4 | B/C | SMD 0.455, MI 0.051 | `claim_agg_paym_sum`과 중복. |
| `cust_claim_dmnd_sum` | 청구 합 | P1, P3, P4 | B/C | SMD 0.453, MI 0.049 | v1 `sum_dmnd_amt`와 중복 가능. |
| `cust_claim_paym_mean` | 지급 평균 | P1, P3, P4 | C | SMD 0.022, MI 0.039 | 평균 단독 EDA는 약하지만 금액 계열이라 후보 가능. |
| `cust_claim_dmnd_mean` | 청구 평균 | P1, P3, P4 | C | SMD 0.021, MI 0.038 | 위와 동일. |
| `cust_claim_paym_max` | 최대 지급액 | P1, P3, P4 | C | SMD 0.136, MI 0.044 | outlier 금액 후보. |
| `cust_claim_dmnd_max` | 최대 청구액 | P1, P3, P4 | C | SMD 0.135, MI 0.045 | `ma3_max_dmnd_amt`와 중복. |
| `cust_claim_paym_dmnd_ratio` | 지급합/청구합 | P1, P4 | X | SMD 0.010, MI 0.002, 음성 평균 이상치 매우 큼 | 0 분모/극단값 영향 가능. 수식 안정화 또는 제거 후보. |
| `cust_claim_hosp_nunique` | 이용 병원 수 | P2 | B/C | SMD 1.008, MI 0.065 | `claim_agg_n_hosp`와 중복. |
| `cust_claim_top_hosp_share` | 최빈 병원 비중 | P2, P5 | B/C | SMD 1.053, MI 0.058 | doctor shopping의 반대 방향 집중도 지표. 로컬 강함. |
| `cust_claim_heed_claim_share` | 유의병원 청구 비중 | 데이터 정의서 | C/D | SMD 0.124, MI 0.010 | `claim_agg_heed_frac` 중복. |
| `cust_claim_vlid_sum` | 유효 입원/통원일수 합 | P1, P4 | C | SMD 0.911, MI 0.087, v4 importance #13 | 로컬 근거 매우 강함. 논문 직접 변수는 아님. |
| `cust_claim_vlid_max` | 최대 유효 입원/통원일수 | P1, P4 | C | SMD 0.859, MI 0.069 | `claim_agg_vlid_max` 중복. |
| `cust_claim_cnt_recent_6m` | 최근 6개월 청구 수 | P1, P4 | C/D | SMD 0.244, MI 0.002 | temporal 후보이나 로컬 MI 낮음. window 타당성 EDA 필요. |
| `cust_claim_cnt_6m_to_12m` | 6~12개월 전 청구 수 | P1, P4 | D | SMD 0.219, MI 0.001 | 위와 동일. |
| `cust_claim_cnt_prior_12m` | 12개월 이전 청구 수 | P1, P4 | C/D | SMD 0.333, MI 0.010 | 위와 동일. |
| `cust_claim_recent6_vs_prior_ratio` | 최근/과거 청구 강도 비율 | P1, P4 | D | SMD 0.123, MI 0.004 | ratio window 직접 근거 약함. |
| `cust_claim_tenure_m` | 고객등록~관측월 기간 | P1 | C/D | SMD 0.019, MI 0.005, importance 상위 | account age/tenure 계열로 우회 설명 가능하나 의료 claims 직접 근거는 약함. |
| `cust_claim_first_claim_lag_m` | 고객등록~첫청구 lag | P1 | C | SMD 0.013, MI 0.012, v4 importance #3 | 모델 importance 근거는 강하지만 논문 직접 근거는 약함. SHAP/ablation 필요. |
| `cust_claim_peer_paym_z` | 직업x연령 peer 지급합 z-score | P1, P5 | B | SMD 0.589, MI 0.044, importance 상위 | peer group standardized deviation 개념 차용. |
| `cust_claim_prem_incm_ratio` | 납입총보험료/추정소득 | 독자 설계 | C | SMD 0.107, MI 0.003, importance 상위 | affordability/financial pressure 가설. 논문 근거 없이 EDA로만 주장. |
| `cust_claim_incm_missing_or_zero` | 소득 결측 또는 0 | 데이터 정의서/결측 EDA | D | SMD 0.137, MI 0.004 | 결측 자체 신호. 논문 근거보다 EDA 필요. |

## 발표/보고서용 안전 문장

- "v3/v4는 논문 수식을 그대로 복제한 것이 아니라, 의료청구 사기탐지 문헌에서 반복되는 claim amount, utilization, provider diversity, doctor shopping, peer-group outlier, temporal intensity 개념을 현재 데이터 정의서의 CLAIM/CUST 컬럼으로 번역한 feature engineering이다."
- "`claim_agg_n_doc`, `claim_agg_n_hosp`, `claim_agg_hosp_switch_count`, `peer_z_*`는 doctor shopping/peer outlier 문헌 근거와 로컬 EDA가 같이 있어 근거가 강하다."
- "`cust_claim_first_claim_lag_m`, `cust_claim_prem_incm_ratio`, `ma3_absdiff_hh_incm_est`는 논문 직접 근거가 아니라 로컬 모델 중요도/ablation으로 방어해야 한다."
- "`ma3_*_siu_train_expo` 계열은 라벨 기반 exposure라서 발표에서는 빼거나, fold 내부 LOO target/stat encoding으로 별도 실험했다는 근거가 있을 때만 사용한다."

## 우선순위

| 우선순위 | 변수군 | 이유 |
| --- | --- | --- |
| 유지 강함 | `claim_agg_n_doc`, `claim_agg_n_hosp`, `claim_agg_hosp_switch_count`, `claim_agg_dsas_nunique`, `claim_agg_hosp_spec_nunique`, `peer_z_*`, `cust_claim_vlid_sum` | 문헌 개념 + 로컬 EDA가 모두 강함 |
| 유지 가능 | 금액 aggregate/CV/max ratio, `claim_agg_claim_intensity`, `claim_agg_weeks_multi_hosp`, `cust_claim_peer_paym_z` | 문헌 개념 또는 로컬 중요도 있음 |
| EDA/ablation 필요 | `cust_claim_first_claim_lag_m`, `cust_claim_tenure_m`, `cust_claim_prem_incm_ratio`, `ma3_absdiff_hh_incm_est`, `ma3_max_nonpay_ratio` | 모델에서는 보이나 논문 직접 근거가 약함 |
| 제거/주의 | `ma3_doc_siu_train_expo`, `ma3_hosp_siu_train_expo`, `ma3_doc_hosp_train_cross`, `cust_claim_paym_dmnd_ratio`, 완전 중복 변수들 | 누수/계산/중복 설명 리스크 |
