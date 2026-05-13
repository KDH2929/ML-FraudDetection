# member_c_strategy_4 (v4) 최적화 보고서
**생성 일자**: 2026-05-13  
**전략 ID**: `member_c_strategy_4`  
**비교 기준 전략**: `member_c_strategy_3`

---

## 개요

`member_c_strategy_4`는 safe한 `v3` 구조를 유지하면서, `v1`에서 중요도가 높았고 `v3.1`, `v3.2` 실험에서도 실제로 기여한 변수만 선별 복원한 버전이다.

- `hospital_freq_min`, `doctor_freq_min` 복원
- `claim_per_year` 복원
- `delay_origin_to_recp`, `delay_resn_to_recp`의 `min/median/max` 복원
- `hosp_days_calc_median`, `hosp_days_calc_std` 복원
- `dmnd_cv`, `valid_hosp_days_log_mean`, `dmnd_per_valid_day` 복원
- 기여가 약했던 `cause_freq_min`은 제외

---

## 성능 확인

### v4 모델별 성능

| 모델 | recall_class1 | f1_class1 | f1_macro |
| --- | ---: | ---: | ---: |
| lgbm | 0.5646 | 0.6567 | 0.8143 |
| rf | 0.4133 | 0.5444 | 0.7559 |
| logistic | 0.1568 | 0.2439 | 0.5994 |

### v3 대비 비교 (LightGBM 기준)

| 지표 | member_c_strategy_3 | member_c_strategy_4 | 변화량 |
| --- | ---: | ---: | ---: |
| recall_class1 | 0.5498 | 0.5646 | +0.0148 |
| f1_class1 | 0.6327 | 0.6567 | +0.0240 |
| f1_macro | 0.8012 | 0.8143 | +0.0131 |

### v1 대비 비교 (LightGBM 기준)

| 지표 | member_c | member_c_strategy_4 | 변화량 |
| --- | ---: | ---: | ---: |
| recall_class1 | 0.5590 | 0.5646 | +0.0055 |
| f1_class1 | 0.6516 | 0.6567 | +0.0050 |
| f1_macro | 0.8116 | 0.8143 | +0.0027 |

---

## 해석

- 현재 기준 최고 성능은 `member_c_strategy_4 + lgbm` 조합이다.
- `v3`에서 안전성은 확보했지만 빠졌던 지연일/변동성/반복성 변수 일부를 복구하니 성능이 다시 올라왔다.
- 특히 `claim_per_year`, `delay_resn_to_recp_max`, `delay_origin_to_recp_max`, `dmnd_cv`, `hosp_days_calc_median` 계열이 실제로 높은 중요도를 보였다.
- 반면 `logistic`은 아직 수렴 경고가 있어, 별도로 `scale_numeric=True` 실험이 필요하다.

---

## 전처리 결과 요약

| 항목 | 값 |
| --- | ---: |
| 전체 고객 수 | 22,400 |
| 학습 대상 (`DIVIDED_SET == 1`) | 20,607 |
| 예측 대상 (`DIVIDED_SET == 2`) | 1,793 |
| 전체 컬럼 수 | 192 |
| 모델 입력 feature 수 | 190 |
| feature 결측 수 | 0 |
| 학습 데이터 사기 고객 비율 | 8.76% |

---

## 다음 단계 제안

1. `member_c_strategy_4`를 member_c의 주력 safe 버전으로 사용
2. `scale_numeric=True` 버전에서 `logistic` 재평가
3. 필요하면 `include_target_encoded_features=True` 옵션을 켠 확장 버전과 비교
