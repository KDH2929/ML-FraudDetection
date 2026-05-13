# member_c_strategy_3 (v3) 최적화 보고서

**생성 날짜**: 2026-05-13  
**전략 ID**: `member_c_strategy_3`  
**비교 기준 전략**: `member_c_strategy_2`

---

## 개요

`member_c_strategy_3`는 v2보다 더 엄격한 safe 전처리 기준을 적용한 버전이다.

- `FP_CAREER` 완전 제외
- 결측치 대체를 train 기준 그룹 중앙값/최빈값으로 재구성
- 결측률이 매우 높은 CLAIM 원본 변수 제거
- 날짜 원본 컬럼은 파생변수 생성 후 제거
- 지급/심사 결과 계열 변수는 기본 safe 버전에서 제외
- target encoding 변수는 옵션형으로만 분리

이번 성능 측정은 아래 safe 설정 기준으로 수행했다.

- `include_leakage_features=False`
- `include_target_encoded_features=False`
- `scale_numeric=False`

---

## 성능 확인

### v3 모델별 성능

| 모델 | recall_class1 | f1_class1 | f1_macro |
| --- | ---: | ---: | ---: |
| lgbm | 0.5498 | 0.6327 | 0.8012 |
| rf | 0.3819 | 0.5136 | 0.7399 |
| logistic | 0.1458 | 0.2303 | 0.5926 |

### v2 대비 비교 (LightGBM 기준)

| 지표 | member_c_strategy_2 | member_c_strategy_3 | 변화량 |
| --- | ---: | ---: | ---: |
| recall_class1 | 0.5535 | 0.5498 | -0.0037 |
| f1_class1 | 0.6410 | 0.6327 | -0.0083 |
| f1_macro | 0.8058 | 0.8012 | -0.0046 |

### 해석

- 이번 v3 safe 버전의 최고 성능 모델도 `lgbm`이었다.
- 다만 v2 대비 `f1_class1`과 `recall_class1`은 소폭 하락했다.
- 결측률이 높은 변수 제거와 safe 규칙 강화로 누수 위험은 더 줄었지만, 그만큼 정보량도 일부 감소한 것으로 보인다.
- `logistic`은 수렴 경고가 있었고 성능도 낮아, 비교용 baseline 성격이 강하다.

---

## 전처리 결과 요약

| 항목 | 값 |
| --- | ---: |
| 전체 고객 수 | 22,400 |
| 학습 대상 (`DIVIDED_SET == 1`) | 20,607 |
| 예측 대상 (`DIVIDED_SET == 2`) | 1,793 |
| 전체 컬럼 수 | 179 |
| 모델 입력 feature 수 | 176 |
| feature 결측 수 | 0 |
| 학습 데이터 사기 고객 비율 | 8.76% |

### feature 규모 비교

| 전략 | 전체 컬럼 수 | 모델 입력 feature 수 |
| --- | ---: | ---: |
| member_c_strategy_2 | 176 | 174 |
| member_c_strategy_3 | 179 | 176 |

---

## 핵심 변경 포인트

1. 고객 결측치 대체를 `Unknown` 중심에서 그룹 중앙값/최빈값 중심으로 바꿨다.
2. `MINCRDT`, `MAXCRDT`의 `99`를 미상값으로 간주하고 별도 flag를 추가했다.
3. 결측률이 매우 높은 CLAIM 원본 변수들을 기본 전략에서 제거했다.
4. 병원/의사/질병/원인 다양성과 집중도는 유지하되, 과도한 통계량 생성은 줄였다.
5. `DOC_SIU_RATIO`, `HOSP_SIU_RATIO`, `HOSP_DOC_SIU`는 기본 safe 버전에서 제외하고 옵션형으로 분리했다.
6. 고객 등록 시점과 첫 청구 시점 차이를 반영하는 `months_from_register_to_first_claim` 계열 변수를 추가했다.

---

## 다음 실험 제안

1. `include_target_encoded_features=True` 버전 성능 비교
2. `scale_numeric=True` 버전으로 `logistic` 재평가
3. safe v3와 leakage 옵션 버전을 분리 실험해 정보 증가폭과 누수 리스크를 비교

---

## 실행 방법

### 전처리만 수행

```bash
python run_experiment.py --strategy member_c_strategy_3 --prepare-only
```

### 모델 평가

```bash
python run_experiment.py --strategy member_c_strategy_3 --model lgbm
python run_experiment.py --strategy member_c_strategy_3 --model all
```
