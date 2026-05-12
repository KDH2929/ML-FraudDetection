# 프로젝트 명세서 — 보험 사기자 탐지 ML 프로젝트

## 개발 환경

- **Python**: 3.13.3
- **패키지**: pandas, numpy, scikit-learn, lightgbm, imbalanced-learn, joblib

---

## 프로젝트 개요

- **목표**: 보험 사기자 탐지 데이터에서 팀원별로 서로 다른 전처리 전략을 적용한 뒤, 동일한 모델로 성능을 비교해 최적 전처리 방식을 찾는다.
- **구조 원칙**: 전처리(바뀌는 것)와 모델링(바뀌지 않는 것)을 완전히 분리한다.
- **비교 공정성**: `data_loader.py`의 train/test split은 `random_state=42`, `test_size=0.3`으로 고정하며 누구도 변경할 수 없다.
- **평가 기준**: 사기자(class=1)의 F1-score를 최우선 지표로 삼는다.

---

## 디렉터리 구조

```
project/
├── data/
│   ├── raw/
│   │   └── original_data.csv
│   └── processed/
│       ├── member_a_preprocessed.csv
│       ├── member_b_preprocessed.csv
│       └── member_c_preprocessed.csv
│
├── artifacts/
│   ├── best_pipeline.pkl
│   ├── best_info.json
│   ├── experiment_result_member_a.csv
│   ├── experiment_result_member_b.csv
│   └── experiment_result_member_c.csv
│
├── src/
│   ├── config.py
│   ├── preprocessing/
│   │   ├── components/
│   │   │   ├── missing_value.py
│   │   │   ├── outlier.py
│   │   │   ├── encoder.py
│   │   │   ├── scaler.py
│   │   │   └── feature_engineer.py
│   │   ├── strategies/
│   │   │   ├── base_strategy.py
│   │   │   ├── member_a_strategy.py
│   │   │   ├── member_b_strategy.py
│   │   │   └── member_c_strategy.py
│   │   └── preprocessor_factory.py
│   ├── optimization/
│   │   ├── feature_selector.py
│   │   ├── sampler_factory.py
│   │   └── param_grid_factory.py
│   ├── models/
│   │   └── model_factory.py
│   ├── pipeline/
│   │   ├── pipeline_builder.py
│   │   ├── data_loader.py
│   │   └── trainer.py
│   ├── utils/
│   │   ├── save_load.py
│   │   └── metrics.py
│   └── experiment/
│       └── experiment_runner.py
│
└── run_experiment.py
```

---

## 내 담당 (각자 작성)

```
# 아래 내용을 자신에 맞게 수정하세요
나는 팀원 [A / B / C]
작성 파일: strategies/member_[a/b/c]_strategy.py
현재 상태: [미작성 / 작성 중 / 완료]
```

## 공통 파일 완료 현황

- [ ] `config.py`
- [ ] `preprocessing/components/missing_value.py`
- [ ] `preprocessing/components/outlier.py`
- [ ] `preprocessing/components/encoder.py`
- [ ] `preprocessing/components/scaler.py`
- [ ] `preprocessing/components/feature_engineer.py`
- [ ] `preprocessing/strategies/base_strategy.py`
- [ ] `preprocessing/preprocessor_factory.py`
- [ ] `optimization/sampler_factory.py`
- [ ] `optimization/feature_selector.py`
- [ ] `optimization/param_grid_factory.py`
- [ ] `models/model_factory.py`
- [ ] `pipeline/data_loader.py`
- [ ] `pipeline/pipeline_builder.py`
- [ ] `pipeline/trainer.py`
- [ ] `utils/save_load.py`
- [ ] `utils/metrics.py`
- [ ] `experiment/experiment_runner.py`
- [ ] `run_experiment.py`

---

## 전체 파일 명세

---

### `config.py`

- **역할**: 프로젝트 전체에서 공통으로 사용하는 상수 및 설정값을 한 곳에 모아 관리한다.
- **포함 내용**:
  - 데이터 경로 (`RAW_DATA_PATH`, `PROCESSED_DIR`, `ARTIFACTS_DIR`)
  - 타겟/제외 컬럼명 (`TARGET_COL = "SIU_CUST_YN"`, `DROP_COLS`, `ID_COL`)
  - train/test split 고정값 (`RANDOM_STATE = 42`, `TEST_SIZE = 0.3`)
  - 공통 모델 파라미터 (`MODEL_PARAMS`)
- **주의**: 팀원 개인이 임의로 수정하지 않는다.

---

### `data/raw/original_data.csv`

- **역할**: 원본 데이터. 절대 수정하지 않는다.
- **포함 내용**: `CUST_DATA`, `CLAIM_DATA`를 병합한 원본 CSV.

### `data/processed/member_*_preprocessed.csv`

- **역할**: 각 팀원이 자신의 전처리 전략을 적용한 결과물.
- **조건**:
  - NaN 없음
  - 모든 컬럼 수치형(float/int)
  - `SIU_CUST_YN`은 Y=1, N=0으로 변환 완료
  - `CUST_ID`, `DIVIDED_SET` 컬럼 포함 (모델 입력 전 제거됨)

---

### `artifacts/`

| 파일 | 설명 |
|------|------|
| `best_pipeline.pkl` | 가장 성능이 좋은 파이프라인 객체 저장 (joblib) |
| `best_info.json` | 최고 성능 멤버명, Recall, F1, 사용 전략 요약 |
| `experiment_result_member_a.csv` | 팀원 A의 실험 결과 (Recall, F1-macro, F1-class1 등) |
| `experiment_result_member_b.csv` | 팀원 B의 실험 결과 |
| `experiment_result_member_c.csv` | 팀원 C의 실험 결과 |

---

### `src/preprocessing/components/`

> **역할**: 전처리의 개별 기능을 독립된 클래스로 구현한다. 팀원이 strategy에서 import해서 선택·조합하는 부품 창고다.

#### `missing_value.py`

- **역할**: 결측치를 처리하는 클래스들을 모아 놓는다.
- **구현 클래스**:
  - `GroupMeanImputer`: 직업(OCCP_GRP_1) + 나이(AGE) 그룹 평균으로 소득 결측치 대체
  - `MedianImputer`: 수치형 변수 결측치를 중앙값으로 대체
  - `ModeImputer`: 범주형 변수 결측치를 최빈값으로 대체
  - `ZeroImputer`: 결측치를 0으로 대체 (CHLD_CNT, TOTALPREM 등에 사용)
- **인터페이스**: 모두 `fit(X, y=None) → self`, `transform(X) → pd.DataFrame` 구현

#### `outlier.py`

- **역할**: 이상치를 처리하는 클래스들을 모아 놓는다.
- **구현 클래스**:
  - `IQRCapper`: Q1-1.5*IQR ~ Q3+1.5*IQR 범위로 수치형 변수 캡핑
  - `QuantileCapper`: 하위 1% ~ 상위 99% 범위로 캡핑
  - `NoOpOutlier`: 이상치 처리 없음 (비교용)
- **인터페이스**: `fit(X, y=None) → self`, `transform(X) → pd.DataFrame`

#### `encoder.py`

- **역할**: 범주형 변수를 수치형으로 변환하는 클래스들을 모아 놓는다.
- **구현 클래스**:
  - `TargetEncoder`: 범주형 변수를 타겟(사기여부) 평균값으로 인코딩. 고카디널리티에 효과적.
  - `OneHotEncoder`: 범주형 변수를 원-핫 벡터로 변환 (`pd.get_dummies` 래퍼)
  - `LabelEncoder`: 범주형 변수를 정수로 레이블 인코딩
- **대상 컬럼**: `SEX`, `RESI_TYPE_CODE`, `CTPR`, `OCCP_GRP_1`, `WEDD_YN`, `MATE_OCCP_GRP_1`
- **인터페이스**: `fit(X, y=None) → self`, `transform(X) → pd.DataFrame`

#### `scaler.py`

- **역할**: 수치형 변수의 스케일을 조정하는 클래스들을 모아 놓는다.
- **구현 클래스**:
  - `RobustScalerWrapper`: 중앙값과 IQR 기반 스케일링. 이상치에 강건.
  - `StandardScalerWrapper`: 평균 0, 표준편차 1로 표준화.
  - `MinMaxScalerWrapper`: 0~1 범위로 정규화.
- **대상 컬럼**: `RESI_COST`, `TOTALPREM`, `MAX_PRM`, `CUST_INCM`, `RCBASE_HSHD_INCM`, `JPBASE_HSHD_INCM`
- **인터페이스**: `fit(X, y=None) → self`, `transform(X) → pd.DataFrame`

#### `feature_engineer.py`

- **역할**: CLAIM 데이터 기반의 파생변수를 생성하는 클래스들을 모아 놓는다.
- **구현 클래스**:
  - `ClaimFeature`: 고객별 청구 횟수, 평균 입원일수, 유의병원 방문 여부, 의사/병원 사기비율 생성
  - `RatioFeature`: ACCI_DVSN 비율, 청구금액 대비 비율 등 비율 파생변수 생성
  - `HospFeature`: 동일병원 반복 방문 횟수, 병원 변경 횟수 등 병원 패턴 파생변수 생성
- **인터페이스**: `fit(X, y=None, claim_df=None) → self`, `transform(X) → pd.DataFrame`
- **주의**: CLAIM 원본 데이터가 필요한 클래스는 `fit()` 호출 시 `claim_df`를 함께 전달한다.

---

### `src/preprocessing/strategies/`

> **역할**: 팀원별 전처리 전략을 정의한다. `components/`의 클래스들을 import해서 조합하는 방식(Composition)으로 구현한다.

#### `base_strategy.py`

- **역할**: 모든 팀원 전략이 반드시 구현해야 하는 인터페이스를 추상 클래스로 정의한다.
- **추상 메서드**:
  - `preprocess(X: pd.DataFrame, y: pd.Series, claim_df: pd.DataFrame = None) → pd.DataFrame`
  - `get_strategy_name() → str`: 전략 이름 반환 (로그/비교표 출력용)
- **주의**: 팀원은 이 클래스를 상속받아 두 메서드를 반드시 구현해야 한다.

#### `member_a_strategy.py`

- **역할**: 팀원 A의 전처리 전략 구현체.
- **사용 컴포넌트**: `GroupMeanImputer` + `IQRCapper` + `TargetEncoder` + `RobustScalerWrapper` + `ClaimFeature`
- **상속**: `BaseStrategy`
- **구현 메서드**: `preprocess()`, `get_strategy_name() → "A: GroupMean+IQR+Target+Robust"`

#### `member_b_strategy.py`

- **역할**: 팀원 B의 전처리 전략 구현체.
- **사용 컴포넌트**: `MedianImputer` + `NoOpOutlier` + `OneHotEncoder` + `StandardScalerWrapper` + `RatioFeature`
- **상속**: `BaseStrategy`
- **구현 메서드**: `preprocess()`, `get_strategy_name() → "B: Median+NoOp+OneHot+Standard"`

#### `member_c_strategy.py`

- **역할**: 팀원 C의 전처리 전략 구현체.
- **사용 컴포넌트**: `ModeImputer` + `QuantileCapper` + `LabelEncoder` + `MinMaxScalerWrapper` + `HospFeature`
- **상속**: `BaseStrategy`
- **구현 메서드**: `preprocess()`, `get_strategy_name() → "C: Mode+Quantile+Label+MinMax"`

---

### `src/preprocessing/preprocessor_factory.py`

- **역할**: 멤버 이름을 입력받아 해당 전략 객체를 반환하는 팩토리 함수를 제공한다.
- **인터페이스**: `get_strategy(member: str) → BaseStrategy`
- **등록 방식**: 각 팀원이 자신의 전략 클래스를 `STRATEGY_REGISTRY` 딕셔너리에 등록한다.
- **사용 예**:
  ```python
  strategy = get_strategy("member_a")   # MemberAStrategy 반환
  ```

---

### `src/optimization/`

> **역할**: 전처리 이후 모델 학습 전에 적용하는 최적화 로직을 모아 놓는다.

#### `sampler_factory.py`

- **역할**: 클래스 불균형 해소를 위한 오버샘플링 객체를 생성한다.
- **지원 방식**: `SMOTE`, `ADASYN`, `BorderlineSMOTE`, `None`(미적용)
- **인터페이스**: `get_sampler(method: str) → sampler 객체`

#### `feature_selector.py`

- **역할**: 불필요한 변수를 제거해 모델 성능을 향상시킨다.
- **지원 방식**: `SelectFromModel`(중요도 기반), `VarianceThreshold`(분산 기반), `None`(미적용)
- **인터페이스**: `fit(X, y) → self`, `transform(X) → pd.DataFrame`

#### `param_grid_factory.py`

- **역할**: 하이퍼파라미터 탐색 범위를 모델별로 반환한다.
- **인터페이스**: `get_param_grid(model_name: str) → dict`

---

### `src/models/model_factory.py`

- **역할**: 모델 이름을 입력받아 sklearn 호환 분류기 객체를 반환한다. 모든 모델은 `config.py`의 파라미터로 초기화된다.
- **지원 모델**: `lgbm`, `rf`(RandomForest), `logistic`
- **인터페이스**: `get_model(model_name: str) → sklearn estimator`

---

### `src/pipeline/`

#### `data_loader.py`

- **역할**: 원본 데이터를 로드하고 train/test split을 수행한다. split은 `config.py`의 고정값을 사용하며 누구도 변경할 수 없다.
- **인터페이스**: `load_and_split(processed_csv: str) → (train_X, test_X, train_y, test_y)`
- **고정값**: `random_state=42`, `test_size=0.3`, `stratify=y`

#### `pipeline_builder.py`

- **역할**: 전략 객체와 모델 객체를 받아 하나의 실행 가능한 파이프라인을 조립한다.
- **인터페이스**: `build(strategy: BaseStrategy, model, sampler, selector) → Pipeline`

#### `trainer.py`

- **역할**: 파이프라인을 학습 데이터로 학습시킨다. 학습 로직을 `experiment_runner.py`와 분리한다.
- **인터페이스**: `train(pipeline, train_X, train_y) → fitted_pipeline`

---

### `src/utils/`

#### `save_load.py`

- **역할**: 파이프라인, CSV 등 산출물을 저장하고 불러오는 유틸 함수를 제공한다.
- **주요 함수**:
  - `save_pipeline(pipeline, path)`: joblib으로 파이프라인 저장
  - `load_pipeline(path)`: 저장된 파이프라인 로드
  - `save_result(df, path)`: 실험 결과 CSV 저장

#### `metrics.py`

- **역할**: 모델 예측 결과로부터 평가 지표를 계산하고 출력한다.
- **주요 함수**:
  - `evaluate(y_true, y_pred) → dict`: Recall, F1-class1, F1-macro 반환
  - `print_report(member, metrics_dict)`: 비교표 형태로 콘솔 출력

---

### `src/experiment/experiment_runner.py`

- **역할**: 실험 전체 흐름을 조율한다. 데이터 로드, 전략 선택, 학습, 평가, 저장을 순서대로 호출하는 컨트롤 타워 역할만 담당한다. 직접 연산 로직을 포함하지 않는다.
- **주요 메서드**:
  - `run(member: str, model_name: str)`: 특정 멤버의 실험 1회 실행
  - `run_all()`: 전체 멤버 실험 실행 후 비교표 출력

---

### `run_experiment.py`

- **역할**: 프로젝트의 유일한 실행 진입점.
- **사용법**:
  ```bash
  # 특정 멤버만 실행
  python run_experiment.py --member member_a --model lgbm

  # 전체 비교 실행
  python run_experiment.py --all
  ```

---

## 팀원이 구현해야 하는 것 (요약)

| 팀원 | 작성 파일 | 해야 할 일 |
|------|----------|-----------|
| 공통 담당 1명 | 위 명세의 모든 공통 파일 | 추상 클래스, 공통 컴포넌트, ml 레이어 구현 |
| 팀원 A | `strategies/member_a_strategy.py` | `BaseStrategy` 상속, `preprocess()` 구현 |
| 팀원 B | `strategies/member_b_strategy.py` | `BaseStrategy` 상속, `preprocess()` 구현 |
| 팀원 C | `strategies/member_c_strategy.py` | `BaseStrategy` 상속, `preprocess()` 구현 |
| 전원 | `preprocessor_factory.py` | 자신의 전략 클래스를 `STRATEGY_REGISTRY`에 등록 |

---

## 절대 변경 금지 항목

| 항목 | 이유 |
|------|------|
| `config.py`의 `RANDOM_STATE`, `TEST_SIZE` | 비교 공정성 |
| `data_loader.py`의 split 로직 | 비교 공정성 |
| `model_factory.py`의 모델 파라미터 | 비교 공정성 |
| `metrics.py`의 평가 지표 계산 로직 | 비교 공정성 |
