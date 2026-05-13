# CatchCatch

보험사기 탐지용 전처리 전략, 모델 학습, 하이퍼파라미터 최적화, threshold 분석, 보고서 생성을 한 저장소에서 관리하는 프로젝트입니다.

이 문서는 다음 내용을 빠르게 확인할 수 있도록 정리했습니다.

- 프로젝트 목적
- 권장 폴더 구조
- 환경 설정 방법
- 전처리 / 학습 / 최적화 / 보고서 생성 명령어

## 프로젝트 개요

이 프로젝트는 고객 데이터와 청구 데이터를 바탕으로 보험사기 탐지 성능을 비교하기 위한 실험 파이프라인입니다.

- `preprocessing`: member별 전처리 전략 관리
- `optimization`: 튜닝, threshold 분석, feature 탐색
- `experiment`: 모델 학습 및 평가 실행
- `docs/optimization_reports`: 전략별 보고서 관리
- `artifacts`: 실험 결과물, 파이프라인, tuning 결과 저장

## 3인 협업 구조

이 프로젝트는 크게 `Member A`, `Member B`, `Member C` 세 트랙으로 나누어 병렬적으로 전처리 전략을 설계하고 비교하는 방식으로 진행했습니다.

| 담당 트랙 | 주요 역할 | 핵심 산출물 |
|---|---|---|
| Member A | 고객 테이블 중심의 기본 전처리와 파생 변수 설계 | `member_a`, `member_a_strategy_2~5` |
| Member B | Feature Engineering 고도화 + Feature Selection + Hyperparameter Tuning + Threshold 최적화 | `member_b`, `member_b_strategy_2`, `member_b_strategy_3` |
| Member C | 고객-청구 집계 기반 safe 전처리 파이프라인 설계 및 성능 안정화 | `member_c`, `member_c_strategy_2~4` |

공통적으로는 아래 흐름을 따릅니다.

1. 각 member가 자신의 전처리 전략을 설계하고 `src/preprocessing/strategies/<member>/` 아래에 구현
2. 전처리 결과를 `data/processed/<member>/` 아래 CSV로 생성
3. 실험 결과와 모델 파이프라인을 `artifacts/<member>/<strategy>/` 아래 저장
4. 전략별 성능 차이를 보고, 필요 시 tuning / threshold 최적화를 추가 적용
5. 최종 결과를 `docs/optimization_reports/<member>/` 및 비교 문서로 정리

즉, 이 저장소는 "한 개의 모델 파이프라인"이라기보다, `A/B/C` 세 사람이 각자 전처리 전략을 실험하고 공통 러너 위에서 동일 기준으로 비교하는 실험 플랫폼에 가깝습니다.

## 성능 비교 요약

아래 표는 현재 저장된 최신 보고서와 산출물 기준으로 정리한 요약입니다.

- Member B: `docs/optimization_reports/member_b/*`, `docs/MEMBER_B_STRATEGY_COMPARISON.md`
- Member C: `docs/optimization_reports/member_c/*`, `artifacts/member_c/member_c_strategy_4/custom_threshold_analysis.json`
- Member A: 아직 대표 성능 결과 미정리

### 전체 전략 비교

| 순위 | 전략 | Recall | F1 | F1 Macro | 비고 |
|---|---|---:|---:|---:|---|
| 1 | `member_c_strategy_4` | 0.5646 | 0.6567 | 0.8143 | 현재 기준 F1 최고 |
| 2 | `member_c` | 0.5627 | 0.6566 | 0.8144 | C 계열 v1 baseline |
| 3 | `member_c_strategy_2` | 0.5535 | 0.6410 | 0.8058 | safe 규칙 강화 버전 |
| 4 | `member_b_strategy_3` | 0.5830 | 0.6345 | 0.8013 | 현재 기준 Recall 최고, 최적화 완료 |
| 5 | `member_c_strategy_3` | 0.5498 | 0.6327 | 0.8012 | v3 safe 강화 버전 |
| 6 | `member_b_strategy_2` | 0.4908 | 0.5924 | 0.7803 | amount/time/medical feature 추가 |
| 7 | `member_b` | 0.4539 | 0.5541 | 0.7598 | B 계열 v1 baseline |
| - | `member_a` | - | - | - | 아직 대표 성능 결과 미정리 |

### member별 대표 전략 요약

| Member | 현재 대표 전략 | 강점 | 상태 |
|---|---|---|---|
| A | `member_a` 계열 | 고객 테이블 중심 전처리 확장 실험 | 성능 비교는 추가 정리 필요 |
| B | `member_b_strategy_3` | Recall 최상위, 최적화 파이프라인 완성 | Feature Selection + Tuning + Threshold 완료 |
| C | `member_c_strategy_4` | F1 최상위, safe 구조 기반 안정적 성능 | 추가 최적화 여지 존재 |

### Member B 전략 비교

| 전략 | Recall | F1 | F1 Macro | 특징 |
|---|---:|---:|---:|---|
| `member_b` | 0.4539 | 0.5541 | 0.7598 | v1 baseline |
| `member_b_strategy_2` | 0.4908 | 0.5924 | 0.7803 | amount/time/medical feature 추가 |
| `member_b_strategy_3` | 0.5830 | 0.6345 | 0.8013 | feature engineering 고도화 + 최적화 완료 |

### Member C 전략 비교

| 전략 | Recall | F1 | F1 Macro | 특징 |
|---|---:|---:|---:|---|
| `member_c` | 0.5627 | 0.6566 | 0.8144 | v1 baseline, 높은 기본 성능 |
| `member_c_strategy_2` | 0.5535 | 0.6410 | 0.8058 | safe 규칙 분리 강화 |
| `member_c_strategy_3` | 0.5498 | 0.6327 | 0.8012 | v3 safe 강화 |
| `member_c_strategy_4` | 0.5646 | 0.6567 | 0.8143 | v3 구조 + v1 유효 변수 복원 |

### 최적화 전후 비교

| 전략 | 구분 | Recall | F1 | F1 Macro | 비고 |
|---|---|---:|---:|---:|---|
| `member_b_strategy_3` | 최적화 전 | 0.5166 | 0.6078 | - | tuning 적용 후 threshold 0.5 기준 |
| `member_b_strategy_3` | 최적화 후 | 0.5830 | 0.6345 | 0.8013 | threshold 0.3819 적용 최종 |
| `member_c_strategy_4` | baseline | 0.5646 | 0.6567 | 0.8143 | 기본 LightGBM |
| `member_c_strategy_4` | tuned @ 0.5 | 0.6642 | 0.6723 | 0.8206 | custom tuning 적용 |
| `member_c_strategy_4` | tuned + best threshold | 0.6827 | 0.6795 | 0.8243 | threshold 0.4643 적용 |

### 해석 포인트

- `Member B`는 최적화 파이프라인이 가장 잘 정리된 트랙입니다. 특히 `member_b_strategy_3`는 feature engineering, feature selection, hyperparameter tuning, threshold optimization이 모두 적용되어 있습니다.
- `Member C`는 기본 전처리 설계의 성능 자체가 강한 트랙입니다. 최적화 없이도 높은 F1을 보였고, `member_c_strategy_4`는 튜닝과 threshold 조정 후 F1 0.6795까지 올라갑니다.
- 현재 저장된 결과만 기준으로 보면 "Recall 최고"는 `member_b_strategy_3`, "F1 최고"는 `member_c_strategy_4`입니다.

## 권장 폴더 구조

아래 구조는 README 문서화 기준으로 정리한 통일형 구조입니다.

```text
CatchCatch/
├─ src/
│  ├─ preprocessing/
│  │  ├─ strategies/
│  │  │  ├─ member_a/
│  │  │  │  ├─ member_a_strategy_1.py
│  │  │  │  ├─ member_a_strategy_2.py
│  │  │  │  ├─ member_a_strategy_3.py
│  │  │  │  ├─ member_a_strategy_4.py
│  │  │  │  └─ member_a_strategy_5.py
│  │  │  ├─ member_b/
│  │  │  │  ├─ member_b_strategy_1.py
│  │  │  │  ├─ member_b_strategy_2.py
│  │  │  │  └─ member_b_strategy_3.py
│  │  │  ├─ member_c/
│  │  │  │  ├─ member_c_strategy_1.py
│  │  │  │  ├─ member_c_strategy_2.py
│  │  │  │  ├─ member_c_strategy_3.py
│  │  │  │  └─ member_c_strategy_4.py
│  │  │  ├─ book/
│  │  │  └─ test/
│  │  ├─ preprocessor_factory.py
│  │  └─ processed_data.py
│  │
│  ├─ optimization/
│  │  ├─ shared/
│  │  │  ├─ hyperparameter_tuner.py
│  │  │  ├─ smote_experiment.py
│  │  │  └─ threshold_optimizer.py
│  │  ├─ member_b/
│  │  │  ├─ hyperparameter_tuner.py
│  │  │  ├─ smote_experiment.py
│  │  │  └─ threshold_optimizer.py
│  │  ├─ member_c/
│  │  │  ├─ hyperparameter_tuner.py
│  │  │  └─ threshold_optimizer.py
│  │  └─ feature_selector.py
│  │
│  └─ project_paths.py
│
├─ data/
│  └─ processed/
│     ├─ member_a/
│     │  ├─ member_a_strategy_1_preprocessed.csv
│     │  ├─ member_a_strategy_2_preprocessed.csv
│     │  ├─ member_a_strategy_3_preprocessed.csv
│     │  ├─ member_a_strategy_4_preprocessed.csv
│     │  └─ member_a_strategy_5_preprocessed.csv
│     ├─ member_b/
│     │  ├─ member_b_strategy_1_preprocessed.csv
│     │  ├─ member_b_strategy_2_preprocessed.csv
│     │  └─ member_b_strategy_3_preprocessed.csv
│     └─ member_c/
│        ├─ member_c_strategy_1_preprocessed.csv
│        ├─ member_c_strategy_2_preprocessed.csv
│        ├─ member_c_strategy_3_preprocessed.csv
│        └─ member_c_strategy_4_preprocessed.csv
│
├─ artifacts/
│  ├─ member_a/
│  │  ├─ member_a_strategy_1/
│  │  ├─ member_a_strategy_2/
│  │  ├─ member_a_strategy_3/
│  │  ├─ member_a_strategy_4/
│  │  └─ member_a_strategy_5/
│  ├─ member_b/
│  │  ├─ member_b_strategy_1/
│  │  ├─ member_b_strategy_2/
│  │  └─ member_b_strategy_3/
│  └─ member_c/
│     ├─ member_c_strategy_1/
│     ├─ member_c_strategy_2/
│     ├─ member_c_strategy_3/
│     └─ member_c_strategy_4/
│
└─ docs/
   └─ optimization_reports/
      ├─ member_a/
      │  ├─ member_a_strategy_1_v1_report.md
      │  ├─ member_a_strategy_2_v2_report.md
      │  ├─ member_a_strategy_3_v3_report.md
      │  ├─ member_a_strategy_4_v4_report.md
      │  └─ member_a_strategy_5_v5_report.md
      ├─ member_b/
      │  ├─ member_b_strategy_1_v1_report.md
      │  ├─ member_b_strategy_2_v2_report.md
      │  └─ member_b_strategy_3_v3_report.md
      └─ member_c/
         ├─ member_c_strategy_1_v1_report.md
         ├─ member_c_strategy_2_v2_report.md
         ├─ member_c_strategy_3_v3_report.md
         └─ member_c_strategy_4_v4_report.md
```

## 환경 설정

### 1. 가상환경 생성

```bash
python -m venv .venv
```

### 2. 가상환경 활성화

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

## 실행 가이드

주의:

- 아래 명령어는 현재 코드 기준의 실제 실행 예시입니다.
- 문서 구조도는 `member_x_strategy_1` 형식으로 통일해 표현했지만, 현재 구현된 전략 ID는 `member_a`, `member_b`, `member_c`처럼 v1이 짧게 남아 있는 경우가 있습니다.
- 따라서 실행 시에는 아래 예시처럼 실제 전략 ID를 사용하면 됩니다.

## 1. 전처리만 실행

### 단일 전략 전처리

```bash
python run_experiment.py --strategy member_b_strategy_2 --prepare-only
```

### 특정 전략 전처리 재생성

```bash
python run_experiment.py --strategy member_c_strategy_4 --prepare-only --force-preprocess
```

### 전체 전략 전처리

```bash
python run_experiment.py --all --prepare-only
```

## 2. 모델 학습 및 평가

### 단일 전략 + LightGBM

```bash
python run_experiment.py --strategy member_b_strategy_2 --model lgbm
```

### 단일 전략 + 전체 모델 비교

```bash
python run_experiment.py --strategy member_c_strategy_4 --model all
```

### 전체 전략 실행

```bash
python run_experiment.py --all --model lgbm
```

## 3. 튜닝 결과를 반영한 실행

튜닝 결과 JSON이 이미 생성되어 있다면 `--use-tuning` 옵션으로 반영할 수 있습니다.

```bash
python run_experiment.py --strategy member_c_strategy_4 --model lgbm --use-tuning
```

## 4. 하이퍼파라미터 튜닝

### 공통 LightGBM 튜닝

```bash
python -m src.optimization.shared.hyperparameter_tuner --strategy member_b_strategy_2 --trials 100
```

### member_b 전략 튜닝

```bash
python -m src.optimization.member_b.hyperparameter_tuner --strategy member_b_strategy_3 --trials 100
```

### member_c_strategy_4 커스텀 튜닝

```bash
python -m src.optimization.member_c.custom_tuning --trials 40 --cv-folds 5
```

## 5. Threshold 최적화

### 공통 threshold 분석

```bash
python -m src.optimization.shared.threshold_optimizer --strategy member_b_strategy_2
```

### member_b threshold 분석

```bash
python -m src.optimization.member_b.threshold_optimizer --strategy member_b_strategy_3
```

### member_c threshold 분석

```bash
python -m src.optimization.member_c.threshold_optimizer --strategy member_c_strategy_4
```

### 전처리까지 다시 포함해서 threshold 분석

```bash
python -m src.optimization.shared.threshold_optimizer --strategy member_c_strategy_4 --force-preprocess
```

## 6. 빠른 LightGBM 확인

Optuna, K-fold 없이 빠르게 한 번 성능만 확인할 때 사용합니다.

```bash
python run_quick_lgbm.py --strategy member_a_strategy_4
```

트리 수를 줄여 더 빠르게 확인:

```bash
python run_quick_lgbm.py --strategy member_a_strategy_4 --few-trees
```

## 7. Feature subset exhaustive search

특정 전략의 전처리 CSV를 기반으로 feature subset을 탐색합니다.

```bash
python run_feature_subset_exhaustive.py --strategy member_a_strategy_5
```

고정 개수 조합만 탐색:

```bash
python run_feature_subset_exhaustive.py --strategy member_a_strategy_5 --mode fixed_k --subset-size 8 --preselect-top 16
```

CSV 경로 직접 지정:

```bash
python run_feature_subset_exhaustive.py --csv data/processed/member_a/member_a_strategy_5_preprocessed.csv --max-exhaustive-p 12
```

## 8. 최적화 보고서 생성

### 단일 보고서 생성

```bash
python generate_report.py --strategy member_b_strategy_2 --baseline member_b
```

### 실험 직후 자동 보고서 생성

```bash
python run_experiment.py --strategy member_b_strategy_2 --model lgbm --generate-report --baseline member_b
```

## 결과물 저장 규칙

대표적인 저장 규칙은 아래처럼 이해하면 됩니다.

- 전처리 CSV: `data/processed/<member>/<strategy>_preprocessed.csv`
- 실험 결과: `artifacts/<member>/<strategy>/experiment_results.csv`
- 파이프라인: `artifacts/<member>/<strategy>/<model>_pipeline.pkl`
- tuning 결과: `artifacts/<member>/<strategy>/tuning_results.json`
- threshold 결과: `artifacts/<member>/<strategy>/threshold_analysis.json`
- 보고서: `docs/optimization_reports/<member>/<strategy>_<version>_report.md`

## 자주 쓰는 예시 모음

```bash
# member_c_strategy_4 전처리
python run_experiment.py --strategy member_c_strategy_4 --prepare-only

# member_c_strategy_4 학습
python run_experiment.py --strategy member_c_strategy_4 --model lgbm

# member_c_strategy_4 커스텀 튜닝
python -m src.optimization.member_c.custom_tuning --trials 40 --cv-folds 5

# member_c_strategy_4 threshold 최적화
python -m src.optimization.member_c.threshold_optimizer --strategy member_c_strategy_4

# member_c_strategy_4 보고서 생성
python generate_report.py --strategy member_c_strategy_4 --baseline member_c
```

## 참고

- 원본 데이터는 `data/raw/` 아래 CSV를 기준으로 사용합니다.
- `requirements.txt`에 없는 의존성이 있다면 현재 환경에 맞게 별도 설치가 필요할 수 있습니다.
- `artifacts/`, `data/processed/`, `docs/optimization_reports/`는 전략 실행 및 최적화 결과에 따라 계속 누적될 수 있습니다.
