"""
member_c_strategy_4 전용 LightGBM 튜닝 스크립트
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold

from src.config import MODEL_PARAMS, RANDOM_STATE
from src.models.model_factory import get_model
from src.pipeline.data_loader import load_and_split
from src.project_paths import custom_tuning_results_path, processed_csv_path, tuning_results_path
from src.utils import metrics


STRATEGY_ID = "member_c_strategy_4"
OUTPUT_PATH = custom_tuning_results_path(STRATEGY_ID)
COMPAT_OUTPUT_PATH = tuning_results_path(STRATEGY_ID)
SUPPORTED_METRICS = ("recall_class1", "f1_class1", "f1_macro")


class MemberCStrategy4CustomTuner:
    """member_c_strategy_4 전용 F1 중심 튜너"""

    def __init__(self, n_trials: int = 40, cv_folds: int = 5):
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.best_params_ = None
        self.best_cv_scores_ = None
        self.study_ = None

    def _fixed_params(self) -> dict:
        params = dict(MODEL_PARAMS["lgbm"])
        params.update(
            {
                "objective": "binary",
                "metric": "binary_logloss",
                "verbosity": -1,
                "random_state": RANDOM_STATE,
                "n_jobs": -1,
            }
        )
        return params

    def _suggest_params(self, trial: optuna.Trial) -> dict:
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 120),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "n_estimators": trial.suggest_int("n_estimators", 200, 900),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 12.0),
        }

    def _cross_validate(self, params: dict, x_train, y_train) -> dict:
        cv = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        fold_scores = []
        for fit_idx, valid_idx in cv.split(x_train, y_train):
            x_fit = x_train.iloc[fit_idx]
            x_valid = x_train.iloc[valid_idx]
            y_fit = y_train.iloc[fit_idx]
            y_valid = y_train.iloc[valid_idx]

            model = lgb.LGBMClassifier(**params)
            model.fit(x_fit, y_fit)
            y_pred = model.predict(x_valid)
            fold_scores.append(metrics.evaluate(y_valid, y_pred))

        averaged = {}
        for metric_name in SUPPORTED_METRICS:
            averaged[metric_name] = sum(
                score[metric_name] for score in fold_scores
            ) / len(fold_scores)
        return averaged

    def _objective(self, trial: optuna.Trial, x_train, y_train):
        params = self._fixed_params()
        params.update(self._suggest_params(trial))

        cv_scores = self._cross_validate(params, x_train, y_train)
        trial.set_user_attr("cv_scores", cv_scores)
        trial.set_user_attr("full_params", params)
        return cv_scores["f1_class1"]

    def tune(self, x_train, y_train):
        self.study_ = optuna.create_study(
            direction="maximize",
            study_name=f"{STRATEGY_ID}_custom_tuning",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        )
        self.study_.optimize(
            lambda trial: self._objective(trial, x_train, y_train),
            n_trials=self.n_trials,
            show_progress_bar=True,
        )

        best_trial = self.study_.best_trial
        self.best_params_ = dict(best_trial.user_attrs["full_params"])
        self.best_cv_scores_ = dict(best_trial.user_attrs["cv_scores"])
        return self.best_params_


def _evaluate_model(model, x_train, y_train, x_test, y_test) -> dict:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return metrics.evaluate(y_test, y_pred)


def run_custom_tuning(n_trials: int = 40, cv_folds: int = 5):
    processed_csv = processed_csv_path(STRATEGY_ID)
    if not processed_csv.exists():
        raise FileNotFoundError(
            f"Preprocessed CSV not found: {processed_csv}. "
            f"Run preprocessing first."
        )

    print(f"Loading preprocessed data: {processed_csv}")
    x_train, x_test, y_train, y_test = load_and_split(str(processed_csv))

    print("\n[Baseline Test Performance]")
    baseline_model = get_model("lgbm")
    baseline_result = _evaluate_model(baseline_model, x_train, y_train, x_test, y_test)
    for key, value in baseline_result.items():
        print(f"  {key}: {value:.4f}")

    tuner = MemberCStrategy4CustomTuner(n_trials=n_trials, cv_folds=cv_folds)
    print(f"\n[Custom Tuning] trials={n_trials}, cv_folds={cv_folds}, optimize=f1_class1")
    best_params = tuner.tune(x_train, y_train)

    print("\n[Best CV Scores]")
    for key, value in tuner.best_cv_scores_.items():
        print(f"  {key}: {value:.4f}")

    print("\n[Tuned Test Performance]")
    tuned_model = lgb.LGBMClassifier(**best_params)
    tuned_result = _evaluate_model(tuned_model, x_train, y_train, x_test, y_test)
    for key, value in tuned_result.items():
        print(f"  {key}: {value:.4f} ({value - baseline_result[key]:+.4f})")

    top_trials = []
    for trial in sorted(
        tuner.study_.trials,
        key=lambda item: item.value if item.value is not None else float("-inf"),
        reverse=True,
    )[:10]:
        top_trials.append(
            {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "cv_scores": trial.user_attrs.get("cv_scores", {}),
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "strategy_id": STRATEGY_ID,
        "mode": "custom_separated_tuning",
        "n_trials": n_trials,
        "cv_folds": cv_folds,
        "optimize_metric": "f1_class1",
        "best_params": best_params,
        "best_cv_scores": tuner.best_cv_scores_,
        "baseline_test_result": baseline_result,
        "tuned_test_result": tuned_result,
        "test_improvement": {
            key: tuned_result[key] - baseline_result[key]
            for key in baseline_result.keys()
        },
        "top_trials": top_trials,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # 공용 experiment_runner 호환용 결과 파일도 함께 저장한다.
    compat_params = {
        key: value
        for key, value in best_params.items()
        if key not in {"random_state", "verbosity"}
    }
    compat_payload = {
        "best_params": compat_params,
        "best_cv_recall": tuner.best_cv_scores_["recall_class1"],
        "n_trials": n_trials,
        "cv_folds": cv_folds,
        "source": "member_c_strategy_4_custom_tuning",
        "optimize_metric": "f1_class1",
        "best_cv_scores": tuner.best_cv_scores_,
    }
    with open(COMPAT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(compat_payload, f, ensure_ascii=False, indent=2)

    print(f"\n[Saved] {OUTPUT_PATH}")
    print(f"[Saved] {COMPAT_OUTPUT_PATH}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="member_c_strategy_4 custom tuning")
    parser.add_argument("--trials", type=int, default=40, help="Optuna trials")
    parser.add_argument("--cv-folds", type=int, default=5, help="CV folds")
    args = parser.parse_args()

    run_custom_tuning(n_trials=args.trials, cv_folds=args.cv_folds)
