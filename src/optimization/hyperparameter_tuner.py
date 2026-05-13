"""
LightGBM Hyperparameter Tuning with Optuna
"""
import json
from pathlib import Path

import lightgbm as lgb
import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold

from src.config import ARTIFACTS_DIR, RANDOM_STATE
from src.pipeline.data_loader import load_and_split


class LGBMTuner:
    """LightGBM Hyperparameter Tuning with Optuna"""

    def __init__(self, n_trials=100, cv_folds=5, random_state=RANDOM_STATE):
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.best_params_ = None
        self.study_ = None

    def _objective(self, trial, X_train, y_train):
        """Optuna objective function"""
        # Hyperparameter search space
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'random_state': self.random_state,
            'n_jobs': -1,
            # Tuning parameters
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        }

        # Cross-validation
        model = lgb.LGBMClassifier(**params)
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        # Recall score (class 1)
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv,
            scoring='recall',
            n_jobs=-1
        )

        return scores.mean()

    def tune(self, X_train, y_train, study_name='lgbm_tuning'):
        """Run hyperparameter tuning"""
        print(f"Starting Optuna hyperparameter tuning ({self.n_trials} trials)...")
        print(f"Training data shape: {X_train.shape}")

        self.study_ = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )

        self.study_.optimize(
            lambda trial: self._objective(trial, X_train, y_train),
            n_trials=self.n_trials,
            show_progress_bar=True
        )

        self.best_params_ = self.study_.best_params

        print(f"\n[Tuning Complete]")
        print(f"  Best CV Recall: {self.study_.best_value:.4f}")
        print(f"  Best Parameters:")
        for key, value in self.best_params_.items():
            print(f"    {key}: {value}")

        return self.best_params_

    def save_results(self, output_path):
        """Save tuning results to JSON"""
        if self.best_params_ is None:
            raise ValueError("No tuning results to save. Run tune() first.")

        results = {
            'best_params': self.best_params_,
            'best_cv_recall': self.study_.best_value,
            'n_trials': self.n_trials,
            'cv_folds': self.cv_folds,
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n[Results Saved] {output_path}")


def tune_strategy(strategy_id: str, n_trials=100, output_dir=None):
    """
    Run hyperparameter tuning for a given strategy

    Args:
        strategy_id: Strategy ID (e.g., 'member_b_strategy_3_optimized')
        n_trials: Number of Optuna trials
        output_dir: Output directory for results (default: ARTIFACTS_DIR)
    """
    if output_dir is None:
        output_dir = ARTIFACTS_DIR

    # Load processed data
    processed_csv = Path(f"data/processed/{strategy_id}_preprocessed.csv")
    if not processed_csv.exists():
        raise FileNotFoundError(
            f"Preprocessed CSV not found: {processed_csv}\n"
            f"Run: python run_experiment.py --strategy {strategy_id} --prepare-only"
        )

    print(f"Loading preprocessed data: {processed_csv}")
    train_X, test_X, train_y, test_y = load_and_split(str(processed_csv))

    # Run tuning
    tuner = LGBMTuner(n_trials=n_trials)
    best_params = tuner.tune(train_X, train_y, study_name=f"{strategy_id}_tuning")

    # Save results
    output_path = Path(output_dir) / f"{strategy_id}_tuning_results.json"
    tuner.save_results(output_path)

    # Test with best parameters
    print(f"\n[Testing with best parameters]")
    model = lgb.LGBMClassifier(**best_params, random_state=RANDOM_STATE, verbosity=-1)
    model.fit(train_X, train_y)

    from src.utils import metrics
    y_pred = model.predict(test_X)
    result = metrics.evaluate(test_y, y_pred)

    print(f"  Test Recall: {result['recall_class1']:.4f}")
    print(f"  Test F1: {result['f1_class1']:.4f}")
    print(f"  Test F1-macro: {result['f1_macro']:.4f}")

    return best_params, result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LightGBM Hyperparameter Tuning")
    parser.add_argument("--strategy", required=True, help="Strategy ID")
    parser.add_argument("--trials", type=int, default=100, help="Number of Optuna trials")
    parser.add_argument("--output-dir", default=None, help="Output directory")

    args = parser.parse_args()

    tune_strategy(
        strategy_id=args.strategy,
        n_trials=args.trials,
        output_dir=args.output_dir
    )
