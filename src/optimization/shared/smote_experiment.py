"""
SMOTE (Synthetic Minority Over-sampling Technique) 실험
"""
import json
from pathlib import Path

import lightgbm as lgb
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from sklearn.metrics import precision_score

from src.config import RANDOM_STATE
from src.pipeline.data_loader import load_and_split
from src.project_paths import processed_csv_path, strategy_artifact_dir, tuning_results_path
from src.utils import metrics


class SMOTEExperiment:
    """SMOTE 샘플링 실험"""

    def __init__(self, strategy='smote', k_neighbors=5, random_state=RANDOM_STATE):
        """
        Args:
            strategy: 'smote', 'adasyn', 'borderline'
            k_neighbors: SMOTE k_neighbors 파라미터
            random_state: Random seed
        """
        self.strategy = strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.sampler_ = None

    def _create_sampler(self):
        """샘플러 생성"""
        if self.strategy == 'smote':
            return SMOTE(
                k_neighbors=self.k_neighbors,
                random_state=self.random_state
            )
        elif self.strategy == 'adasyn':
            return ADASYN(
                n_neighbors=self.k_neighbors,
                random_state=self.random_state
            )
        elif self.strategy == 'borderline':
            return BorderlineSMOTE(
                k_neighbors=self.k_neighbors,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def run_experiment(self, train_X, train_y, test_X, test_y, model_params=None):
        """
        SMOTE 실험 실행

        Returns:
            dict: 실험 결과
        """
        # Original class distribution
        original_dist = train_y.value_counts().to_dict()
        print(f"\n[Original Distribution]")
        print(f"  Class 0: {original_dist.get(0, 0)}")
        print(f"  Class 1: {original_dist.get(1, 0)}")
        print(f"  Ratio: {original_dist.get(1, 0) / original_dist.get(0, 1):.4f}")

        # Apply SMOTE
        print(f"\n[Applying {self.strategy.upper()}...]")
        self.sampler_ = self._create_sampler()
        train_X_resampled, train_y_resampled = self.sampler_.fit_resample(train_X, train_y)

        # Resampled distribution
        resampled_dist = train_y_resampled.value_counts().to_dict()
        print(f"\n[Resampled Distribution]")
        print(f"  Class 0: {resampled_dist.get(0, 0)}")
        print(f"  Class 1: {resampled_dist.get(1, 0)}")
        print(f"  Ratio: {resampled_dist.get(1, 0) / resampled_dist.get(0, 1):.4f}")
        print(f"  Total samples: {len(train_X)} -> {len(train_X_resampled)} (+{len(train_X_resampled) - len(train_X)})")

        # Train without SMOTE (baseline)
        print(f"\n[Training Baseline (No SMOTE)]")
        if model_params is None:
            model_params = {}
        model_baseline = lgb.LGBMClassifier(**model_params, random_state=self.random_state, verbosity=-1)
        model_baseline.fit(train_X, train_y)

        y_pred_baseline = model_baseline.predict(test_X)
        result_baseline = metrics.evaluate(test_y, y_pred_baseline)
        precision_baseline = precision_score(test_y, y_pred_baseline, pos_label=1, zero_division=0)

        print(f"  Recall: {result_baseline['recall_class1']:.4f}")
        print(f"  F1: {result_baseline['f1_class1']:.4f}")
        print(f"  Precision: {precision_baseline:.4f}")

        # Train with SMOTE
        print(f"\n[Training with {self.strategy.upper()}]")
        model_smote = lgb.LGBMClassifier(**model_params, random_state=self.random_state, verbosity=-1)
        model_smote.fit(train_X_resampled, train_y_resampled)

        y_pred_smote = model_smote.predict(test_X)
        result_smote = metrics.evaluate(test_y, y_pred_smote)
        precision_smote = precision_score(test_y, y_pred_smote, pos_label=1, zero_division=0)

        print(f"  Recall: {result_smote['recall_class1']:.4f}")
        print(f"  F1: {result_smote['f1_class1']:.4f}")
        print(f"  Precision: {precision_smote:.4f}")

        # Improvement
        improvement = {
            'recall': result_smote['recall_class1'] - result_baseline['recall_class1'],
            'f1': result_smote['f1_class1'] - result_baseline['f1_class1'],
            'precision': precision_smote - precision_baseline
        }

        print(f"\n[Improvement]")
        print(f"  Recall: {improvement['recall']:+.4f} ({improvement['recall']/result_baseline['recall_class1']*100:+.2f}%)")
        print(f"  F1: {improvement['f1']:+.4f} ({improvement['f1']/result_baseline['f1_class1']*100:+.2f}%)")
        print(f"  Precision: {improvement['precision']:+.4f}")

        return {
            'strategy': self.strategy,
            'k_neighbors': self.k_neighbors,
            'original_distribution': original_dist,
            'resampled_distribution': resampled_dist,
            'baseline': {
                'recall': float(result_baseline['recall_class1']),
                'f1': float(result_baseline['f1_class1']),
                'precision': float(precision_baseline)
            },
            'smote': {
                'recall': float(result_smote['recall_class1']),
                'f1': float(result_smote['f1_class1']),
                'precision': float(precision_smote)
            },
            'improvement': {
                'recall': float(improvement['recall']),
                'f1': float(improvement['f1']),
                'precision': float(improvement['precision'])
            }
        }


def run_smote_experiment(strategy_id: str, smote_strategy='smote', k_neighbors=5, output_dir=None):
    """
    SMOTE 실험 실행

    Args:
        strategy_id: 전략 ID
        smote_strategy: 'smote', 'adasyn', 'borderline'
        k_neighbors: SMOTE k_neighbors
        output_dir: 결과 저장 디렉토리
    """
    base_output_dir = Path(output_dir) if output_dir is not None else strategy_artifact_dir(strategy_id)

    # Load data
    processed_csv = processed_csv_path(strategy_id)
    if not processed_csv.exists():
        raise FileNotFoundError(f"Preprocessed CSV not found: {processed_csv}")

    print(f"Loading data: {processed_csv}")
    train_X, test_X, train_y, test_y = load_and_split(str(processed_csv))

    # Load tuning results for model params
    tuning_path = tuning_results_path(strategy_id)
    model_params = {}
    if tuning_path.exists():
        print(f"Loading tuning results: {tuning_path}")
        with open(tuning_path, 'r') as f:
            tuning_results = json.load(f)
        model_params = tuning_results['best_params']
        print(f"Using best params from tuning")
    else:
        print(f"No tuning results found. Using default params")

    # Run experiment
    experiment = SMOTEExperiment(strategy=smote_strategy, k_neighbors=k_neighbors)
    results = experiment.run_experiment(train_X, train_y, test_X, test_y, model_params=model_params)

    # Save results
    results_path = base_output_dir / f"smote_{smote_strategy}_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[Results Saved] {results_path}")

    return results


def compare_smote_strategies(strategy_id: str, output_dir=None):
    """여러 SMOTE 전략 비교"""
    strategies = ['smote', 'adasyn', 'borderline']
    results_all = {}

    for smote_strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Testing {smote_strategy.upper()}")
        print(f"{'='*60}")

        try:
            results = run_smote_experiment(
                strategy_id=strategy_id,
                smote_strategy=smote_strategy,
                output_dir=output_dir
            )
            results_all[smote_strategy] = results
        except Exception as e:
            print(f"Error with {smote_strategy}: {e}")
            continue

    # Summary comparison
    print(f"\n{'='*60}")
    print("SMOTE Strategy Comparison")
    print(f"{'='*60}")
    print(f"{'Strategy':<15} {'Recall':>10} {'F1':>10} {'Precision':>10}")
    print(f"{'-'*60}")

    for strategy, result in results_all.items():
        print(f"{strategy:<15} "
              f"{result['smote']['recall']:>10.4f} "
              f"{result['smote']['f1']:>10.4f} "
              f"{result['smote']['precision']:>10.4f}")

    # Save comparison
    if output_dir:
        comparison_path = Path(output_dir) / "smote_comparison.json"
        with open(comparison_path, 'w') as f:
            json.dump(results_all, f, indent=2)
        print(f"\n[Comparison Saved] {comparison_path}")

    return results_all


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SMOTE Sampling Experiment")
    parser.add_argument("--strategy", required=True, help="Strategy ID")
    parser.add_argument("--smote", default='smote', choices=['smote', 'adasyn', 'borderline', 'all'],
                        help="SMOTE strategy")
    parser.add_argument("--k-neighbors", type=int, default=5, help="k_neighbors parameter")
    parser.add_argument("--output-dir", default=None, help="Output directory")

    args = parser.parse_args()

    if args.smote == 'all':
        compare_smote_strategies(
            strategy_id=args.strategy,
            output_dir=args.output_dir
        )
    else:
        run_smote_experiment(
            strategy_id=args.strategy,
            smote_strategy=args.smote,
            k_neighbors=args.k_neighbors,
            output_dir=args.output_dir
        )
