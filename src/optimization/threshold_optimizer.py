"""
Threshold Optimization and Misclassification Analysis
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix
from pathlib import Path

from src.config import ARTIFACTS_DIR
from src.pipeline.data_loader import load_and_split
from src.utils import metrics


class ThresholdOptimizer:
    """최적 threshold 탐색 및 오분류 분석"""

    def __init__(self):
        self.best_threshold_ = None
        self.threshold_metrics_ = None

    def find_optimal_threshold(self, y_true, y_pred_proba, metric='recall'):
        """
        최적 threshold 탐색

        Args:
            y_true: 실제 레이블
            y_pred_proba: 예측 확률
            metric: 최적화 대상 ('recall', 'f1', 'precision')
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)

        # F1 score 계산
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

        # Threshold별 metrics
        threshold_metrics = pd.DataFrame({
            'threshold': np.append(thresholds, 1.0),
            'precision': precision,
            'recall': recall,
            'f1': f1_scores
        })

        self.threshold_metrics_ = threshold_metrics

        # 최적 threshold 선택
        if metric == 'recall':
            # Precision >= 0.4 조건 하에서 Recall 최대화
            valid_idx = threshold_metrics['precision'] >= 0.4
            if valid_idx.sum() > 0:
                best_idx = threshold_metrics[valid_idx]['recall'].idxmax()
            else:
                best_idx = threshold_metrics['recall'].idxmax()
        elif metric == 'f1':
            best_idx = threshold_metrics['f1'].idxmax()
        elif metric == 'precision':
            best_idx = threshold_metrics['precision'].idxmax()
        else:
            raise ValueError(f"Unknown metric: {metric}")

        self.best_threshold_ = threshold_metrics.loc[best_idx, 'threshold']

        return self.best_threshold_

    def analyze_misclassifications(self, X, y_true, y_pred, y_pred_proba, feature_names=None):
        """
        오분류 케이스 분석

        Returns:
            dict: FN/FP 케이스 통계
        """
        if feature_names is None:
            feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else list(range(X.shape[1]))

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # False Negative 분석
        fn_mask = (y_true == 1) & (y_pred == 0)
        fn_cases = X[fn_mask]
        fn_proba = y_pred_proba[fn_mask]

        # False Positive 분석
        fp_mask = (y_true == 0) & (y_pred == 1)
        fp_cases = X[fp_mask]
        fp_proba = y_pred_proba[fp_mask]

        # True Positive/Negative 분석
        tp_mask = (y_true == 1) & (y_pred == 1)
        tn_mask = (y_true == 0) & (y_pred == 0)

        analysis = {
            'confusion_matrix': {
                'TN': int(tn), 'FP': int(fp),
                'FN': int(fn), 'TP': int(tp)
            },
            'fn_count': int(fn),
            'fp_count': int(fp),
            'fn_avg_proba': float(fn_proba.mean()) if len(fn_proba) > 0 else 0,
            'fp_avg_proba': float(fp_proba.mean()) if len(fp_proba) > 0 else 0,
        }

        # Feature 평균 비교 (FN vs TP)
        if len(fn_cases) > 0 and len(X[tp_mask]) > 0:
            if isinstance(X, pd.DataFrame):
                fn_means = fn_cases.mean()
                tp_means = X[tp_mask].mean()
                diff = ((fn_means - tp_means) / (tp_means + 1e-10) * 100).abs()
                top_diff_features = diff.nlargest(10)
                analysis['fn_distinctive_features'] = top_diff_features.to_dict()

        return analysis

    def plot_threshold_curve(self, output_path=None):
        """Threshold별 metrics 시각화"""
        if self.threshold_metrics_ is None:
            raise ValueError("Run find_optimal_threshold first")

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(self.threshold_metrics_['threshold'],
                self.threshold_metrics_['precision'],
                label='Precision', linewidth=2)
        ax.plot(self.threshold_metrics_['threshold'],
                self.threshold_metrics_['recall'],
                label='Recall', linewidth=2)
        ax.plot(self.threshold_metrics_['threshold'],
                self.threshold_metrics_['f1'],
                label='F1', linewidth=2)

        if self.best_threshold_ is not None:
            ax.axvline(self.best_threshold_, color='red', linestyle='--',
                      label=f'Best Threshold: {self.best_threshold_:.3f}')

        ax.set_xlabel('Threshold', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Threshold vs Metrics', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved: {output_path}")
        else:
            plt.show()

        plt.close()


def analyze_strategy(
    strategy_id: str,
    model_path=None,
    output_dir=None,
    *,
    force_preprocess: bool = False,
):
    """
    전략 성능 분석 및 threshold 최적화

    Args:
        strategy_id: 전략 ID
        model_path: 학습된 모델 경로 (None이면 tuning_results로 새로 학습)
        output_dir: 결과 저장 디렉토리
        force_preprocess: True이면 `ensure_processed_csv` 로 CSV를 다시 생성
    """
    import lightgbm as lgb
    import json
    from src.config import MODEL_PARAMS, RANDOM_STATE
    from src.preprocessing.processed_data import ensure_processed_csv

    if output_dir is None:
        output_dir = ARTIFACTS_DIR

    processed_path = ensure_processed_csv(strategy_id, force=force_preprocess)
    if processed_path is None:
        raise FileNotFoundError(
            f"전처리 CSV를 만들 수 없습니다: {strategy_id}. 전략 구현 여부를 확인하세요."
        )
    processed_csv = Path(processed_path)
    if not processed_csv.is_file():
        raise FileNotFoundError(f"Preprocessed CSV not found: {processed_csv}")

    print(f"Loading data: {processed_csv}")
    train_X, test_X, train_y, test_y = load_and_split(str(processed_csv))

    # Train model with best params
    tuning_results_path = ARTIFACTS_DIR / f"{strategy_id}_tuning_results.json"
    if tuning_results_path.exists():
        print(f"Loading tuning results: {tuning_results_path}")
        with open(tuning_results_path, 'r') as f:
            tuning_results = json.load(f)
        best_params = tuning_results['best_params']
        print(f"Training model with best params...")
        merged = {**MODEL_PARAMS["lgbm"], **best_params}
        model = lgb.LGBMClassifier(**merged, random_state=RANDOM_STATE, verbosity=-1)
    else:
        print(f"No tuning results found. Using default params...")
        model = lgb.LGBMClassifier(**MODEL_PARAMS["lgbm"], random_state=RANDOM_STATE, verbosity=-1)

    model.fit(train_X, train_y)

    # Predict
    print("Generating predictions...")
    y_pred = model.predict(test_X)
    y_pred_proba = model.predict_proba(test_X)[:, 1]

    # Current performance
    from sklearn.metrics import precision_score
    print("\n[Current Performance (threshold=0.5)]")
    result = metrics.evaluate(test_y, y_pred)
    precision = precision_score(test_y, y_pred, pos_label=1, zero_division=0)
    print(f"  Recall: {result['recall_class1']:.4f}")
    print(f"  F1: {result['f1_class1']:.4f}")
    print(f"  Precision: {precision:.4f}")

    # Threshold optimization
    print("\n[Threshold Optimization - F1 Maximization]")
    optimizer = ThresholdOptimizer()
    best_threshold = optimizer.find_optimal_threshold(test_y, y_pred_proba, metric='f1')

    print(f"  Best threshold: {best_threshold:.4f}")

    # Re-predict with best threshold
    y_pred_optimized = (y_pred_proba >= best_threshold).astype(int)
    result_optimized = metrics.evaluate(test_y, y_pred_optimized)
    precision_optimized = precision_score(test_y, y_pred_optimized, pos_label=1, zero_division=0)

    print(f"\n[Optimized Performance (threshold={best_threshold:.4f})]")
    print(f"  Recall: {result_optimized['recall_class1']:.4f} ({result_optimized['recall_class1'] - result['recall_class1']:+.4f})")
    print(f"  F1: {result_optimized['f1_class1']:.4f} ({result_optimized['f1_class1'] - result['f1_class1']:+.4f})")
    print(f"  Precision: {precision_optimized:.4f} ({precision_optimized - precision:+.4f})")

    # Misclassification analysis
    print("\n[Misclassification Analysis]")
    analysis = optimizer.analyze_misclassifications(
        test_X, test_y, y_pred_optimized, y_pred_proba
    )

    cm = analysis['confusion_matrix']
    print(f"  Confusion Matrix:")
    print(f"    TN={cm['TN']:4d}  FP={cm['FP']:4d}")
    print(f"    FN={cm['FN']:4d}  TP={cm['TP']:4d}")
    print(f"  FN count: {analysis['fn_count']} (avg proba: {analysis['fn_avg_proba']:.3f})")
    print(f"  FP count: {analysis['fp_count']} (avg proba: {analysis['fp_avg_proba']:.3f})")

    if 'fn_distinctive_features' in analysis:
        print(f"\n  Top 10 FN distinctive features (vs TP):")
        for feat, diff in list(analysis['fn_distinctive_features'].items())[:10]:
            print(f"    {feat}: {diff:.1f}% difference")

    # Save plot
    plot_path = Path(output_dir) / f"{strategy_id}_threshold_curve.png"
    optimizer.plot_threshold_curve(output_path=plot_path)

    # Save results
    results = {
        'strategy_id': strategy_id,
        'best_threshold': float(best_threshold),
        'default_performance': {
            'recall': float(result['recall_class1']),
            'f1': float(result['f1_class1']),
            'precision': float(precision)
        },
        'optimized_performance': {
            'recall': float(result_optimized['recall_class1']),
            'f1': float(result_optimized['f1_class1']),
            'precision': float(precision_optimized)
        },
        'improvement': {
            'recall': float(result_optimized['recall_class1'] - result['recall_class1']),
            'f1': float(result_optimized['f1_class1'] - result['f1_class1']),
            'precision': float(precision_optimized - precision)
        },
        'misclassification_analysis': analysis
    }

    import json
    results_path = Path(output_dir) / f"{strategy_id}_threshold_analysis.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[Results Saved]")
    print(f"  JSON: {results_path}")
    print(f"  Plot: {plot_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Threshold Optimization and Analysis")
    parser.add_argument("--strategy", required=True, help="Strategy ID")
    parser.add_argument("--model", default=None, help="Model path")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument(
        "--force-preprocess",
        action="store_true",
        help="전처리 CSV를 다시 생성한 뒤 실행 (run_experiment 와 동일)",
    )

    args = parser.parse_args()

    analyze_strategy(
        strategy_id=args.strategy,
        model_path=args.model,
        output_dir=args.output_dir,
        force_preprocess=args.force_preprocess,
    )
