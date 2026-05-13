"""
Backward-compatible CLI for threshold search.

실행 예:
  python -m src.optimization.threshold_optimizer --strategy test_member_abc

로직은 ``src.optimization.shared.threshold_optimizer`` 에 있습니다.
"""
from __future__ import annotations

from src.optimization.shared.threshold_optimizer import ThresholdOptimizer, analyze_strategy

__all__ = ["ThresholdOptimizer", "analyze_strategy"]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Threshold Optimization and Analysis")
    parser.add_argument("--strategy", required=True, help="Strategy ID")
    parser.add_argument("--model", default=None, help="Model path")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument(
        "--force-preprocess",
        action="store_true",
        help="Regenerate the preprocessed CSV before analysis.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=None,
        metavar="N",
        help="Override LGBM n_estimators (lower = faster).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Shorthand for --n-estimators 200.",
    )
    args = parser.parse_args()
    n_est = args.n_estimators
    if args.quick and n_est is None:
        n_est = 200
    analyze_strategy(
        strategy_id=args.strategy,
        model_path=args.model,
        output_dir=args.output_dir,
        force_preprocess=args.force_preprocess,
        n_estimators=n_est,
    )
