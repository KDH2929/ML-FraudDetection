from src.optimization.shared.threshold_optimizer import ThresholdOptimizer, analyze_strategy


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Member B Threshold Optimization and Analysis")
    parser.add_argument("--strategy", required=True, help="Strategy ID")
    parser.add_argument("--model", default=None, help="Model path")
    parser.add_argument("--output-dir", default=None, help="Output directory")

    args = parser.parse_args()

    analyze_strategy(
        strategy_id=args.strategy,
        model_path=args.model,
        output_dir=args.output_dir,
    )
