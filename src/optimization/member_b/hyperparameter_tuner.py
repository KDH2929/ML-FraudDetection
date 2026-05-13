from src.optimization.shared.hyperparameter_tuner import LGBMTuner, tune_strategy


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Member B LightGBM Hyperparameter Tuning")
    parser.add_argument("--strategy", required=True, help="Strategy ID")
    parser.add_argument("--trials", type=int, default=100, help="Number of Optuna trials")
    parser.add_argument("--output-dir", default=None, help="Output directory")

    args = parser.parse_args()

    tune_strategy(
        strategy_id=args.strategy,
        n_trials=args.trials,
        output_dir=args.output_dir,
    )
