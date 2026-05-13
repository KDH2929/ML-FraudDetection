from src.optimization.shared.smote_experiment import (
    SMOTEExperiment,
    compare_smote_strategies,
    run_smote_experiment,
)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Member B SMOTE Sampling Experiment")
    parser.add_argument("--strategy", required=True, help="Strategy ID")
    parser.add_argument(
        "--smote",
        default="smote",
        choices=["smote", "adasyn", "borderline", "all"],
        help="SMOTE strategy",
    )
    parser.add_argument("--k-neighbors", type=int, default=5, help="k_neighbors parameter")
    parser.add_argument("--output-dir", default=None, help="Output directory")

    args = parser.parse_args()

    if args.smote == "all":
        compare_smote_strategies(
            strategy_id=args.strategy,
            output_dir=args.output_dir,
        )
    else:
        run_smote_experiment(
            strategy_id=args.strategy,
            smote_strategy=args.smote,
            k_neighbors=args.k_neighbors,
            output_dir=args.output_dir,
        )
