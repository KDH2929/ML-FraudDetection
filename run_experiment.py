import argparse

from src.experiment.experiment_runner import (
    MODEL_OPTIONS,
    STRATEGIES,
    run,
    run_all,
)
from src.preprocessing.processed_data import ensure_processed_csv, ensure_processed_csvs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="보험사기 실험 실행기")
    parser.add_argument("--member", type=str, choices=STRATEGIES, help="실행할 전략 ID 1개")
    parser.add_argument("--strategy", type=str, choices=STRATEGIES, help="실행할 전략 ID 1개")
    parser.add_argument(
        "--model",
        type=str,
        default="lgbm",
        choices=MODEL_OPTIONS + ["all"],
        help="실행할 모델 또는 'all'",
    )
    parser.add_argument("--all", action="store_true", help="구현된 모든 전략 실행")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="전처리 CSV만 만들고 실험은 실행하지 않음",
    )
    parser.add_argument(
        "--force-preprocess",
        action="store_true",
        help="이미 CSV가 있어도 전처리를 다시 실행함",
    )
    args = parser.parse_args()

    target_strategy = args.strategy or args.member

    if args.prepare_only:
        if args.all:
            ensure_processed_csvs(force=args.force_preprocess)
        elif target_strategy:
            ensure_processed_csv(target_strategy, force=args.force_preprocess)
        else:
            parser.error("--prepare-only 는 --strategy/--member 또는 --all 과 함께 사용해야 합니다.")
    elif args.all:
        run_all(
            model_name=args.model,
            force_preprocess=args.force_preprocess,
        )
    elif target_strategy:
        run(
            target_strategy,
            model_name=args.model,
            force_preprocess=args.force_preprocess,
        )
    else:
        parser.print_help()
