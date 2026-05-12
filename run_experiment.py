import argparse
from src.experiment.experiment_runner import run, run_all

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="보험 사기자 탐지 실험 실행")
    parser.add_argument("--member", type=str, choices=["member_a", "member_b", "member_c"],
                        help="특정 멤버만 실행")
    parser.add_argument("--model", type=str, default="lgbm", choices=["lgbm", "rf", "logistic"],
                        help="사용할 모델 (기본: lgbm)")
    parser.add_argument("--sampler", type=str, default="smote",
                        help="오버샘플러 방법: smote, adasyn, borderlinesmote, none (기본: smote)")
    parser.add_argument("--selector", type=str, default=None,
                        help="피처 선택 방법: variance, model, none (기본: none)")
    parser.add_argument("--all", action="store_true",
                        help="전체 멤버 비교 실험 실행")
    args = parser.parse_args()

    if args.all:
        run_all(model_name=args.model, sampler_method=args.sampler, selector_method=args.selector)
    elif args.member:
        run(args.member, model_name=args.model, sampler_method=args.sampler, selector_method=args.selector)
    else:
        parser.print_help()
