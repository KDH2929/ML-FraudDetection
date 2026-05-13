"""
빠른 스모크: Optuna·K-fold 없이, 기본 `MODEL_PARAMS['lgbm']` + `load_and_split` 한 번만.

  python run_quick_lgbm.py --strategy member_a_strategy_4
  python run_quick_lgbm.py --strategy member_a --few-trees

`--few-trees` 는 n_estimators=120 정도로만 줄여 더 빨리 돌린다(지표는 참고용).
"""

from __future__ import annotations

import argparse

from lightgbm import LGBMClassifier

from src.config import MODEL_PARAMS
from src.pipeline.data_loader import load_and_split
from src.preprocessing.processed_data import ensure_processed_csv
from src.utils import metrics


def main() -> None:
    p = argparse.ArgumentParser(description="Quick LGBM eval (no Optuna, no K-fold)")
    p.add_argument("--strategy", type=str, default="member_a_strategy_4")
    p.add_argument("--force-preprocess", action="store_true")
    p.add_argument(
        "--few-trees",
        action="store_true",
        help="n_estimators를 줄여 더 빠르게 (기본 파라미터는 config 그대로)",
    )
    args = p.parse_args()

    path = ensure_processed_csv(args.strategy, force=args.force_preprocess)
    if path is None:
        raise SystemExit("전처리 CSV를 만들 수 없습니다.")

    train_X, test_X, train_y, test_y = load_and_split(str(path))
    params = dict(MODEL_PARAMS["lgbm"])
    if args.few_trees:
        params["n_estimators"] = min(120, int(params.get("n_estimators", 500)))

    clf = LGBMClassifier(**params)
    clf.fit(train_X, train_y)
    pred = clf.predict(test_X)
    out = metrics.evaluate(test_y, pred)
    out["strategy"] = args.strategy
    out["n_estimators_used"] = params["n_estimators"]

    metrics.print_report(args.strategy, {k: v for k, v in out.items() if k != "strategy"})


if __name__ == "__main__":
    main()
