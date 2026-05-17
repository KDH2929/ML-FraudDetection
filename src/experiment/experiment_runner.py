import json

import pandas as pd

from src.config import MODEL_PARAMS, RANDOM_STATE, STRATEGY_THRESHOLDS
from src.models.model_factory import get_model
from src.pipeline.data_loader import load_and_split
from src.preprocessing.preprocessor_factory import list_strategies
from src.preprocessing.processed_data import ensure_processed_csv, ensure_processed_csvs
from src.project_paths import (
    experiment_result_path,
    pipeline_path,
    summary_best_info_path,
    summary_best_pipeline_path,
    summary_experiment_results_path,
    threshold_analysis_path,
    tuning_results_path,
)
from src.utils import metrics, save_load


STRATEGIES = list_strategies()
MODEL_OPTIONS = ["lgbm", "rf", "logistic", "xgboost", "catboost", "voting", "stacking"]


def _threshold_for_strategy(strategy_id: str):
    if strategy_id in STRATEGY_THRESHOLDS:
        return float(STRATEGY_THRESHOLDS[strategy_id])

    path = threshold_analysis_path(strategy_id)
    if not path.is_file():
        return None

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    best_threshold = data.get("best_threshold")
    return float(best_threshold) if best_threshold is not None else None


def _resolve_models(model_name="lgbm"):
    normalized = str(model_name).lower()
    if normalized == "all":
        return list(MODEL_OPTIONS)
    return [normalized]


def _run_single_model(strategy_id: str, processed_csv: str, model_name: str, use_optimization: bool = False):
    from src.pipeline import pipeline_builder, trainer

    train_X, test_X, train_y, test_y = load_and_split(str(processed_csv))
    pos = int((train_y == 1).sum())
    neg = int((train_y == 0).sum())
    scale_pos_weight = float(neg) / max(1, pos)

    tuning_path = tuning_results_path(strategy_id)
    if use_optimization and tuning_path.exists() and model_name == "lgbm":
        import lightgbm as lgb

        with open(tuning_path, encoding="utf-8") as f:
            tuning_results = json.load(f)
        best_params = tuning_results["best_params"]
        merged = {**MODEL_PARAMS["lgbm"], **best_params}
        merged["random_state"] = RANDOM_STATE
        merged["verbosity"] = -1
        print(f"  [Using tuned params from {tuning_path.name}]")
        model = lgb.LGBMClassifier(**merged)
    else:
        model = get_model(model_name, scale_pos_weight=scale_pos_weight)

    pipeline = pipeline_builder.build(strategy=None, model=model, sampler=None, selector=None)
    fitted = trainer.train(pipeline, train_X, train_y)

    threshold = None
    if use_optimization and model_name == "lgbm" and hasattr(fitted, "predict_proba"):
        threshold = _threshold_for_strategy(strategy_id)

    if threshold is not None and model_name == "lgbm" and hasattr(fitted, "predict_proba"):
        y_pred_proba = fitted.predict_proba(test_X)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
    else:
        y_pred = fitted.predict(test_X)

    result = metrics.evaluate(test_y, y_pred)
    result["strategy"] = strategy_id
    result["model"] = model_name
    if threshold is not None:
        result["threshold"] = threshold
    return result, fitted


def run_strategy(
    strategy_id: str,
    model_name: str = "lgbm",
    force_preprocess: bool = False,
    use_optimization: bool = False,
    force_stage_cache: bool = False,
):
    processed_csv = ensure_processed_csv(
        strategy_id,
        force=force_preprocess,
        force_stage_cache=force_stage_cache,
    )
    if processed_csv is None:
        return []

    results = []
    best_result = None
    best_pipeline = None

    for current_model in _resolve_models(model_name=model_name):
        result, fitted = _run_single_model(
            strategy_id=strategy_id,
            processed_csv=processed_csv,
            model_name=current_model,
            use_optimization=use_optimization,
        )
        metrics.print_report(
            strategy_id,
            {
                "model": result["model"],
                "recall_class1": result["recall_class1"],
                "f1 (class 1)": result["f1_class1"],
                "f1_macro": result["f1_macro"],
            },
        )
        results.append(result)
        if best_result is None or result["f1_class1"] > best_result["f1_class1"]:
            best_result = result
            best_pipeline = fitted

    result_df = pd.DataFrame(results).sort_values(
        by=["f1_class1", "recall_class1", "f1_macro"],
        ascending=False,
    )
    save_load.save_result(result_df, experiment_result_path(strategy_id))
    if best_pipeline is not None:
        save_load.save_pipeline(
            best_pipeline,
            pipeline_path(strategy_id, best_result["model"]),
        )
    return results


def run(
    strategy_id: str,
    model_name: str = "lgbm",
    force_preprocess: bool = False,
    use_optimization: bool = False,
    force_stage_cache: bool = False,
):
    return run_strategy(
        strategy_id=strategy_id,
        model_name=model_name,
        force_preprocess=force_preprocess,
        use_optimization=use_optimization,
        force_stage_cache=force_stage_cache,
    )


def run_all(
    model_name: str = "lgbm",
    force_preprocess: bool = False,
    use_optimization: bool = False,
    force_stage_cache: bool = False,
):
    ensure_processed_csvs(
        STRATEGIES,
        force=force_preprocess,
        force_stage_cache=force_stage_cache,
    )

    all_results = []
    for strategy_id in STRATEGIES:
        strategy_results = run_strategy(
            strategy_id=strategy_id,
            model_name=model_name,
            force_preprocess=False,
            use_optimization=use_optimization,
            force_stage_cache=False,
        )
        all_results.extend(strategy_results)

    if not all_results:
        print("No runnable strategies were found. Check strategy implementation and preprocessing outputs.")
        return []

    summary = pd.DataFrame(all_results).sort_values(
        by=["f1_class1", "recall_class1", "f1_macro"],
        ascending=False,
    )
    save_load.save_result(summary, summary_experiment_results_path())

    best = summary.iloc[0].to_dict()
    best_info = {
        "best_strategy": best.get("strategy"),
        "model": best.get("model"),
        "f1_class1": best.get("f1_class1"),
        "recall_class1": best.get("recall_class1"),
        "f1_macro": best.get("f1_macro"),
    }

    summary_best_info_path().parent.mkdir(parents=True, exist_ok=True)
    with open(summary_best_info_path(), "w", encoding="utf-8") as f:
        json.dump(best_info, f, ensure_ascii=False, indent=2)

    best_pipeline_source = pipeline_path(best["strategy"], best["model"])
    if best_pipeline_source.exists():
        save_load.save_pipeline(
            save_load.load_pipeline(best_pipeline_source),
            summary_best_pipeline_path(),
        )

    print("\n===== Overall Comparison =====")
    print(summary.to_string(index=False))
    print(f"\nBest combination: {best_info}")
    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="단일/전체 전략 실험 (루트에서: python -m src.experiment.experiment_runner ...)",
    )
    parser.add_argument("--strategy", type=str, default=None, help="전략 ID (예: test_member_abc). --all 이면 생략 가능")
    parser.add_argument("--model", type=str, default="lgbm", choices=MODEL_OPTIONS + ["all"])
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument(
        "--force-stage-cache",
        action="store_true",
        help="Rebuild cached intermediate preprocessing stages too.",
    )
    parser.add_argument(
        "--use-tuning",
        action="store_true",
        help="artifacts/.../tuning_results.json 있으면 LGBM 파라미터 병합",
    )
    parser.add_argument(
        "--use-optimization",
        action="store_true",
        help="--use-tuning 과 동일(예전 이름 호환)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="등록된 모든 전략에 대해 실행",
    )
    args = parser.parse_args()
    use_opt = args.use_tuning or args.use_optimization
    if args.all:
        run_all(
            model_name=args.model,
            force_preprocess=args.force_preprocess,
            use_optimization=use_opt,
            force_stage_cache=args.force_stage_cache,
        )
    elif args.strategy:
        run(
            args.strategy,
            model_name=args.model,
            force_preprocess=args.force_preprocess,
            use_optimization=use_opt,
            force_stage_cache=args.force_stage_cache,
        )
    else:
        parser.error("--strategy 가 필요합니다 (--all 이 아닐 때).")
