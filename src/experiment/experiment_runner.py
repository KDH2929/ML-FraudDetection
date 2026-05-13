import json

import pandas as pd

from src.config import ARTIFACTS_DIR, STRATEGY_THRESHOLDS, RANDOM_STATE
from src.models.model_factory import get_model
from src.pipeline import pipeline_builder, trainer
from src.pipeline.data_loader import load_and_split
from src.preprocessing.preprocessor_factory import list_strategies
from src.preprocessing.processed_data import ensure_processed_csv, ensure_processed_csvs
from src.utils import metrics, save_load


STRATEGIES = list_strategies()
MODEL_OPTIONS = ["lgbm", "rf", "logistic", "xgboost", "catboost", "voting", "stacking"]


def _resolve_models(model_name="lgbm"):
    normalized = str(model_name).lower()
    if normalized == "all":
        return list(MODEL_OPTIONS)
    return [normalized]


def _result_filename(strategy_id: str):
    return ARTIFACTS_DIR / f"experiment_result_{strategy_id}.csv"


def _run_single_model(strategy_id: str, processed_csv: str, model_name: str):
    train_X, test_X, train_y, test_y = load_and_split(str(processed_csv))

    # tuning 결과가 있으면 자동 로드
    tuning_results_path = ARTIFACTS_DIR / f"{strategy_id}_tuning_results.json"
    if tuning_results_path.exists() and model_name == "lgbm":
        import lightgbm as lgb
        with open(tuning_results_path, 'r', encoding='utf-8') as f:
            tuning_results = json.load(f)
        best_params = tuning_results['best_params']
        model = lgb.LGBMClassifier(**best_params, random_state=RANDOM_STATE, verbosity=-1)
        print(f"  [Using tuned params from {tuning_results_path.name}]")
    else:
        model = get_model(model_name)

    # 현재 공통 실행은 모델 비교까지만 담당하고, 세부 최적화는 수동으로 진행한다.
    pipeline = pipeline_builder.build(strategy=None, model=model, sampler=None, selector=None)
    fitted = trainer.train(pipeline, train_X, train_y)

    # 전략별 최적 threshold 적용
    if strategy_id in STRATEGY_THRESHOLDS:
        threshold = STRATEGY_THRESHOLDS[strategy_id]
        y_pred_proba = fitted.predict_proba(test_X)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
    else:
        y_pred = fitted.predict(test_X)

    result = metrics.evaluate(test_y, y_pred)
    result["strategy"] = strategy_id
    result["model"] = model_name
    if strategy_id in STRATEGY_THRESHOLDS:
        result["threshold"] = threshold
    return result, fitted


def run_strategy(
    strategy_id: str,
    model_name: str = "lgbm",
    force_preprocess: bool = False,
):
    # CSV가 없으면 전략을 실행해 만들고, 있으면 그대로 재사용한다.
    processed_csv = ensure_processed_csv(strategy_id, force=force_preprocess)
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
        )
        metrics.print_report(
            strategy_id,
            {
                "model": result["model"],
                "recall_class1": result["recall_class1"],
                "f1_class1": result["f1_class1"],
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
    save_load.save_result(result_df, _result_filename(strategy_id))
    if best_pipeline is not None:
        save_load.save_pipeline(
            best_pipeline,
            ARTIFACTS_DIR / f"{strategy_id}_{best_result['model']}_pipeline.pkl",
        )
    return results


def run(
    strategy_id: str,
    model_name: str = "lgbm",
    force_preprocess: bool = False,
):
    return run_strategy(
        strategy_id=strategy_id,
        model_name=model_name,
        force_preprocess=force_preprocess,
    )


def run_all(
    model_name: str = "lgbm",
    force_preprocess: bool = False,
):
    # 구현된 모든 전략에 대해 CSV 생성 여부를 먼저 맞춘다.
    ensure_processed_csvs(STRATEGIES, force=force_preprocess)

    all_results = []
    for strategy_id in STRATEGIES:
        strategy_results = run_strategy(
            strategy_id=strategy_id,
            model_name=model_name,
            force_preprocess=False,
        )
        all_results.extend(strategy_results)

    if not all_results:
        print("실행 가능한 실험이 없습니다. 전략을 구현하거나 전처리 CSV를 준비해주세요.")
        return []

    summary = pd.DataFrame(all_results).sort_values(
        by=["f1_class1", "recall_class1", "f1_macro"],
        ascending=False,
    )
    save_load.save_result(summary, ARTIFACTS_DIR / "experiment_result_all.csv")

    best = summary.iloc[0].to_dict()
    best_info = {
        "best_strategy": best.get("strategy"),
        "model": best.get("model"),
        "f1_class1": best.get("f1_class1"),
        "recall_class1": best.get("recall_class1"),
        "f1_macro": best.get("f1_macro"),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS_DIR / "best_info.json", "w", encoding="utf-8") as f:
        json.dump(best_info, f, ensure_ascii=False, indent=2)

    best_pipeline_path = ARTIFACTS_DIR / f"{best['strategy']}_{best['model']}_pipeline.pkl"
    if best_pipeline_path.exists():
        save_load.save_pipeline(
            save_load.load_pipeline(best_pipeline_path),
            ARTIFACTS_DIR / "best_pipeline.pkl",
        )

    print("\n===== 전체 비교 결과 =====")
    print(summary.to_string(index=False))
    print(f"\n최고 성능 조합: {best_info}")
    return all_results
