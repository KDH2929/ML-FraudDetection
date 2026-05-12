import json
import pandas as pd
from pathlib import Path
from src.config import PROCESSED_DIR, ARTIFACTS_DIR, TARGET_COL, DROP_COLS
from src.preprocessing.preprocessor_factory import get_strategy
from src.pipeline.data_loader import load_and_split
from src.pipeline import pipeline_builder, trainer
from src.models.model_factory import get_model
from src.optimization.sampler_factory import get_sampler
from src.optimization.feature_selector import FeatureSelector
from src.utils import save_load, metrics


MEMBERS = ["member_a", "member_b", "member_c"]


def run(member: str, model_name: str = "lgbm", sampler_method: str = "smote", selector_method: str = None):
    """특정 멤버의 실험 1회 실행"""
    processed_csv = PROCESSED_DIR / f"{member}_preprocessed.csv"
    if not processed_csv.exists():
        print(f"[SKIP] {processed_csv} 파일이 없습니다.")
        return None

    train_X, test_X, train_y, test_y = load_and_split(str(processed_csv))

    sampler = get_sampler(sampler_method)
    selector = FeatureSelector(method=selector_method)
    selector.fit(train_X, train_y)
    train_X = selector.transform(train_X)
    test_X = selector.transform(test_X)

    model = get_model(model_name)
    pipeline = pipeline_builder.build(get_strategy(member), model, sampler=sampler)
    fitted = trainer.train(pipeline, train_X, train_y)

    y_pred = fitted.predict(test_X)
    result = metrics.evaluate(test_y, y_pred)
    result["member"] = member
    result["model"] = model_name
    metrics.print_report(member, result)

    result_df = pd.DataFrame([result])
    save_load.save_result(result_df, ARTIFACTS_DIR / f"experiment_result_{member}.csv")
    save_load.save_pipeline(fitted, ARTIFACTS_DIR / "best_pipeline.pkl")

    return result


def run_all(model_name: str = "lgbm", sampler_method: str = "smote", selector_method: str = None):
    """전체 멤버 실험 실행 후 비교표 출력"""
    all_results = []
    best = {"f1_class1": -1}

    for member in MEMBERS:
        result = run(member, model_name=model_name, sampler_method=sampler_method, selector_method=selector_method)
        if result is None:
            continue
        all_results.append(result)
        if result["f1_class1"] > best["f1_class1"]:
            best = result

    if not all_results:
        print("실행된 실험이 없습니다. processed CSV 파일을 먼저 준비하세요.")
        return

    summary = pd.DataFrame(all_results)
    print("\n===== 전체 비교표 =====")
    print(summary.to_string(index=False))

    best_info = {
        "best_member": best.get("member"),
        "f1_class1": best.get("f1_class1"),
        "recall_class1": best.get("recall_class1"),
        "f1_macro": best.get("f1_macro"),
        "model": best.get("model"),
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS_DIR / "best_info.json", "w", encoding="utf-8") as f:
        json.dump(best_info, f, ensure_ascii=False, indent=2)
    print(f"\n최고 성능: {best_info}")
