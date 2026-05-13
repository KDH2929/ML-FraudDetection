from pathlib import Path

from src.config import ARTIFACTS_DIR, BASE_DIR, PROCESSED_DIR


REPORTS_DIR = BASE_DIR / "docs" / "optimization_reports"
SUMMARY_ARTIFACTS_DIR = ARTIFACTS_DIR / "summary"


def strategy_member_dir(strategy_id: str) -> str:
    normalized = strategy_id.lower()
    for prefix in ("member_a", "member_b", "member_c"):
        if normalized.startswith(prefix):
            return prefix
    if normalized.startswith("book_"):
        return "book"
    if normalized.startswith("test_"):
        return "test"
    return "shared"


def processed_dir(strategy_id: str) -> Path:
    return PROCESSED_DIR / strategy_member_dir(strategy_id)


def processed_csv_path(strategy_id: str) -> Path:
    return processed_dir(strategy_id) / f"{strategy_id}_preprocessed.csv"


def strategy_artifact_dir(strategy_id: str) -> Path:
    return ARTIFACTS_DIR / strategy_member_dir(strategy_id) / strategy_id


def experiment_result_path(strategy_id: str) -> Path:
    return strategy_artifact_dir(strategy_id) / "experiment_results.csv"


def tuning_results_path(strategy_id: str) -> Path:
    return strategy_artifact_dir(strategy_id) / "tuning_results.json"


def custom_tuning_results_path(strategy_id: str) -> Path:
    return strategy_artifact_dir(strategy_id) / "custom_tuning_results.json"


def threshold_analysis_path(strategy_id: str, *, label: str | None = None) -> Path:
    filename = "threshold_analysis.json" if label is None else f"{label}_threshold_analysis.json"
    return strategy_artifact_dir(strategy_id) / filename


def threshold_curve_path(strategy_id: str, *, label: str | None = None) -> Path:
    filename = "threshold_curve.png" if label is None else f"{label}_threshold_curve.png"
    return strategy_artifact_dir(strategy_id) / filename


def pipeline_path(strategy_id: str, model_name: str) -> Path:
    return strategy_artifact_dir(strategy_id) / f"{model_name}_pipeline.pkl"


def manual_results_path(strategy_id: str) -> Path:
    return strategy_artifact_dir(strategy_id) / "manual_model_results.csv"


def report_dir(strategy_id: str) -> Path:
    return REPORTS_DIR / strategy_member_dir(strategy_id)


def report_path(strategy_id: str, version: str) -> Path:
    return report_dir(strategy_id) / f"{strategy_id}_{version}_report.md"


def summary_experiment_results_path() -> Path:
    return SUMMARY_ARTIFACTS_DIR / "experiment_results.csv"


def summary_best_info_path() -> Path:
    return SUMMARY_ARTIFACTS_DIR / "best_info.json"


def summary_best_pipeline_path() -> Path:
    return SUMMARY_ARTIFACTS_DIR / "best_pipeline.pkl"
