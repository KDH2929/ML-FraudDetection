from numbers import Real

import numpy as np
from sklearn.metrics import f1_score, recall_score


def evaluate(y_true, y_pred) -> dict:
    """프로젝트 비교에 사용할 평가 지표를 반환한다."""
    return {
        "recall_class1": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_class1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def _format_metric_value(value) -> str:
    """sklearn/numpy 스칼라까지 소수 4자리로 통일."""
    if isinstance(value, Real) and not isinstance(value, bool):
        return f"{float(value):.4f}"
    if isinstance(value, (np.floating, np.integer)):
        return f"{float(value):.4f}"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def print_report(member: str, metrics_dict: dict):
    sep = "-" * 50
    print(sep)
    print(f"[{member}] 실험 결과")
    print(sep)
    for key, value in metrics_dict.items():
        formatted = _format_metric_value(value)
        print(f"  {key:<20}: {formatted}")
    print(sep)
