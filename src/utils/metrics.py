import pandas as pd
from sklearn.metrics import recall_score, f1_score, classification_report


def evaluate(y_true, y_pred) -> dict:
    """Recall, F1-class1, F1-macro 반환"""
    return {
        "recall_class1": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_class1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def print_report(member: str, metrics_dict: dict):
    sep = "-" * 50
    print(sep)
    print(f"[{member}] 실험 결과")
    print(sep)
    for k, v in metrics_dict.items():
        print(f"  {k:<20}: {v:.4f}")
    print(sep)
