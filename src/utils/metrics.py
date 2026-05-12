from numbers import Real

from sklearn.metrics import f1_score, recall_score


def evaluate(y_true, y_pred) -> dict:
    """프로젝트 비교에 사용할 평가 지표를 반환한다."""
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
    for key, value in metrics_dict.items():
        formatted = f"{value:.4f}" if isinstance(value, Real) else str(value)
        print(f"  {key:<20}: {formatted}")
    print(sep)
