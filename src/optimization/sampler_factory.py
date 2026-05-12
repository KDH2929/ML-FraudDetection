from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from src.config import RANDOM_STATE


def get_sampler(method: str):
    """클래스 불균형 해소용 오버샘플러 반환. method=None이면 None 반환."""
    if method is None or method.lower() == "none":
        return None
    method = method.lower()
    if method == "smote":
        return SMOTE(random_state=RANDOM_STATE)
    if method == "adasyn":
        return ADASYN(random_state=RANDOM_STATE)
    if method == "borderlinesmote":
        return BorderlineSMOTE(random_state=RANDOM_STATE)
    raise ValueError(f"Unknown sampler method: {method}")
