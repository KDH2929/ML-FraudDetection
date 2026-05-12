from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from src.config import MODEL_PARAMS


def get_model(model_name: str):
    """모델 이름을 입력받아 sklearn 호환 분류기 반환"""
    key = model_name.lower()
    if key == "lgbm":
        return LGBMClassifier(**MODEL_PARAMS["lgbm"])
    if key == "rf":
        return RandomForestClassifier(**MODEL_PARAMS["rf"])
    if key == "logistic":
        return LogisticRegression(**MODEL_PARAMS["logistic"])
    raise ValueError(f"Unknown model: {model_name}. Available: lgbm, rf, logistic")
