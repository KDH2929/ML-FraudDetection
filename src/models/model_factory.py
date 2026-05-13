from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from src.config import MODEL_PARAMS, RANDOM_STATE


def get_model(model_name: str):
    """모델 이름을 입력받아 sklearn 호환 분류기 반환"""
    key = model_name.lower()

    if key == "lgbm":
        return LGBMClassifier(**MODEL_PARAMS["lgbm"])

    if key == "rf":
        return RandomForestClassifier(**MODEL_PARAMS["rf"])

    if key == "logistic":
        return LogisticRegression(**MODEL_PARAMS["logistic"])

    if key == "xgboost":
        return XGBClassifier(**MODEL_PARAMS["xgboost"])

    if key == "catboost":
        return CatBoostClassifier(**MODEL_PARAMS["catboost"])

    if key == "voting":
        # Voting Classifier: LightGBM + XGBoost + CatBoost
        estimators = [
            ('lgbm', LGBMClassifier(**MODEL_PARAMS["lgbm"])),
            ('xgb', XGBClassifier(**MODEL_PARAMS["xgboost"])),
            ('cat', CatBoostClassifier(**MODEL_PARAMS["catboost"])),
        ]
        return VotingClassifier(estimators=estimators, **MODEL_PARAMS["voting"])

    if key == "stacking":
        # Stacking Classifier: base=(LGBM, XGB, CAT), meta=Logistic
        estimators = [
            ('lgbm', LGBMClassifier(**MODEL_PARAMS["lgbm"])),
            ('xgb', XGBClassifier(**MODEL_PARAMS["xgboost"])),
            ('cat', CatBoostClassifier(**MODEL_PARAMS["catboost"])),
        ]
        final_estimator = LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        return StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            **MODEL_PARAMS["stacking"]
        )

    raise ValueError(
        f"Unknown model: {model_name}. "
        f"Available: lgbm, rf, logistic, xgboost, catboost, voting, stacking"
    )
