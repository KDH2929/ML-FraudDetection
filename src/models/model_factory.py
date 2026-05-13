from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from src.config import MODEL_PARAMS, RANDOM_STATE


def _xgb_params(scale_pos_weight: float | None) -> dict:
    p = dict(MODEL_PARAMS["xgboost"])
    if scale_pos_weight is not None:
        p["scale_pos_weight"] = float(scale_pos_weight)
    return p


def get_model(model_name: str, *, scale_pos_weight: float | None = None):
    """모델 이름을 입력받아 sklearn 호환 분류기 반환.

    scale_pos_weight
        이진 불균형용 XGBoost 가중. ``experiment_runner`` 등에서
        ``neg/pos`` 로 채워 전달한다. voting/stacking 내부 XGB에도 동일 적용.
    """
    key = model_name.lower()

    if key == "lgbm":
        return LGBMClassifier(**MODEL_PARAMS["lgbm"])

    if key == "rf":
        return RandomForestClassifier(**MODEL_PARAMS["rf"])

    if key == "logistic":
        return LogisticRegression(**MODEL_PARAMS["logistic"])

    if key == "xgboost":
        return XGBClassifier(**_xgb_params(scale_pos_weight))

    if key == "catboost":
        return CatBoostClassifier(**MODEL_PARAMS["catboost"])

    if key == "voting":
        xgb_p = _xgb_params(scale_pos_weight)
        estimators = [
            ("lgbm", LGBMClassifier(**MODEL_PARAMS["lgbm"])),
            ("xgb", XGBClassifier(**xgb_p)),
            ("cat", CatBoostClassifier(**MODEL_PARAMS["catboost"])),
        ]
        return VotingClassifier(estimators=estimators, **MODEL_PARAMS["voting"])

    if key == "stacking":
        xgb_p = _xgb_params(scale_pos_weight)
        estimators = [
            ("lgbm", LGBMClassifier(**MODEL_PARAMS["lgbm"])),
            ("xgb", XGBClassifier(**xgb_p)),
            ("cat", CatBoostClassifier(**MODEL_PARAMS["catboost"])),
        ]
        final_estimator = LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
            solver="lbfgs",
        )
        return StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            **MODEL_PARAMS["stacking"],
        )

    raise ValueError(
        f"Unknown model: {model_name}. "
        f"Available: lgbm, rf, logistic, xgboost, catboost, voting, stacking"
    )
