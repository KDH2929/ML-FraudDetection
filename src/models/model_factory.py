from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

from src.config import MODEL_PARAMS, RANDOM_STATE


def _xgb_params(scale_pos_weight: float | None) -> dict:
    params = dict(MODEL_PARAMS["xgboost"])
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = float(scale_pos_weight)
    return params


def _build_lgbm():
    from lightgbm import LGBMClassifier

    return LGBMClassifier(**MODEL_PARAMS["lgbm"])


def _build_xgboost(scale_pos_weight: float | None):
    from xgboost import XGBClassifier

    return XGBClassifier(**_xgb_params(scale_pos_weight))


def _build_catboost():
    from catboost import CatBoostClassifier

    return CatBoostClassifier(**MODEL_PARAMS["catboost"])


def get_model(model_name: str, *, scale_pos_weight: float | None = None):
    """Return a sklearn-compatible classifier by name."""

    key = model_name.lower()

    if key == "lgbm":
        return _build_lgbm()

    if key == "rf":
        return RandomForestClassifier(**MODEL_PARAMS["rf"])

    if key == "logistic":
        return LogisticRegression(**MODEL_PARAMS["logistic"])

    if key == "xgboost":
        return _build_xgboost(scale_pos_weight)

    if key == "catboost":
        return _build_catboost()

    if key == "voting":
        estimators = [
            ("lgbm", _build_lgbm()),
            ("xgb", _build_xgboost(scale_pos_weight)),
            ("cat", _build_catboost()),
        ]
        return VotingClassifier(estimators=estimators, **MODEL_PARAMS["voting"])

    if key == "stacking":
        estimators = [
            ("lgbm", _build_lgbm()),
            ("xgb", _build_xgboost(scale_pos_weight)),
            ("cat", _build_catboost()),
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
        "Available: lgbm, rf, logistic, xgboost, catboost, voting, stacking"
    )
