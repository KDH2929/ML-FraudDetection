import pandas as pd
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from src.config import RANDOM_STATE


class FeatureSelector:
    """불필요한 변수를 제거해 모델 성능을 향상시킨다."""

    def __init__(self, method: str = None):
        self.method = method
        self.selector_ = None
        self.selected_cols_ = None

    def fit(self, X: pd.DataFrame, y):
        if self.method is None or self.method.lower() == "none":
            self.selected_cols_ = X.columns.tolist()
            return self
        if self.method == "variance":
            self.selector_ = VarianceThreshold()
            self.selector_.fit(X)
            self.selected_cols_ = X.columns[self.selector_.get_support()].tolist()
        elif self.method == "model":
            estimator = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
            self.selector_ = SelectFromModel(estimator)
            self.selector_.fit(X, y)
            self.selected_cols_ = X.columns[self.selector_.get_support()].tolist()
        else:
            raise ValueError(f"Unknown feature selection method: {self.method}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_cols_]
