import pandas as pd
import numpy as np
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


class CorrelationRemover:
    """상관관계가 높은 feature 쌍 중 하나를 제거"""

    def __init__(self, threshold=0.95, method='pearson'):
        self.threshold = threshold
        self.method = method
        self.selected_cols_ = None

    def fit(self, X: pd.DataFrame, y=None):
        corr_matrix = X.corr(method=self.method).abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > self.threshold)]
        self.selected_cols_ = [col for col in X.columns if col not in to_drop]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_cols_]


class VIFRemover:
    """VIF(Variance Inflation Factor)가 높은 feature를 제거 (다중공선성 제거)"""

    def __init__(self, threshold=10.0, max_iter=10):
        self.threshold = threshold
        self.max_iter = max_iter
        self.selected_cols_ = None

    def fit(self, X: pd.DataFrame, y=None):
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
        except ImportError:
            # statsmodels가 없으면 모든 feature 유지
            self.selected_cols_ = X.columns.tolist()
            return self

        features = X.select_dtypes(include='number').columns.tolist()
        selected = features.copy()

        for _ in range(self.max_iter):
            if len(selected) <= 1:
                break

            X_subset = X[selected].fillna(0)
            vif_data = pd.DataFrame()
            vif_data["feature"] = selected
            vif_data["VIF"] = [
                variance_inflation_factor(X_subset.values, i)
                for i in range(len(selected))
            ]

            max_vif = vif_data["VIF"].max()
            if max_vif <= self.threshold:
                break

            # VIF가 가장 높은 feature 제거
            feature_to_remove = vif_data.loc[vif_data["VIF"].idxmax(), "feature"]
            selected.remove(feature_to_remove)

        self.selected_cols_ = selected
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_cols_]
