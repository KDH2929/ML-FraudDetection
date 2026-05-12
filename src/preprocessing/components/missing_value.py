import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, IterativeImputer


class GroupMeanImputer:
    """직업(OCCP_GRP_1) + 나이(AGE) 그룹 평균으로 소득 결측치 대체"""

    def __init__(self, group_cols=("OCCP_GRP_1", "AGE"), target_col="CUST_INCM"):
        self.group_cols = list(group_cols)
        self.target_col = target_col
        self.group_means_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        self.group_means_ = (
            X.groupby(self.group_cols)[self.target_col].mean().to_dict()
        )
        self.global_mean_ = X[self.target_col].mean()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        mask = X[self.target_col].isna()
        for idx in X[mask].index:
            key = tuple(X.loc[idx, col] for col in self.group_cols)
            X.loc[idx, self.target_col] = self.group_means_.get(key, self.global_mean_)
        return X


class MedianImputer:
    """수치형 변수 결측치를 중앙값으로 대체"""

    def __init__(self, cols=None):
        self.cols = cols
        self.medians_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        cols = self.cols if self.cols else X.select_dtypes(include="number").columns.tolist()
        self.medians_ = {col: X[col].median() for col in cols}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, val in self.medians_.items():
            if col in X.columns:
                X[col] = X[col].fillna(val)
        return X


class ModeImputer:
    """범주형 변수 결측치를 최빈값으로 대체"""

    def __init__(self, cols=None):
        self.cols = cols
        self.modes_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        cols = self.cols if self.cols else X.select_dtypes(include="object").columns.tolist()
        self.modes_ = {col: X[col].mode()[0] for col in cols if not X[col].mode().empty}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, val in self.modes_.items():
            if col in X.columns:
                X[col] = X[col].fillna(val)
        return X


class ZeroImputer:
    """결측치를 0으로 대체"""

    def __init__(self, cols=None):
        self.cols = cols

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        cols = self.cols if self.cols else X.columns.tolist()
        for col in cols:
            if col in X.columns:
                X[col] = X[col].fillna(0)
        return X


class MissingIndicator:
    """결측치 여부를 binary feature로 추가"""

    def __init__(self, cols=None):
        self.cols = cols
        self.missing_cols_ = []

    def fit(self, X: pd.DataFrame, y=None):
        cols = self.cols if self.cols else X.columns.tolist()
        self.missing_cols_ = [col for col in cols if col in X.columns and X[col].isna().any()]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.missing_cols_:
            if col in X.columns:
                X[f"IS_MISSING_{col}"] = X[col].isna().astype(int)
        return X


class KNNImputerWrapper:
    """KNN 기반 결측치 대체 (scikit-learn KNNImputer 래퍼)"""

    def __init__(self, cols=None, n_neighbors=5):
        self.cols = cols
        self.n_neighbors = n_neighbors
        self.imputer_ = None
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y=None):
        cols = self.cols if self.cols else X.select_dtypes(include="number").columns.tolist()
        self.feature_names_ = cols
        self.imputer_ = KNNImputer(n_neighbors=self.n_neighbors)
        self.imputer_.fit(X[cols])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.feature_names_:
            X[self.feature_names_] = self.imputer_.transform(X[self.feature_names_])
        return X


class IterativeImputerWrapper:
    """반복적 회귀 기반 결측치 대체 (scikit-learn IterativeImputer 래퍼)"""

    def __init__(self, cols=None, max_iter=10, random_state=42):
        self.cols = cols
        self.max_iter = max_iter
        self.random_state = random_state
        self.imputer_ = None
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y=None):
        cols = self.cols if self.cols else X.select_dtypes(include="number").columns.tolist()
        self.feature_names_ = cols
        self.imputer_ = IterativeImputer(max_iter=self.max_iter, random_state=self.random_state)
        self.imputer_.fit(X[cols])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.feature_names_:
            X[self.feature_names_] = self.imputer_.transform(X[self.feature_names_])
        return X
