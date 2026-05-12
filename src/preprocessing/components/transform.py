import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer


class LogTransformer:
    """로그 변환 (log1p)으로 right-skewed 분포 완화"""

    def __init__(self, cols=None):
        self.cols = cols

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        cols = self.cols if self.cols else X.select_dtypes(include="number").columns.tolist()
        for col in cols:
            if col in X.columns:
                X[col] = np.log1p(X[col])
        return X


class YeoJohnsonTransformer:
    """Yeo-Johnson 변환으로 분포 정규성 개선 (음수 허용)"""

    def __init__(self, cols=None):
        self.cols = cols
        self.transformer_ = None
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y=None):
        cols = self.cols if self.cols else X.select_dtypes(include="number").columns.tolist()
        self.feature_names_ = cols
        self.transformer_ = PowerTransformer(method='yeo-johnson', standardize=False)
        self.transformer_.fit(X[cols])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.feature_names_:
            X[self.feature_names_] = self.transformer_.transform(X[self.feature_names_])
        return X


class RankTransformer:
    """순위 변환으로 이상치 영향 제거"""

    def __init__(self, cols=None):
        self.cols = cols

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        cols = self.cols if self.cols else X.select_dtypes(include="number").columns.tolist()
        for col in cols:
            if col in X.columns:
                X[col] = X[col].rank(method='average')
        return X
