import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from src.config import NUM_COLS


class RobustScalerWrapper:
    """중앙값과 IQR 기반 스케일링 — 이상치에 강건"""

    def __init__(self, cols=None):
        self.cols = cols if cols else NUM_COLS
        self.scaler = RobustScaler()
        self.fit_cols_ = []

    def fit(self, X: pd.DataFrame, y=None):
        self.fit_cols_ = [c for c in self.cols if c in X.columns]
        self.scaler.fit(X[self.fit_cols_])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.fit_cols_] = self.scaler.transform(X[self.fit_cols_])
        return X


class StandardScalerWrapper:
    """평균 0, 표준편차 1로 표준화"""

    def __init__(self, cols=None):
        self.cols = cols if cols else NUM_COLS
        self.scaler = StandardScaler()
        self.fit_cols_ = []

    def fit(self, X: pd.DataFrame, y=None):
        self.fit_cols_ = [c for c in self.cols if c in X.columns]
        self.scaler.fit(X[self.fit_cols_])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.fit_cols_] = self.scaler.transform(X[self.fit_cols_])
        return X


class MinMaxScalerWrapper:
    """0~1 범위로 정규화"""

    def __init__(self, cols=None):
        self.cols = cols if cols else NUM_COLS
        self.scaler = MinMaxScaler()
        self.fit_cols_ = []

    def fit(self, X: pd.DataFrame, y=None):
        self.fit_cols_ = [c for c in self.cols if c in X.columns]
        self.scaler.fit(X[self.fit_cols_])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.fit_cols_] = self.scaler.transform(X[self.fit_cols_])
        return X
