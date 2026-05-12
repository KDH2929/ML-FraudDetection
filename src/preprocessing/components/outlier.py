import pandas as pd
import numpy as np


class IQRCapper:
    """Q1-1.5*IQR ~ Q3+1.5*IQR 범위로 수치형 변수 캡핑"""

    def __init__(self, cols=None):
        self.cols = cols
        self.bounds_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        cols = self.cols if self.cols else X.select_dtypes(include="number").columns.tolist()
        for col in cols:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            self.bounds_[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, (lo, hi) in self.bounds_.items():
            if col in X.columns:
                X[col] = X[col].clip(lo, hi)
        return X


class QuantileCapper:
    """하위 1% ~ 상위 99% 범위로 캡핑"""

    def __init__(self, cols=None, lower=0.01, upper=0.99):
        self.cols = cols
        self.lower = lower
        self.upper = upper
        self.bounds_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        cols = self.cols if self.cols else X.select_dtypes(include="number").columns.tolist()
        for col in cols:
            self.bounds_[col] = (X[col].quantile(self.lower), X[col].quantile(self.upper))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, (lo, hi) in self.bounds_.items():
            if col in X.columns:
                X[col] = X[col].clip(lo, hi)
        return X


class NoOpOutlier:
    """이상치 처리 없음 (비교용)"""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.copy()
