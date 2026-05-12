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


class OutlierFlagger:
    """이상치 여부를 binary feature로 추가 (IQR 기준)"""

    def __init__(self, cols=None, method='iqr'):
        self.cols = cols
        self.method = method
        self.bounds_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        cols = self.cols if self.cols else X.select_dtypes(include="number").columns.tolist()
        for col in cols:
            if self.method == 'iqr':
                q1 = X[col].quantile(0.25)
                q3 = X[col].quantile(0.75)
                iqr = q3 - q1
                self.bounds_[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
            elif self.method == 'quantile':
                self.bounds_[col] = (X[col].quantile(0.01), X[col].quantile(0.99))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, (lo, hi) in self.bounds_.items():
            if col in X.columns:
                X[f"IS_OUTLIER_{col}_LOWER"] = (X[col] < lo).astype(int)
                X[f"IS_OUTLIER_{col}_UPPER"] = (X[col] > hi).astype(int)
        return X


class AdaptiveQuantileCapper:
    """Feature 타입별로 다른 임계값을 적용하는 캡퍼"""

    def __init__(self, amount_cols=None, ratio_cols=None, count_cols=None):
        self.amount_cols = amount_cols or []
        self.ratio_cols = ratio_cols or []
        self.count_cols = count_cols or []
        self.bounds_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        # 금액 변수: 0.5%-99.5% (극단값 보존)
        for col in self.amount_cols:
            if col in X.columns:
                self.bounds_[col] = (X[col].quantile(0.005), X[col].quantile(0.995))

        # 비율 변수: [0, 1] 범위로 자연 제한
        for col in self.ratio_cols:
            if col in X.columns:
                self.bounds_[col] = (0, 1)

        # 카운트 변수: 5%-95% (0이 자연스러운 값)
        for col in self.count_cols:
            if col in X.columns:
                self.bounds_[col] = (X[col].quantile(0.05), X[col].quantile(0.95))

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, (lo, hi) in self.bounds_.items():
            if col in X.columns:
                X[col] = X[col].clip(lo, hi)
        return X
