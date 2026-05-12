import pandas as pd
import numpy as np
from src.config import CAT_COLS


class TargetEncoder:
    """범주형 변수를 타겟 평균값으로 인코딩 (고카디널리티에 효과적)"""

    def __init__(self, cols=None):
        self.cols = cols if cols else CAT_COLS
        self.encoding_map_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        if y is None:
            raise ValueError("TargetEncoder requires y.")
        for col in self.cols:
            if col in X.columns:
                self.encoding_map_[col] = X.groupby(col)[y.name if hasattr(y, "name") else "target"].mean().to_dict() if False else (
                    pd.concat([X[[col]], y.rename("_y")], axis=1)
                    .groupby(col)["_y"].mean().to_dict()
                )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, mapping in self.encoding_map_.items():
            if col in X.columns:
                X[col] = X[col].map(mapping).fillna(0.0)
        return X


class OneHotEncoder:
    """범주형 변수를 원-핫 벡터로 변환 (pd.get_dummies 래퍼)"""

    def __init__(self, cols=None):
        self.cols = cols if cols else CAT_COLS
        self.dummy_cols_ = None

    def fit(self, X: pd.DataFrame, y=None):
        cols = [c for c in self.cols if c in X.columns]
        dummies = pd.get_dummies(X[cols], columns=cols, drop_first=False, dtype=int)
        self.dummy_cols_ = dummies.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        cols = [c for c in self.cols if c in X.columns]
        dummies = pd.get_dummies(X[cols], columns=cols, drop_first=False, dtype=int)
        dummies = dummies.reindex(columns=self.dummy_cols_, fill_value=0)
        X = X.drop(columns=cols)
        X = pd.concat([X, dummies], axis=1)
        return X


class LabelEncoder:
    """범주형 변수를 정수로 레이블 인코딩"""

    def __init__(self, cols=None):
        self.cols = cols if cols else CAT_COLS
        self.label_map_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        for col in self.cols:
            if col in X.columns:
                categories = X[col].astype(str).unique()
                self.label_map_[col] = {v: i for i, v in enumerate(sorted(categories))}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, mapping in self.label_map_.items():
            if col in X.columns:
                X[col] = X[col].astype(str).map(mapping).fillna(-1).astype(int)
        return X
