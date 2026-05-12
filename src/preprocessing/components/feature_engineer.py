import pandas as pd
import numpy as np


class ClaimFeature:
    """고객별 청구 횟수, 평균 입원일수, 유의병원 방문 여부, 의사/병원 사기비율 생성"""

    def __init__(self):
        self.claim_features_ = None

    def fit(self, X: pd.DataFrame, y=None, claim_df: pd.DataFrame = None):
        if claim_df is None:
            raise ValueError("ClaimFeature requires claim_df.")
        self.claim_features_ = self._build_features(claim_df)
        return self

    def _build_features(self, claim_df: pd.DataFrame) -> pd.DataFrame:
        grp = claim_df.groupby("CUST_ID")
        feats = pd.DataFrame(index=claim_df["CUST_ID"].unique())
        feats["claim_count"] = grp.size()
        if "HSPZ_DAYS" in claim_df.columns:
            feats["avg_hspz_days"] = grp["HSPZ_DAYS"].mean()
        if "SUSPCT_HOSP_YN" in claim_df.columns:
            feats["has_suspct_hosp"] = grp["SUSPCT_HOSP_YN"].max()
        if "DR_FRAUD_RATE" in claim_df.columns:
            feats["avg_dr_fraud_rate"] = grp["DR_FRAUD_RATE"].mean()
        if "HOSP_FRAUD_RATE" in claim_df.columns:
            feats["avg_hosp_fraud_rate"] = grp["HOSP_FRAUD_RATE"].mean()
        return feats.reset_index().rename(columns={"index": "CUST_ID"})

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = X.merge(self.claim_features_, on="CUST_ID", how="left")
        new_cols = [c for c in self.claim_features_.columns if c != "CUST_ID"]
        X[new_cols] = X[new_cols].fillna(0)
        return X


class RatioFeature:
    """ACCI_DVSN 비율, 청구금액 대비 비율 등 비율 파생변수 생성"""

    def __init__(self):
        self.claim_features_ = None

    def fit(self, X: pd.DataFrame, y=None, claim_df: pd.DataFrame = None):
        if claim_df is None:
            raise ValueError("RatioFeature requires claim_df.")
        self.claim_features_ = self._build_features(claim_df)
        return self

    def _build_features(self, claim_df: pd.DataFrame) -> pd.DataFrame:
        grp = claim_df.groupby("CUST_ID")
        feats = pd.DataFrame(index=claim_df["CUST_ID"].unique())
        if "ACCI_DVSN" in claim_df.columns:
            total = grp["ACCI_DVSN"].count()
            acci_cnt = grp["ACCI_DVSN"].apply(lambda x: (x == "사고").sum())
            feats["acci_ratio"] = (acci_cnt / total.replace(0, np.nan)).fillna(0)
        if "PAYM_AMT" in claim_df.columns and "DMND_AMT" in claim_df.columns:
            paym = grp["PAYM_AMT"].sum()
            dmnd = grp["DMND_AMT"].sum()
            feats["paym_dmnd_ratio"] = (paym / dmnd.replace(0, np.nan)).fillna(0)
        return feats.reset_index().rename(columns={"index": "CUST_ID"})

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = X.merge(self.claim_features_, on="CUST_ID", how="left")
        new_cols = [c for c in self.claim_features_.columns if c != "CUST_ID"]
        X[new_cols] = X[new_cols].fillna(0)
        return X


class HospFeature:
    """동일병원 반복 방문 횟수, 병원 변경 횟수 등 병원 패턴 파생변수 생성"""

    def __init__(self):
        self.claim_features_ = None

    def fit(self, X: pd.DataFrame, y=None, claim_df: pd.DataFrame = None):
        if claim_df is None:
            raise ValueError("HospFeature requires claim_df.")
        self.claim_features_ = self._build_features(claim_df)
        return self

    def _build_features(self, claim_df: pd.DataFrame) -> pd.DataFrame:
        grp = claim_df.groupby("CUST_ID")
        feats = pd.DataFrame(index=claim_df["CUST_ID"].unique())
        if "HOSP_CD" in claim_df.columns:
            feats["unique_hosp_cnt"] = grp["HOSP_CD"].nunique()
            feats["max_same_hosp_visits"] = grp["HOSP_CD"].apply(
                lambda x: x.value_counts().max() if len(x) > 0 else 0
            )
        return feats.reset_index().rename(columns={"index": "CUST_ID"})

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = X.merge(self.claim_features_, on="CUST_ID", how="left")
        new_cols = [c for c in self.claim_features_.columns if c != "CUST_ID"]
        X[new_cols] = X[new_cols].fillna(0)
        return X
