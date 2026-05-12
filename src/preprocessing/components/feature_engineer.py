import numpy as np
import pandas as pd


def _first_existing_column(df: pd.DataFrame, *candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


class ClaimFeature:
    """고객 단위 청구 요약 파생변수."""

    def __init__(self):
        self.claim_features_ = None

    def fit(self, X: pd.DataFrame, y=None, claim_df: pd.DataFrame = None):
        if claim_df is None:
            raise ValueError("ClaimFeature 는 claim_df가 필요합니다.")
        self.claim_features_ = self._build_features(claim_df)
        return self

    def _build_features(self, claim_df: pd.DataFrame) -> pd.DataFrame:
        grp = claim_df.groupby("CUST_ID")
        feats = pd.DataFrame(index=claim_df["CUST_ID"].unique())
        feats["claim_count"] = grp.size()
        if "HSPZ_DAYS" in claim_df.columns:
            feats["avg_hspz_days"] = grp["HSPZ_DAYS"].mean()

        # 실제 원천 컬럼명이 다를 수 있어 후보 컬럼 중 먼저 발견되는 값을 사용한다.
        suspicious_hosp_col = _first_existing_column(claim_df, "SUSPCT_HOSP_YN", "HEED_HOSP_YN")
        if suspicious_hosp_col is not None:
            feats["has_suspct_hosp"] = grp[suspicious_hosp_col].max()

        if "DR_FRAUD_RATE" in claim_df.columns:
            feats["avg_dr_fraud_rate"] = grp["DR_FRAUD_RATE"].mean()
        if "HOSP_FRAUD_RATE" in claim_df.columns:
            feats["avg_hosp_fraud_rate"] = grp["HOSP_FRAUD_RATE"].mean()

        feats.index.name = "CUST_ID"
        return feats.reset_index()

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = X.merge(self.claim_features_, on="CUST_ID", how="left")
        new_cols = [c for c in self.claim_features_.columns if c != "CUST_ID"]
        X[new_cols] = X[new_cols].fillna(0)
        return X


class RatioFeature:
    """고객 단위 청구 비율 파생변수."""

    def __init__(self):
        self.claim_features_ = None

    def fit(self, X: pd.DataFrame, y=None, claim_df: pd.DataFrame = None):
        if claim_df is None:
            raise ValueError("RatioFeature 는 claim_df가 필요합니다.")
        self.claim_features_ = self._build_features(claim_df)
        return self

    def _build_features(self, claim_df: pd.DataFrame) -> pd.DataFrame:
        grp = claim_df.groupby("CUST_ID")
        feats = pd.DataFrame(index=claim_df["CUST_ID"].unique())

        if "ACCI_DVSN" in claim_df.columns:
            acci_ratio = pd.crosstab(
                claim_df["CUST_ID"],
                claim_df["ACCI_DVSN"].astype("string").fillna("missing"),
                normalize="index",
            )
            acci_ratio.columns = [
                f"acci_dvsn_ratio_{str(col).strip().replace(' ', '_')}"
                for col in acci_ratio.columns
            ]
            feats = feats.join(acci_ratio, how="left")

        if "PAYM_AMT" in claim_df.columns and "DMND_AMT" in claim_df.columns:
            paym = grp["PAYM_AMT"].sum()
            dmnd = grp["DMND_AMT"].sum()
            feats["paym_dmnd_ratio"] = (paym / dmnd.replace(0, np.nan)).fillna(0)

        feats.index.name = "CUST_ID"
        return feats.reset_index()

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = X.merge(self.claim_features_, on="CUST_ID", how="left")
        new_cols = [c for c in self.claim_features_.columns if c != "CUST_ID"]
        X[new_cols] = X[new_cols].fillna(0)
        return X


class HospFeature:
    """병원 이용 패턴 파생변수."""

    def __init__(self):
        self.claim_features_ = None

    def fit(self, X: pd.DataFrame, y=None, claim_df: pd.DataFrame = None):
        if claim_df is None:
            raise ValueError("HospFeature 는 claim_df가 필요합니다.")
        self.claim_features_ = self._build_features(claim_df)
        return self

    def _build_features(self, claim_df: pd.DataFrame) -> pd.DataFrame:
        grp = claim_df.groupby("CUST_ID")
        feats = pd.DataFrame(index=claim_df["CUST_ID"].unique())

        hosp_col = _first_existing_column(claim_df, "HOSP_CD", "HOSP_CODE")
        if hosp_col is not None:
            feats["unique_hosp_cnt"] = grp[hosp_col].nunique()
            feats["max_same_hosp_visits"] = grp[hosp_col].apply(
                lambda x: x.value_counts().max() if len(x) > 0 else 0
            )

        feats.index.name = "CUST_ID"
        return feats.reset_index()

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = X.merge(self.claim_features_, on="CUST_ID", how="left")
        new_cols = [c for c in self.claim_features_.columns if c != "CUST_ID"]
        X[new_cols] = X[new_cols].fillna(0)
        return X
