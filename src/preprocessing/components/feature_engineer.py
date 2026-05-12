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


class DeviationFeature:
    """그룹별 평균 대비 z-score 편차 특성."""

    def __init__(self):
        self.claim_features_ = None

    def fit(self, X: pd.DataFrame, y=None, claim_df: pd.DataFrame = None):
        if claim_df is None:
            raise ValueError("DeviationFeature 는 claim_df가 필요합니다.")
        self.claim_features_ = self._build_features(claim_df)
        return self

    def _build_features(self, claim_df: pd.DataFrame) -> pd.DataFrame:
        df_claim = claim_df.copy()
        deviation_features = []

        # 1. DSAS_NAME 그룹별 DMND_AMT z-score
        if 'DSAS_NAME' in df_claim.columns and 'DMND_AMT' in df_claim.columns:
            grp_stats = df_claim.groupby('DSAS_NAME')['DMND_AMT'].agg(['mean', 'std']).reset_index()
            grp_stats.columns = ['DSAS_NAME', 'DSAS_DMND_MEAN', 'DSAS_DMND_STD']
            grp_stats['DSAS_DMND_STD'] = grp_stats['DSAS_DMND_STD'].replace(0, 1e-6).fillna(1e-6)
            df_claim = df_claim.merge(grp_stats, on='DSAS_NAME', how='left')
            df_claim['DEV_DSAS_DMND_Z'] = (df_claim['DMND_AMT'] - df_claim['DSAS_DMND_MEAN']) / df_claim['DSAS_DMND_STD']
            df_claim['DEV_DSAS_DMND_Z'] = df_claim['DEV_DSAS_DMND_Z'].replace([np.inf, -np.inf], np.nan).fillna(0)
            deviation_features.append('DEV_DSAS_DMND_Z')

        # 2. ACCI_DVSN 그룹별 PAYM_AMT z-score
        if 'ACCI_DVSN' in df_claim.columns and 'PAYM_AMT' in df_claim.columns:
            grp_stats = df_claim.groupby('ACCI_DVSN')['PAYM_AMT'].agg(['mean', 'std']).reset_index()
            grp_stats.columns = ['ACCI_DVSN', 'ACCI_PAYM_MEAN', 'ACCI_PAYM_STD']
            grp_stats['ACCI_PAYM_STD'] = grp_stats['ACCI_PAYM_STD'].replace(0, 1e-6).fillna(1e-6)
            df_claim = df_claim.merge(grp_stats, on='ACCI_DVSN', how='left')
            df_claim['DEV_ACCI_PAYM_Z'] = (df_claim['PAYM_AMT'] - df_claim['ACCI_PAYM_MEAN']) / df_claim['ACCI_PAYM_STD']
            df_claim['DEV_ACCI_PAYM_Z'] = df_claim['DEV_ACCI_PAYM_Z'].replace([np.inf, -np.inf], np.nan).fillna(0)
            deviation_features.append('DEV_ACCI_PAYM_Z')

        # 3. HOSP_SPEC_DVSN 그룹별 NON_PAY_RATIO z-score
        if 'HOSP_SPEC_DVSN' in df_claim.columns and 'NON_PAY_RATIO' in df_claim.columns:
            grp_stats = df_claim.groupby('HOSP_SPEC_DVSN')['NON_PAY_RATIO'].agg(['mean', 'std']).reset_index()
            grp_stats.columns = ['HOSP_SPEC_DVSN', 'HOSP_SPEC_NONPAY_MEAN', 'HOSP_SPEC_NONPAY_STD']
            grp_stats['HOSP_SPEC_NONPAY_STD'] = grp_stats['HOSP_SPEC_NONPAY_STD'].replace(0, 1e-6).fillna(1e-6)
            df_claim = df_claim.merge(grp_stats, on='HOSP_SPEC_DVSN', how='left')
            df_claim['DEV_HOSP_SPEC_NONPAY_Z'] = (df_claim['NON_PAY_RATIO'] - df_claim['HOSP_SPEC_NONPAY_MEAN']) / df_claim['HOSP_SPEC_NONPAY_STD']
            df_claim['DEV_HOSP_SPEC_NONPAY_Z'] = df_claim['DEV_HOSP_SPEC_NONPAY_Z'].replace([np.inf, -np.inf], np.nan).fillna(0)
            deviation_features.append('DEV_HOSP_SPEC_NONPAY_Z')

        # 고객별 집계
        deviation_agg = {}
        for feat in deviation_features:
            deviation_agg[f'{feat}_MEAN'] = (feat, 'mean')
            deviation_agg[f'{feat}_MAX'] = (feat, 'max')

        deviation_df = df_claim.groupby('CUST_ID').agg(**deviation_agg).reset_index()
        deviation_df = deviation_df.fillna(0)

        return deviation_df

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = X.merge(self.claim_features_, on="CUST_ID", how="left")
        new_cols = [c for c in self.claim_features_.columns if c != "CUST_ID"]
        X[new_cols] = X[new_cols].fillna(0)
        return X


class BurstFeature:
    """짧은 기간 내 청구 집중도 특성."""

    def __init__(self):
        self.claim_features_ = None

    def fit(self, X: pd.DataFrame, y=None, claim_df: pd.DataFrame = None):
        if claim_df is None:
            raise ValueError("BurstFeature 는 claim_df가 필요합니다.")
        self.claim_features_ = self._build_features(claim_df)
        return self

    def _build_features(self, claim_df: pd.DataFrame) -> pd.DataFrame:
        df_claim = claim_df.copy()

        if 'RECP_DATE' not in df_claim.columns:
            return pd.DataFrame({'CUST_ID': claim_df['CUST_ID'].unique()})

        # RECP_DATE를 datetime으로 변환
        df_claim['RECP_DATE'] = pd.to_datetime(df_claim['RECP_DATE'], errors='coerce')
        df_claim = df_claim.dropna(subset=['RECP_DATE'])
        df_claim = df_claim.sort_values(['CUST_ID', 'RECP_DATE'])

        # 청구 간격 계산
        df_claim['CLAIM_INTERVAL'] = df_claim.groupby('CUST_ID')['RECP_DATE'].diff().dt.days

        grp = df_claim.groupby('CUST_ID')

        # 청구 간격 통계
        burst_df = grp.agg(
            BURST_CLAIM_INTERVAL_MEAN=('CLAIM_INTERVAL', 'mean'),
            BURST_CLAIM_INTERVAL_MIN=('CLAIM_INTERVAL', 'min'),
            BURST_CLAIM_INTERVAL_STD=('CLAIM_INTERVAL', 'std'),
        ).reset_index()

        # 청구 활동 기간
        date_range = grp['RECP_DATE'].agg(lambda x: (x.max() - x.min()).days).reset_index(name='BURST_CLAIM_ACTIVE_DAYS')
        burst_df = burst_df.merge(date_range, on='CUST_ID', how='left')

        # 월평균 청구 횟수
        claim_count = grp.size().reset_index(name='claim_count')
        burst_df = burst_df.merge(claim_count, on='CUST_ID', how='left')
        burst_df['BURST_CLAIM_PER_ACTIVE_MONTH'] = burst_df['claim_count'] / (burst_df['BURST_CLAIM_ACTIVE_DAYS'].replace(0, 1) / 30)
        burst_df = burst_df.drop(columns=['claim_count'])

        # 30일/90일 내 최대 청구 횟수
        def max_claims_in_window(dates, window_days=30):
            dates = sorted(dates)
            if not dates:
                return 0
            max_count = 1
            left = 0
            for right in range(len(dates)):
                while dates[right] >= dates[left] + pd.Timedelta(days=window_days):
                    left += 1
                max_count = max(max_count, right - left + 1)
            return max_count

        burst_30d = grp['RECP_DATE'].apply(lambda x: max_claims_in_window(x.tolist(), 30)).reset_index(name='BURST_CLAIM_BURST_30D')
        burst_df = burst_df.merge(burst_30d, on='CUST_ID', how='left')

        burst_90d = grp['RECP_DATE'].apply(lambda x: max_claims_in_window(x.tolist(), 90)).reset_index(name='BURST_CLAIM_BURST_90D')
        burst_df = burst_df.merge(burst_90d, on='CUST_ID', how='left')

        burst_df = burst_df.fillna(0)

        return burst_df

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = X.merge(self.claim_features_, on="CUST_ID", how="left")
        new_cols = [c for c in self.claim_features_.columns if c != "CUST_ID"]
        X[new_cols] = X[new_cols].fillna(0)
        return X


class GraphFeature:
    """고객-병원-의사 네트워크 특성."""

    def __init__(self):
        self.claim_features_ = None

    def fit(self, X: pd.DataFrame, y=None, claim_df: pd.DataFrame = None):
        if claim_df is None:
            raise ValueError("GraphFeature 는 claim_df가 필요합니다.")
        self.claim_features_ = self._build_features(claim_df)
        return self

    def _build_features(self, claim_df: pd.DataFrame) -> pd.DataFrame:
        df_claim = claim_df.copy()

        # 결측치를 UNKNOWN으로 처리
        df_claim['HOSP_CODE'] = df_claim.get('HOSP_CODE', pd.Series()).fillna('UNKNOWN')
        df_claim['CHME_LICE_NO'] = df_claim.get('CHME_LICE_NO', pd.Series()).fillna('UNKNOWN')
        df_claim['DSAS_NAME'] = df_claim.get('DSAS_NAME', pd.Series()).fillna('UNKNOWN')

        grp = df_claim.groupby('CUST_ID')
        graph_df = pd.DataFrame({'CUST_ID': df_claim['CUST_ID'].unique()})

        # 고객별 degree (방문 장소 수)
        hosp_degree = grp['HOSP_CODE'].nunique().reset_index(name='GRAPH_CUST_HOSP_DEGREE')
        graph_df = graph_df.merge(hosp_degree, on='CUST_ID', how='left')

        doc_degree = grp['CHME_LICE_NO'].nunique().reset_index(name='GRAPH_CUST_DOC_DEGREE')
        graph_df = graph_df.merge(doc_degree, on='CUST_ID', how='left')

        disease_degree = grp['DSAS_NAME'].nunique().reset_index(name='GRAPH_CUST_DISEASE_DEGREE')
        graph_df = graph_df.merge(disease_degree, on='CUST_ID', how='left')

        # 병원/의사별 공유 고객 수
        hosp_cust_count = df_claim.groupby('HOSP_CODE')['CUST_ID'].nunique().to_dict()
        df_claim['HOSP_SHARED_CUSTOMER'] = df_claim['HOSP_CODE'].map(hosp_cust_count)
        hosp_shared = grp['HOSP_SHARED_CUSTOMER'].mean().reset_index(name='GRAPH_HOSP_SHARED_CUSTOMER_MEAN')
        graph_df = graph_df.merge(hosp_shared, on='CUST_ID', how='left')

        doc_cust_count = df_claim.groupby('CHME_LICE_NO')['CUST_ID'].nunique().to_dict()
        df_claim['DOC_SHARED_CUSTOMER'] = df_claim['CHME_LICE_NO'].map(doc_cust_count)
        doc_shared = grp['DOC_SHARED_CUSTOMER'].mean().reset_index(name='GRAPH_DOC_SHARED_CUSTOMER_MEAN')
        graph_df = graph_df.merge(doc_shared, on='CUST_ID', how='left')

        # 최다 방문 병원/의사 비율
        def get_max_repeat_ratio(series):
            if len(series) == 0:
                return 0
            value_counts = series.value_counts()
            return value_counts.iloc[0] / len(series) if len(value_counts) > 0 else 0

        hosp_repeat = grp['HOSP_CODE'].apply(get_max_repeat_ratio).reset_index(name='GRAPH_CUST_HOSP_REPEAT_RATIO')
        graph_df = graph_df.merge(hosp_repeat, on='CUST_ID', how='left')

        doc_repeat = grp['CHME_LICE_NO'].apply(get_max_repeat_ratio).reset_index(name='GRAPH_CUST_DOC_REPEAT_RATIO')
        graph_df = graph_df.merge(doc_repeat, on='CUST_ID', how='left')

        # 전체 그래프 degree
        graph_df['GRAPH_CUST_GRAPH_DEGREE'] = (
            graph_df['GRAPH_CUST_HOSP_DEGREE'] +
            graph_df['GRAPH_CUST_DOC_DEGREE'] +
            graph_df['GRAPH_CUST_DISEASE_DEGREE']
        )

        graph_df = graph_df.fillna(0)

        return graph_df

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = X.merge(self.claim_features_, on="CUST_ID", how="left")
        new_cols = [c for c in self.claim_features_.columns if c != "CUST_ID"]
        X[new_cols] = X[new_cols].fillna(0)
        return X
