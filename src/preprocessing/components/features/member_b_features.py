"""
Member B Feature Engineering 컴포넌트

- DeviationFeature: 그룹별 z-score 편차 (v1)
- BurstFeature: 청구 집중도 (v1)
- GraphFeature: 네트워크 특성 (v1)
- AmountFeature: 금액 패턴 (v2)
- TimeFeature: 시계열 패턴 (v2)
- MedicalFeature: 의료 행태 (v2)
"""
import numpy as np
import pandas as pd


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


class AmountFeature:
    """청구/지급 금액 패턴 특성."""

    def __init__(self):
        self.claim_features_ = None

    def fit(self, X: pd.DataFrame, y=None, claim_df: pd.DataFrame = None):
        if claim_df is None:
            raise ValueError("AmountFeature 는 claim_df가 필요합니다.")
        self.claim_features_ = self._build_features(claim_df)
        return self

    def _build_features(self, claim_df: pd.DataFrame) -> pd.DataFrame:
        df_claim = claim_df.copy()
        grp = df_claim.groupby('CUST_ID')
        amount_df = pd.DataFrame({'CUST_ID': df_claim['CUST_ID'].unique()})

        # 청구/지급 금액 관련
        if 'PAYM_AMT' in df_claim.columns and 'DMND_AMT' in df_claim.columns:
            # 지급/청구 비율
            df_claim['PAYM_DMND_RATIO'] = df_claim['PAYM_AMT'] / (df_claim['DMND_AMT'].replace(0, np.nan))
            df_claim['PAYM_DMND_RATIO'] = df_claim['PAYM_DMND_RATIO'].replace([np.inf, -np.inf], np.nan)

            paym_dmnd_stats = grp['PAYM_DMND_RATIO'].agg(['mean', 'std']).reset_index()
            paym_dmnd_stats.columns = ['CUST_ID', 'AMT_PAYM_DMND_RATIO_MEAN', 'AMT_PAYM_DMND_RATIO_STD']
            amount_df = amount_df.merge(paym_dmnd_stats, on='CUST_ID', how='left')

            # 총 청구금액, 총 지급금액
            total_amt = grp.agg(
                AMT_TOTAL_DMND=('DMND_AMT', 'sum'),
                AMT_TOTAL_PAYM=('PAYM_AMT', 'sum'),
            ).reset_index()
            amount_df = amount_df.merge(total_amt, on='CUST_ID', how='left')

            # 고액 청구 비율 (상위 10% 기준)
            dmnd_threshold = df_claim['DMND_AMT'].quantile(0.9)
            df_claim['IS_HIGH_DMND'] = (df_claim['DMND_AMT'] > dmnd_threshold).astype(int)
            high_dmnd_ratio = grp['IS_HIGH_DMND'].mean().reset_index(name='AMT_HIGH_DMND_RATIO')
            amount_df = amount_df.merge(high_dmnd_ratio, on='CUST_ID', how='left')

        # NON_PAY_RATIO 통계
        if 'NON_PAY_RATIO' in df_claim.columns:
            non_pay_stats = grp['NON_PAY_RATIO'].agg(['mean', 'max', 'std']).reset_index()
            non_pay_stats.columns = ['CUST_ID', 'AMT_NON_PAY_RATIO_MEAN', 'AMT_NON_PAY_RATIO_MAX', 'AMT_NON_PAY_RATIO_STD']
            amount_df = amount_df.merge(non_pay_stats, on='CUST_ID', how='left')

        amount_df = amount_df.fillna(0)
        return amount_df

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = X.merge(self.claim_features_, on="CUST_ID", how="left")
        new_cols = [c for c in self.claim_features_.columns if c != "CUST_ID"]
        X[new_cols] = X[new_cols].fillna(0)
        return X


class TimeFeature:
    """시계열 청구 패턴 특성."""

    def __init__(self):
        self.claim_features_ = None

    def fit(self, X: pd.DataFrame, y=None, claim_df: pd.DataFrame = None):
        if claim_df is None:
            raise ValueError("TimeFeature 는 claim_df가 필요합니다.")
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

        grp = df_claim.groupby('CUST_ID')
        time_df = pd.DataFrame({'CUST_ID': df_claim['CUST_ID'].unique()})

        # 요일 패턴
        df_claim['WEEKDAY'] = df_claim['RECP_DATE'].dt.weekday  # 0=월요일, 6=일요일
        df_claim['IS_WEEKEND'] = (df_claim['WEEKDAY'] >= 5).astype(int)

        weekday_stats = grp['IS_WEEKEND'].agg(['mean']).reset_index()
        weekday_stats.columns = ['CUST_ID', 'TIME_WEEKEND_RATIO']
        time_df = time_df.merge(weekday_stats, on='CUST_ID', how='left')

        # 월별 청구 변동성
        df_claim['YEAR_MONTH'] = df_claim['RECP_DATE'].dt.to_period('M')
        monthly_counts = df_claim.groupby(['CUST_ID', 'YEAR_MONTH']).size().reset_index(name='MONTHLY_COUNT')
        monthly_std = monthly_counts.groupby('CUST_ID')['MONTHLY_COUNT'].std().reset_index(name='TIME_MONTHLY_CLAIM_STD')
        time_df = time_df.merge(monthly_std, on='CUST_ID', how='left')

        # 첫 청구 이후 경과 일수
        date_range = grp['RECP_DATE'].agg(
            first_claim=('min'),
            last_claim=('max')
        ).reset_index()
        date_range['TIME_DAYS_SINCE_FIRST'] = (date_range['last_claim'] - date_range['first_claim']).dt.days
        time_df = time_df.merge(date_range[['CUST_ID', 'TIME_DAYS_SINCE_FIRST']], on='CUST_ID', how='left')

        # 월평균 청구 빈도
        claim_count = grp.size().reset_index(name='claim_count')
        time_df = time_df.merge(claim_count, on='CUST_ID', how='left')
        time_df['TIME_CLAIM_FREQ_PER_MONTH'] = time_df['claim_count'] / (time_df['TIME_DAYS_SINCE_FIRST'].replace(0, 1) / 30)
        time_df = time_df.drop(columns=['claim_count'])

        # 최근 청구 가중 평균 (exponential decay)
        def recent_weighted_count(dates):
            if len(dates) == 0:
                return 0
            dates = sorted(dates)
            max_date = dates[-1]
            weights = [np.exp(-0.01 * (max_date - d).days) for d in dates]
            return sum(weights) / len(dates) if len(dates) > 0 else 0

        recent_weighted = grp['RECP_DATE'].apply(recent_weighted_count).reset_index(name='TIME_RECENT_WEIGHTED_AVG')
        time_df = time_df.merge(recent_weighted, on='CUST_ID', how='left')

        time_df = time_df.fillna(0)
        return time_df

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = X.merge(self.claim_features_, on="CUST_ID", how="left")
        new_cols = [c for c in self.claim_features_.columns if c != "CUST_ID"]
        X[new_cols] = X[new_cols].fillna(0)
        return X


class MedicalFeature:
    """의료 행태 패턴 특성."""

    def __init__(self):
        self.claim_features_ = None

    def fit(self, X: pd.DataFrame, y=None, claim_df: pd.DataFrame = None):
        if claim_df is None:
            raise ValueError("MedicalFeature 는 claim_df가 필요합니다.")
        self.claim_features_ = self._build_features(claim_df)
        return self

    def _build_features(self, claim_df: pd.DataFrame) -> pd.DataFrame:
        df_claim = claim_df.copy()
        grp = df_claim.groupby('CUST_ID')
        medical_df = pd.DataFrame({'CUST_ID': df_claim['CUST_ID'].unique()})

        # 병명(DSAS_NAME) 다양성 - 엔트로피
        if 'DSAS_NAME' in df_claim.columns:
            def calculate_entropy(series):
                value_counts = series.value_counts()
                probs = value_counts / len(series)
                return -np.sum(probs * np.log2(probs + 1e-10))

            disease_entropy = grp['DSAS_NAME'].apply(calculate_entropy).reset_index(name='MED_DISEASE_ENTROPY')
            medical_df = medical_df.merge(disease_entropy, on='CUST_ID', how='left')

            # 고유 병명 개수
            disease_count = grp['DSAS_NAME'].nunique().reset_index(name='MED_DISEASE_COUNT')
            medical_df = medical_df.merge(disease_count, on='CUST_ID', how='left')

        # 사고구분(ACCI_DVSN) 분포
        if 'ACCI_DVSN' in df_claim.columns:
            df_claim['ACCI_DVSN'] = df_claim['ACCI_DVSN'].fillna('UNKNOWN')
            acci_ratio = pd.crosstab(
                df_claim['CUST_ID'],
                df_claim['ACCI_DVSN'],
                normalize='index'
            )
            acci_ratio.columns = [f'MED_ACCI_{col}' for col in acci_ratio.columns]
            acci_ratio = acci_ratio.reset_index()
            medical_df = medical_df.merge(acci_ratio, on='CUST_ID', how='left')

        # 병원전문구분(HOSP_SPEC_DVSN) 분포
        if 'HOSP_SPEC_DVSN' in df_claim.columns:
            df_claim['HOSP_SPEC_DVSN'] = df_claim['HOSP_SPEC_DVSN'].fillna('UNKNOWN')
            hosp_spec_ratio = pd.crosstab(
                df_claim['CUST_ID'],
                df_claim['HOSP_SPEC_DVSN'],
                normalize='index'
            )
            hosp_spec_ratio.columns = [f'MED_HOSP_{col}' for col in hosp_spec_ratio.columns]
            hosp_spec_ratio = hosp_spec_ratio.reset_index()
            medical_df = medical_df.merge(hosp_spec_ratio, on='CUST_ID', how='left')

        medical_df = medical_df.fillna(0)
        return medical_df

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = X.merge(self.claim_features_, on="CUST_ID", how="left")
        new_cols = [c for c in self.claim_features_.columns if c != "CUST_ID"]
        X[new_cols] = X[new_cols].fillna(0)
        return X


# ===== v2 개선 버전 =====


class DeviationFeature_v2:
    """그룹별 편차 특성 v2: 다층 그룹화, 백분위수, 시간 구간별 z-score"""

    def __init__(self):
        self.claim_features_ = None

    def fit(self, X: pd.DataFrame, y=None, claim_df: pd.DataFrame = None):
        if claim_df is None:
            raise ValueError("DeviationFeature_v2 는 claim_df가 필요합니다.")
        self.claim_features_ = self._build_features(claim_df)
        return self

    def _build_features(self, claim_df: pd.DataFrame) -> pd.DataFrame:
        df_claim = claim_df.copy()
        deviation_features = []

        # 기존 v1 feature (단일 그룹)
        if 'DSAS_NAME' in df_claim.columns and 'DMND_AMT' in df_claim.columns:
            grp_stats = df_claim.groupby('DSAS_NAME')['DMND_AMT'].agg(['mean', 'std']).reset_index()
            grp_stats.columns = ['DSAS_NAME', 'DSAS_DMND_MEAN', 'DSAS_DMND_STD']
            grp_stats['DSAS_DMND_STD'] = grp_stats['DSAS_DMND_STD'].replace(0, 1e-6).fillna(1e-6)
            df_claim = df_claim.merge(grp_stats, on='DSAS_NAME', how='left')
            df_claim['DEV2_DSAS_DMND_Z'] = (df_claim['DMND_AMT'] - df_claim['DSAS_DMND_MEAN']) / df_claim['DSAS_DMND_STD']
            df_claim['DEV2_DSAS_DMND_Z'] = df_claim['DEV2_DSAS_DMND_Z'].replace([np.inf, -np.inf], np.nan).fillna(0)
            deviation_features.append('DEV2_DSAS_DMND_Z')

            # 백분위수 기반 편차 추가
            grp_percentile = df_claim.groupby('DSAS_NAME')['DMND_AMT'].transform(
                lambda x: x.rank(pct=True)
            )
            df_claim['DEV2_DSAS_DMND_PERCENTILE'] = grp_percentile
            deviation_features.append('DEV2_DSAS_DMND_PERCENTILE')

        # 다층 그룹화: DSAS_NAME × ACCI_DVSN
        if 'DSAS_NAME' in df_claim.columns and 'ACCI_DVSN' in df_claim.columns and 'DMND_AMT' in df_claim.columns:
            df_claim['DSAS_ACCI_GROUP'] = df_claim['DSAS_NAME'].astype(str) + '_' + df_claim['ACCI_DVSN'].astype(str)
            grp_stats = df_claim.groupby('DSAS_ACCI_GROUP')['DMND_AMT'].agg(['mean', 'std']).reset_index()
            grp_stats.columns = ['DSAS_ACCI_GROUP', 'DSAS_ACCI_MEAN', 'DSAS_ACCI_STD']
            grp_stats['DSAS_ACCI_STD'] = grp_stats['DSAS_ACCI_STD'].replace(0, 1e-6).fillna(1e-6)
            df_claim = df_claim.merge(grp_stats, on='DSAS_ACCI_GROUP', how='left')
            df_claim['DEV2_DSAS_ACCI_Z'] = (df_claim['DMND_AMT'] - df_claim['DSAS_ACCI_MEAN']) / df_claim['DSAS_ACCI_STD']
            df_claim['DEV2_DSAS_ACCI_Z'] = df_claim['DEV2_DSAS_ACCI_Z'].replace([np.inf, -np.inf], np.nan).fillna(0)
            deviation_features.append('DEV2_DSAS_ACCI_Z')

        # 시간 구간별 편차 (RECP_DATE가 있을 경우)
        if 'RECP_DATE' in df_claim.columns and 'DMND_AMT' in df_claim.columns:
            df_claim['RECP_DATE'] = pd.to_datetime(df_claim['RECP_DATE'], errors='coerce')
            df_claim_with_date = df_claim.dropna(subset=['RECP_DATE']).copy()

            if len(df_claim_with_date) > 0:
                df_claim_with_date = df_claim_with_date.sort_values(['CUST_ID', 'RECP_DATE'])

                # 고객별 첫 청구일 기준 6개월 구간 나누기
                first_claim = df_claim_with_date.groupby('CUST_ID')['RECP_DATE'].transform('min')
                df_claim_with_date['MONTHS_SINCE_FIRST'] = ((df_claim_with_date['RECP_DATE'] - first_claim).dt.days / 30).astype(int)
                df_claim_with_date['TIME_PERIOD'] = (df_claim_with_date['MONTHS_SINCE_FIRST'] // 6).clip(upper=1)

                # 고객별 시간 구간별 청구금액 평균
                time_period_stats = df_claim_with_date.groupby(['CUST_ID', 'TIME_PERIOD'])['DMND_AMT'].mean().reset_index()
                time_period_stats.columns = ['CUST_ID', 'TIME_PERIOD', 'PERIOD_DMND_MEAN']
                df_claim = df_claim.merge(time_period_stats[['CUST_ID', 'PERIOD_DMND_MEAN']].groupby('CUST_ID').mean().reset_index(),
                                         on='CUST_ID', how='left')
                deviation_features.append('PERIOD_DMND_MEAN')

        # 고객별 집계
        deviation_agg = {}
        for feat in deviation_features:
            if feat in df_claim.columns:
                deviation_agg[f'{feat}_MEAN'] = (feat, 'mean')
                deviation_agg[f'{feat}_MAX'] = (feat, 'max')
                deviation_agg[f'{feat}_STD'] = (feat, 'std')

        deviation_df = df_claim.groupby('CUST_ID').agg(**deviation_agg).reset_index()
        deviation_df = deviation_df.fillna(0)

        return deviation_df

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = X.merge(self.claim_features_, on="CUST_ID", how="left")
        new_cols = [c for c in self.claim_features_.columns if c != "CUST_ID"]
        X[new_cols] = X[new_cols].fillna(0)
        return X
"""
Member B v2 Features - Part 2
나머지 v2 Feature 클래스들
"""
import numpy as np
import pandas as pd


class BurstFeature_v2:
    """청구 집중도 v2: 7/14일 윈도우, 변동계수"""

    def __init__(self):
        self.claim_features_ = None

    def fit(self, X: pd.DataFrame, y=None, claim_df: pd.DataFrame = None):
        if claim_df is None:
            raise ValueError("BurstFeature_v2 는 claim_df가 필요합니다.")
        self.claim_features_ = self._build_features(claim_df)
        return self

    def _build_features(self, claim_df: pd.DataFrame) -> pd.DataFrame:
        df_claim = claim_df.copy()

        if 'RECP_DATE' not in df_claim.columns:
            return pd.DataFrame({'CUST_ID': claim_df['CUST_ID'].unique()})

        df_claim['RECP_DATE'] = pd.to_datetime(df_claim['RECP_DATE'], errors='coerce')
        df_claim = df_claim.dropna(subset=['RECP_DATE'])
        df_claim = df_claim.sort_values(['CUST_ID', 'RECP_DATE'])

        # 청구 간격 계산
        df_claim['CLAIM_INTERVAL'] = df_claim.groupby('CUST_ID')['RECP_DATE'].diff().dt.days

        grp = df_claim.groupby('CUST_ID')

        # 청구 간격 통계 (기존)
        burst_df = grp.agg(
            BURST2_CLAIM_INTERVAL_MEAN=('CLAIM_INTERVAL', 'mean'),
            BURST2_CLAIM_INTERVAL_MIN=('CLAIM_INTERVAL', 'min'),
            BURST2_CLAIM_INTERVAL_STD=('CLAIM_INTERVAL', 'std'),
        ).reset_index()

        # 변동계수 추가 (CV = std / mean)
        burst_df['BURST2_CLAIM_INTERVAL_CV'] = (
            burst_df['BURST2_CLAIM_INTERVAL_STD'] /
            burst_df['BURST2_CLAIM_INTERVAL_MEAN'].replace(0, np.nan)
        )
        burst_df['BURST2_CLAIM_INTERVAL_CV'] = burst_df['BURST2_CLAIM_INTERVAL_CV'].fillna(0)

        # 청구 활동 기간
        date_range = grp['RECP_DATE'].agg(lambda x: (x.max() - x.min()).days).reset_index(name='BURST2_CLAIM_ACTIVE_DAYS')
        burst_df = burst_df.merge(date_range, on='CUST_ID', how='left')

        # 월평균 청구 횟수
        claim_count = grp.size().reset_index(name='claim_count')
        burst_df = burst_df.merge(claim_count, on='CUST_ID', how='left')
        burst_df['BURST2_CLAIM_PER_ACTIVE_MONTH'] = burst_df['claim_count'] / (burst_df['BURST2_CLAIM_ACTIVE_DAYS'].replace(0, 1) / 30)
        burst_df = burst_df.drop(columns=['claim_count'])

        # 다양한 윈도우 크기로 최대 청구 횟수
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

        # 7일 윈도우 (신규)
        burst_7d = grp['RECP_DATE'].apply(lambda x: max_claims_in_window(x.tolist(), 7)).reset_index(name='BURST2_CLAIM_BURST_7D')
        burst_df = burst_df.merge(burst_7d, on='CUST_ID', how='left')

        # 14일 윈도우 (신규)
        burst_14d = grp['RECP_DATE'].apply(lambda x: max_claims_in_window(x.tolist(), 14)).reset_index(name='BURST2_CLAIM_BURST_14D')
        burst_df = burst_df.merge(burst_14d, on='CUST_ID', how='left')

        # 30일 윈도우 (기존)
        burst_30d = grp['RECP_DATE'].apply(lambda x: max_claims_in_window(x.tolist(), 30)).reset_index(name='BURST2_CLAIM_BURST_30D')
        burst_df = burst_df.merge(burst_30d, on='CUST_ID', how='left')

        # 90일 윈도우 (기존)
        burst_90d = grp['RECP_DATE'].apply(lambda x: max_claims_in_window(x.tolist(), 90)).reset_index(name='BURST2_CLAIM_BURST_90D')
        burst_df = burst_df.merge(burst_90d, on='CUST_ID', how='left')

        burst_df = burst_df.fillna(0)

        return burst_df

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = X.merge(self.claim_features_, on="CUST_ID", how="left")
        new_cols = [c for c in self.claim_features_.columns if c != "CUST_ID"]
        X[new_cols] = X[new_cols].fillna(0)
        return X


class GraphFeature_v2:
    """네트워크 특성 v2: 병원-의사 페어, 희귀도"""

    def __init__(self):
        self.claim_features_ = None

    def fit(self, X: pd.DataFrame, y=None, claim_df: pd.DataFrame = None):
        if claim_df is None:
            raise ValueError("GraphFeature_v2 는 claim_df가 필요합니다.")
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

        # 기존 v1 features (고객별 degree)
        hosp_degree = grp['HOSP_CODE'].nunique().reset_index(name='GRAPH2_CUST_HOSP_DEGREE')
        graph_df = graph_df.merge(hosp_degree, on='CUST_ID', how='left')

        doc_degree = grp['CHME_LICE_NO'].nunique().reset_index(name='GRAPH2_CUST_DOC_DEGREE')
        graph_df = graph_df.merge(doc_degree, on='CUST_ID', how='left')

        disease_degree = grp['DSAS_NAME'].nunique().reset_index(name='GRAPH2_CUST_DISEASE_DEGREE')
        graph_df = graph_df.merge(disease_degree, on='CUST_ID', how='left')

        # 병원-의사 페어 빈도 (신규)
        df_claim['HOSP_DOC_PAIR'] = df_claim['HOSP_CODE'].astype(str) + '_' + df_claim['CHME_LICE_NO'].astype(str)
        pair_degree = grp['HOSP_DOC_PAIR'].nunique().reset_index(name='GRAPH2_HOSP_DOC_PAIR_COUNT')
        graph_df = graph_df.merge(pair_degree, on='CUST_ID', how='left')

        # 병원 대비 의사 비율 (신규)
        graph_df['GRAPH2_DOC_PER_HOSP'] = (
            graph_df['GRAPH2_CUST_DOC_DEGREE'] /
            graph_df['GRAPH2_CUST_HOSP_DEGREE'].replace(0, 1)
        )

        # 병원/의사별 공유 고객 수 (기존)
        hosp_cust_count = df_claim.groupby('HOSP_CODE')['CUST_ID'].nunique().to_dict()
        df_claim['HOSP_SHARED_CUSTOMER'] = df_claim['HOSP_CODE'].map(hosp_cust_count)
        hosp_shared = grp['HOSP_SHARED_CUSTOMER'].mean().reset_index(name='GRAPH2_HOSP_SHARED_CUSTOMER_MEAN')
        graph_df = graph_df.merge(hosp_shared, on='CUST_ID', how='left')

        doc_cust_count = df_claim.groupby('CHME_LICE_NO')['CUST_ID'].nunique().to_dict()
        df_claim['DOC_SHARED_CUSTOMER'] = df_claim['CHME_LICE_NO'].map(doc_cust_count)
        doc_shared = grp['DOC_SHARED_CUSTOMER'].mean().reset_index(name='GRAPH2_DOC_SHARED_CUSTOMER_MEAN')
        graph_df = graph_df.merge(doc_shared, on='CUST_ID', how='left')

        # 희귀 병원/의사 방문 비율 (신규 - 전체 데이터에서 하위 20% 빈도)
        hosp_freq = df_claim['HOSP_CODE'].value_counts()
        hosp_rare_threshold = hosp_freq.quantile(0.2)
        df_claim['IS_RARE_HOSP'] = df_claim['HOSP_CODE'].map(lambda x: 1 if hosp_freq.get(x, 0) < hosp_rare_threshold else 0)
        rare_hosp_ratio = grp['IS_RARE_HOSP'].mean().reset_index(name='GRAPH2_RARE_HOSP_RATIO')
        graph_df = graph_df.merge(rare_hosp_ratio, on='CUST_ID', how='left')

        doc_freq = df_claim['CHME_LICE_NO'].value_counts()
        doc_rare_threshold = doc_freq.quantile(0.2)
        df_claim['IS_RARE_DOC'] = df_claim['CHME_LICE_NO'].map(lambda x: 1 if doc_freq.get(x, 0) < doc_rare_threshold else 0)
        rare_doc_ratio = grp['IS_RARE_DOC'].mean().reset_index(name='GRAPH2_RARE_DOC_RATIO')
        graph_df = graph_df.merge(rare_doc_ratio, on='CUST_ID', how='left')

        # 최다 방문 병원/의사 비율 (기존)
        def get_max_repeat_ratio(series):
            if len(series) == 0:
                return 0
            value_counts = series.value_counts()
            return value_counts.iloc[0] / len(series) if len(value_counts) > 0 else 0

        hosp_repeat = grp['HOSP_CODE'].apply(get_max_repeat_ratio).reset_index(name='GRAPH2_CUST_HOSP_REPEAT_RATIO')
        graph_df = graph_df.merge(hosp_repeat, on='CUST_ID', how='left')

        doc_repeat = grp['CHME_LICE_NO'].apply(get_max_repeat_ratio).reset_index(name='GRAPH2_CUST_DOC_REPEAT_RATIO')
        graph_df = graph_df.merge(doc_repeat, on='CUST_ID', how='left')

        # 전체 그래프 degree
        graph_df['GRAPH2_CUST_GRAPH_DEGREE'] = (
            graph_df['GRAPH2_CUST_HOSP_DEGREE'] +
            graph_df['GRAPH2_CUST_DOC_DEGREE'] +
            graph_df['GRAPH2_CUST_DISEASE_DEGREE']
        )

        graph_df = graph_df.fillna(0)

        return graph_df

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = X.merge(self.claim_features_, on="CUST_ID", how="left")
        new_cols = [c for c in self.claim_features_.columns if c != "CUST_ID"]
        X[new_cols] = X[new_cols].fillna(0)
        return X


class AmountFeature_v2:
    """금액 패턴 v2: 변동계수, MAD, 재청구 패턴"""

    def __init__(self):
        self.claim_features_ = None

    def fit(self, X: pd.DataFrame, y=None, claim_df: pd.DataFrame = None):
        if claim_df is None:
            raise ValueError("AmountFeature_v2 는 claim_df가 필요합니다.")
        self.claim_features_ = self._build_features(claim_df)
        return self

    def _build_features(self, claim_df: pd.DataFrame) -> pd.DataFrame:
        df_claim = claim_df.copy()
        grp = df_claim.groupby('CUST_ID')
        amount_df = pd.DataFrame({'CUST_ID': df_claim['CUST_ID'].unique()})

        # 청구/지급 금액 관련 (기존)
        if 'PAYM_AMT' in df_claim.columns and 'DMND_AMT' in df_claim.columns:
            # 지급/청구 비율
            df_claim['PAYM_DMND_RATIO'] = df_claim['PAYM_AMT'] / (df_claim['DMND_AMT'].replace(0, np.nan))
            df_claim['PAYM_DMND_RATIO'] = df_claim['PAYM_DMND_RATIO'].replace([np.inf, -np.inf], np.nan)

            paym_dmnd_stats = grp['PAYM_DMND_RATIO'].agg(['mean', 'std']).reset_index()
            paym_dmnd_stats.columns = ['CUST_ID', 'AMT2_PAYM_DMND_RATIO_MEAN', 'AMT2_PAYM_DMND_RATIO_STD']
            amount_df = amount_df.merge(paym_dmnd_stats, on='CUST_ID', how='left')

            # 총 청구금액, 총 지급금액
            total_amt = grp.agg(
                AMT2_TOTAL_DMND=('DMND_AMT', 'sum'),
                AMT2_TOTAL_PAYM=('PAYM_AMT', 'sum'),
            ).reset_index()
            amount_df = amount_df.merge(total_amt, on='CUST_ID', how='left')

            # 고액 청구 비율 (상위 10% 기준)
            dmnd_threshold = df_claim['DMND_AMT'].quantile(0.9)
            df_claim['IS_HIGH_DMND'] = (df_claim['DMND_AMT'] > dmnd_threshold).astype(int)
            high_dmnd_ratio = grp['IS_HIGH_DMND'].mean().reset_index(name='AMT2_HIGH_DMND_RATIO')
            amount_df = amount_df.merge(high_dmnd_ratio, on='CUST_ID', how='left')

            # 변동계수 추가 (신규)
            dmnd_cv = grp['DMND_AMT'].agg(['mean', 'std']).reset_index()
            dmnd_cv['AMT2_DMND_CV'] = dmnd_cv['std'] / dmnd_cv['mean'].replace(0, np.nan)
            dmnd_cv['AMT2_DMND_CV'] = dmnd_cv['AMT2_DMND_CV'].fillna(0)
            amount_df = amount_df.merge(dmnd_cv[['CUST_ID', 'AMT2_DMND_CV']], on='CUST_ID', how='left')

            # MAD (Median Absolute Deviation) 추가 (신규)
            def mad(series):
                median = series.median()
                return (series - median).abs().median()

            dmnd_mad = grp['DMND_AMT'].apply(mad).reset_index(name='AMT2_DMND_MAD')
            amount_df = amount_df.merge(dmnd_mad, on='CUST_ID', how='left')

        # NON_PAY_RATIO 통계 (기존)
        if 'NON_PAY_RATIO' in df_claim.columns:
            non_pay_stats = grp['NON_PAY_RATIO'].agg(['mean', 'max', 'std']).reset_index()
            non_pay_stats.columns = ['CUST_ID', 'AMT2_NON_PAY_RATIO_MEAN', 'AMT2_NON_PAY_RATIO_MAX', 'AMT2_NON_PAY_RATIO_STD']
            amount_df = amount_df.merge(non_pay_stats, on='CUST_ID', how='left')

        amount_df = amount_df.fillna(0)
        return amount_df

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = X.merge(self.claim_features_, on="CUST_ID", how="left")
        new_cols = [c for c in self.claim_features_.columns if c != "CUST_ID"]
        X[new_cols] = X[new_cols].fillna(0)
        return X


class TimeFeature_v2:
    """시계열 패턴 v2: 월말/월초, 요일 엔트로피"""

    def __init__(self):
        self.claim_features_ = None

    def fit(self, X: pd.DataFrame, y=None, claim_df: pd.DataFrame = None):
        if claim_df is None:
            raise ValueError("TimeFeature_v2 는 claim_df가 필요합니다.")
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

        grp = df_claim.groupby('CUST_ID')
        time_df = pd.DataFrame({'CUST_ID': df_claim['CUST_ID'].unique()})

        # 요일 패턴 (기존)
        df_claim['WEEKDAY'] = df_claim['RECP_DATE'].dt.weekday
        df_claim['IS_WEEKEND'] = (df_claim['WEEKDAY'] >= 5).astype(int)

        weekday_stats = grp['IS_WEEKEND'].agg(['mean']).reset_index()
        weekday_stats.columns = ['CUST_ID', 'TIME2_WEEKEND_RATIO']
        time_df = time_df.merge(weekday_stats, on='CUST_ID', how='left')

        # 요일 분포 엔트로피 (신규)
        def calculate_entropy(series):
            value_counts = series.value_counts()
            probs = value_counts / len(series)
            return -np.sum(probs * np.log2(probs + 1e-10))

        weekday_entropy = grp['WEEKDAY'].apply(calculate_entropy).reset_index(name='TIME2_WEEKDAY_ENTROPY')
        time_df = time_df.merge(weekday_entropy, on='CUST_ID', how='left')

        # 월말/월초 패턴 (신규)
        df_claim['DAY_OF_MONTH'] = df_claim['RECP_DATE'].dt.day
        df_claim['IS_MONTH_START'] = (df_claim['DAY_OF_MONTH'] <= 10).astype(int)
        df_claim['IS_MONTH_END'] = (df_claim['DAY_OF_MONTH'] >= 21).astype(int)

        month_start_ratio = grp['IS_MONTH_START'].mean().reset_index(name='TIME2_MONTH_START_RATIO')
        time_df = time_df.merge(month_start_ratio, on='CUST_ID', how='left')

        month_end_ratio = grp['IS_MONTH_END'].mean().reset_index(name='TIME2_MONTH_END_RATIO')
        time_df = time_df.merge(month_end_ratio, on='CUST_ID', how='left')

        # 월별 청구 변동성 (기존)
        df_claim['YEAR_MONTH'] = df_claim['RECP_DATE'].dt.to_period('M')
        monthly_counts = df_claim.groupby(['CUST_ID', 'YEAR_MONTH']).size().reset_index(name='MONTHLY_COUNT')
        monthly_std = monthly_counts.groupby('CUST_ID')['MONTHLY_COUNT'].std().reset_index(name='TIME2_MONTHLY_CLAIM_STD')
        time_df = time_df.merge(monthly_std, on='CUST_ID', how='left')

        # 첫 청구 이후 경과 일수 (기존)
        date_range = grp['RECP_DATE'].agg(
            first_claim=('min'),
            last_claim=('max')
        ).reset_index()
        date_range['TIME2_DAYS_SINCE_FIRST'] = (date_range['last_claim'] - date_range['first_claim']).dt.days
        time_df = time_df.merge(date_range[['CUST_ID', 'TIME2_DAYS_SINCE_FIRST']], on='CUST_ID', how='left')

        # 월평균 청구 빈도 (기존)
        claim_count = grp.size().reset_index(name='claim_count')
        time_df = time_df.merge(claim_count, on='CUST_ID', how='left')
        time_df['TIME2_CLAIM_FREQ_PER_MONTH'] = time_df['claim_count'] / (time_df['TIME2_DAYS_SINCE_FIRST'].replace(0, 1) / 30)
        time_df = time_df.drop(columns=['claim_count'])

        # 최근 청구 가중 평균 (기존)
        def recent_weighted_count(dates):
            if len(dates) == 0:
                return 0
            dates = sorted(dates)
            max_date = dates[-1]
            weights = [np.exp(-0.01 * (max_date - d).days) for d in dates]
            return sum(weights) / len(dates) if len(dates) > 0 else 0

        recent_weighted = grp['RECP_DATE'].apply(recent_weighted_count).reset_index(name='TIME2_RECENT_WEIGHTED_AVG')
        time_df = time_df.merge(recent_weighted, on='CUST_ID', how='left')

        time_df = time_df.fillna(0)
        return time_df

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = X.merge(self.claim_features_, on="CUST_ID", how="left")
        new_cols = [c for c in self.claim_features_.columns if c != "CUST_ID"]
        X[new_cols] = X[new_cols].fillna(0)
        return X


class MedicalFeature_v2:
    """의료 행태 v2: 병명 전이, 희귀 질병"""

    def __init__(self):
        self.claim_features_ = None

    def fit(self, X: pd.DataFrame, y=None, claim_df: pd.DataFrame = None):
        if claim_df is None:
            raise ValueError("MedicalFeature_v2 는 claim_df가 필요합니다.")
        self.claim_features_ = self._build_features(claim_df)
        return self

    def _build_features(self, claim_df: pd.DataFrame) -> pd.DataFrame:
        df_claim = claim_df.copy()
        grp = df_claim.groupby('CUST_ID')
        medical_df = pd.DataFrame({'CUST_ID': df_claim['CUST_ID'].unique()})

        # 병명(DSAS_NAME) 다양성 - 엔트로피 (기존)
        if 'DSAS_NAME' in df_claim.columns:
            def calculate_entropy(series):
                value_counts = series.value_counts()
                probs = value_counts / len(series)
                return -np.sum(probs * np.log2(probs + 1e-10))

            disease_entropy = grp['DSAS_NAME'].apply(calculate_entropy).reset_index(name='MED2_DISEASE_ENTROPY')
            medical_df = medical_df.merge(disease_entropy, on='CUST_ID', how='left')

            # 고유 병명 개수 (기존)
            disease_count = grp['DSAS_NAME'].nunique().reset_index(name='MED2_DISEASE_COUNT')
            medical_df = medical_df.merge(disease_count, on='CUST_ID', how='left')

            # 희귀 질병 비율 (신규 - 전체 데이터에서 하위 10% 빈도)
            disease_freq = df_claim['DSAS_NAME'].value_counts()
            disease_rare_threshold = disease_freq.quantile(0.1)
            df_claim['IS_RARE_DISEASE'] = df_claim['DSAS_NAME'].map(
                lambda x: 1 if disease_freq.get(x, 0) < disease_rare_threshold else 0
            )
            rare_disease_ratio = grp['IS_RARE_DISEASE'].mean().reset_index(name='MED2_RARE_DISEASE_RATIO')
            medical_df = medical_df.merge(rare_disease_ratio, on='CUST_ID', how='left')

        # 사고구분(ACCI_DVSN) 분포 (기존)
        if 'ACCI_DVSN' in df_claim.columns:
            df_claim['ACCI_DVSN'] = df_claim['ACCI_DVSN'].fillna('UNKNOWN')
            acci_ratio = pd.crosstab(
                df_claim['CUST_ID'],
                df_claim['ACCI_DVSN'],
                normalize='index'
            )
            acci_ratio.columns = [f'MED2_ACCI_{col}' for col in acci_ratio.columns]
            acci_ratio = acci_ratio.reset_index()
            medical_df = medical_df.merge(acci_ratio, on='CUST_ID', how='left')

        # 병원전문구분(HOSP_SPEC_DVSN) 분포 (기존)
        if 'HOSP_SPEC_DVSN' in df_claim.columns:
            df_claim['HOSP_SPEC_DVSN'] = df_claim['HOSP_SPEC_DVSN'].fillna('UNKNOWN')
            hosp_spec_ratio = pd.crosstab(
                df_claim['CUST_ID'],
                df_claim['HOSP_SPEC_DVSN'],
                normalize='index'
            )
            hosp_spec_ratio.columns = [f'MED2_HOSP_{col}' for col in hosp_spec_ratio.columns]
            hosp_spec_ratio = hosp_spec_ratio.reset_index()
            medical_df = medical_df.merge(hosp_spec_ratio, on='CUST_ID', how='left')

        medical_df = medical_df.fillna(0)
        return medical_df

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = X.merge(self.claim_features_, on="CUST_ID", how="left")
        new_cols = [c for c in self.claim_features_.columns if c != "CUST_ID"]
        X[new_cols] = X[new_cols].fillna(0)
        return X
"""
Member B v3 Features - Phase 3
신규 Feature 클래스들
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


# ===== Phase 3: 신규 Feature =====


class InteractionFeature:
    """Feature 간 상호작용 패턴 (v3 신규)"""

    def __init__(self):
        self.claim_features_ = None

    def fit(self, X: pd.DataFrame, y=None, claim_df: pd.DataFrame = None):
        if claim_df is None:
            raise ValueError("InteractionFeature 는 claim_df가 필요합니다.")
        self.claim_features_ = self._build_features(claim_df, X)
        return self

    def _build_features(self, claim_df: pd.DataFrame, cust_df: pd.DataFrame) -> pd.DataFrame:
        df_claim = claim_df.copy()
        grp = df_claim.groupby('CUST_ID')

        # 먼저 기본 집계
        interaction_df = pd.DataFrame({'CUST_ID': df_claim['CUST_ID'].unique()})

        # 1. 고액 청구 × 집중도
        if 'DMND_AMT' in df_claim.columns and 'RECP_DATE' in df_claim.columns:
            dmnd_threshold = df_claim['DMND_AMT'].quantile(0.9)
            df_claim['IS_HIGH_DMND'] = (df_claim['DMND_AMT'] > dmnd_threshold).astype(int)

            high_dmnd_ratio = grp['IS_HIGH_DMND'].mean().reset_index(name='high_dmnd_ratio')
            interaction_df = interaction_df.merge(high_dmnd_ratio, on='CUST_ID', how='left')

            # 30일 내 청구 횟수 계산
            df_claim['RECP_DATE'] = pd.to_datetime(df_claim['RECP_DATE'], errors='coerce')
            df_claim_with_date = df_claim.dropna(subset=['RECP_DATE']).copy()

            if len(df_claim_with_date) > 0:
                df_claim_with_date = df_claim_with_date.sort_values(['CUST_ID', 'RECP_DATE'])

                def max_claims_in_30d(dates):
                    dates = sorted(dates)
                    if not dates:
                        return 0
                    max_count = 1
                    left = 0
                    for right in range(len(dates)):
                        while dates[right] >= dates[left] + pd.Timedelta(days=30):
                            left += 1
                        max_count = max(max_count, right - left + 1)
                    return max_count

                burst_30d = df_claim_with_date.groupby('CUST_ID')['RECP_DATE'].apply(
                    lambda x: max_claims_in_30d(x.tolist())
                ).reset_index(name='burst_30d')
                interaction_df = interaction_df.merge(burst_30d, on='CUST_ID', how='left')

                # 상호작용: 고액 청구 비율 × 집중도
                interaction_df['INTER_HIGH_DMND_X_BURST'] = (
                    interaction_df['high_dmnd_ratio'] * interaction_df['burst_30d']
                )

        # 2. 네트워크 분산도 × 금액
        if 'HOSP_CODE' in df_claim.columns and 'CHME_LICE_NO' in df_claim.columns and 'DMND_AMT' in df_claim.columns:
            hosp_degree = grp['HOSP_CODE'].nunique().reset_index(name='hosp_degree')
            doc_degree = grp['CHME_LICE_NO'].nunique().reset_index(name='doc_degree')
            interaction_df = interaction_df.merge(hosp_degree, on='CUST_ID', how='left')
            interaction_df = interaction_df.merge(doc_degree, on='CUST_ID', how='left')

            graph_degree = interaction_df['hosp_degree'] + interaction_df['doc_degree']
            claim_count = grp.size().reset_index(name='claim_count')
            interaction_df = interaction_df.merge(claim_count, on='CUST_ID', how='left')

            total_dmnd = grp['DMND_AMT'].sum().reset_index(name='total_dmnd')
            interaction_df = interaction_df.merge(total_dmnd, on='CUST_ID', how='left')

            # 상호작용: 그래프 분산도 × 평균 청구금액
            interaction_df['INTER_GRAPH_X_AVG_DMND'] = (
                graph_degree * (interaction_df['total_dmnd'] / interaction_df['claim_count'].replace(0, 1))
            )

        # 3. 병원-의사 일치도
        if 'HOSP_CODE' in df_claim.columns and 'CHME_LICE_NO' in df_claim.columns:
            df_claim['HOSP_CODE'] = df_claim['HOSP_CODE'].fillna('UNKNOWN')
            df_claim['CHME_LICE_NO'] = df_claim['CHME_LICE_NO'].fillna('UNKNOWN')

            # 병원 대비 의사 비율
            interaction_df['INTER_DOC_HOSP_MISMATCH'] = (
                interaction_df['doc_degree'] / interaction_df['hosp_degree'].replace(0, 1)
            )

        # 임시 컬럼 제거
        cols_to_drop = ['high_dmnd_ratio', 'burst_30d', 'hosp_degree', 'doc_degree', 'claim_count', 'total_dmnd']
        interaction_df = interaction_df.drop(columns=[c for c in cols_to_drop if c in interaction_df.columns], errors='ignore')

        interaction_df = interaction_df.fillna(0)
        return interaction_df

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = X.merge(self.claim_features_, on="CUST_ID", how="left")
        new_cols = [c for c in self.claim_features_.columns if c != "CUST_ID"]
        X[new_cols] = X[new_cols].fillna(0)
        return X


class SequenceFeature:
    """청구 시퀀스 및 시계열 추세 (v3 신규)"""

    def __init__(self):
        self.claim_features_ = None

    def fit(self, X: pd.DataFrame, y=None, claim_df: pd.DataFrame = None):
        if claim_df is None:
            raise ValueError("SequenceFeature 는 claim_df가 필요합니다.")
        self.claim_features_ = self._build_features(claim_df)
        return self

    def _build_features(self, claim_df: pd.DataFrame) -> pd.DataFrame:
        df_claim = claim_df.copy()

        if 'RECP_DATE' not in df_claim.columns or 'DMND_AMT' not in df_claim.columns:
            return pd.DataFrame({'CUST_ID': claim_df['CUST_ID'].unique()})

        df_claim['RECP_DATE'] = pd.to_datetime(df_claim['RECP_DATE'], errors='coerce')
        df_claim = df_claim.dropna(subset=['RECP_DATE', 'DMND_AMT'])
        df_claim = df_claim.sort_values(['CUST_ID', 'RECP_DATE'])

        grp = df_claim.groupby('CUST_ID')
        sequence_df = pd.DataFrame({'CUST_ID': df_claim['CUST_ID'].unique()})

        # 1. 청구금액 선형 회귀 기울기 (시간에 따른 추세)
        def calculate_slope(group):
            if len(group) < 2:
                return 0
            dates = group['RECP_DATE']
            amounts = group['DMND_AMT']

            # 날짜를 숫자로 변환 (첫 날짜부터의 일수)
            x = (dates - dates.min()).dt.days.values
            y = amounts.values

            if len(x) < 2 or np.std(x) == 0:
                return 0

            # 선형 회귀 기울기
            slope = np.polyfit(x, y, 1)[0]
            return slope

        dmnd_slope = grp.apply(calculate_slope).reset_index(name='SEQ_DMND_SLOPE')
        sequence_df = sequence_df.merge(dmnd_slope, on='CUST_ID', how='left')

        # 2. 청구 간격 가속도 (간격이 점점 짧아지는지)
        df_claim['CLAIM_INTERVAL'] = df_claim.groupby('CUST_ID')['RECP_DATE'].diff().dt.days

        def calculate_interval_slope(group):
            intervals = group['CLAIM_INTERVAL'].dropna()
            if len(intervals) < 2:
                return 0
            x = np.arange(len(intervals))
            y = intervals.values
            if np.std(x) == 0:
                return 0
            return np.polyfit(x, y, 1)[0]

        interval_slope = df_claim.groupby('CUST_ID').apply(calculate_interval_slope).reset_index(name='SEQ_INTERVAL_ACCEL')
        sequence_df = sequence_df.merge(interval_slope, on='CUST_ID', how='left')

        # 3. 최근 집중도 (최근 3개월 청구 비율)
        if len(df_claim) > 0:
            max_date = df_claim['RECP_DATE'].max()
            df_claim['IS_RECENT_3M'] = (df_claim['RECP_DATE'] >= max_date - pd.Timedelta(days=90)).astype(int)

            recent_ratio = df_claim.groupby('CUST_ID')['IS_RECENT_3M'].mean().reset_index(name='SEQ_RECENT_3M_RATIO')
            sequence_df = sequence_df.merge(recent_ratio, on='CUST_ID', how='left')

        sequence_df = sequence_df.fillna(0)
        return sequence_df

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = X.merge(self.claim_features_, on="CUST_ID", how="left")
        new_cols = [c for c in self.claim_features_.columns if c != "CUST_ID"]
        X[new_cols] = X[new_cols].fillna(0)
        return X


class AnomalyScoreFeature:
    """통계적 이상 탐지 점수 (v3 신규)"""

    def __init__(self, contamination=0.1):
        self.contamination = contamination
        self.claim_features_ = None

    def fit(self, X: pd.DataFrame, y=None, claim_df: pd.DataFrame = None):
        if claim_df is None:
            raise ValueError("AnomalyScoreFeature 는 claim_df가 필요합니다.")
        self.claim_features_ = self._build_features(claim_df)
        return self

    def _build_features(self, claim_df: pd.DataFrame) -> pd.DataFrame:
        df_claim = claim_df.copy()
        grp = df_claim.groupby('CUST_ID')
        anomaly_df = pd.DataFrame({'CUST_ID': df_claim['CUST_ID'].unique()})

        # 1. Isolation Forest score (청구금액, 지급금액)
        if 'DMND_AMT' in df_claim.columns and 'PAYM_AMT' in df_claim.columns:
            amount_agg = grp.agg(
                dmnd_mean=('DMND_AMT', 'mean'),
                dmnd_max=('DMND_AMT', 'max'),
                paym_mean=('PAYM_AMT', 'mean'),
                paym_max=('PAYM_AMT', 'max'),
            ).reset_index()

            # Isolation Forest
            features_for_iso = amount_agg[['dmnd_mean', 'dmnd_max', 'paym_mean', 'paym_max']].fillna(0)

            if len(features_for_iso) > 10:  # 최소 샘플 수
                iso = IsolationForest(contamination=self.contamination, random_state=42)
                anomaly_scores = iso.fit_predict(features_for_iso)
                anomaly_scores = iso.score_samples(features_for_iso)  # 이상치 점수 (-1에 가까울수록 이상)
                amount_agg['ANOM_ISO_AMOUNT_SCORE'] = anomaly_scores
            else:
                amount_agg['ANOM_ISO_AMOUNT_SCORE'] = 0

            anomaly_df = anomaly_df.merge(amount_agg[['CUST_ID', 'ANOM_ISO_AMOUNT_SCORE']], on='CUST_ID', how='left')

        # 2. 청구금액 분포 왜도/첨도
        if 'DMND_AMT' in df_claim.columns:
            from scipy import stats

            def safe_skew(series):
                try:
                    return stats.skew(series) if len(series) > 3 else 0
                except:
                    return 0

            def safe_kurtosis(series):
                try:
                    return stats.kurtosis(series) if len(series) > 3 else 0
                except:
                    return 0

            skew_kurt = grp['DMND_AMT'].agg([safe_skew, safe_kurtosis]).reset_index()
            skew_kurt.columns = ['CUST_ID', 'ANOM_DMND_SKEW', 'ANOM_DMND_KURT']
            anomaly_df = anomaly_df.merge(skew_kurt, on='CUST_ID', how='left')

        anomaly_df = anomaly_df.fillna(0)
        return anomaly_df

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = X.merge(self.claim_features_, on="CUST_ID", how="left")
        new_cols = [c for c in self.claim_features_.columns if c != "CUST_ID"]
        X[new_cols] = X[new_cols].fillna(0)
        return X


class RatioFeature:
    """비율 및 정규화 특성 (v3 신규)"""

    def __init__(self):
        self.claim_features_ = None

    def fit(self, X: pd.DataFrame, y=None, claim_df: pd.DataFrame = None):
        if claim_df is None:
            raise ValueError("RatioFeature 는 claim_df가 필요합니다.")
        self.claim_features_ = self._build_features(claim_df, X)
        return self

    def _build_features(self, claim_df: pd.DataFrame, cust_df: pd.DataFrame) -> pd.DataFrame:
        df_claim = claim_df.copy()
        grp = df_claim.groupby('CUST_ID')
        ratio_df = pd.DataFrame({'CUST_ID': df_claim['CUST_ID'].unique()})

        # 고객 데이터와 조인 (연령, 직업 정보 필요)
        if 'AGE' in cust_df.columns and 'OCCP_GRP_1' in cust_df.columns and 'DMND_AMT' in df_claim.columns:
            # 고객별 평균 청구금액
            cust_avg_dmnd = grp['DMND_AMT'].mean().reset_index(name='cust_avg_dmnd')
            ratio_df = ratio_df.merge(cust_avg_dmnd, on='CUST_ID', how='left')

            # 연령대×직업 그룹별 평균 계산을 위해 고객 정보 병합
            ratio_df = ratio_df.merge(cust_df[['CUST_ID', 'AGE', 'OCCP_GRP_1']], on='CUST_ID', how='left')

            # 연령대 생성 (10년 단위)
            ratio_df['AGE_GROUP'] = (ratio_df['AGE'] // 10) * 10

            # 연령대×직업 그룹별 평균
            peer_avg = ratio_df.groupby(['AGE_GROUP', 'OCCP_GRP_1'])['cust_avg_dmnd'].transform('mean')
            ratio_df['RATIO_DMND_VS_PEER'] = ratio_df['cust_avg_dmnd'] / peer_avg.replace(0, 1)

            # 임시 컬럼 제거
            ratio_df = ratio_df.drop(columns=['AGE', 'OCCP_GRP_1', 'AGE_GROUP', 'cust_avg_dmnd'])

        # 최대/평균 비율
        if 'DMND_AMT' in df_claim.columns:
            dmnd_stats = grp['DMND_AMT'].agg(['mean', 'max']).reset_index()
            dmnd_stats['RATIO_DMND_MAX_MEAN'] = dmnd_stats['max'] / dmnd_stats['mean'].replace(0, 1)
            ratio_df = ratio_df.merge(dmnd_stats[['CUST_ID', 'RATIO_DMND_MAX_MEAN']], on='CUST_ID', how='left')

        # 집중도 지수 (상위 20% 청구가 전체에서 차지하는 비율)
        if 'DMND_AMT' in df_claim.columns:
            def concentration_index(series):
                if len(series) == 0:
                    return 0
                sorted_series = series.sort_values(ascending=False)
                top_20_count = max(1, len(series) // 5)
                top_20_sum = sorted_series.head(top_20_count).sum()
                total_sum = series.sum()
                return top_20_sum / total_sum if total_sum > 0 else 0

            concentration = grp['DMND_AMT'].apply(concentration_index).reset_index(name='RATIO_CONCENTRATION_INDEX')
            ratio_df = ratio_df.merge(concentration, on='CUST_ID', how='left')

        ratio_df = ratio_df.fillna(0)
        return ratio_df

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = X.merge(self.claim_features_, on="CUST_ID", how="left")
        new_cols = [c for c in self.claim_features_.columns if c != "CUST_ID"]
        X[new_cols] = X[new_cols].fillna(0)
        return X


class TemporalDecayFeature:
    """시간 감쇠 가중 특성 (v3 신규)"""

    def __init__(self, decay_rates=None):
        self.decay_rates = decay_rates or [0.01, 0.05, 0.1]
        self.claim_features_ = None

    def fit(self, X: pd.DataFrame, y=None, claim_df: pd.DataFrame = None):
        if claim_df is None:
            raise ValueError("TemporalDecayFeature 는 claim_df가 필요합니다.")
        self.claim_features_ = self._build_features(claim_df)
        return self

    def _build_features(self, claim_df: pd.DataFrame) -> pd.DataFrame:
        df_claim = claim_df.copy()

        if 'RECP_DATE' not in df_claim.columns or 'DMND_AMT' not in df_claim.columns:
            return pd.DataFrame({'CUST_ID': claim_df['CUST_ID'].unique()})

        df_claim['RECP_DATE'] = pd.to_datetime(df_claim['RECP_DATE'], errors='coerce')
        df_claim = df_claim.dropna(subset=['RECP_DATE', 'DMND_AMT'])
        df_claim = df_claim.sort_values(['CUST_ID', 'RECP_DATE'])

        grp = df_claim.groupby('CUST_ID')
        temporal_df = pd.DataFrame({'CUST_ID': df_claim['CUST_ID'].unique()})

        # 지수 감쇠 가중 청구금액
        for decay_rate in self.decay_rates:
            def exponential_decay_weighted(group):
                if len(group) == 0:
                    return 0
                max_date = group['RECP_DATE'].max()
                days_ago = (max_date - group['RECP_DATE']).dt.days
                weights = np.exp(-decay_rate * days_ago)
                weighted_sum = (group['DMND_AMT'] * weights).sum()
                return weighted_sum / weights.sum() if weights.sum() > 0 else 0

            decay_feature = grp.apply(exponential_decay_weighted).reset_index(name=f'TEMP_DECAY_{int(decay_rate*100):02d}_DMND')
            temporal_df = temporal_df.merge(decay_feature, on='CUST_ID', how='left')

        # 최근 6개월 vs 이전 차이
        if len(df_claim) > 0:
            max_date = df_claim['RECP_DATE'].max()
            df_claim['IS_RECENT_6M'] = df_claim['RECP_DATE'] >= max_date - pd.Timedelta(days=180)

            recent_avg = df_claim[df_claim['IS_RECENT_6M']].groupby('CUST_ID')['DMND_AMT'].mean().reset_index(name='recent_avg')
            old_avg = df_claim[~df_claim['IS_RECENT_6M']].groupby('CUST_ID')['DMND_AMT'].mean().reset_index(name='old_avg')

            temporal_df = temporal_df.merge(recent_avg, on='CUST_ID', how='left')
            temporal_df = temporal_df.merge(old_avg, on='CUST_ID', how='left')

            temporal_df['TEMP_RECENT_VS_OLD_DIFF'] = temporal_df['recent_avg'] - temporal_df['old_avg']
            temporal_df = temporal_df.drop(columns=['recent_avg', 'old_avg'])

        temporal_df = temporal_df.fillna(0)
        return temporal_df

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = X.merge(self.claim_features_, on="CUST_ID", how="left")
        new_cols = [c for c in self.claim_features_.columns if c != "CUST_ID"]
        X[new_cols] = X[new_cols].fillna(0)
        return X
