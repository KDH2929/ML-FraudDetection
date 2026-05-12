import pandas as pd
from src.preprocessing.strategies.base_strategy import BaseStrategy
from src.preprocessing.components.missing_value import MedianImputer
from src.preprocessing.components.outlier import QuantileCapper
from src.preprocessing.components.encoder import TargetEncoder
from src.preprocessing.components.scaler import RobustScalerWrapper
from src.preprocessing.components.feature_engineer import (
    DeviationFeature,
    BurstFeature,
    GraphFeature,
    AmountFeature,
    TimeFeature,
    MedicalFeature,
)
from src.config import ID_COL, DIVIDED_SET_COL


class MemberB2Strategy(BaseStrategy):

    def get_strategy_name(self) -> str:
        return "B2: Deviation+Burst+Graph+Amount+Time+Medical"

    def preprocess(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        claim_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Member B 전처리 전략 v2: 1차 Feature + 2차 Feature

        [1차 Feature]
        1. DeviationFeature: 그룹별 평균 대비 z-score 편차
        2. BurstFeature: 짧은 기간 내 청구 집중도
        3. GraphFeature: 고객-병원-의사 네트워크

        [2차 Feature - 추가]
        4. AmountFeature: 청구/지급 금액 패턴
        5. TimeFeature: 시계열 청구 패턴
        6. MedicalFeature: 의료 행태 패턴

        [전처리]
        7. TargetEncoder: 범주형 변수를 타겟 평균으로 인코딩
        8. MedianImputer: 결측치를 중앙값으로 대체
        9. QuantileCapper: 이상치를 1%~99% 범위로 캡핑
        10. RobustScaler: 중앙값과 IQR 기반 스케일링
        """
        if claim_df is None:
            raise ValueError("Member B2 전략은 claim_df가 필요합니다.")

        X = X.copy()

        # 1차 Feature Engineering
        deviation_feat = DeviationFeature()
        X = deviation_feat.fit(X, y, claim_df=claim_df).transform(X)

        burst_feat = BurstFeature()
        X = burst_feat.fit(X, y, claim_df=claim_df).transform(X)

        graph_feat = GraphFeature()
        X = graph_feat.fit(X, y, claim_df=claim_df).transform(X)

        # 2차 Feature Engineering (추가)
        amount_feat = AmountFeature()
        X = amount_feat.fit(X, y, claim_df=claim_df).transform(X)

        time_feat = TimeFeature()
        X = time_feat.fit(X, y, claim_df=claim_df).transform(X)

        medical_feat = MedicalFeature()
        X = medical_feat.fit(X, y, claim_df=claim_df).transform(X)

        # 범주형 인코딩 (타겟 인코더)
        # 메타 컬럼 제외하고 모든 범주형 컬럼 자동 감지
        meta_cols = [ID_COL, DIVIDED_SET_COL]
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        cat_cols = [col for col in cat_cols if col not in meta_cols]

        if cat_cols:
            # 범주형 결측치를 먼저 'MISSING'으로 채움
            for col in cat_cols:
                X[col] = X[col].fillna('MISSING')

            encoder = TargetEncoder(cols=cat_cols)
            X = encoder.fit(X, y).transform(X)

        # 결측치 처리 (중앙값)
        # 메타 컬럼 제외하고 수치형만 처리
        num_cols = [col for col in X.select_dtypes(include='number').columns
                    if col not in meta_cols]

        imputer = MedianImputer(cols=num_cols)
        X = imputer.fit(X).transform(X)

        # 이상치 캡핑 (1%~99%)
        capper = QuantileCapper(cols=num_cols, lower=0.01, upper=0.99)
        X = capper.fit(X).transform(X)

        # 스케일링 (RobustScaler)
        scaler = RobustScalerWrapper(cols=num_cols)
        X = scaler.fit(X).transform(X)

        return X
