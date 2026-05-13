import pandas as pd
from src.preprocessing.strategies.base_strategy import BaseStrategy
from src.preprocessing.components.missing_value import MedianImputer, MissingIndicator
from src.preprocessing.components.outlier import QuantileCapper, OutlierFlagger
from src.preprocessing.components.encoder import TargetEncoder, FrequencyEncoder, RarityEncoder
from src.preprocessing.components.scaler import RobustScalerWrapper
from src.preprocessing.components.features.member_b_features import (
    # v2 개선 Feature
    DeviationFeature_v2,
    BurstFeature_v2,
    GraphFeature_v2,
    AmountFeature_v2,
    TimeFeature_v2,
    MedicalFeature_v2,
    # v3 신규 Feature
    InteractionFeature,
    SequenceFeature,
    AnomalyScoreFeature,
    RatioFeature,
    TemporalDecayFeature,
)
from src.optimization.feature_selector import CorrelationRemover, ImportanceSelector
from src.config import ID_COL, DIVIDED_SET_COL


class MemberB3Strategy(BaseStrategy):

    def get_strategy_name(self) -> str:
        return "B3: v3 최적화 (Feature Selection + Hyperparameter Tuning)"

    def preprocess(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        claim_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Member B 전처리 전략 v3: 최적화 (Feature Selection + Tuning)

        [Phase 1: v2 개선 Feature]
        1. DeviationFeature_v2: 다층 그룹, 백분위, 시간 구간별
        2. BurstFeature_v2: 7/14일 윈도우, 변동계수
        3. GraphFeature_v2: 병원-의사 페어, 희귀도
        4. AmountFeature_v2: 변동계수, MAD
        5. TimeFeature_v2: 월말/월초, 요일 엔트로피
        6. MedicalFeature_v2: 희귀 질병

        [Phase 2: v3 신규 Feature]
        7. InteractionFeature: 고액×집중, 네트워크×금액
        8. SequenceFeature: 청구 추세, 가속도
        9. AnomalyScoreFeature: Isolation Forest, 왜도/첨도
        10. RatioFeature: 또래 대비, 집중도
        11. TemporalDecayFeature: 시간 감쇠 가중

        [Phase 3: 전처리 파이프라인 + Feature Selection]
        12. MissingIndicator: 결측치 flag
        13. TargetEncoder: 범주형 인코딩
        14. FrequencyEncoder, RarityEncoder: 고카디널리티 보강
        15. MedianImputer: 결측치 대체
        16. OutlierFlagger: 이상치 flag
        17. QuantileCapper: 이상치 캡핑
        18. CorrelationRemover: 고상관 feature 제거 (threshold=0.95)
        19. ImportanceSelector: LightGBM 기반 상위 100개 선택
        20. RobustScaler: 스케일링

        [성능]
        - Features: 100개 (326개에서 69% 감소)
        - Recall: 0.5148 (v2 대비 +4.9%, v3 full 대비 +0.7%)
        - F1: 0.6078 (v2 대비 +2.6%, v3 full 대비 +0.2%)
        - Hyperparameter: Optuna 50 trials로 최적화
        """
        if claim_df is None:
            raise ValueError("Member B3 전략은 claim_df가 필요합니다.")

        X = X.copy()
        meta_cols = [ID_COL, DIVIDED_SET_COL]

        # ===== 1. 결측치 지시자 생성 (Feature Engineering 전) =====
        missing_indicator = MissingIndicator()
        X = missing_indicator.fit(X).transform(X)

        # ===== 2. Feature Engineering (원본 범주형 사용) =====

        # v2 개선 Feature
        print("  [v3] DeviationFeature_v2...")
        deviation_v2 = DeviationFeature_v2()
        X = deviation_v2.fit(X, y, claim_df=claim_df).transform(X)

        print("  [v3] BurstFeature_v2...")
        burst_v2 = BurstFeature_v2()
        X = burst_v2.fit(X, y, claim_df=claim_df).transform(X)

        print("  [v3] GraphFeature_v2...")
        graph_v2 = GraphFeature_v2()
        X = graph_v2.fit(X, y, claim_df=claim_df).transform(X)

        print("  [v3] AmountFeature_v2...")
        amount_v2 = AmountFeature_v2()
        X = amount_v2.fit(X, y, claim_df=claim_df).transform(X)

        print("  [v3] TimeFeature_v2...")
        time_v2 = TimeFeature_v2()
        X = time_v2.fit(X, y, claim_df=claim_df).transform(X)

        print("  [v3] MedicalFeature_v2...")
        medical_v2 = MedicalFeature_v2()
        X = medical_v2.fit(X, y, claim_df=claim_df).transform(X)

        # v3 신규 Feature
        print("  [v3] InteractionFeature...")
        interaction = InteractionFeature()
        X = interaction.fit(X, y, claim_df=claim_df).transform(X)

        print("  [v3] SequenceFeature...")
        sequence = SequenceFeature()
        X = sequence.fit(X, y, claim_df=claim_df).transform(X)

        print("  [v3] AnomalyScoreFeature...")
        anomaly = AnomalyScoreFeature()
        X = anomaly.fit(X, y, claim_df=claim_df).transform(X)

        print("  [v3] RatioFeature...")
        ratio = RatioFeature()
        X = ratio.fit(X, y, claim_df=claim_df).transform(X)

        print("  [v3] TemporalDecayFeature...")
        temporal = TemporalDecayFeature()
        X = temporal.fit(X, y, claim_df=claim_df).transform(X)

        # ===== 3. 이상치 처리 (Feature 생성 완료 후) =====
        num_cols = [col for col in X.select_dtypes(include='number').columns
                    if col not in meta_cols]

        # 이상치 플래그 생성
        outlier_flagger = OutlierFlagger(cols=num_cols, method='quantile')
        X = outlier_flagger.fit(X).transform(X)

        # 이상치 캡핑
        capper = QuantileCapper(cols=num_cols, lower=0.01, upper=0.99)
        X = capper.fit(X).transform(X)

        # ===== 4. 범주형 인코딩 =====
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        cat_cols = [col for col in cat_cols if col not in meta_cols]

        if cat_cols:
            # 결측치 먼저 처리
            for col in cat_cols:
                X[col] = X[col].fillna('MISSING')

            # 카디널리티 분석
            low_card = [c for c in cat_cols if X[c].nunique() < 10]
            high_card = [c for c in cat_cols if X[c].nunique() >= 10]

            # Target Encoding
            encoder = TargetEncoder(cols=cat_cols)
            X = encoder.fit(X, y).transform(X)

            # 고카디널리티 변수에 대해 빈도/희귀도 추가
            if high_card:
                freq_encoder = FrequencyEncoder(cols=high_card)
                X = freq_encoder.fit(X).transform(X)

                rarity_encoder = RarityEncoder(cols=high_card)
                X = rarity_encoder.fit(X).transform(X)

        # ===== 5. 결측치 2차 처리 =====
        num_cols = [col for col in X.select_dtypes(include='number').columns
                    if col not in meta_cols]

        imputer = MedianImputer(cols=num_cols)
        X = imputer.fit(X).transform(X)

        # ===== 6. Feature 선택 =====
        # 6-1. 고상관 제거
        print("  [v3] CorrelationRemover...")
        num_cols = [col for col in X.select_dtypes(include='number').columns
                    if col not in meta_cols]
        X_features = X[num_cols]

        corr_remover = CorrelationRemover(threshold=0.95)
        X_features = corr_remover.fit(X_features).transform(X_features)
        print(f"    -> CorrelationRemover: {len(num_cols)} -> {len(X_features.columns)} features")

        # 6-2. Feature Importance 기반 선택 (상위 100개)
        # train 데이터만 사용 (y가 유효한 행만)
        print("  [v3] ImportanceSelector (top 100)...")
        valid_mask = y.notna()
        X_train = X_features[valid_mask]
        y_train = y[valid_mask]

        importance_selector = ImportanceSelector(top_k=100)
        importance_selector.fit(X_train, y_train)
        X_features = importance_selector.transform(X_features)
        print(f"    -> ImportanceSelector: {len(corr_remover.selected_cols_)} -> {len(X_features.columns)} features")

        # meta_cols와 결합
        X = pd.concat([X[meta_cols], X_features], axis=1)

        # ===== 7. 스케일링 =====
        num_cols = [col for col in X.select_dtypes(include='number').columns
                    if col not in meta_cols]

        scaler = RobustScalerWrapper(cols=num_cols)
        X = scaler.fit(X).transform(X)

        print(f"  [v3] 최종 Feature 개수: {len(X.columns) - len(meta_cols)}")

        return X
