import pandas as pd
from src.preprocessing.strategies.base_strategy import BaseStrategy
from src.preprocessing.components.missing_value import GroupMeanImputer, ZeroImputer
from src.preprocessing.components.outlier import IQRCapper
from src.preprocessing.components.encoder import TargetEncoder
from src.preprocessing.components.scaler import RobustScalerWrapper
from src.preprocessing.components.feature_engineer import ClaimFeature


class MemberAStrategy(BaseStrategy):

    def get_strategy_name(self) -> str:
        return "A: GroupMean+IQR+Target+Robust"

    def preprocess(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        claim_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        # TODO: 팀원 A가 구현
        raise NotImplementedError("팀원 A가 preprocess()를 구현해야 합니다.")
