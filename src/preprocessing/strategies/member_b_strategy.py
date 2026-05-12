import pandas as pd
from src.preprocessing.strategies.base_strategy import BaseStrategy
from src.preprocessing.components.missing_value import MedianImputer
from src.preprocessing.components.outlier import NoOpOutlier
from src.preprocessing.components.encoder import OneHotEncoder
from src.preprocessing.components.scaler import StandardScalerWrapper
from src.preprocessing.components.feature_engineer import RatioFeature


class MemberBStrategy(BaseStrategy):

    def get_strategy_name(self) -> str:
        return "B: Median+NoOp+OneHot+Standard"

    def preprocess(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        claim_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        # TODO: 팀원 B가 구현
        raise NotImplementedError("팀원 B가 preprocess()를 구현해야 합니다.")
