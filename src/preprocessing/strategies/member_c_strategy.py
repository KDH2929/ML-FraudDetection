import pandas as pd
from src.preprocessing.strategies.base_strategy import BaseStrategy
from src.preprocessing.components.missing_value import ModeImputer
from src.preprocessing.components.outlier import QuantileCapper
from src.preprocessing.components.encoder import LabelEncoder
from src.preprocessing.components.scaler import MinMaxScalerWrapper
from src.preprocessing.components.feature_engineer import HospFeature


class MemberCStrategy(BaseStrategy):

    def get_strategy_name(self) -> str:
        return "C: Mode+Quantile+Label+MinMax"

    def preprocess(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        claim_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        # TODO: 팀원 C가 구현
        raise NotImplementedError("팀원 C가 preprocess()를 구현해야 합니다.")
