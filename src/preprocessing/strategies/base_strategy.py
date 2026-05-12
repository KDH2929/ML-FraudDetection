from abc import ABC, abstractmethod
import pandas as pd


class BaseStrategy(ABC):

    @abstractmethod
    def preprocess(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        claim_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """전처리 수행 후 완성된 X DataFrame 반환 (y 포함 안 함)"""

    @abstractmethod
    def get_strategy_name(self) -> str:
        """전략 이름 반환 (로그/비교표 출력용)"""
