"""
Combine the final A/B/C feature blocks with stable prefixes.

- A: `MemberA4Strategy` with `ma4_`
- B: `MemberB3Strategy` with `mb3_`
- C: `MemberCStrategy4` with `mc4_`
"""

from __future__ import annotations

import pandas as pd

from src.config import DIVIDED_SET_COL, ID_COL
from src.preprocessing.strategies.base_strategy import BaseStrategy
from src.preprocessing.strategies.member_a.member_a_strategy_4 import MemberA4Strategy
from src.preprocessing.strategies.member_b.member_b_strategy_3 import MemberB3Strategy
from src.preprocessing.strategies.member_c.member_c_strategy_4 import MemberCStrategy4


def _feature_block_prefixed(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    drop = [c for c in (ID_COL, DIVIDED_SET_COL) if c in df.columns]
    part = df.drop(columns=drop, errors="ignore")
    return part.add_prefix(prefix)


class TestMemberAbcStrategy(BaseStrategy):
    """Merge A4, B3, and C4 feature blocks into one strategy output."""

    def __init__(self) -> None:
        self._a = MemberA4Strategy()
        self._b = MemberB3Strategy()
        self._c = MemberCStrategy4()

    def get_strategy_name(self) -> str:
        return "test_member_abc: prefixed A4+B3+C4 feature merge"

    def preprocess(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        claim_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        if claim_df is None or len(claim_df) == 0:
            raise ValueError("test_member_abc requires claim_df because B3 features depend on claims.")

        a = self._a.preprocess(X, y, claim_df=claim_df)
        b = self._b.preprocess(X, y, claim_df=claim_df)
        c = self._c.preprocess(X, y, claim_df=claim_df)

        if len(a) != len(X) or len(b) != len(X) or len(c) != len(X):
            raise ValueError(
                f"Length mismatch detected: X={len(X)}, A={len(a)}, B={len(b)}, C={len(c)}. "
                "All strategy outputs must stay row-aligned before concatenation."
            )

        base = pd.DataFrame({ID_COL: a[ID_COL].values}, index=a.index)
        if DIVIDED_SET_COL in a.columns:
            base[DIVIDED_SET_COL] = a[DIVIDED_SET_COL].values

        ba = _feature_block_prefixed(a, "ma4_")
        bb = _feature_block_prefixed(b, "mb3_")
        bc = _feature_block_prefixed(c, "mc4_")

        out = pd.concat([base, ba, bb, bc], axis=1)
        out = out.loc[:, ~out.columns.duplicated(keep="first")]
        return out
