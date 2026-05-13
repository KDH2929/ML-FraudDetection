"""
member A·B·C **최종 계열** 전처리를 한 번에 돌린 뒤, 특징 열만 접두어로 붙여 **가로 병합**한 실험용 전략.

- **A**: `MemberA4Strategy` → 열 접두어 ``ma4_``
- **B**: `MemberB3Strategy` → ``mb3_`` (청구 필수)
- **C**: `MemberCStrategy4` → ``mc4_``

``ID_COL`` 과 (있으면) ``DIVIDED_SET_COL`` 은 A4 결과 기준으로 **한 번만** 두고,
나머지는 전부 접두어 붙여 세 블록을 ``concat`` 한다. 인덱스·행 순서는 A4와 동일해야 한다.

주의: 열 수·다중공선이 매우 커지고 학습이 무거워질 수 있다. 비교·실험용.

실험 CLI ID: test_member_abc

LGBM 임계값: ``python -m src.optimization.threshold_optimizer --strategy test_member_abc`` 실행 후
``artifacts/test_member_abc_threshold_analysis.json`` 이 있으면 실험 시 자동 반영된다 (또는 ``STRATEGY_THRESHOLDS``).
"""

from __future__ import annotations

import pandas as pd

from src.config import DIVIDED_SET_COL, ID_COL
from src.preprocessing.strategies.base_strategy import BaseStrategy
from src.preprocessing.strategies.member_a_strategy_4 import MemberA4Strategy
from src.preprocessing.strategies.member_b_strategy_3 import MemberB3Strategy
from src.preprocessing.strategies.member_c_strategy_4 import MemberCStrategy4


def _feature_block_prefixed(
    df: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    """ID·DIVIDED_SET 제외 열에만 prefix."""
    drop = [c for c in (ID_COL, DIVIDED_SET_COL) if c in df.columns]
    part = df.drop(columns=drop, errors="ignore")
    return part.add_prefix(prefix)


class TestMemberAbcStrategy(BaseStrategy):
    """A4 + B3 + C4 산출 특징을 접두어 병합."""

    def __init__(self) -> None:
        self._a = MemberA4Strategy()
        self._b = MemberB3Strategy()
        self._c = MemberCStrategy4()

    def get_strategy_name(self) -> str:
        return "test_member_abc: A4+B3+C4 특징 접두어 병합 (실험용)"

    def preprocess(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        claim_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        if claim_df is None or len(claim_df) == 0:
            raise ValueError("test_member_abc 전략은 B3 의존으로 claim_df 가 필요합니다.")

        a = self._a.preprocess(X, y, claim_df=claim_df)
        b = self._b.preprocess(X, y, claim_df=claim_df)
        c = self._c.preprocess(X, y, claim_df=claim_df)

        if len(a) != len(X) or len(b) != len(X) or len(c) != len(X):
            raise ValueError(
                f"행 길이 불일치: X={len(X)}, A={len(a)}, B={len(b)}, C={len(c)}. "
                "세 전략이 동일 행 수를 유지해야 병합 가능합니다."
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
