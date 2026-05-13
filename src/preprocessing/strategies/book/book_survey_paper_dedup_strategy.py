"""
book_survey_paper 와 동일한 파이프라인 후, 다중공선성 진단에 따른 논문 파생(paper_*) 축소.

제거 원칙 (VIF·상관 기반):
- CLAIM_COUNT 와 사실상 동일: paper_claim_rows
- 청구(dmnd) 쪽은 지급(paym)과 초고상관 → dmnd_* 전부 제거, paym 은 합계만 유지
- paym 의 mean/max 는 sum 과 쌍둥이 → paper_paym_mean, paper_paym_max 제거
- paper_paym_dmnd_ratio 는 dmnd 의존·sum 들과 강결합 → 제거
- 기존 CLAIM 파생과 강결합: paper_vlid_sum (vlid_max 는 유지)
- tenure vs 첫청구 lag 고상관 → paper_first_claim_lag_m 제거, paper_tenure_m 유지
- 병원 nunique vs top share 음의 강상관 → paper_top_hosp_share 제거, paper_hosp_nunique 유지
- 최근6m 비율은 건수 파생과 고상관 → paper_recent6_vs_prior_ratio 제거 (건수 3개 유지)

실험 CLI ID: book_survey_paper_dedup
"""

from __future__ import annotations

import pandas as pd

from src.preprocessing.strategies.base_strategy import BaseStrategy
from .book_survey_paper_strategy import BookSurveyPaperStrategy

_PAPER_DROP_FOR_MULTICOLLINEARITY = frozenset(
    {
        "paper_claim_rows",
        "paper_dmnd_sum",
        "paper_dmnd_mean",
        "paper_dmnd_max",
        "paper_paym_mean",
        "paper_paym_max",
        "paper_paym_dmnd_ratio",
        "paper_vlid_sum",
        "paper_first_claim_lag_m",
        "paper_top_hosp_share",
        "paper_recent6_vs_prior_ratio",
    }
)


class BookSurveyPaperDedupStrategy(BaseStrategy):
    """book_survey_paper + 논문 파생 중복·초고상관 열 제거."""

    def __init__(self) -> None:
        self._survey = BookSurveyPaperStrategy()

    def get_strategy_name(self) -> str:
        return "6장 3절 + 논문 서베이 파생, 다중공선성 축소(book_survey_paper_dedup)"

    def preprocess(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        claim_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        out = self._survey.preprocess(X, y, claim_df=claim_df)
        drop = [c for c in _PAPER_DROP_FOR_MULTICOLLINEARITY if c in out.columns]
        if drop:
            out = out.drop(columns=drop, errors="ignore")
        return out
