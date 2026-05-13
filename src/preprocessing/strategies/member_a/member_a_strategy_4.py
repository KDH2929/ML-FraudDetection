"""
Member A 전처리 v4 — v3 위에 논문(paper) 계열 파생을 얹은 확장.

- **v3 동일**: `member_a_strategy_3`의 ma3 도메인 집계·고객 파생·SIU 노출 규칙( CSV 경로에서는 SIU 끔 ).
- **논문 참조(복제 아님, paper strategy 모듈에서 블록만 호출)**:
  - `book_dupreez_paper_strategy._dupreez_claim_insight_features` — du Preez et al. (2025) 문헌고찰식 CLAIM-only `drz_*`.
  - `book_bauder_paper_strategy._bauder_within_spec_peer_z` — Bauder류 동일 과목 내 Z-score `bdr_*` (라벨 미사용).
  - `book_survey_paper_strategy`: `_paper_survey_block` + `BookSurveyPaperStrategy` 와 동일한 `paper_meta`·`peer_frame`·
    `GroupMeanImputer` 이후 `_paper_survey_customer_addons` (`paper_prem_incm_ratio`, `paper_incm_missing_or_zero`).

튜닝: `python -m src.optimization.hyperparameter_tuner --strategy member_a_strategy_4`

실험 CLI ID: member_a_strategy_4
"""

from __future__ import annotations

import pandas as pd

from src.config import CAT_COLS, DIVIDED_SET_COL, ID_COL, NUM_COLS
from src.preprocessing.components.encoder import TargetEncoder
from src.preprocessing.components.missing_value import GroupMeanImputer, MedianImputer, ModeImputer
from src.preprocessing.components.outlier import IQRCapper
from src.preprocessing.components.scaler import RobustScalerWrapper
from src.preprocessing.strategies.base_strategy import BaseStrategy
from src.preprocessing.strategies.book.book_bauder_paper_strategy import _bauder_within_spec_peer_z
from src.preprocessing.strategies.book.book_dupreez_paper_strategy import _dupreez_claim_insight_features
from src.preprocessing.strategies.book.book_survey_paper_strategy import (
    _paper_survey_block,
    _paper_survey_customer_addons,
    _paper_survey_paper_meta_from_X,
    _paper_survey_peer_for_block,
)
from .member_a_strategy import (
    _drop_unused_cust_columns,
    _fill_residual_missing_for_ml,
    _merge_claim_features,
    _y_to_float01,
)
from .member_a_strategy_3 import (
    _ma3_claim_domain_block,
    _ma3_customer_domain_features,
    _train_customer_ids,
    _y_binary_int,
)


def _merge_paper_blocks(
    X: pd.DataFrame,
    claim_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(dupreez, bauder, survey) 고객×ID 병합용 테이블. 비어 있으면 빈 프레임."""
    drz = _dupreez_claim_insight_features(claim_df) if claim_df is not None else pd.DataFrame()

    if claim_df is None or claim_df.empty:
        return drz, pd.DataFrame(), pd.DataFrame()

    meta_b = X[[ID_COL]].copy()
    if DIVIDED_SET_COL in X.columns:
        meta_b[DIVIDED_SET_COL] = pd.to_numeric(X[DIVIDED_SET_COL], errors="coerce")
    else:
        meta_b[DIVIDED_SET_COL] = 1
    bdr = _bauder_within_spec_peer_z(claim_df, meta_b)

    paper_meta = _paper_survey_paper_meta_from_X(X)
    peer_frame = _paper_survey_peer_for_block(X)
    surv = _paper_survey_block(claim_df, paper_meta, peer_frame)

    return drz, bdr, surv


class MemberA4Strategy(BaseStrategy):

    def get_strategy_name(self) -> str:
        return "A4: v3 + 논문 파생(dupreez drz / bauder bdr / survey paper) — 튜닝은 src.optimization.hyperparameter_tuner"

    def preprocess(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        claim_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        train_ids = _train_customer_ids(X)
        y_bin = _y_binary_int(y)

        X_out = _merge_claim_features(X, claim_df)
        X_out = _drop_unused_cust_columns(X_out)

        extra = _ma3_claim_domain_block(
            claim_df, X[ID_COL], y_bin, train_ids, include_label_dependent_exposure=False
        )
        if not extra.empty:
            X_out = X_out.merge(extra, on=ID_COL, how="left")

        X_out = _ma3_customer_domain_features(X_out)

        drz, bdr, surv = _merge_paper_blocks(X, claim_df)
        for block in (drz, bdr, surv):
            if block is not None and not block.empty:
                X_out = X_out.merge(block, on=ID_COL, how="left")
        X_out = X_out.loc[:, ~X_out.columns.duplicated(keep="first")]

        if "RESI_COST" in X_out.columns:
            X_out["RESI_COST"] = X_out["RESI_COST"].replace(0, pd.NA)

        for c in NUM_COLS:
            if c in X_out.columns:
                X_out[c] = pd.to_numeric(X_out[c], errors="coerce")

        if all(c in X_out.columns for c in ("OCCP_GRP_1", "AGE", "CUST_INCM")):
            gim = GroupMeanImputer()
            gim.fit(X_out, y)
            X_out = gim.transform(X_out)

        addons = _paper_survey_customer_addons(X_out)
        if not addons.empty:
            ac = [c for c in addons.columns if c != ID_COL]
            if ac:
                X_out = X_out.merge(addons[[ID_COL] + ac], on=ID_COL, how="left")

        num_cols = [
            c
            for c in X_out.columns
            if c != ID_COL and pd.api.types.is_numeric_dtype(X_out[c])
        ]
        if num_cols:
            med = MedianImputer(cols=num_cols)
            med.fit(X_out, y)
            X_out = med.transform(X_out)
            X_out[num_cols] = X_out[num_cols].apply(pd.to_numeric, errors="coerce")
            med2 = X_out[num_cols].median()
            X_out[num_cols] = X_out[num_cols].fillna(med2).fillna(0.0).astype("float64")

        cat_cols = [c for c in CAT_COLS if c in X_out.columns]
        if cat_cols:
            mode_imp = ModeImputer(cols=cat_cols)
            mode_imp.fit(X_out, y)
            X_out = mode_imp.transform(X_out)

        if cat_cols and y is not None and len(y) == len(X_out):
            y01 = _y_to_float01(y).reindex(X_out.index)
            te = TargetEncoder(cols=cat_cols)
            te.fit(X_out, y01)
            X_out = te.transform(X_out)

        iqr_cols = [c for c in NUM_COLS if c in X_out.columns]
        if iqr_cols:
            cap = IQRCapper(cols=iqr_cols)
            cap.fit(X_out, y)
            X_out = cap.transform(X_out)

        scale_cols = [c for c in NUM_COLS if c in X_out.columns]
        if scale_cols:
            sc = RobustScalerWrapper(cols=scale_cols)
            sc.fit(X_out, y)
            X_out = sc.transform(X_out)

        X_out = _fill_residual_missing_for_ml(X_out)
        return X_out

    def preprocess_train_test(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        claim_df: pd.DataFrame = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_ids = frozenset(
            pd.to_numeric(X_train[ID_COL], errors="coerce").dropna().astype(int).unique().tolist()
        )

        def _path(df: pd.DataFrame, y_block: pd.Series) -> pd.DataFrame:
            z = _merge_claim_features(df, claim_df)
            z = _drop_unused_cust_columns(z)
            extra = _ma3_claim_domain_block(
                claim_df, df[ID_COL], y_block, train_ids, include_label_dependent_exposure=True
            )
            if not extra.empty:
                z = z.merge(extra, on=ID_COL, how="left")
            z = _ma3_customer_domain_features(z)
            drz, bdr, surv = _merge_paper_blocks(df, claim_df)
            for block in (drz, bdr, surv):
                if block is not None and not block.empty:
                    z = z.merge(block, on=ID_COL, how="left")
            z = z.loc[:, ~z.columns.duplicated(keep="first")]
            if "RESI_COST" in z.columns:
                z["RESI_COST"] = z["RESI_COST"].replace(0, pd.NA)
            for c in NUM_COLS:
                if c in z.columns:
                    z[c] = pd.to_numeric(z[c], errors="coerce")
            return z

        tr = _path(X_train, _y_binary_int(y_train))
        te = _path(X_test, pd.Series(0, index=X_test.index, dtype=int))

        if all(c in tr.columns for c in ("OCCP_GRP_1", "AGE", "CUST_INCM")):
            gim = GroupMeanImputer()
            gim.fit(tr, y_train)
            tr = gim.transform(tr)
            te = gim.transform(te)

        for z_name, zdf in (("tr", tr), ("te", te)):
            addons = _paper_survey_customer_addons(zdf)
            if not addons.empty:
                ac = [c for c in addons.columns if c != ID_COL]
                if ac:
                    merged = zdf.merge(addons[[ID_COL] + ac], on=ID_COL, how="left")
                    if z_name == "tr":
                        tr = merged
                    else:
                        te = merged

        num_cols = [
            c
            for c in tr.columns
            if c != ID_COL and pd.api.types.is_numeric_dtype(tr[c])
        ]
        if num_cols:
            med = MedianImputer(cols=num_cols)
            med.fit(tr, y_train)
            tr = med.transform(tr)
            te = med.transform(te)
            tr[num_cols] = tr[num_cols].apply(pd.to_numeric, errors="coerce")
            med2 = tr[num_cols].median()
            tr[num_cols] = tr[num_cols].fillna(med2).fillna(0.0).astype("float64")
            te[num_cols] = te[num_cols].apply(pd.to_numeric, errors="coerce")
            te[num_cols] = te[num_cols].fillna(med2).fillna(0.0).astype("float64")

        cat_cols = [c for c in CAT_COLS if c in tr.columns]
        if cat_cols:
            mode_imp = ModeImputer(cols=cat_cols)
            mode_imp.fit(tr, y_train)
            tr = mode_imp.transform(tr)
            te = mode_imp.transform(te)

        if cat_cols and len(y_train) == len(tr):
            y01 = _y_to_float01(y_train).reindex(tr.index)
            te_enc = TargetEncoder(cols=cat_cols)
            te_enc.fit(tr, y01)
            tr = te_enc.transform(tr)
            te = te_enc.transform(te)

        iqr_cols = [c for c in NUM_COLS if c in tr.columns]
        if iqr_cols:
            cap = IQRCapper(cols=iqr_cols)
            cap.fit(tr, y_train)
            tr = cap.transform(tr)
            te = cap.transform(te)

        scale_cols = [c for c in NUM_COLS if c in tr.columns]
        if scale_cols:
            sc = RobustScalerWrapper(cols=scale_cols)
            sc.fit(tr, y_train)
            tr = sc.transform(tr)
            te = sc.transform(te)

        te = te.reindex(columns=tr.columns, fill_value=0.0)
        tr = _fill_residual_missing_for_ml(tr)
        te = _fill_residual_missing_for_ml(te)
        return tr, te
