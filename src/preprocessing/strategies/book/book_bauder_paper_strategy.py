"""
6장 book_pdf 파이프라인 동일 + 원 논문식 블록 1개.

원 논문(예: Bauder & Khoshgoftaar, J Big Data 2018; 관련 Medicare 사기·이상치 연구)에서
반복되는 아이디어 중 하나는 **동일 전문과(또는 동일 집단) 안에서의 상대적 이탈**을 보는 것이다.
데이터에 맞게 `HOSP_SPEC_DVSN`(병원 과목) 최빈값으로 고객을 집단에 넣고,
DIVIDED_SET==1 구간에서 동일 집단 내 **지급액 합·청구 건수·병원 수**의 평균·표준편차를 구한 뒤
고객 단위 Z-score(`bdr_*`)로 붙인다. (라벨은 동료 통계에 사용하지 않는다.)

실험 CLI ID: book_bauder_paper
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import DIVIDED_SET_COL, ID_COL
from src.preprocessing.components.encoder import OneHotEncoder
from src.preprocessing.components.missing_value import GroupMeanImputer
from src.preprocessing.strategies.base_strategy import BaseStrategy
from .book_pdf_strategy import (
    _OHE_COLS,
    _SCALE_COLS,
    _book_claim_block,
    _drop_cust_dates,
    _strip_two_chars,
    _y_binary,
)


def _modal_value(s: pd.Series) -> object:
    s = s.dropna()
    if len(s) == 0:
        return np.nan
    m = s.mode()
    return m.iloc[0] if len(m) else np.nan


def _bauder_within_spec_peer_z(
    claim_df: pd.DataFrame | None,
    meta: pd.DataFrame,
) -> pd.DataFrame:
    """
    claim_df: 원본 CLAIM (날짜 컬럼 있어도 됨).
    meta: ID_COL, DIVIDED_SET_COL 만 포함. 행 순서는 X와 동일해야 함(길이 동일).
    """
    if claim_df is None or len(claim_df) == 0 or ID_COL not in claim_df.columns:
        return pd.DataFrame()
    if "HOSP_SPEC_DVSN" not in claim_df.columns:
        return pd.DataFrame()
    if DIVIDED_SET_COL not in meta.columns or ID_COL not in meta.columns:
        return pd.DataFrame()

    cl = claim_df.copy()
    cl[ID_COL] = pd.to_numeric(cl[ID_COL], errors="coerce").astype("Int64")
    cl = cl[cl[ID_COL].notna()].copy()
    if cl.empty:
        return pd.DataFrame()
    cl[ID_COL] = cl[ID_COL].astype(int)

    spec = cl["HOSP_SPEC_DVSN"]
    cl["_spec"] = spec

    g = cl.groupby(ID_COL, sort=False)
    base = g.size().to_frame("_n_claim")
    if "PAYM_AMT" in cl.columns:
        cl["_paym"] = pd.to_numeric(cl["PAYM_AMT"], errors="coerce").fillna(0.0)
        base["_paym_sum"] = cl.groupby(ID_COL, sort=False)["_paym"].sum()
    else:
        base["_paym_sum"] = 0.0

    if "HOSP_CODE" in cl.columns:
        base["_n_hosp"] = g["HOSP_CODE"].nunique()
    else:
        base["_n_hosp"] = 0

    base["_spec_mode"] = g["_spec"].apply(_modal_value)
    cust = base.reset_index()

    m = meta[[ID_COL, DIVIDED_SET_COL]].copy()
    m[ID_COL] = pd.to_numeric(m[ID_COL], errors="coerce").astype(int)
    m[DIVIDED_SET_COL] = pd.to_numeric(m[DIVIDED_SET_COL], errors="coerce")

    merged = cust.merge(m, on=ID_COL, how="left")
    tr_mask = merged[DIVIDED_SET_COL] == 1
    tr = merged.loc[tr_mask & merged["_spec_mode"].notna()].copy()
    if tr.empty:
        out = cust[[ID_COL]].copy()
        out["bdr_paym_sum_z_spec"] = 0.0
        out["bdr_claim_count_z_spec"] = 0.0
        out["bdr_n_hosp_z_spec"] = 0.0
        return out

    def _z_cols(group_col: str, value_cols: list[str], prefix: str) -> pd.DataFrame:
        stats = tr.groupby(group_col, observed=False)[value_cols].agg(["mean", "std"])
        stats.columns = [f"{a}_{b}" for a, b in stats.columns]
        stats = stats.reset_index()
        wide = merged.merge(stats, on=group_col, how="left")
        out_z = pd.DataFrame({ID_COL: wide[ID_COL]})
        for vc in value_cols:
            mu = wide[f"{vc}_mean"]
            sd = wide[f"{vc}_std"].replace(0, np.nan)
            out_z[f"{prefix}{vc}_z"] = ((wide[vc] - mu) / (sd + 1e-6)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out_z

    zpay = _z_cols("_spec_mode", ["_paym_sum"], "bdr_")
    zpay = zpay.rename(columns={"bdr__paym_sum_z": "bdr_paym_sum_z_spec"})

    zct = _z_cols("_spec_mode", ["_n_claim"], "bdr_")
    zct = zct.rename(columns={"bdr__n_claim_z": "bdr_claim_count_z_spec"})

    zh = _z_cols("_spec_mode", ["_n_hosp"], "bdr_")
    zh = zh.rename(columns={"bdr__n_hosp_z": "bdr_n_hosp_z_spec"})

    out = cust[[ID_COL]].merge(zpay, on=ID_COL, how="left").merge(zct, on=ID_COL, how="left").merge(zh, on=ID_COL, how="left")
    for c in ("bdr_paym_sum_z_spec", "bdr_claim_count_z_spec", "bdr_n_hosp_z_spec"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
        else:
            out[c] = 0.0
    return out


class BookBauderPaperStrategy(BaseStrategy):

    def get_strategy_name(self) -> str:
        return "6장 3절 PDF 동일 + Bauder류 동일과목(HOSP_SPEC) 동료 Z-score bdr_*"

    def preprocess(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        claim_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        y_bin = _y_binary(y)
        X_out = X.copy()

        meta_peer = X_out[[ID_COL, DIVIDED_SET_COL]].copy() if DIVIDED_SET_COL in X_out.columns else None
        if meta_peer is not None:
            meta_peer = meta_peer.drop_duplicates(subset=[ID_COL], keep="first")

        X_out = _drop_cust_dates(X_out)

        if meta_peer is None and DIVIDED_SET_COL in X_out.columns:
            meta_peer = X_out[[ID_COL, DIVIDED_SET_COL]].copy().drop_duplicates(subset=[ID_COL], keep="first")

        if "OCCP_GRP_2" in X_out.columns:
            X_out = X_out.drop(columns=["OCCP_GRP_2"])
        if "MATE_OCCP_GRP_2" in X_out.columns:
            X_out = X_out.drop(columns=["MATE_OCCP_GRP_2"])

        if "OCCP_GRP_1" in X_out.columns:
            X_out["OCCP_GRP_1"] = X_out["OCCP_GRP_1"].map(_strip_two_chars)
        if "MATE_OCCP_GRP_1" in X_out.columns:
            X_out["MATE_OCCP_GRP_1"] = X_out["MATE_OCCP_GRP_1"].map(_strip_two_chars)

        if "AGE" in X_out.columns:
            X_out["AGE"] = X_out["AGE"].map(lambda x: int(pd.to_numeric(x, errors="coerce") or 0) // 10)
        if "LTBN_CHLD_AGE" in X_out.columns:
            X_out["LTBN_CHLD_AGE"] = X_out["LTBN_CHLD_AGE"].map(
                lambda x: (pd.to_numeric(x, errors="coerce") or 0) // 10 if pd.notna(x) else np.nan
            )

        for c in ("MINCRDT", "MAXCRDT"):
            if c in X_out.columns:
                X_out = X_out.drop(columns=[c])

        if all(c in X_out.columns for c in ("OCCP_GRP_1", "AGE", "CUST_INCM")):
            gim = GroupMeanImputer(group_cols=("AGE", "OCCP_GRP_1"), target_col="CUST_INCM")
            gim.fit(X_out, y)
            X_out = gim.transform(X_out)

        if "OCCP_GRP_1" in X_out.columns and "JPBASE_HSHD_INCM" in X_out.columns:
            gj = GroupMeanImputer(group_cols=["OCCP_GRP_1"], target_col="JPBASE_HSHD_INCM")
            gj.fit(X_out, y)
            X_out = gj.transform(X_out)

        if "RESI_TYPE_CODE" in X_out.columns:
            X_out["RESI_TYPE_CODE"] = X_out["RESI_TYPE_CODE"].fillna(20)
        if "CTPR" in X_out.columns:
            X_out["CTPR"] = X_out["CTPR"].fillna("경기")
        if "WEDD_YN" in X_out.columns:
            X_out["WEDD_YN"] = X_out["WEDD_YN"].fillna("N")
        for c in ("LTBN_CHLD_AGE", "CHLD_CNT", "TOTALPREM", "MAX_PRM"):
            if c in X_out.columns:
                X_out[c] = pd.to_numeric(X_out[c], errors="coerce").fillna(0.0)

        if "FP_CAREER" in X_out.columns:
            X_out = X_out.drop(columns=["FP_CAREER"])

        ohe_use = [c for c in _OHE_COLS if c in X_out.columns]
        for c in ohe_use:
            X_out[c] = X_out[c].astype("string").fillna("unknown").replace({"<NA>": "unknown"}).astype(str)

        if ohe_use:
            ohe = OneHotEncoder(cols=ohe_use)
            ohe.fit(X_out, y)
            X_out = ohe.transform(X_out)

        agg = _book_claim_block(claim_df, X_out[ID_COL], y_bin)
        if not agg.empty:
            X_out = X_out.merge(agg, on=ID_COL, how="left")
            new_cols = [c for c in agg.columns if c != ID_COL]
            if new_cols:
                X_out[new_cols] = X_out[new_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        if meta_peer is not None:
            bdr = _bauder_within_spec_peer_z(claim_df, meta_peer)
            if not bdr.empty:
                X_out = X_out.merge(bdr, on=ID_COL, how="left")
                for c in ("bdr_paym_sum_z_spec", "bdr_claim_count_z_spec", "bdr_n_hosp_z_spec"):
                    if c in X_out.columns:
                        X_out[c] = pd.to_numeric(X_out[c], errors="coerce").fillna(0.0)

        scale_present = [c for c in _SCALE_COLS if c in X_out.columns]
        if scale_present:
            scaler = StandardScaler()
            mat = scaler.fit_transform(X_out[scale_present].astype(float))
            X_out = X_out.drop(columns=scale_present)
            X_out = pd.concat([X_out, pd.DataFrame(mat, columns=scale_present, index=X_out.index)], axis=1)

        num_cols = [c for c in X_out.columns if c != ID_COL]
        if num_cols:
            med = X_out[num_cols].median(numeric_only=True)
            X_out[num_cols] = X_out[num_cols].apply(pd.to_numeric, errors="coerce")
            X_out[num_cols] = X_out[num_cols].fillna(med).fillna(0.0)

        for c in X_out.columns:
            if c == ID_COL:
                continue
            X_out[c] = np.asarray(X_out[c], dtype=np.float64)

        return X_out
