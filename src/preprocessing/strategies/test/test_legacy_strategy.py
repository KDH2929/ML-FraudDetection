"""
PDF 정렬 전의 book_pdf 전처리(legacy) 복구본.

현재 `book_pdf_strategy`(PDF 동일)와 비교용:
- RESI_TYPE_CODE / CTPR: fillna(20)·'경기' 고정 대신 ModeImputer
- CLAIM 파생: 원-핫 이후 병합이 아니라 그 이전에 merge
- 입원일수 열: HOSP_DAYS 대신 HOST_DAYS (VLID_HOSP_OTDA 평균)
- ACCI_RATIO: ACCI_RATIO_1~3 고정 대신 crosstab 기반 동적 열
- 사고×청구 조합: 책과 동일한 pivot 이름 대신 ACCI_DMND_* 슬러그 열
- 표준화: sklearn 직접 concat 대신 StandardScalerWrapper
- 초기 단계에서 MATE_OCCP_GRP_2·FP_CAREER·OCCP_GRP_2·신용한도 등을 한꺼번에 제거

실험 CLI ID: test_legacy (파일명 test_legacy_strategy.py)
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.config import ID_COL
from src.preprocessing.components.encoder import OneHotEncoder
from src.preprocessing.components.missing_value import GroupMeanImputer, ModeImputer
from src.preprocessing.components.scaler import StandardScalerWrapper
from src.preprocessing.strategies.base_strategy import BaseStrategy

_SCALE_COLS = [
    "RESI_COST",
    "TOTALPREM",
    "MAX_PRM",
    "CUST_INCM",
    "RCBASE_HSHD_INCM",
    "JPBASE_HSHD_INCM",
]

_CLAIM_DROP_EXPLICIT = [
    "SELF_CHAM",
    "NON_PAY",
    "TAMT_SFCA",
    "PATT_CHRG_TOTA",
    "DSCT_AMT",
    "COUNT_TRMT_ITEM",
    "DCAF_CMPS_XCPA",
    "HOSP_OTPA_STDT",
    "HOSP_OTPA_ENDT",
]

_OHE_COLS = [
    "SEX",
    "RESI_TYPE_CODE",
    "CTPR",
    "OCCP_GRP_1",
    "WEDD_YN",
    "MATE_OCCP_GRP_1",
]


def _strip_two_chars(val) -> object:
    if pd.isna(val):
        return val
    s = str(val).strip()
    if s.lower() in ("nan", "<na>", "none"):
        return np.nan
    return s[2:] if len(s) > 2 else s


def _y_binary(y: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(y):
        return pd.to_numeric(y, errors="coerce").fillna(0).astype(int).clip(0, 1)
    m = y.astype(str).str.upper().str.strip()
    return m.map({"Y": 1, "N": 0, "1": 1, "0": 0}).fillna(0).astype(int)


def _drop_cust_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    drop_names = []
    for c in out.columns:
        cu = c.upper()
        if "DATE" in cu or c.endswith("_YM") or c in ("CUST_RGST", "MAX_PAYM_YM"):
            drop_names.append(c)
    if drop_names:
        out = out.drop(columns=drop_names, errors="ignore")
    return out


def _slug_col_part(x) -> str:
    return re.sub(r"[^\w]+", "_", str(x))[:64]


def _legacy_book_claim_block(
    claim_df: pd.DataFrame,
    cust_ids: pd.Series,
    y_bin: pd.Series,
) -> pd.DataFrame:
    """PDF 정렬 전 book_pdf용 CLAIM 파생(동적 ACCI·슬러그 교차열·HOST_DAYS)."""
    if claim_df is None or len(claim_df) == 0:
        return pd.DataFrame()

    cl = claim_df.copy()
    cl = cl.drop(columns=[c for c in _CLAIM_DROP_EXPLICIT if c in cl.columns], errors="ignore")
    date_cols = [c for c in cl.columns if "DATE" in c.upper() or c.endswith("_STDT") or c.endswith("_ENDT")]
    cl = cl.drop(columns=date_cols, errors="ignore")

    if ID_COL not in cl.columns:
        return pd.DataFrame()

    cl[ID_COL] = pd.to_numeric(cl[ID_COL], errors="coerce").astype("Int64")
    cl = cl[cl[ID_COL].notna()].copy()
    cl[ID_COL] = cl[ID_COL].astype(int)

    siu_map = pd.Series(y_bin.values, index=pd.to_numeric(cust_ids, errors="coerce").values)
    siu_map = siu_map.groupby(level=0).first()
    cl["_siu"] = cl[ID_COL].map(siu_map).fillna(0).astype(int)

    base = cl.groupby(ID_COL, sort=False).size().to_frame("CLAIM_COUNT")

    if "VLID_HOSP_OTDA" in cl.columns:
        hd = cl.groupby(ID_COL, sort=False)["VLID_HOSP_OTDA"].mean()
        base["HOST_DAYS"] = hd

    if "HEED_HOSP_YN" in cl.columns:
        yn = cl["HEED_HOSP_YN"].astype(str).str.upper().str.strip()
        flag = (yn == "Y").astype(int)
        base["HEED_HOSP"] = flag.groupby(cl[ID_COL], sort=False).max()

    if "CHME_LICE_NO" in cl.columns:
        doc_tot = cl.groupby("CHME_LICE_NO", dropna=False).size()
        doc_f = cl.groupby("CHME_LICE_NO", dropna=False)["_siu"].sum()
        doc_ratio = (doc_f / doc_tot.replace(0, np.nan)).fillna(0.0)
        cl["_dr"] = cl["CHME_LICE_NO"].map(doc_ratio).fillna(0.0)
        base["DOC_SIU_RATIO"] = cl.groupby(ID_COL, sort=False)["_dr"].max()

    hosp_key = "HOSP_CODE" if "HOSP_CODE" in cl.columns else None
    if hosp_key:
        h_tot = cl.groupby(hosp_key, dropna=False).size()
        h_f = cl.groupby(hosp_key, dropna=False)["_siu"].sum()
        h_ratio = (h_f / h_tot.replace(0, np.nan)).fillna(0.0)
        cl["_hr"] = cl[hosp_key].map(h_ratio).fillna(0.0)
        base["HOSP_SIU_RATIO"] = cl.groupby(ID_COL, sort=False)["_hr"].max()

    if "DOC_SIU_RATIO" in base.columns and "HOSP_SIU_RATIO" in base.columns:
        base["HOSP_DOC_SIU"] = base["DOC_SIU_RATIO"] * base["HOSP_SIU_RATIO"]

    if "ACCI_DVSN" in cl.columns:
        acci_pct = pd.crosstab(cl[ID_COL], cl["ACCI_DVSN"], normalize="index")
        acci_pct = acci_pct.add_prefix("ACCI_RATIO_")
        base = base.join(acci_pct, how="left")

    if "ACCI_DVSN" in cl.columns and "DMND_RESN_CODE" in cl.columns:
        sub = cl[[ID_COL, "ACCI_DVSN", "DMND_RESN_CODE"]].dropna()
        if not sub.empty:
            ct = pd.crosstab(
                index=sub[ID_COL],
                columns=[sub["ACCI_DVSN"].astype(str), sub["DMND_RESN_CODE"].astype(str)],
                normalize="index",
            )
            ct.columns = [f"ACCI_DMND_{_slug_col_part(a)}__{_slug_col_part(b)}" for a, b in ct.columns]
            base = base.join(ct, how="left")

    out = base.reset_index().fillna(0.0)
    return out


class TestLegacyStrategy(BaseStrategy):

    def get_strategy_name(self) -> str:
        return "6장 book_pdf — PDF 정렬 이전(legacy) 전처리"

    def preprocess(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        claim_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        y_bin = _y_binary(y)
        X_out = X.copy()

        X_out = _drop_cust_dates(X_out)

        for c in ("MATE_OCCP_GRP_2", "FP_CAREER", "OCCP_GRP_2", "MINCRDT", "MAXCRDT"):
            if c in X_out.columns:
                X_out = X_out.drop(columns=[c])

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

        if all(c in X_out.columns for c in ("OCCP_GRP_1", "AGE", "CUST_INCM")):
            gim = GroupMeanImputer()
            gim.fit(X_out, y)
            X_out = gim.transform(X_out)

        if "OCCP_GRP_1" in X_out.columns and "JPBASE_HSHD_INCM" in X_out.columns:
            gj = GroupMeanImputer(group_cols=["OCCP_GRP_1"], target_col="JPBASE_HSHD_INCM")
            gj.fit(X_out, y)
            X_out = gj.transform(X_out)

        if "RESI_TYPE_CODE" in X_out.columns and "CTPR" in X_out.columns:
            mi = ModeImputer(cols=["RESI_TYPE_CODE", "CTPR"])
            mi.fit(X_out, y)
            X_out = mi.transform(X_out)

        if "WEDD_YN" in X_out.columns:
            X_out["WEDD_YN"] = X_out["WEDD_YN"].fillna("N")
        for c in ("LTBN_CHLD_AGE", "CHLD_CNT", "TOTALPREM", "MAX_PRM"):
            if c in X_out.columns:
                X_out[c] = pd.to_numeric(X_out[c], errors="coerce").fillna(0.0)

        agg = _legacy_book_claim_block(claim_df, X_out[ID_COL], y_bin)
        if not agg.empty:
            X_out = X_out.merge(agg, on=ID_COL, how="left")
            new_cols = [c for c in agg.columns if c != ID_COL]
            if new_cols:
                X_out[new_cols] = X_out[new_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        ohe_use = [c for c in _OHE_COLS if c in X_out.columns]
        for c in ohe_use:
            X_out[c] = X_out[c].astype("string").fillna("unknown").replace({"<NA>": "unknown"}).astype(str)

        if ohe_use:
            ohe = OneHotEncoder(cols=ohe_use)
            ohe.fit(X_out, y)
            X_out = ohe.transform(X_out)

        scale_present = [c for c in _SCALE_COLS if c in X_out.columns]
        if scale_present:
            scaler = StandardScalerWrapper(cols=scale_present)
            scaler.fit(X_out, y)
            X_out = scaler.transform(X_out)

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
