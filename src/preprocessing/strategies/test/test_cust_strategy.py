"""
교재 스펙에 가깝게: CUST 전처리(연령대, 드롭, 결측, 원-핫, 표준화) +
CLAIM 파생(입원일수, 유의병원, 청구수, 의사/병원 SIU 비율, 교차 비율 등).

`book_pdf_strategy`(PDF 6장 3절 그대로)와 CLAIM·CUST 세부가 다른 실험용 변형이다.

실험 CLI ID: test_cust (`test_cust_strategy.py`)
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.config import DIVIDED_SET_COL, ID_COL
from src.preprocessing.components.encoder import OneHotEncoder
from src.preprocessing.components.missing_value import GroupMeanImputer, MedianImputer, ModeImputer
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

_CUST_DATE_DROP_PATTERNS = (
    re.compile(r"DATE", re.I),
    re.compile(r"_YM$"),
    re.compile(r"^CUST_RGST$"),
)

_AGE_BINS = [-1, 19, 29, 39, 49, 59, 69, 150]
_AGE_LABELS = ["<=19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]


def _y_fraud_series(y: pd.Series) -> pd.Series:
    if y is None:
        return pd.Series(dtype=float)
    if pd.api.types.is_numeric_dtype(y):
        return pd.to_numeric(y, errors="coerce").fillna(0.0).clip(0, 1)
    m = y.astype(str).str.upper().str.strip()
    return m.map({"Y": 1.0, "N": 0.0, "1": 1.0, "0": 0.0}).fillna(0.0)


def _drop_cust_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    drop_list = []
    for c in out.columns:
        if any(p.search(c) for p in _CUST_DATE_DROP_PATTERNS):
            drop_list.append(c)
    if drop_list:
        out = out.drop(columns=drop_list, errors="ignore")
    return out


def _age_to_band(s: pd.Series, prefix: str) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce").fillna(0)
    band = pd.cut(v, bins=_AGE_BINS, labels=_AGE_LABELS, right=True)
    return prefix + band.astype(str).replace("nan", "unknown")


def _jpbase_by_occupation(X: pd.DataFrame) -> pd.DataFrame:
    if "OCCP_GRP_1" not in X.columns or "JPBASE_HSHD_INCM" not in X.columns:
        return X
    out = X.copy()
    gm = GroupMeanImputer(group_cols=["OCCP_GRP_1"], target_col="JPBASE_HSHD_INCM")
    gm.fit(out, None)
    return gm.transform(out)


def _train_customer_ids(X: pd.DataFrame) -> frozenset | None:
    """데이터 사전: 학습용=1, 평가용=2. SIU 비율 집계에는 1번 고객만 사용."""
    if DIVIDED_SET_COL not in X.columns:
        return None
    d = pd.to_numeric(X[DIVIDED_SET_COL], errors="coerce")
    mask = d == 1
    if not mask.any():
        return frozenset()
    ids = X.loc[mask, ID_COL]
    return frozenset(pd.to_numeric(ids, errors="coerce").dropna().astype(int).unique())


def _clean_claim_for_agg(claim_df: pd.DataFrame) -> pd.DataFrame:
    """DATE 계열·결측 과다 컬럼 제거 후, 집계에 필요한 컬럼만 유지."""
    if claim_df is None or claim_df.empty:
        return pd.DataFrame()
    df = claim_df.copy()
    miss = df.isna().mean()
    high_miss = miss[miss > 0.5].index.tolist()
    df = df.drop(columns=high_miss, errors="ignore")
    date_cols = [c for c in df.columns if "DATE" in c.upper() or c.endswith("_STDT") or c.endswith("_ENDT")]
    df = df.drop(columns=date_cols, errors="ignore")
    return df


def _claim_book_features(
    claim_df: pd.DataFrame,
    fraud_by_cust: pd.Series,
    train_cust_ids: frozenset | None,
) -> pd.DataFrame:
    """고객 단위 CLAIM 파생변수.

    DOC_SIU_RATIO / HOSP_SIU_RATIO / HOSP_DOC_SIU 는 **DIVIDED_SET==1(학습용) 고객의 청구만**으로
    의사·병원별 사기 비율을 추정한 뒤, 전체 고객의 방문 의사·병원에 그 비율을 매핑한다.
    """
    cl = _clean_claim_for_agg(claim_df)
    if cl.empty or ID_COL not in cl.columns:
        return pd.DataFrame()

    cl[ID_COL] = pd.to_numeric(cl[ID_COL], errors="coerce").astype("Int64")

    base = cl.groupby(ID_COL, sort=False).size().to_frame("CLAIM_COUNT")

    hosp_days_col = None
    for c in ("HOST_DAYS", "VLID_HOSP_OTDA", "HSPZ_DAYS"):
        if c in cl.columns:
            hosp_days_col = c
            break
    if hosp_days_col:
        base["HOST_DAYS"] = cl.groupby(ID_COL, sort=False)[hosp_days_col].mean()

    heed_col = None
    if "HEED_HOSP_YN" in cl.columns:
        heed_col = "HEED_HOSP_YN"
    elif "SUSPCT_HOSP_YN" in cl.columns:
        heed_col = "SUSPCT_HOSP_YN"
    if heed_col is not None:
        yn = cl[heed_col].astype(str).str.upper().str.strip()
        flag = (yn == "Y").astype(int)
        base["HEED_HOSP"] = flag.groupby(cl[ID_COL], sort=False).max()

    cl = cl.copy()
    if train_cust_ids is None:
        cl_siu = cl
    else:
        cl_siu = cl[cl[ID_COL].isin(list(train_cust_ids))].copy()

    doc_ratio = pd.Series(dtype=np.float64)
    h_ratio = pd.Series(dtype=np.float64)
    if len(cl_siu) > 0:
        cl_siu = cl_siu.copy()
        cl_siu["_fraud"] = (
            pd.to_numeric(cl_siu[ID_COL], errors="coerce")
            .map(fraud_by_cust)
            .fillna(0.0)
            .astype(float)
            .clip(0, 1)
        )
        if "CHME_LICE_NO" in cl_siu.columns:
            doc_tot = cl_siu.groupby("CHME_LICE_NO", dropna=False).size()
            doc_f = cl_siu.groupby("CHME_LICE_NO", dropna=False)["_fraud"].sum()
            doc_ratio = (doc_f / doc_tot.replace(0, np.nan)).fillna(0.0)
        hosp_key = (
            "HOSP_CODE"
            if "HOSP_CODE" in cl_siu.columns
            else ("HOSP_CD" if "HOSP_CD" in cl_siu.columns else None)
        )
        if hosp_key:
            h_tot = cl_siu.groupby(hosp_key, dropna=False).size()
            h_f = cl_siu.groupby(hosp_key, dropna=False)["_fraud"].sum()
            h_ratio = (h_f / h_tot.replace(0, np.nan)).fillna(0.0)

    if "CHME_LICE_NO" in cl.columns:
        cl["_doc_r"] = cl["CHME_LICE_NO"].map(doc_ratio).fillna(0.0)
        base["DOC_SIU_RATIO"] = cl.groupby(ID_COL, sort=False)["_doc_r"].max().fillna(0.0)

    hosp_key_full = "HOSP_CODE" if "HOSP_CODE" in cl.columns else ("HOSP_CD" if "HOSP_CD" in cl.columns else None)
    if hosp_key_full:
        cl["_hosp_r"] = cl[hosp_key_full].map(h_ratio).fillna(0.0)
        base["HOSP_SIU_RATIO"] = cl.groupby(ID_COL, sort=False)["_hosp_r"].max().fillna(0.0)

    if "DOC_SIU_RATIO" in base.columns and "HOSP_SIU_RATIO" in base.columns:
        base["HOSP_DOC_SIU"] = base["DOC_SIU_RATIO"] * base["HOSP_SIU_RATIO"]

    if "ACCI_DVSN" in cl.columns:
        acci_pct = pd.crosstab(cl[ID_COL], cl["ACCI_DVSN"], normalize="index")
        acci_pct = acci_pct.add_prefix("ACCI_RATIO_")
        base = base.join(acci_pct, how="left")

    need_cross = "ACCI_DVSN" in cl.columns and "DMND_RESN_CODE" in cl.columns
    if need_cross:
        sub = cl[[ID_COL, "ACCI_DVSN", "DMND_RESN_CODE"]].dropna()
        if not sub.empty:
            ct = pd.crosstab(
                index=sub[ID_COL],
                columns=[sub["ACCI_DVSN"].astype(str), sub["DMND_RESN_CODE"].astype(str)],
                normalize="index",
            )
            ct.columns = [f"ACCI_DMND_{_slug_col_part(a)}__{_slug_col_part(b)}" for a, b in ct.columns]
            base = base.join(ct, how="left")

    return base.reset_index()


def _slug_col_part(x) -> str:
    return re.sub(r"[^\w]+", "_", str(x))[:64]


def _merge_claims(
    X: pd.DataFrame,
    claim_df: pd.DataFrame | None,
    fraud_by_cust: pd.Series,
    train_cust_ids: frozenset | None,
) -> pd.DataFrame:
    if claim_df is None or len(claim_df) == 0:
        return X.copy()
    agg = _claim_book_features(claim_df, fraud_by_cust, train_cust_ids)
    if agg.empty:
        return X.copy()
    out = X.merge(agg, on=ID_COL, how="left")
    num_new = [c for c in agg.columns if c != ID_COL]
    if num_new:
        out[num_new] = out[num_new].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return out


class TestCustStrategy(BaseStrategy):

    def get_strategy_name(self) -> str:
        return "교재 근접: Set1 청구만으로 SIU비율 + 연령대/원핫/표준화 등"

    def preprocess(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        claim_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        fraud_map = pd.Series(_y_fraud_series(y).values, index=X[ID_COL].values)
        fraud_map = fraud_map.groupby(level=0).first()
        fraud_map.index = pd.to_numeric(fraud_map.index, errors="coerce").astype("Int64")
        fraud_map = fraud_map[fraud_map.index.notna()]
        fraud_map.index = fraud_map.index.astype(int)

        train_cust_ids = _train_customer_ids(X)
        X_out = _merge_claims(X, claim_df, fraud_map, train_cust_ids)

        X_out = _drop_cust_date_columns(X_out)
        drop_static = [c for c in ("MATE_OCCP_GRP_2", "FP_CAREER") if c in X_out.columns]
        if drop_static:
            X_out = X_out.drop(columns=drop_static)

        if "RESI_TYPE_CODE" in X_out.columns and "CTPR" in X_out.columns:
            mi = ModeImputer(cols=["RESI_TYPE_CODE", "CTPR"])
            mi.fit(X_out, y)
            X_out = mi.transform(X_out)

        if "WEDD_YN" in X_out.columns:
            X_out["WEDD_YN"] = X_out["WEDD_YN"].fillna("N")

        for c in ("LTBN_CHLD_AGE", "CHLD_CNT", "TOTALPREM", "MAX_PRM"):
            if c in X_out.columns:
                X_out[c] = pd.to_numeric(X_out[c], errors="coerce").fillna(0.0)

        if all(c in X_out.columns for c in ("OCCP_GRP_1", "AGE", "CUST_INCM")):
            gim = GroupMeanImputer()
            gim.fit(X_out, y)
            X_out = gim.transform(X_out)

        X_out = _jpbase_by_occupation(X_out)

        if "AGE" in X_out.columns:
            X_out["AGE_BAND"] = _age_to_band(X_out["AGE"], "AGE_")
            X_out = X_out.drop(columns=["AGE"])

        if "LTBN_CHLD_AGE" in X_out.columns:
            X_out["LTBN_BAND"] = _age_to_band(X_out["LTBN_CHLD_AGE"], "LTBN_")
            X_out = X_out.drop(columns=["LTBN_CHLD_AGE"])

        ohe_cols = [
            c
            for c in (
                "SEX",
                "RESI_TYPE_CODE",
                "CTPR",
                "OCCP_GRP_1",
                "OCCP_GRP_2",
                "WEDD_YN",
                "MATE_OCCP_GRP_1",
                "AGE_BAND",
                "LTBN_BAND",
            )
            if c in X_out.columns
        ]
        for c in ohe_cols:
            X_out[c] = (
                X_out[c]
                .astype("string")
                .fillna("unknown")
                .replace({"<NA>": "unknown"})
                .astype(str)
            )

        other_num = [
            c
            for c in X_out.columns
            if c != ID_COL and c not in ohe_cols and pd.api.types.is_numeric_dtype(X_out[c])
        ]
        if other_num:
            med = MedianImputer(cols=other_num)
            med.fit(X_out, y)
            X_out = med.transform(X_out)

        if ohe_cols:
            ohe = OneHotEncoder(cols=ohe_cols)
            ohe.fit(X_out, y)
            X_out = ohe.transform(X_out)

        scale_present = [c for c in _SCALE_COLS if c in X_out.columns]
        if scale_present:
            scaler = StandardScalerWrapper(cols=scale_present)
            scaler.fit(X_out, y)
            X_out = scaler.transform(X_out)

        for c in X_out.columns:
            if c == ID_COL:
                continue
            X_out[c] = np.asarray(X_out[c], dtype=np.float64)

        num_cols = [c for c in X_out.columns if c != ID_COL and pd.api.types.is_numeric_dtype(X_out[c])]
        if num_cols:
            X_out[num_cols] = X_out[num_cols].apply(pd.to_numeric, errors="coerce")
            med2 = X_out[num_cols].median()
            X_out[num_cols] = X_out[num_cols].fillna(med2).fillna(0.0)

        return X_out
