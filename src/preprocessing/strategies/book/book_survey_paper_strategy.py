"""
book_pdf(PDF 6장 3절) 전처리와 동일한 뒤, 논문 서베이 표 A/B 요지 파생을 추가.

- A: 가입~관측월·첫청구까지 기간, 고객 단위 청구·지급 요약, 납입/소득 비율
- B: 직업×연령(//10) 그룹 내 총지급 Z-score(DIVIDED_SET==1 로만 μ·σ), 최근 6/6~12/12개월 이전 청구 강도,
  병원 다양도·유의병원 비중·상위 병원 청구비중, VLID 입원일수 합·최대

실험 CLI ID: book_survey_paper
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


def _month_end_from_yyyymm(ym: pd.Series) -> pd.Series:
    s_clean = pd.to_numeric(ym, errors="coerce")
    parts = []
    for v in s_clean:
        if pd.isna(v):
            parts.append(pd.NaT)
        else:
            iv = int(round(float(v)))
            parts.append(f"{iv:06d}")
    ts = pd.to_datetime(pd.Series(parts, index=ym.index), format="%Y%m", errors="coerce")
    return ts + pd.offsets.MonthEnd(0)


def _month_start_from_yyyymm(ym: pd.Series) -> pd.Series:
    s_clean = pd.to_numeric(ym, errors="coerce")
    parts = []
    for v in s_clean:
        if pd.isna(v):
            parts.append(pd.NaT)
        else:
            iv = int(round(float(v)))
            parts.append(f"{iv:06d}")
    return pd.to_datetime(pd.Series(parts, index=ym.index), format="%Y%m", errors="coerce")


def _paper_survey_block(
    claim_df: pd.DataFrame | None,
    paper_meta: pd.DataFrame,
    peer_frame: pd.DataFrame,
) -> pd.DataFrame:
    if claim_df is None or len(claim_df) == 0 or ID_COL not in claim_df.columns:
        return pd.DataFrame()

    cl = claim_df.copy()
    has_signal = any(
        c in cl.columns for c in ("RECP_DATE", "PAYM_AMT", "DMND_AMT", "HOSP_CODE", "HEED_HOSP_YN", "VLID_HOSP_OTDA")
    )
    if not has_signal:
        return pd.DataFrame()

    cl[ID_COL] = pd.to_numeric(cl[ID_COL], errors="coerce").astype("Int64")
    cl = cl[cl[ID_COL].notna()].copy()
    cl[ID_COL] = cl[ID_COL].astype(int)

    if "RECP_DATE" in cl.columns:
        rd = pd.to_numeric(cl["RECP_DATE"], errors="coerce")
        rd_str = rd.map(lambda x: f"{int(round(x)):08d}" if pd.notna(x) else np.nan)
        cl["_recp_dt"] = pd.to_datetime(rd_str, format="%Y%m%d", errors="coerce")
    else:
        cl["_recp_dt"] = pd.NaT

    meta = paper_meta.drop_duplicates(subset=[ID_COL], keep="first").copy()
    meta[ID_COL] = pd.to_numeric(meta[ID_COL], errors="coerce").astype(int)
    cl = cl.merge(meta, on=ID_COL, how="left")

    obs_col = "_obs_end"
    if "MAX_PAYM_YM" in cl.columns:
        cl[obs_col] = _month_end_from_yyyymm(cl["MAX_PAYM_YM"])
    else:
        cl[obs_col] = cl.groupby(ID_COL, sort=False)["_recp_dt"].transform("max")

    if "PAYM_AMT" in cl.columns:
        cl["_paym"] = pd.to_numeric(cl["PAYM_AMT"], errors="coerce").fillna(0.0)
    else:
        cl["_paym"] = 0.0
    if "DMND_AMT" in cl.columns:
        cl["_dmnd"] = pd.to_numeric(cl["DMND_AMT"], errors="coerce").fillna(0.0)
    else:
        cl["_dmnd"] = 0.0

    g = cl.groupby(ID_COL, sort=False)
    base = g.size().to_frame("paper_claim_rows")
    base["paper_paym_sum"] = g["_paym"].sum()
    base["paper_dmnd_sum"] = g["_dmnd"].sum()
    base["paper_paym_mean"] = g["_paym"].mean()
    base["paper_dmnd_mean"] = g["_dmnd"].mean()
    base["paper_paym_max"] = g["_paym"].max()
    base["paper_dmnd_max"] = g["_dmnd"].max()
    base["paper_paym_dmnd_ratio"] = base["paper_paym_sum"] / (base["paper_dmnd_sum"] + 1e-6)

    if cl["_recp_dt"].notna().any():
        base["paper_first_recp"] = g["_recp_dt"].min()
        base["paper_last_recp"] = g["_recp_dt"].max()
    else:
        base["paper_first_recp"] = pd.NaT
        base["paper_last_recp"] = pd.NaT

    if "HOSP_CODE" in cl.columns:
        base["paper_hosp_nunique"] = g["HOSP_CODE"].nunique()
        top_series = cl.groupby(ID_COL, sort=False)["HOSP_CODE"].apply(
            lambda s: float(s.value_counts(normalize=True).iloc[0]) if len(s) else 0.0
        )
        base["paper_top_hosp_share"] = top_series.reindex(base.index).fillna(0.0)
    if "HEED_HOSP_YN" in cl.columns:
        yn = cl["HEED_HOSP_YN"].astype(str).str.upper().str.strip()
        cl["_heed"] = (yn == "Y").astype(float)
        base["paper_heed_claim_share"] = g["_heed"].mean()

    if "VLID_HOSP_OTDA" in cl.columns:
        vl = pd.to_numeric(cl["VLID_HOSP_OTDA"], errors="coerce").fillna(0.0)
        cl["_vlid"] = vl
        base["paper_vlid_sum"] = g["_vlid"].sum()
        base["paper_vlid_max"] = g["_vlid"].max()

    if cl["_recp_dt"].notna().any() or "MAX_PAYM_YM" in cl.columns:
        m6, m6_12, m12p = [], [], []
        for cid in base.index.tolist():
            sub = cl.loc[cl[ID_COL] == cid]
            obs = sub[obs_col].iloc[0] if len(sub) else pd.NaT
            if pd.isna(obs):
                m6.append(0.0)
                m6_12.append(0.0)
                m12p.append(0.0)
                continue
            d = sub["_recp_dt"]
            c6 = int(((d > obs - pd.DateOffset(months=6)) & (d <= obs)).sum())
            c612 = int(((d > obs - pd.DateOffset(months=12)) & (d <= obs - pd.DateOffset(months=6))).sum())
            cold = int((d <= obs - pd.DateOffset(months=12)).sum())
            m6.append(float(c6))
            m6_12.append(float(c612))
            m12p.append(float(cold))
        base["paper_cnt_recent_6m"] = m6
        base["paper_cnt_6m_to_12m"] = m6_12
        base["paper_cnt_prior_12m"] = m12p
        denom = base["paper_cnt_6m_to_12m"] + base["paper_cnt_prior_12m"] + 1.0
        base["paper_recent6_vs_prior_ratio"] = base["paper_cnt_recent_6m"] / denom

    base = base.reset_index()
    meta_ix = paper_meta.drop_duplicates(subset=[ID_COL], keep="first").set_index(ID_COL)

    if "CUST_RGST" in meta_ix.columns:
        cust_start = _month_start_from_yyyymm(base[ID_COL].map(meta_ix["CUST_RGST"]))
    else:
        cust_start = pd.Series(pd.NaT, index=base.index)

    if "MAX_PAYM_YM" in meta_ix.columns:
        obs_m = _month_end_from_yyyymm(base[ID_COL].map(meta_ix["MAX_PAYM_YM"]))
    else:
        obs_m = pd.Series(pd.NaT, index=base.index)

    ok_t = cust_start.notna() & obs_m.notna()
    delta_days = (obs_m - cust_start) / np.timedelta64(1, "D")
    base["paper_tenure_m"] = np.where(ok_t, delta_days.astype(float) / 30.4375, np.nan)

    fr = pd.to_datetime(base["paper_first_recp"], errors="coerce")
    ok_l = cust_start.notna() & fr.notna()
    lag_days = (fr - cust_start) / np.timedelta64(1, "D")
    base["paper_first_claim_lag_m"] = np.where(ok_l, lag_days.astype(float) / 30.4375, np.nan)

    peer = peer_frame.copy()
    peer[ID_COL] = pd.to_numeric(peer[ID_COL], errors="coerce").astype(int)
    merged = base.merge(peer, on=ID_COL, how="left")

    paym_col = "paper_paym_sum"
    if paym_col in merged.columns and "OCCP_GRP_1" in merged.columns and "AGE" in merged.columns:
        tr_mask = pd.Series(True, index=merged.index)
        if DIVIDED_SET_COL in merged.columns:
            tr_mask = pd.to_numeric(merged[DIVIDED_SET_COL], errors="coerce") == 1
        sub = merged.loc[tr_mask, ["OCCP_GRP_1", "AGE", paym_col]]
        if len(sub) > 0:
            stats = (
                sub.groupby(["OCCP_GRP_1", "AGE"], observed=False)[paym_col]
                .agg(mean_train="mean", std_train="std")
                .reset_index()
            )
            stats["std_train"] = stats["std_train"].replace(0, np.nan)
            merged = merged.merge(stats, on=["OCCP_GRP_1", "AGE"], how="left")
            merged["paper_peer_paym_z"] = (merged[paym_col] - merged["mean_train"]) / (merged["std_train"] + 1e-6)
            merged = merged.drop(columns=["mean_train", "std_train"], errors="ignore")
            merged["paper_peer_paym_z"] = merged["paper_peer_paym_z"].fillna(0.0)
        else:
            merged["paper_peer_paym_z"] = 0.0
    else:
        merged["paper_peer_paym_z"] = 0.0

    drop_peer = [c for c in (DIVIDED_SET_COL, "OCCP_GRP_1", "AGE") if c in merged.columns and c != ID_COL]
    out = merged.drop(columns=[c for c in drop_peer if c in merged.columns], errors="ignore")

    dt_drop = [c for c in out.columns if c in ("paper_first_recp", "paper_last_recp") or str(c).startswith("_")]
    out = out.drop(columns=[c for c in dt_drop if c in out.columns], errors="ignore")

    num_cols = [c for c in out.columns if c != ID_COL]
    if num_cols:
        out[num_cols] = out[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return out


def _paper_survey_customer_addons(X: pd.DataFrame) -> pd.DataFrame:
    """
    서베이 논문과 연계된 고객표 파생(납입/소득 비율 등). `book_pdf` CLAIM 집계와 무관.
    `BookSurveyPaperStrategy` preprocess 의 동일 단계를 함수로 분리.
    """
    if ID_COL not in X.columns:
        return pd.DataFrame()
    out = pd.DataFrame({ID_COL: X[ID_COL]})
    if "TOTALPREM" in X.columns and "CUST_INCM" in X.columns:
        inc = pd.to_numeric(X["CUST_INCM"], errors="coerce").clip(lower=0)
        prm = pd.to_numeric(X["TOTALPREM"], errors="coerce").fillna(0.0)
        out["paper_prem_incm_ratio"] = prm / (inc + 1.0)
        out["paper_incm_missing_or_zero"] = ((inc <= 0) | X["CUST_INCM"].isna()).astype(float)
    else:
        out["paper_prem_incm_ratio"] = 0.0
        out["paper_incm_missing_or_zero"] = 0.0
    return out


def _paper_survey_paper_meta_from_X(X: pd.DataFrame) -> pd.DataFrame:
    """`_paper_survey_block` 용 paper_meta (가입·관측월). 날짜 컬럼 삭제 전과 동일 스냅샷."""
    cols = [ID_COL]
    for c in ("CUST_RGST", "MAX_PAYM_YM"):
        if c in X.columns:
            cols.append(c)
    return X[cols].drop_duplicates(subset=[ID_COL], keep="first").copy()


def _paper_survey_peer_for_block(X: pd.DataFrame) -> pd.DataFrame:
    """
    `_paper_survey_block` 직전 `BookSurveyPaperStrategy` 와 동일한 peer_frame:
    직업 앞 2글자 제거, 연령 10세 단위, DIVIDED_SET.
    """
    peer_cols = [ID_COL, "OCCP_GRP_1", "AGE"]
    if DIVIDED_SET_COL in X.columns:
        peer_cols.append(DIVIDED_SET_COL)
    peer = X[[c for c in peer_cols if c in X.columns]].copy()
    if peer.empty or ID_COL not in peer.columns:
        return pd.DataFrame({ID_COL: X[ID_COL]}) if ID_COL in X.columns else pd.DataFrame()
    if "OCCP_GRP_1" in peer.columns:
        peer["OCCP_GRP_1"] = peer["OCCP_GRP_1"].map(_strip_two_chars)
    if "AGE" in peer.columns:
        peer["AGE"] = peer["AGE"].map(lambda x: int(pd.to_numeric(x, errors="coerce") or 0) // 10)
    if DIVIDED_SET_COL in peer.columns:
        peer[DIVIDED_SET_COL] = pd.to_numeric(peer[DIVIDED_SET_COL], errors="coerce")
    return peer


class BookSurveyPaperStrategy(BaseStrategy):

    def get_strategy_name(self) -> str:
        return "6장 3절 PDF 동일 + 논문 서베이 파생(book_survey_paper)"

    def preprocess(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        claim_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        y_bin = _y_binary(y)
        X_out = X.copy()

        paper_meta = _paper_survey_paper_meta_from_X(X_out)

        X_out = _drop_cust_dates(X_out)

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

        addons = _paper_survey_customer_addons(X_out)
        if not addons.empty:
            X_out = X_out.merge(addons, on=ID_COL, how="left")

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

        peer_frame = _paper_survey_peer_for_block(X_out)
        paper_agg = _paper_survey_block(claim_df, paper_meta, peer_frame)
        if not paper_agg.empty:
            X_out = X_out.merge(paper_agg, on=ID_COL, how="left")
            pnew = [c for c in paper_agg.columns if c != ID_COL]
            if pnew:
                X_out[pnew] = X_out[pnew].apply(pd.to_numeric, errors="coerce").fillna(0.0)

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
