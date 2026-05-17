"""
허진경, 『머신러닝을 이용한 데이터 분석』 6장 3절(book_pdf)과 동일한 전처리 뒤,
청구(CLAIM) 데이터에서 직접 계산할 수 있는 일반적인 고객 단위 집계 파생
`claim_agg_*` 열을 추가한다.

- `claim_agg_*` 는 라벨 y 를 사용하지 않는다(다기관·시간 밀도·진단/과목 다양도·
  지급/청구 이질성·거리·유의병원 비중 등 청구 행동 지표).
- 책 PDF 재현용 CLAIM 파생(`CLAIM_COUNT`, SIU 비율 등)은 `book_pdf_strategy`
  의 `_book_claim_block` 과 동일하게 유지한다.

실험 CLI ID: book_claim_agg
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import ID_COL
from src.preprocessing.components.encoder import OneHotEncoder
from src.preprocessing.components.missing_value import GroupMeanImputer
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


def _book_claim_block(
    claim_df: pd.DataFrame,
    cust_ids: pd.Series,
    y_bin: pd.Series,
) -> pd.DataFrame:
    """3.2 CLAIM 파생 — PDF 코드와 동일한 연산."""
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
        base["HOSP_DAYS"] = hd

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

    if "ACCI_DVSN" in cl.columns and "POLY_NO" in cl.columns:
        ac = cl.pivot_table(index=ID_COL, columns="ACCI_DVSN", values="POLY_NO", aggfunc="count", fill_value=0)
        cc = base["CLAIM_COUNT"].replace(0, np.nan)
        for d in (1, 2, 3):
            colv = ac[d] if d in ac.columns else pd.Series(0, index=base.index)
            colv = colv.reindex(base.index, fill_value=0).fillna(0)
            base[f"ACCI_RATIO_{d}"] = (colv / cc).fillna(0.0)

    if "ACCI_DVSN" in cl.columns and "DMND_RESN_CODE" in cl.columns:
        claim_acci = cl.loc[:, [ID_COL, "ACCI_DVSN", "DMND_RESN_CODE"]].dropna()
        if not claim_acci.empty:
            claim_acci = claim_acci.copy()
            claim_acci["value"] = 1
            cust_claim_df = claim_acci.pivot_table(
                index=[ID_COL],
                columns=["ACCI_DVSN", "DMND_RESN_CODE"],
                values=["value"],
                aggfunc="count",
                fill_value=0,
            )
            cust_claim_df = cust_claim_df.reset_index(level=[ID_COL], col_level=1)
            cust_claim_df.columns = cust_claim_df.columns.droplevel(level=0)
            cust_claim_df.columns = pd.Index(
                ["_".join([str(col) for col in cols]) for cols in cust_claim_df.columns]
            )
            rename_key = f"{ID_COL}_"
            if rename_key in cust_claim_df.columns:
                cust_claim_df = cust_claim_df.rename(columns={rename_key: ID_COL})
            elif ID_COL not in cust_claim_df.columns and len(cust_claim_df.columns) > 0:
                first = str(cust_claim_df.columns[0])
                if first.startswith(str(ID_COL)):
                    cust_claim_df = cust_claim_df.rename(columns={first: ID_COL})
            base = base.join(cust_claim_df.set_index(ID_COL), how="left")

    out = base.reset_index().fillna(0.0)
    return out


def _dsas_top_share(s: pd.Series) -> float:
    if s is None or len(s) == 0:
        return 0.0
    vc = s.value_counts()
    return float(vc.iloc[0] / len(s))


def _claim_aggregation_features(claim_df: pd.DataFrame | None) -> pd.DataFrame:
    """청구 데이터에서 라벨 없이 계산 가능한 고객 단위 집계 파생.

    포함되는 시그널: 청구 건수, 다기관/다의사 노출, 지급액 평균·표준편차·최대,
    지급/청구 spread, 진단·사고구분·과목 다양도, 유의병원 비중, 입원일수 분포,
    집-병원 평균 거리, 청구 강도(시간 밀도), 병원 전환 횟수, 청구 간격 통계,
    같은 주 다기관 방문 등.
    """
    if claim_df is None or len(claim_df) == 0 or ID_COL not in claim_df.columns:
        return pd.DataFrame()

    cl = claim_df.copy()
    cl[ID_COL] = pd.to_numeric(cl[ID_COL], errors="coerce").astype("Int64")
    cl = cl[cl[ID_COL].notna()].copy()
    if cl.empty:
        return pd.DataFrame()
    cl[ID_COL] = cl[ID_COL].astype(int)

    g = cl.groupby(ID_COL, sort=False)
    base = g.size().to_frame("claim_agg_n_claims")

    if "HOSP_CODE" in cl.columns:
        base["claim_agg_n_hosp"] = g["HOSP_CODE"].nunique()
        base["claim_agg_n_hosp_per_10_claims"] = (
            base["claim_agg_n_hosp"] / (base["claim_agg_n_claims"].replace(0, np.nan)) * 10.0
        )
    if "CHME_LICE_NO" in cl.columns:
        base["claim_agg_n_doc"] = g["CHME_LICE_NO"].nunique()

    if "PAYM_AMT" in cl.columns:
        cl["_paym"] = pd.to_numeric(cl["PAYM_AMT"], errors="coerce")
        gp = cl.groupby(ID_COL, sort=False)["_paym"]
        mn = gp.mean()
        sd = gp.std()
        mx = gp.max()
        base["claim_agg_paym_sum"] = gp.sum()
        base["claim_agg_paym_cv"] = (sd / mn.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        base["claim_agg_paym_max_ratio"] = mx / (mn + 1e-6)

    if "DMND_AMT" in cl.columns and "PAYM_AMT" in cl.columns:
        cl["_dmnd"] = pd.to_numeric(cl["DMND_AMT"], errors="coerce")
        if "_paym" not in cl.columns:
            cl["_paym"] = pd.to_numeric(cl["PAYM_AMT"], errors="coerce")
        cl["_spread"] = cl["_dmnd"].fillna(0.0) - cl["_paym"].fillna(0.0)
        base["claim_agg_dmnd_paym_spread_mean"] = cl.groupby(ID_COL, sort=False)["_spread"].mean()

    if "DSAS_NAME" in cl.columns:
        base["claim_agg_dsas_nunique"] = g["DSAS_NAME"].nunique()
        base["claim_agg_dsas_top_share"] = g["DSAS_NAME"].apply(_dsas_top_share)

    if "ACCI_DVSN" in cl.columns:
        base["claim_agg_acci_nunique"] = g["ACCI_DVSN"].nunique()

    if "HOSP_SPEC_DVSN" in cl.columns:
        base["claim_agg_hosp_spec_nunique"] = g["HOSP_SPEC_DVSN"].nunique()

    if "HEED_HOSP_YN" in cl.columns:
        yn = cl["HEED_HOSP_YN"].astype(str).str.upper().str.strip()
        cl["_heed"] = (yn == "Y").astype(float)
        base["claim_agg_heed_frac"] = cl.groupby(ID_COL, sort=False)["_heed"].mean()

    if "VLID_HOSP_OTDA" in cl.columns:
        cl["_vlid"] = pd.to_numeric(cl["VLID_HOSP_OTDA"], errors="coerce")
        gv = cl.groupby(ID_COL, sort=False)["_vlid"]
        base["claim_agg_vlid_std"] = gv.std()
        base["claim_agg_vlid_max"] = gv.max()

    if "HOUSE_HOSP_DIST" in cl.columns:
        cl["_hdist"] = pd.to_numeric(cl["HOUSE_HOSP_DIST"], errors="coerce")
        base["claim_agg_house_hosp_dist_mean"] = cl.groupby(ID_COL, sort=False)["_hdist"].mean()

    if "RECP_DATE" in cl.columns:
        rd = pd.to_numeric(cl["RECP_DATE"], errors="coerce")
        rd_str = rd.map(lambda x: f"{int(round(x)):08d}" if pd.notna(x) else np.nan)
        cl["_recp_dt"] = pd.to_datetime(rd_str, format="%Y%m%d", errors="coerce")
    else:
        cl["_recp_dt"] = pd.NaT

    span_stats = cl.groupby(ID_COL, sort=False)["_recp_dt"].agg(["min", "max"])
    span_days = (span_stats["max"] - span_stats["min"]).dt.days.astype("float64")
    base["claim_agg_claim_intensity"] = base["claim_agg_n_claims"] / (span_days.clip(lower=1.0) / 30.0 + 1e-6)

    cl_s = cl.sort_values([ID_COL, "_recp_dt"], kind="mergesort")
    if "HOSP_CODE" in cl_s.columns:
        ph = cl_s.groupby(ID_COL, sort=False)["HOSP_CODE"].shift(1)
        sw = (cl_s["HOSP_CODE"].astype(str) != ph.astype(str)).fillna(False).astype(int)
        sw_sum = cl_s.assign(_sw=sw).groupby(ID_COL, sort=False)["_sw"].sum()
        base["claim_agg_hosp_switch_count"] = (sw_sum - 1).clip(lower=0).reindex(base.index, fill_value=0)

    cl_s = cl.sort_values([ID_COL, "_recp_dt"], kind="mergesort")
    cl_s["_gd"] = cl_s.groupby(ID_COL, sort=False)["_recp_dt"].diff().dt.total_seconds() / 86400.0
    base["claim_agg_recp_gap_mean_days"] = cl_s.groupby(ID_COL, sort=False)["_gd"].mean()
    base["claim_agg_recp_gap_min_days"] = cl_s.groupby(ID_COL, sort=False)["_gd"].min()

    if "HOSP_CODE" in cl.columns:
        cl_w = cl.loc[cl["_recp_dt"].notna()].copy()
        if not cl_w.empty:
            cl_w["_week"] = cl_w["_recp_dt"].dt.to_period("W").astype(str)
            wk = cl_w.groupby([ID_COL, "_week"], sort=False)["HOSP_CODE"].nunique()
            wk = wk[wk > 1].reset_index(name="_n")
            if not wk.empty:
                multi = wk.groupby(ID_COL, sort=False).size()
                base["claim_agg_weeks_multi_hosp"] = multi.reindex(base.index, fill_value=0)
            else:
                base["claim_agg_weeks_multi_hosp"] = 0
        else:
            base["claim_agg_weeks_multi_hosp"] = 0

    out = base.reset_index()
    feat_cols = [c for c in out.columns if c.startswith("claim_agg_")]
    out[feat_cols] = (
        out[feat_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    return out


class BookClaimAggStrategy(BaseStrategy):

    def get_strategy_name(self) -> str:
        return "6장 3절 PDF 동일 + 청구 데이터 일반 집계 파생 claim_agg_*"

    def preprocess(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        claim_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        y_bin = _y_binary(y)
        X_out = X.copy()

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

        claim_agg = _claim_aggregation_features(claim_df)
        if not claim_agg.empty:
            X_out = X_out.merge(claim_agg, on=ID_COL, how="left")
            cols = [c for c in claim_agg.columns if c != ID_COL]
            if cols:
                X_out[cols] = X_out[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

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
