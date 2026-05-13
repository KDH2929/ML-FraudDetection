"""
허진경, 『머신러닝을 이용한 데이터 분석』 6장 부정탐지 분석 프로젝트(3절 Feature Engineering) 절차.

PDF 기준(357p~)과 동일한 순서·규칙:
- 3.1: 연령 //10, OCCP_GRP_2·MATE_OCCP_GRP_2 삭제, 직업 str[2:], 날짜·신용 삭제,
  소득·JPBASE 그룹 평균, RESI=20·CTPR='경기' 고정, WEDD=N, LTBN/CHLD/TOTALPREM/MAX_PRM=0,
  FP_CAREER 삭제 후 원-핫(SEX, RESI_TYPE_CODE, CTPR, OCCP_GRP_1, WEDD_YN, MATE_OCCP_GRP_1)
- 3.2: CLAIM 컬럼 삭제·날짜 삭제, HOSP_DAYS, HEED_HOSP, CLAIM_COUNT, DOC/HOSP 비율(전 청구×SIU),
  HOSP_DOC_SIU, ACCI_RATIO_1~3, 사고×청구코드 조합(열 이름 책과 동일 패턴)
- 3.3: 6개 수치 StandardScaler (책은 sklearn StandardScaler 직접 사용과 동치)

※ SIU 라벨은 raw y(정규화 전)를 병합용으로 쓰지 않고, 전달된 y를 0/1로 두어 비율 합산에 사용한다.

[초기 구현 대비 PDF 정렬로 바뀐 점 — 수치 재현·해석용]
- RESI_TYPE_CODE / CTPR: 최빈값(ModeImputer) → 책과 같이 fillna(20), fillna('경기').
- CLAIM 파생 병합: 원-핫 이전 → 원-핫 이후(책 셀 순서).
- 입원일수 열 이름: HOST_DAYS → HOSP_DAYS.
- ACCI_RATIO: 동적 열 이름 → ACCI_RATIO_1/2/3 고정 + CLAIM_COUNT로 나눔.
- 사고×청구 조합열: 슬러그 접두 → 책과 같은 pivot_table + MultiIndex droplevel + join 이름.
- 표준화: StandardScalerWrapper → sklearn StandardScaler 후 열 삭제·concat.
- FP·신용·직업 드롭 순서를 책 3.1 단계 순서에 맞춤.

[이전 버전에서도 F1·정확도가 책과 비슷하게 “높게” 보였던 이유]
- 두 버전 모두 의사·병원 SIU 비율 등이 “전체 청구 × 고객 라벨”로 만들어져, 트리/부스팅에 강한 신호가 들어간다(책 노트북과 같은 구조). RESI/CTPR 처리·OHE 순서는 이 신호에 비해 부차적이라, 지표는 대체로 비슷한 구간에 머물 수 있다.
- “책보다 유독 높다”가 아니라 책 출력 자체가 이미 그 구간(다수 클래스 정확도 ~0.97 등)이다.

논문 서베이 기반 파생은 `book_survey_paper_strategy` 를 사용한다 (이 파일은 PDF 재현 전용).

실험 CLI ID: book_pdf (`book_pdf_strategy.py`)
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


class BookPdfStrategy(BaseStrategy):

    def get_strategy_name(self) -> str:
        return "『머신러닝을 이용한 데이터 분석』 6장 3절 Feature Engineering (PDF 동일 순서)"

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
