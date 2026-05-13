"""
Member A 전처리 v3 — v1 파이프라인 + 도메인 집계·누수 완화 SIU 노출·소득/주택 파생.

`book_pdf_strategy`와는 다른 설계(원-핫·책 3.2 피벗 재현 없음). 목표는 동일 데이터에서
트리 모델이 쓰기 쉬운 **수치 파생**과 **학습 구간에서만 추정한 병원·의사 SIU 압력**을 더하는 것.

포함
- **청구(변수정의서 기반)**: 고객별 청구액 분산·최대, 유의병원 청구 비율, 사고구분 다양도,
  행별 지급/청구 비율 평균, (있으면) 비급여비율 요약.
- **SIU 노출(라벨 의존)**: `preprocess`(전역 CSV 생성)에서는 **끈다**. 이유는 `DIVIDED_SET==1` 전체로
  병원·의사 비율을 만든 뒤 `train_test_split`을 하면, **폴드 밖 동료 라벨이 테스트 피처에 섞이는**
  교차 누수가 생기기 때문이다. LOO로 자기 청구만 제외해도 이 문제는 남는다.
  반면 `preprocess_train_test`에서는 전달된 학습 고객 ID로만 비율표를 맞추므로 **SIU 노출을 켠다**.
- **고객(정의서)**: `RESI_COST==0` 또는 결측 → `ma3_resi_unknown`, 가구소득 두 추정치 차이,
  개인추정소득과 JPBASE 가구소득 괴리.

고급(HTML 체크리스트) 중 본 파일에서 다루는 것
- 그룹 기반 대치(GroupMean), 중앙값·최빈값, 타깃 인코딩(과적합 주의), IQR, Robust, 잔여 숫자 강제.

로드맵(미구현)
- MCAR/MAR/MNAR 가정별 별도 모형, VIF·필터 검정, SMOTE, K-Fold 내 중첩 전처리 등.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CAT_COLS, DIVIDED_SET_COL, ID_COL, NUM_COLS
from src.preprocessing.components.encoder import TargetEncoder
from src.preprocessing.components.missing_value import GroupMeanImputer, MedianImputer, ModeImputer
from src.preprocessing.components.outlier import IQRCapper
from src.preprocessing.components.scaler import RobustScalerWrapper
from src.preprocessing.strategies.base_strategy import BaseStrategy
from src.preprocessing.strategies.member_a_strategy import (
    _drop_unused_cust_columns,
    _fill_residual_missing_for_ml,
    _merge_claim_features,
    _y_to_float01,
)


def _y_binary_int(y: pd.Series) -> pd.Series:
    s = _y_to_float01(y)
    return s.fillna(0.0).clip(0, 1).astype(int)


def _train_customer_ids(X: pd.DataFrame) -> frozenset | None:
    if DIVIDED_SET_COL not in X.columns:
        return None
    d = pd.to_numeric(X[DIVIDED_SET_COL], errors="coerce")
    mask = d == 1
    if not mask.any():
        return frozenset()
    ids = X.loc[mask, ID_COL]
    return frozenset(pd.to_numeric(ids, errors="coerce").dropna().astype(int).unique())


def _ma3_claim_domain_block(
    claim_df: pd.DataFrame,
    cust_ids: pd.Series,
    y_bin: pd.Series,
    train_cust_ids: frozenset | None,
    *,
    include_label_dependent_exposure: bool = True,
) -> pd.DataFrame:
    """고객 단위 추가 집계. SIU 노출은 include_label_dependent_exposure=False면 생략(단일 preprocess·내부 holdout 불일치 시 누수 방지)."""
    if claim_df is None or claim_df.empty or ID_COL not in claim_df.columns:
        return pd.DataFrame()

    cl = claim_df.copy()
    cl[ID_COL] = pd.to_numeric(cl[ID_COL], errors="coerce").astype("Int64")
    cl = cl[cl[ID_COL].notna()].copy()
    cl[ID_COL] = cl[ID_COL].astype(int)

    gby = cl.groupby(ID_COL, sort=False)
    parts: list[pd.DataFrame] = []

    if "DMND_AMT" in cl.columns:
        mx = gby["DMND_AMT"].max().rename("ma3_max_dmnd_amt")
        st = gby["DMND_AMT"].std().rename("ma3_std_dmnd_amt")
        parts.extend([mx, st])

    if "PAYM_AMT" in cl.columns and "DMND_AMT" in cl.columns:
        cl2 = cl.copy()
        ratio = cl2["PAYM_AMT"] / pd.to_numeric(cl2["DMND_AMT"], errors="coerce").replace(0, np.nan)
        ratio = ratio.replace([np.inf, -np.inf], np.nan)
        cl2["_pay_ratio"] = ratio
        pr_mean = cl2.groupby(ID_COL, sort=False)["_pay_ratio"].mean().rename("ma3_mean_paym_dmnd_ratio")
        parts.append(pr_mean)

    heed_col = "HEED_HOSP_YN" if "HEED_HOSP_YN" in cl.columns else None
    if heed_col:
        yn = cl[heed_col].astype(str).str.upper().str.strip()
        cl["_heed_row"] = (yn == "Y").astype(np.float64)
        hf = cl.groupby(ID_COL, sort=False)["_heed_row"].mean().rename("ma3_frac_heed_claim_rows")
        parts.append(hf)

    if "ACCI_DVSN" in cl.columns:
        div = gby["ACCI_DVSN"].nunique(dropna=True).rename("ma3_nunique_acci_dvsn")
        parts.append(div)

    if "NON_PAY_RATIO" in cl.columns:
        nr = pd.to_numeric(cl["NON_PAY_RATIO"], errors="coerce")
        cl["_npr"] = nr
        nmax = cl.groupby(ID_COL, sort=False)["_npr"].max().rename("ma3_max_nonpay_ratio")
        parts.append(nmax)

    if not parts:
        base = pd.DataFrame({ID_COL: cl[ID_COL].unique()})
    else:
        base = pd.concat(parts, axis=1).reset_index()

    if include_label_dependent_exposure:
        y_map = pd.Series(_y_binary_int(y_bin).values, index=pd.to_numeric(cust_ids, errors="coerce").values)
        y_map = y_map.groupby(level=0).first()
        cl["_siu"] = cl[ID_COL].map(y_map).fillna(0).astype(int)

        if train_cust_ids is None:
            cl_fit = cl
        else:
            cl_fit = cl[cl[ID_COL].isin(train_cust_ids)].copy()

        def _loo_max_siu_exposure(
            cl_all: pd.DataFrame,
            cl_fit_: pd.DataFrame,
            key_col: str,
            out_name: str,
        ) -> pd.DataFrame:
            """학습구간 청구만으로 key별 SIU 비율을 만들되, 동일 고객 청구는 분모·분자에서 제외(자기 라벨 누수 완화)."""
            if key_col not in cl_all.columns or len(cl_fit_) == 0:
                return pd.DataFrame({ID_COL: cl_all[ID_COL].unique(), out_name: 0.0})
            tot = cl_fit_.groupby(key_col, dropna=False).agg(n_all=(ID_COL, "size"), siu_all=("_siu", "sum"))
            ch = (
                cl_fit_.groupby([ID_COL, key_col], dropna=False)
                .agg(n_ch=(ID_COL, "size"), s_ch=("_siu", "sum"))
                .reset_index()
            )
            ch = ch.merge(tot.reset_index(), on=key_col, how="left")
            den = (ch["n_all"] - ch["n_ch"]).astype(np.float64)
            num = (ch["siu_all"] - ch["s_ch"]).astype(np.float64)
            ch["_r"] = np.where(den > 0, num / den, 0.0)
            mm = ch.groupby(ID_COL, sort=False)["_r"].max().reset_index(name=out_name)
            all_ids = pd.DataFrame({ID_COL: cl_all[ID_COL].unique()})
            return all_ids.merge(mm, on=ID_COL, how="left")

        dmax = pd.DataFrame({ID_COL: cl[ID_COL].unique(), "ma3_doc_siu_train_expo": 0.0})
        hmax = pd.DataFrame({ID_COL: cl[ID_COL].unique(), "ma3_hosp_siu_train_expo": 0.0})
        dmax = (
            _loo_max_siu_exposure(cl, cl_fit, "CHME_LICE_NO", "ma3_doc_siu_train_expo")
            if "CHME_LICE_NO" in cl.columns
            else dmax
        )
        hk_all = "HOSP_CODE" if "HOSP_CODE" in cl.columns else ("HOSP_CD" if "HOSP_CD" in cl.columns else None)
        if hk_all:
            hmax = _loo_max_siu_exposure(cl, cl_fit, hk_all, "ma3_hosp_siu_train_expo")
    else:
        dmax = pd.DataFrame({ID_COL: cl[ID_COL].unique(), "ma3_doc_siu_train_expo": 0.0})
        hmax = pd.DataFrame({ID_COL: cl[ID_COL].unique(), "ma3_hosp_siu_train_expo": 0.0})

    base = base.merge(dmax, on=ID_COL, how="outer").merge(hmax, on=ID_COL, how="outer")
    for c in ("ma3_doc_siu_train_expo", "ma3_hosp_siu_train_expo"):
        if c in base.columns:
            base[c] = pd.to_numeric(base[c], errors="coerce").fillna(0.0)
    base["ma3_doc_hosp_train_cross"] = (
        pd.to_numeric(base["ma3_doc_siu_train_expo"], errors="coerce").fillna(0.0)
        * pd.to_numeric(base["ma3_hosp_siu_train_expo"], errors="coerce").fillna(0.0)
    )
    return base


def _ma3_customer_domain_features(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    if "RESI_COST" in out.columns:
        rc = pd.to_numeric(out["RESI_COST"], errors="coerce")
        out["ma3_resi_unknown"] = (rc.isna() | (rc == 0)).astype(np.float64)

    if all(c in out.columns for c in ("RCBASE_HSHD_INCM", "JPBASE_HSHD_INCM")):
        a = pd.to_numeric(out["RCBASE_HSHD_INCM"], errors="coerce")
        b = pd.to_numeric(out["JPBASE_HSHD_INCM"], errors="coerce")
        out["ma3_absdiff_hh_incm_est"] = (a - b).abs()

    if all(c in out.columns for c in ("CUST_INCM", "JPBASE_HSHD_INCM")):
        c = pd.to_numeric(out["CUST_INCM"], errors="coerce")
        j = pd.to_numeric(out["JPBASE_HSHD_INCM"], errors="coerce")
        out["ma3_cust_minus_jpbase_incm"] = c - j

    return out


class MemberA3Strategy(BaseStrategy):

    def get_strategy_name(self) -> str:
        return "A3: member_a + 도메인 집계·소득·주택 파생 (CSV 경로는 SIU병원·의사 노출 제외, train_test 누수 방지)"

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

        if "RESI_COST" in X_out.columns:
            X_out["RESI_COST"] = X_out["RESI_COST"].replace(0, pd.NA)

        for c in NUM_COLS:
            if c in X_out.columns:
                X_out[c] = pd.to_numeric(X_out[c], errors="coerce")

        if all(c in X_out.columns for c in ("OCCP_GRP_1", "AGE", "CUST_INCM")):
            gim = GroupMeanImputer()
            gim.fit(X_out, y)
            X_out = gim.transform(X_out)

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
            X_out[num_cols] = X_out[num_cols].fillna(med2).fillna(0.0).astype(np.float64)

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
            if "RESI_COST" in z.columns:
                z["RESI_COST"] = z["RESI_COST"].replace(0, pd.NA)
            for c in NUM_COLS:
                if c in z.columns:
                    z[c] = pd.to_numeric(z[c], errors="coerce")
            return z

        tr = _path(X_train, _y_binary_int(y_train))
        te_y = pd.Series(0, index=X_test.index, dtype=int)
        te = _path(X_test, te_y)

        if all(c in tr.columns for c in ("OCCP_GRP_1", "AGE", "CUST_INCM")):
            gim = GroupMeanImputer()
            gim.fit(tr, y_train)
            tr = gim.transform(tr)
            te = gim.transform(te)

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
            tr[num_cols] = tr[num_cols].fillna(med2).fillna(0.0).astype(np.float64)
            te[num_cols] = te[num_cols].apply(pd.to_numeric, errors="coerce")
            te[num_cols] = te[num_cols].fillna(med2).fillna(0.0).astype(np.float64)

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
