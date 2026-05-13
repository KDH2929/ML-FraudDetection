from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CAT_COLS, ID_COL, NUM_COLS
from src.preprocessing.components.encoder import TargetEncoder
from src.preprocessing.components.missing_value import GroupMeanImputer, MedianImputer, ModeImputer
from src.preprocessing.components.outlier import IQRCapper
from src.preprocessing.components.scaler import RobustScalerWrapper
from src.preprocessing.strategies.base_strategy import BaseStrategy


def _y_to_float01(y: pd.Series) -> pd.Series:
    """타깃 인코딩용: Y/N → 1.0/0.0, 이미 숫자면 그대로."""
    if y is None:
        return pd.Series(dtype=float)
    if pd.api.types.is_numeric_dtype(y):
        return pd.to_numeric(y, errors="coerce").fillna(0.0)
    m = y.astype(str).str.upper().str.strip()
    return m.map({"Y": 1.0, "N": 0.0}).fillna(0.0)


def claim_to_customer_table(claim_df: pd.DataFrame) -> pd.DataFrame:
    """
    클레임(여러 행) → 고객당 한 행 요약.
    컬럼이 없으면 그 단계는 건너뜁니다.
    """
    if claim_df is None or claim_df.empty:
        return pd.DataFrame()

    if ID_COL not in claim_df.columns:
        raise ValueError(f"클레임 데이터에 '{ID_COL}' 컬럼이 필요합니다.")

    # 0) 고객별 청구 건수로 뼈대 만들기 (인덱스 = 고객 ID)
    agg = claim_df.groupby(ID_COL, sort=False).size().to_frame("claim_cnt")

    # 1) 평균 입원·통원 일수 (컬럼 이름이 데이터마다 다를 수 있어서 순서대로 확인)
    for col in ("VLID_HOSP_OTDA", "HSPZ_DAYS", "HOST_DAYS"):
        if col in claim_df.columns:
            agg["mean_hosp_days"] = claim_df.groupby(ID_COL, sort=False)[col].mean()
            break

    # 2) 집–병원 거리 평균
    if "HOUSE_HOSP_DIST" in claim_df.columns:
        agg["mean_hosp_dist"] = claim_df.groupby(ID_COL, sort=False)["HOUSE_HOSP_DIST"].mean()

    # 3) 유의병원: Y면 1, 나머지는 0 → 고객별로 한 번이라도 1이면 1
    heed_col = None
    if "HEED_HOSP_YN" in claim_df.columns:
        heed_col = "HEED_HOSP_YN"
    elif "SUSPCT_HOSP_YN" in claim_df.columns:
        heed_col = "SUSPCT_HOSP_YN"
    if heed_col is not None:
        yn = claim_df[heed_col].astype(str).str.upper().str.strip()
        flag = (yn == "Y").astype(int)
        agg["is_heed_hosp"] = flag.groupby(claim_df[ID_COL], sort=False).max()

    # 4) 청구액·지급액 합, 지급 비율
    if "DMND_AMT" in claim_df.columns and "PAYM_AMT" in claim_df.columns:
        g = claim_df.groupby(ID_COL, sort=False)
        agg["sum_dmnd_amt"] = g["DMND_AMT"].sum()
        agg["sum_paym_amt"] = g["PAYM_AMT"].sum()
        agg["paym_rate"] = agg["sum_paym_amt"] / agg["sum_dmnd_amt"].replace(0, pd.NA)
        agg["paym_rate"] = agg["paym_rate"].fillna(0)

    # 5) 서로 다른 병원 개수
    hosp_col = "HOSP_CODE" if "HOSP_CODE" in claim_df.columns else "HOSP_CD"
    if hosp_col in claim_df.columns:
        agg["nunique_hosp"] = claim_df.groupby(ID_COL, sort=False)[hosp_col].nunique()

    # 6) 사고 구분 비율 (고객별로 합이 1이 되는 비율)
    if "ACCI_DVSN" in claim_df.columns:
        acci_pct = pd.crosstab(claim_df[ID_COL], claim_df["ACCI_DVSN"], normalize="index")
        acci_pct = acci_pct.add_prefix("acci_pct_")
        agg = agg.join(acci_pct, how="left")

    # 인덱스였던 고객 ID를 일반 컬럼으로 빼기
    return agg.reset_index()


def _merge_claim_features(X: pd.DataFrame, claim_df: pd.DataFrame | None) -> pd.DataFrame:
    X_out = X.copy()
    if claim_df is not None and len(claim_df) > 0:
        if ID_COL not in X_out.columns:
            raise ValueError(
                f"claim_df로 병합하려면 X에 '{ID_COL}' 컬럼이 포함되어야 합니다."
            )
        agg = claim_to_customer_table(claim_df)
        if not agg.empty:
            X_out = X_out.merge(agg, on=ID_COL, how="left")
            new_cols = [c for c in agg.columns if c != ID_COL]
            if new_cols:
                X_out[new_cols] = X_out[new_cols].fillna(0)
    return X_out


# 보조 직업·FP 경력: A 파이프라인에서 제외(요구사항·결측/고카디널리티 정리).
_UNUSED_CUST_FOR_A_PIPELINE = ("OCCP_GRP_2", "MATE_OCCP_GRP_2", "FP_CAREER")


def _drop_unused_cust_columns(df: pd.DataFrame) -> pd.DataFrame:
    drops = [c for c in _UNUSED_CUST_FOR_A_PIPELINE if c in df.columns]
    if not drops:
        return df
    return df.drop(columns=drops)


def _fill_residual_missing_for_ml(X: pd.DataFrame) -> pd.DataFrame:
    """processed_data 검증: ID 제외 모든 열을 float로. 수치=중앙값 대치, 그 외=숫자 파싱 또는 factorize."""
    out = X.copy()
    skip = {ID_COL}
    n = len(out)
    for c in list(out.columns):
        if c in skip:
            continue
        col = out[c]
        if pd.api.types.is_numeric_dtype(col):
            m = col.median()
            fill = 0.0 if pd.isna(m) else float(m)
            out[c] = pd.to_numeric(col, errors="coerce").fillna(fill)
            continue
        coerced = pd.to_numeric(col, errors="coerce")
        if coerced.notna().sum() >= max(1, n // 100):
            m = coerced.median()
            fill = 0.0 if pd.isna(m) else float(m)
            out[c] = coerced.fillna(fill).astype(np.float64)
        else:
            codes, _ = pd.factorize(col, use_na_sentinel=True)
            s = pd.Series(codes, index=out.index, dtype=np.float64)
            out[c] = s.mask(s < 0, 0.0)
    return out


class MemberAStrategy(BaseStrategy):

    def get_strategy_name(self) -> str:
        return "A: merge → 특수값 → 결측 → 타깃인코딩 → IQR → Robust"

    def preprocess(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        claim_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        X_out = _merge_claim_features(X, claim_df)
        X_out = _drop_unused_cust_columns(X_out)

        # --- 2) 도메인 특수값 (집값 0 = 추정 불가 → 결측으로 두고 아래에서 채움) ---
        if "RESI_COST" in X_out.columns:
            X_out["RESI_COST"] = X_out["RESI_COST"].replace(0, pd.NA)

        # config에 적힌 수치형은 미리 숫자 dtype으로 통일 (0→NA 이후 object로 남는 경우 방지)
        for c in NUM_COLS:
            if c in X_out.columns:
                X_out[c] = pd.to_numeric(X_out[c], errors="coerce")

        # --- 3) 소득 결측: 직업·나이 그룹 평균 (컬럼이 모두 있을 때만) ---
        if all(c in X_out.columns for c in ("OCCP_GRP_1", "AGE", "CUST_INCM")):
            gim = GroupMeanImputer()
            gim.fit(X_out, y)
            X_out = gim.transform(X_out)

        # --- 4) 수치형 결측: 중앙값 (고객 ID 제외한 모든 숫자 컬럼) ---
        num_cols = [
            c
            for c in X_out.columns
            if c != ID_COL and pd.api.types.is_numeric_dtype(X_out[c])
        ]
        if num_cols:
            med = MedianImputer(cols=num_cols)
            med.fit(X_out, y)
            X_out = med.transform(X_out)
            # sklearn은 pandas의 pd.NA를 싫어해서 float NaN으로 맞춤
            X_out[num_cols] = X_out[num_cols].apply(pd.to_numeric, errors="coerce")
            med2 = X_out[num_cols].median()
            X_out[num_cols] = X_out[num_cols].fillna(med2).fillna(0.0).astype(np.float64)

        # --- 5) 범주형 결측: 최빈값 ---
        cat_cols = [c for c in CAT_COLS if c in X_out.columns]
        if cat_cols:
            mode_imp = ModeImputer(cols=cat_cols)
            mode_imp.fit(X_out, y)
            X_out = mode_imp.transform(X_out)

        # --- 6) 범주형 → 숫자 (타깃 평균 인코딩). y가 없으면 건너뜀 ---
        if cat_cols and y is not None and len(y) == len(X_out):
            y01 = _y_to_float01(y).reindex(X_out.index)
            te = TargetEncoder(cols=cat_cols)
            te.fit(X_out, y01)
            X_out = te.transform(X_out)

        # --- 7) 이상치: 설정된 수치형만 IQR로 캡 ---
        iqr_cols = [c for c in NUM_COLS if c in X_out.columns]
        if iqr_cols:
            cap = IQRCapper(cols=iqr_cols)
            cap.fit(X_out, y)
            X_out = cap.transform(X_out)

        # --- 8) 스케일: 같은 수치형만 Robust (트리에 필수는 아니지만 비교용으로 유지) ---
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
        """
        train에만 통계를 맞춘 뒤 test에는 동일 규칙만 적용 (타깃 인코딩·중앙값 등 누수 완화).
        """
        tr = _merge_claim_features(X_train, claim_df)
        te = _merge_claim_features(X_test, claim_df)
        tr = _drop_unused_cust_columns(tr)
        te = _drop_unused_cust_columns(te)

        for Xs in (tr, te):
            if "RESI_COST" in Xs.columns:
                Xs["RESI_COST"] = Xs["RESI_COST"].replace(0, pd.NA)
            for c in NUM_COLS:
                if c in Xs.columns:
                    Xs[c] = pd.to_numeric(Xs[c], errors="coerce")

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
