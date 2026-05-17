from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CAT_COLS, ID_COL, NUM_COLS
from src.preprocessing.components.encoder import TargetEncoder
from src.preprocessing.components.missing_value import GroupMeanImputer, MedianImputer, ModeImputer
from src.preprocessing.components.outlier import IQRCapper
from src.preprocessing.components.scaler import RobustScalerWrapper
from src.preprocessing.strategies.base_strategy import BaseStrategy
from src.preprocessing.stage_cache import (
    combined_fingerprint,
    dataframe_fingerprint,
    file_fingerprint,
    load_or_build_stage,
)


_A_SHARED_CACHE_ID = "member_a_shared"
_A1_SOURCE_FILES = (Path(__file__).resolve(),)


def _source_fingerprint(paths: tuple[Path, ...]) -> str:
    return combined_fingerprint(file_fingerprint(p) for p in paths)


# 결측 자체가 MNAR 신호인 컬럼 2개만 유지:
# - MATE_OCCP_GRP_1 (~52.8%): 결측 = 배우자 없음 (미혼/이혼/사별) — 구조적 MNAR
# - MINCRDT        (~42.3%): 결측 = 신용 이력 없음 (금융 비주류 계층) — MNAR
#   (MAXCRDT 는 MINCRDT 와 결측 행 완전 동일 → 중복 제거)
# 나머지 제외 근거:
#   MAX_PRM/MAX_PAYM_YM/TOTALPREM: MAR (상품 유형 조건부)
#   WEDD_YN/CHLD_CNT/LTBN_CHLD_AGE: MCAR (수집 채널 오류, 2.1% 동일 비율)
KEY_MISSING_COLS = (
    "MATE_OCCP_GRP_1",
    "MINCRDT",
)


def _add_key_missing_indicators(X: pd.DataFrame) -> pd.DataFrame:
    """MNAR 확실한 2개 컬럼에만 IS_MISSING_* 지시변수 추가.

    호출 위치: merge/drop 직후, imputation 이전. dtype은 모두 int(0/1).
    누수 안전성: .isna() 만 사용하므로 train/test fitting 불필요.
    """
    out = X.copy()
    for col in KEY_MISSING_COLS:
        if col in out.columns:
            out[f"IS_MISSING_{col}"] = out[col].isna().astype(int)
    return out


# 극단값 자체가 SIU 신호인 컬럼들 (도메인 직관 기반):
# - 청구 건수/금액/병원수가 비정상적으로 높음 = SIU 의심
# - 의사·진단명·과목 다양성이 비정상적으로 큼 = 다기관 doctor-shopping
# - 청구 강도(단위시간 청구 횟수)가 비정상적 = 단기 burst
# 캡핑은 하지 않고 IS_OUTLIER_*_HIGH 플래그만 추가 → 모델이 원본+플래그 결합 학습.
# UPPER 만 표기 (LOWER 는 SIU 신호 아님 — 청구 적은 일반인).
HIGH_SIGNAL_OUTLIER_COLS = (
    # v1 baseline (claim_to_customer_table)
    "claim_cnt",
    "sum_dmnd_amt",
    "nunique_hosp",
    # v3 ma3_*
    "ma3_max_dmnd_amt",
    "ma3_max_nonpay_ratio",
    "ma3_nunique_acci_dvsn",
    # v4 claim_agg_*
    "claim_agg_n_doc",
    "claim_agg_dsas_nunique",
    "claim_agg_hosp_switch_count",
    "claim_agg_claim_intensity",
)


def _compute_upper_bounds(df: pd.DataFrame) -> dict[str, float]:
    """HIGH_SIGNAL_OUTLIER_COLS 의 IQR upper bound (Q3 + 1.5*IQR) 사전 계산.

    NaN 은 분위수 계산에서 자연 제외됨. 데이터에 없는 컬럼은 dict 에서 누락.
    """
    bounds: dict[str, float] = {}
    for col in HIGH_SIGNAL_OUTLIER_COLS:
        if col not in df.columns:
            continue
        num = pd.to_numeric(df[col], errors="coerce")
        q1 = num.quantile(0.25)
        q3 = num.quantile(0.75)
        if pd.isna(q1) or pd.isna(q3):
            continue
        iqr = q3 - q1
        bounds[col] = float(q3 + 1.5 * iqr)
    return bounds


def _apply_upper_outlier_flags(df: pd.DataFrame, bounds: dict[str, float]) -> pd.DataFrame:
    """주어진 upper bound 로 IS_OUTLIER_<col>_HIGH 플래그 컬럼 추가."""
    out = df.copy()
    for col, hi in bounds.items():
        if col not in out.columns:
            continue
        num = pd.to_numeric(out[col], errors="coerce")
        out[f"IS_OUTLIER_{col}_HIGH"] = (num > hi).fillna(False).astype(int)
    return out


def _add_high_signal_outlier_flags(X: pd.DataFrame) -> pd.DataFrame:
    """단일 preprocess 경로: 전체 X 로 IQR 적합 후 IS_OUTLIER_*_HIGH 플래그 추가.

    원본 컬럼은 캡핑하지 않음. 플래그는 binary(0/1).
    호출 위치: imputation 직후, IQRCapper(NUM_COLS 캡핑) 직전.
    """
    bounds = _compute_upper_bounds(X)
    return _apply_upper_outlier_flags(X, bounds)


def _add_high_signal_outlier_flags_train_test(
    tr: pd.DataFrame, te: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """train/test 경로: train 으로만 IQR 적합, test 에 동일 bound 적용 (누수 방지)."""
    bounds = _compute_upper_bounds(tr)
    return _apply_upper_outlier_flags(tr, bounds), _apply_upper_outlier_flags(te, bounds)


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


def _build_member_a_raw_features(
    X: pd.DataFrame,
    claim_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Build the shared v1 raw feature frame before fitted transforms."""
    X_out = _merge_claim_features(X, claim_df)
    X_out = _drop_unused_cust_columns(X_out)
    X_out = _add_key_missing_indicators(X_out)
    return X_out


def _load_or_build_member_a_raw_features(
    X: pd.DataFrame,
    claim_df: pd.DataFrame | None,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    if not use_cache:
        return _build_member_a_raw_features(X, claim_df)

    fp = combined_fingerprint(
        (
            "a1_raw_features_v1",
            dataframe_fingerprint(X),
            dataframe_fingerprint(claim_df),
            _source_fingerprint(_A1_SOURCE_FILES),
        )
    )
    return load_or_build_stage(
        strategy_id=_A_SHARED_CACHE_ID,
        stage_name="a1_raw_features",
        fingerprint=fp,
        builder=lambda: _build_member_a_raw_features(X, claim_df),
    )


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
        X_out = _load_or_build_member_a_raw_features(X, claim_df)

        # --- 1.5) 결측 지시변수: 결측 자체가 신호인 컬럼들 (imputation 이전에 캡처) ---
        X_out = _add_key_missing_indicators(X_out)

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

        # 참고: 도메인 이상치 플래그 (`_add_high_signal_outlier_flags`) 는 helper 로 남겨뒀지만
        # LightGBM 이 split 으로 임계값을 학습하므로 redundant — F1 평가에서 5개 버전 모두 하락 확인됨.
        # 호출은 비활성. 실험 시 헬퍼를 직접 끼워 쓰면 됨.

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

        # 결측 지시변수: imputation 이전, train/test 동일 규칙으로 적용 (.isna() 만 사용)
        tr = _add_key_missing_indicators(tr)
        te = _add_key_missing_indicators(te)

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

        # 참고: `_add_high_signal_outlier_flags_train_test` 도 비활성 (위의 사유 동일).

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
