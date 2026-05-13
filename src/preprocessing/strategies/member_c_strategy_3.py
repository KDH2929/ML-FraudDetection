from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import DIVIDED_SET_COL, ID_COL, TARGET_COL
from src.pipeline.data_loader import normalize_target
from src.preprocessing.components.encoder import OneHotEncoder
from src.preprocessing.components.features.member_c_features import MemberCV3FeaturePipeline
from src.preprocessing.components.missing_value import MedianImputer, ZeroImputer
from src.preprocessing.components.outlier import QuantileCapper
from src.preprocessing.strategies.base_strategy import BaseStrategy


class MemberCStrategy3(BaseStrategy):
    """
    member_c v3 전처리 전략

    핵심 원칙
    1. 고객이 최종 예측 단위이므로 CLAIM_DATA는 반드시 고객 단위로 집계한다.
    2. 결측치 대체, 인코딩, cap, scaler는 train(DIVIDED_SET == 1)에서만 fit한다.
    3. 지급/심사 결과와 target encoding 계열은 옵션으로만 분리해 safe 버전 누수를 막는다.
    4. 파생변수는 청구 반복성, 다양성, 금액, 간격, 경제적 부담, 보험 유지 기간 중심으로 제한한다.
    """

    HIGH_MISSING_CLAIM_COLS = [
        "DCAF_CMPS_XCPA",
        "COUNT_TRMT_ITEM",
        "TAMT_SFCA",
        "NON_PAY",
        "DSCT_AMT",
        "PATT_CHRG_TOTA",
        "SELF_CHAM",
    ]

    CUSTOMER_DROP_COLS = ["FP_CAREER", "OCCP_GRP_2", "MATE_OCCP_GRP_2"]
    CUSTOMER_DATE_COLS = ["CUST_RGST", "MAX_PAYM_YM"]
    CLAIM_DATE_COLS = [
        "RECP_DATE",
        "ORIG_RESN_DATE",
        "RESN_DATE",
        "HOSP_OTPA_STDT",
        "HOSP_OTPA_ENDT",
        "PAYM_DATE",
    ]
    CLAIM_LEAKAGE_COLS = [
        "PAYM_AMT",
        "PAYM_DATE",
        "RESL_CD1",
        "RESL_NM1",
        "PMMI_DLNG_YN",
        "CRNT_PROG_DVSN",
    ]
    HIGH_CARD_COLS = [
        "HOSP_CODE",
        "CHME_LICE_NO",
        "DSAS_NAME",
        "CAUS_CODE",
        "CAUS_CODE_DTAL",
        "ACCI_HOSP_ADDR",
    ]
    FREQ_COLS = ["HOSP_CODE", "CHME_LICE_NO", "DSAS_NAME", "CAUS_CODE", "ACCI_HOSP_ADDR"]
    CUSTOMER_OHE_COLS = [
        "SEX",
        "RESI_TYPE_CODE",
        "CTPR",
        "OCCP_GRP_1",
        "WEDD_YN",
        "MATE_OCCP_GRP_1",
        "AGE_GROUP",
        "LTBN_CHLD_AGE_GROUP",
    ]
    TARGET_ENCODING_ALPHA = 10.0

    def __init__(
        self,
        include_leakage_features: bool = False,
        include_target_encoded_features: bool = False,
        scale_numeric: bool = False,
        target_encode_n_splits: int = 5,
        target_encode_alpha: float = TARGET_ENCODING_ALPHA,
    ):
        self.include_leakage_features = include_leakage_features
        self.include_target_encoded_features = include_target_encoded_features
        self.scale_numeric = scale_numeric
        self.target_encode_n_splits = int(target_encode_n_splits)
        self.target_encode_alpha = float(target_encode_alpha)

    def get_strategy_name(self) -> str:
        parts = ["C3", "safe"]
        if self.include_leakage_features:
            parts.append("leakage")
        if self.include_target_encoded_features:
            parts.append(f"te{self.target_encode_n_splits}")
        if self.scale_numeric:
            parts.append("scaled")
        return " ".join(parts)

    def preprocess(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        claim_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        pipeline = MemberCV3FeaturePipeline()
        return pipeline.run(self, X, y, claim_df)

        if claim_df is None:
            raise ValueError("member_c_strategy_3 는 claim_df가 필요합니다.")

        cust = X.copy()
        claim = claim_df.copy()
        target = normalize_target(y).reindex(cust.index)

        if DIVIDED_SET_COL not in cust.columns:
            raise ValueError("member_c_strategy_3 는 DIVIDED_SET 컬럼이 필요합니다.")

        train_mask = cust[DIVIDED_SET_COL].astype("string").str.strip().eq("1") & target.notna()
        test_mask = cust[DIVIDED_SET_COL].astype("string").str.strip().eq("2")
        train_customer_ids = set(cust.loc[train_mask, ID_COL].tolist())

        report = {
            "include_target_encoded_features": self.include_target_encoded_features,
            "include_leakage_features": self.include_leakage_features,
            "scale_numeric": self.scale_numeric,
            "deleted_customer_columns": [c for c in self.CUSTOMER_DROP_COLS if c in cust.columns],
            "deleted_customer_date_columns": [c for c in self.CUSTOMER_DATE_COLS if c in cust.columns],
        }

        cust_processed, cust_report = self._prepare_customer_data(cust, train_mask)
        claim_agg, claim_report = self._prepare_claim_data(claim, cust_processed, target, train_mask, train_customer_ids)
        report.update(cust_report)
        report.update(claim_report)

        merged = cust_processed.merge(claim_agg, on=ID_COL, how="left")
        claim_cols = [c for c in claim_agg.columns if c != ID_COL]
        if claim_cols:
            merged = ZeroImputer(cols=claim_cols).transform(merged)

        merged = self._add_cross_customer_claim_features(merged)

        report["missing_before_final_fill"] = int(merged.isna().sum().sum())

        ohe_cols = [c for c in self.CUSTOMER_OHE_COLS if c in merged.columns]
        ohe_dummy_cols: list[str] = []
        if ohe_cols:
            encoder = OneHotEncoder(cols=ohe_cols)
            encoder.fit(merged.loc[train_mask])
            ohe_dummy_cols = list(encoder.dummy_cols_ or [])
            merged = encoder.transform(merged)

        numeric_cols = [
            c
            for c in merged.columns
            if c not in {ID_COL}
            and pd.api.types.is_numeric_dtype(merged[c])
        ]
        median_imputer = MedianImputer(cols=numeric_cols)
        median_imputer.fit(merged.loc[train_mask, numeric_cols])
        merged = median_imputer.transform(merged)

        cap_excluded = self._cap_excluded_columns(merged, ohe_dummy_cols)
        cap_cols = [c for c in numeric_cols if c not in cap_excluded]
        capper = QuantileCapper(cols=cap_cols, lower=0.01, upper=0.99)
        capper.fit(merged.loc[train_mask, cap_cols])
        merged = capper.transform(merged)

        if self.scale_numeric:
            scale_cols = [c for c in self._scale_target_columns() if c in merged.columns]
            scale_cols = [c for c in scale_cols if c not in cap_excluded]
            if scale_cols:
                scaler = StandardScaler()
                scaler.fit(merged.loc[train_mask, scale_cols])
                merged[scale_cols] = scaler.transform(merged[scale_cols])
        report["standard_scaler_applied"] = self.scale_numeric

        bool_cols = merged.select_dtypes(include="bool").columns.tolist()
        if bool_cols:
            merged[bool_cols] = merged[bool_cols].astype(int)

        non_numeric_cols = merged.select_dtypes(exclude="number").columns.tolist()
        if non_numeric_cols:
            merged = merged.drop(columns=non_numeric_cols)

        feature_cols = [c for c in merged.columns if c not in {ID_COL, DIVIDED_SET_COL}]
        train_df = merged.loc[train_mask].copy()
        test_df = merged.loc[test_mask].copy()
        x_train = train_df[feature_cols]
        y_train = target.loc[train_mask].astype(int)
        x_test = test_df[feature_cols]

        report["missing_after_final_fill"] = int(merged[feature_cols].isna().sum().sum())
        report["remaining_missing_total"] = int(merged.isna().sum().sum())
        report["remaining_string_columns"] = merged.select_dtypes(exclude="number").columns.tolist()
        if feature_cols:
            train_var = x_train.var(numeric_only=True)
            report["near_zero_variance_columns"] = train_var[train_var <= 1e-8].index.tolist()
            final_missing_rate = x_train.isna().mean()
            report["high_missing_rate_columns"] = (
                final_missing_rate[final_missing_rate > 0.1].sort_values(ascending=False).round(4).to_dict()
            )
        else:
            report["near_zero_variance_columns"] = []
            report["high_missing_rate_columns"] = {}

        report["created_customer_features"] = self._created_customer_features()
        report["created_claim_features"] = self._created_claim_features(merged.columns)
        report["excluded_variables_and_reasons"] = self._excluded_variables_and_reasons()
        report["target_encoding_leakage_prevention"] = self._target_encoding_leakage_prevention_text()
        report["final_managed_shape"] = list(merged.shape)
        report["x_train_shape"] = list(x_train.shape)
        report["y_train_shape"] = [int(y_train.shape[0])]
        report["x_test_shape"] = list(x_test.shape)
        report["target_distribution"] = {
            "0": int((y_train == 0).sum()),
            "1": int((y_train == 1).sum()),
        }
        pos_cnt = int((y_train == 1).sum())
        neg_cnt = int((y_train == 0).sum())
        report["imbalance_ratio"] = round(float(neg_cnt / pos_cnt), 4) if pos_cnt else None
        report["final_save_file_name"] = "member_c_strategy_3_preprocessed.csv"

        self._print_report(report)
        return merged

    def _prepare_customer_data(self, cust: pd.DataFrame, train_mask: pd.Series):
        report: dict[str, object] = {}
        df = cust.copy()

        df = df.drop(columns=[c for c in self.CUSTOMER_DROP_COLS if c in df.columns], errors="ignore")

        numeric_cols = [
            "AGE",
            "RESI_COST",
            "TOTALPREM",
            "MINCRDT",
            "MAXCRDT",
            "CHLD_CNT",
            "LTBN_CHLD_AGE",
            "MAX_PRM",
            "CUST_INCM",
            "RCBASE_HSHD_INCM",
            "JPBASE_HSHD_INCM",
        ]
        self._coerce_numeric(df, numeric_cols)

        for col in ["MINCRDT", "MAXCRDT"]:
            if col in df.columns:
                df[f"{col}_is_99"] = df[col].eq(99).astype(int)
                df.loc[df[col].eq(99), col] = np.nan

        date_parse = {}
        for col in self.CUSTOMER_DATE_COLS:
            if col in df.columns:
                date_parse[col] = self._parse_year_month(df[col])
                df[col] = date_parse[col]

        indicator_map = {
            "AGE": "AGE_isna",
            "CHLD_CNT": "CHLD_CNT_isna",
            "LTBN_CHLD_AGE": "LTBN_CHLD_AGE_isna",
            "CUST_INCM": "CUST_INCM_isna",
            "TOTALPREM": "TOTALPREM_isna",
            "MAX_PRM": "MAX_PRM_isna",
            "MAXCRDT": "MAXCRDT_isna",
            "MINCRDT": "MINCRDT_isna",
            "JPBASE_HSHD_INCM": "JPBASE_HSHD_INCM_isna",
            "RCBASE_HSHD_INCM": "RCBASE_HSHD_INCM_isna",
            "RESI_COST": "RESI_COST_isna",
            "RESI_TYPE_CODE": "RESI_TYPE_CODE_isna",
            "CTPR": "CTPR_isna",
            "OCCP_GRP_1": "OCCP_GRP_1_isna",
            "WEDD_YN": "WEDD_YN_isna",
            "MATE_OCCP_GRP_1": "MATE_OCCP_GRP_1_isna",
            "CUST_RGST": "CUST_RGST_isna",
            "MAX_PAYM_YM": "MAX_PAYM_YM_isna",
        }
        for source, indicator in indicator_map.items():
            if source in df.columns:
                df[indicator] = df[source].isna().astype(int)

        train_df = df.loc[train_mask].copy()

        for col in ["SEX", "RESI_TYPE_CODE", "CTPR", "OCCP_GRP_1", "MATE_OCCP_GRP_1"]:
            if col in df.columns:
                mode_value = self._mode_from_train(train_df[col], default="0")
                df[col] = self._canonical_category(df[col]).fillna(mode_value)
                train_df[col] = self._canonical_category(train_df[col]).fillna(mode_value)

        if "WEDD_YN" in df.columns:
            df["WEDD_YN"] = self._canonical_category(df["WEDD_YN"]).fillna("N")
            train_df["WEDD_YN"] = self._canonical_category(train_df["WEDD_YN"]).fillna("N")

        if "AGE" in df.columns:
            age_median = float(train_df["AGE"].median())
            df["AGE"] = df["AGE"].fillna(age_median)
            train_df["AGE"] = train_df["AGE"].fillna(age_median)

        if "AGE" in df.columns:
            df["AGE_GROUP"] = self._age_group(df["AGE"])
            train_df["AGE_GROUP"] = self._age_group(train_df["AGE"])

        if "CHLD_CNT" in df.columns:
            df["CHLD_CNT"] = df["CHLD_CNT"].fillna(0)
            train_df["CHLD_CNT"] = train_df["CHLD_CNT"].fillna(0)
        if "LTBN_CHLD_AGE" in df.columns:
            df["LTBN_CHLD_AGE"] = df["LTBN_CHLD_AGE"].fillna(0)
            train_df["LTBN_CHLD_AGE"] = train_df["LTBN_CHLD_AGE"].fillna(0)
            df["LTBN_CHLD_AGE_GROUP"] = self._child_age_group(df["LTBN_CHLD_AGE"])
            train_df["LTBN_CHLD_AGE_GROUP"] = self._child_age_group(train_df["LTBN_CHLD_AGE"])

        for col, groups in [
            ("CUST_INCM", [["OCCP_GRP_1", "AGE_GROUP"], ["OCCP_GRP_1"], ["AGE_GROUP"], []]),
            ("JPBASE_HSHD_INCM", [["OCCP_GRP_1"], ["AGE_GROUP"], []]),
            ("RCBASE_HSHD_INCM", [["CTPR"], ["OCCP_GRP_1"], []]),
            ("RESI_COST", [["CTPR", "RESI_TYPE_CODE"], ["CTPR"], ["RESI_TYPE_CODE"], []]),
            ("TOTALPREM", [["AGE_GROUP", "OCCP_GRP_1"], ["OCCP_GRP_1"], ["AGE_GROUP"], []]),
            ("MINCRDT", [["OCCP_GRP_1"], ["AGE_GROUP"], []]),
            ("MAXCRDT", [["OCCP_GRP_1"], ["AGE_GROUP"], []]),
        ]:
            if col in df.columns:
                df[col] = self._hierarchical_group_median_fill(df, train_mask, col, groups)

        if "MAX_PRM" in df.columns:
            if "TOTALPREM" in df.columns:
                df["_TOTALPREM_BIN"] = self._make_totalprem_bins(df["TOTALPREM"], train_mask)
            groups = [["_TOTALPREM_BIN"], ["AGE_GROUP", "OCCP_GRP_1"], ["OCCP_GRP_1"], []]
            df["MAX_PRM"] = self._hierarchical_group_median_fill(df, train_mask, "MAX_PRM", groups)
            df = df.drop(columns=["_TOTALPREM_BIN"], errors="ignore")

        for col in ["RESI_TYPE_CODE", "CTPR", "OCCP_GRP_1", "MATE_OCCP_GRP_1"]:
            if col in df.columns:
                mode_value = self._mode_from_train(df.loc[train_mask, col], default="0")
                df[col] = df[col].fillna(mode_value)

        if {"CUST_RGST", "MAX_PAYM_YM"}.issubset(df.columns):
            df["payment_active_months"] = self._month_diff(df["CUST_RGST"], df["MAX_PAYM_YM"]).fillna(0)
            df["register_year"] = df["CUST_RGST"].dt.year.fillna(0)
            df["register_month"] = df["CUST_RGST"].dt.month.fillna(0)
        else:
            df["payment_active_months"] = 0
            df["register_year"] = 0
            df["register_month"] = 0
        df["payment_active_years"] = self._safe_divide(df["payment_active_months"], 12).fillna(0)

        df["has_child"] = df["CHLD_CNT"].fillna(0).gt(0).astype(int)

        for src, tgt in [
            ("CUST_INCM", "CUST_INCM_log"),
            ("TOTALPREM", "TOTALPREM_log"),
            ("MAX_PRM", "MAX_PRM_log"),
            ("RESI_COST", "RESI_COST_log"),
            ("RCBASE_HSHD_INCM", "RCBASE_HSHD_INCM_log"),
            ("JPBASE_HSHD_INCM", "JPBASE_HSHD_INCM_log"),
        ]:
            if src in df.columns:
                df[tgt] = np.log1p(df[src].clip(lower=0))

        if {"TOTALPREM", "CUST_INCM"}.issubset(df.columns):
            df["premium_income_ratio"] = self._safe_divide(df["TOTALPREM"], df["CUST_INCM"])
        if {"MAX_PRM", "TOTALPREM"}.issubset(df.columns):
            df["max_prm_totalprem_ratio"] = self._safe_divide(df["MAX_PRM"], df["TOTALPREM"])
        if {"RESI_COST", "CUST_INCM"}.issubset(df.columns):
            df["resi_cost_income_ratio"] = self._safe_divide(df["RESI_COST"], df["CUST_INCM"])
        if {"RCBASE_HSHD_INCM", "JPBASE_HSHD_INCM"}.issubset(df.columns):
            df["household_income_gap"] = df["RCBASE_HSHD_INCM"] - df["JPBASE_HSHD_INCM"]
        if {"MAXCRDT", "MINCRDT"}.issubset(df.columns):
            df["credit_gap"] = df["MAXCRDT"] - df["MINCRDT"]
            df["credit_mean"] = (df["MAXCRDT"] + df["MINCRDT"]) / 2

        df = df.drop(columns=[c for c in self.CUSTOMER_DATE_COLS if c in df.columns], errors="ignore")

        report["customer_missing_before"] = int(cust.isna().sum().sum())
        report["customer_missing_after"] = int(df.isna().sum().sum())
        return df, report

    def _prepare_claim_data(
        self,
        claim: pd.DataFrame,
        cust_processed: pd.DataFrame,
        target: pd.Series,
        train_mask: pd.Series,
        train_customer_ids: set,
    ):
        report: dict[str, object] = {}
        df = claim.copy()

        missing_rate_before = {
            col: round(float(df[col].isna().mean()), 4)
            for col in self.HIGH_MISSING_CLAIM_COLS + ["NON_PAY_RATIO"]
            if col in df.columns
        }
        drop_cols = [c for c in self.HIGH_MISSING_CLAIM_COLS if c in df.columns]
        df = df.drop(columns=drop_cols, errors="ignore")
        report["deleted_claim_high_missing_columns"] = drop_cols
        report["claim_missing_rate_before_drop"] = missing_rate_before
        report["deleted_claim_date_columns"] = [c for c in self.CLAIM_DATE_COLS if c in claim.columns]

        numeric_cols = [
            "VLID_HOSP_OTDA",
            "HOUSE_HOSP_DIST",
            "DMND_AMT",
            "NON_PAY_RATIO",
            "HOSP_SPEC_DVSN",
            "ACCI_DVSN",
            "DMND_RESN_CODE",
            "HOSP_CODE",
            "CHME_LICE_NO",
        ]
        if self.include_leakage_features:
            numeric_cols += ["PAYM_AMT"]
        self._coerce_numeric(df, numeric_cols)

        for col in self.CLAIM_DATE_COLS:
            if col in df.columns:
                df[col] = self._parse_yyyymmdd(df[col])

        train_claim_mask = df[ID_COL].isin(train_customer_ids)
        train_claim = df.loc[train_claim_mask].copy()

        if {"RECP_DATE", "ORIG_RESN_DATE"}.issubset(df.columns):
            df["delay_origin_to_recp"] = (df["RECP_DATE"] - df["ORIG_RESN_DATE"]).dt.days.clip(lower=0)
        if {"RECP_DATE", "RESN_DATE"}.issubset(df.columns):
            df["delay_resn_to_recp"] = (df["RECP_DATE"] - df["RESN_DATE"]).dt.days.clip(lower=0)
        if self.include_leakage_features and {"PAYM_DATE", "RECP_DATE"}.issubset(df.columns):
            df["delay_recp_to_paym"] = (df["PAYM_DATE"] - df["RECP_DATE"]).dt.days.clip(lower=0)
        if {"HOSP_OTPA_STDT", "HOSP_OTPA_ENDT"}.issubset(df.columns):
            df["hosp_days_calc"] = (df["HOSP_OTPA_ENDT"] - df["HOSP_OTPA_STDT"]).dt.days.clip(lower=0)

        if "RECP_DATE" in df.columns:
            df["recp_month"] = df["RECP_DATE"].dt.month
            df["recp_year"] = df["RECP_DATE"].dt.year
            df["is_weekend_claim"] = df["RECP_DATE"].dt.weekday.isin([5, 6]).astype(int)
            df = df.sort_values([ID_COL, "RECP_DATE"]).copy()
            df["prev_recp_date"] = df.groupby(ID_COL)["RECP_DATE"].shift(1)
            df["claim_interval"] = (df["RECP_DATE"] - df["prev_recp_date"]).dt.days.clip(lower=0)

        if "DMND_AMT" in df.columns:
            df["dmnd_amt_log"] = np.log1p(df["DMND_AMT"].clip(lower=0))
        if self.include_leakage_features and "PAYM_AMT" in df.columns:
            df["paym_amt_log"] = np.log1p(df["PAYM_AMT"].clip(lower=0))
        if "VLID_HOSP_OTDA" in df.columns:
            df["vlid_hosp_otda_log"] = np.log1p(df["VLID_HOSP_OTDA"].clip(lower=0))

        key_map = {}
        for col in [
            "POLY_NO",
            "HOSP_CODE",
            "CHME_LICE_NO",
            "DSAS_NAME",
            "CAUS_CODE",
            "CAUS_CODE_DTAL",
            "ACCI_HOSP_ADDR",
            "ACCI_DVSN",
            "DMND_RESN_CODE",
            "HOSP_SPEC_DVSN",
        ]:
            if col in df.columns:
                key_col = f"__{col}_KEY"
                df[key_col] = self._canonical_category(df[col])
                key_map[col] = key_col

        freq_mapping = {}
        for col in self.FREQ_COLS:
            if col in key_map:
                mapping = self._fit_frequency_mapping(df.loc[train_claim_mask, key_map[col]])
                df[f"__{col}_FREQ"] = df[key_map[col]].map(mapping).fillna(0.0)
                freq_mapping[col] = mapping

        if {"ACCI_DVSN", "DMND_RESN_CODE"}.issubset(key_map):
            df["__ACCI_DMND_COMBO"] = (
                df[key_map["ACCI_DVSN"]].fillna("NA") + "_" + df[key_map["DMND_RESN_CODE"]].fillna("NA")
            )
            combo_mapping = self._fit_frequency_mapping(df.loc[train_claim_mask, "__ACCI_DMND_COMBO"])
            df["__ACCI_DMND_COMBO_FREQ"] = df["__ACCI_DMND_COMBO"].map(combo_mapping).fillna(0.0)
        else:
            combo_mapping = {}

        if self.include_target_encoded_features:
            te_df = self._build_target_encoded_claim_features(df, cust_processed[[ID_COL]], target, train_mask, key_map)
            for col in te_df.columns:
                if col != ID_COL:
                    df = df.merge(te_df[[ID_COL, col]], on=ID_COL, how="left")

        agg = pd.DataFrame({ID_COL: sorted(df[ID_COL].dropna().unique().tolist())})
        claim_count = df.groupby(ID_COL).size().rename("claim_count").reset_index()
        agg = agg.merge(claim_count, on=ID_COL, how="left")
        agg["claim_count_log"] = np.log1p(agg["claim_count"].fillna(0))

        if "POLY_NO" in key_map:
            policy_nunique = df.groupby(ID_COL)[key_map["POLY_NO"]].nunique().rename("policy_nunique").reset_index()
            agg = agg.merge(policy_nunique, on=ID_COL, how="left")
        if {"claim_count", "policy_nunique"}.issubset(agg.columns):
            agg["claim_per_policy"] = self._safe_divide(agg["claim_count"], agg["policy_nunique"])

        if "VLID_HOSP_OTDA" in df.columns:
            hosp_days = (
                df.groupby(ID_COL)["VLID_HOSP_OTDA"]
                .agg(["mean", "sum", "max"])
                .reset_index()
                .rename(columns={"mean": "hosp_days_mean", "sum": "hosp_days_sum", "max": "hosp_days_max"})
            )
            agg = agg.merge(hosp_days, on=ID_COL, how="left")

        if "DMND_AMT" in df.columns:
            dmnd = (
                df.groupby(ID_COL)["DMND_AMT"]
                .agg(["sum", "mean", "max", "std"])
                .reset_index()
                .rename(columns={"sum": "dmnd_sum", "mean": "dmnd_mean", "max": "dmnd_max", "std": "dmnd_std"})
            )
            dmnd["dmnd_sum_log"] = np.log1p(dmnd["dmnd_sum"].clip(lower=0))
            agg = agg.merge(dmnd, on=ID_COL, how="left")

        if "HOUSE_HOSP_DIST" in df.columns:
            dist = (
                df.groupby(ID_COL)["HOUSE_HOSP_DIST"]
                .agg(["mean", "max"])
                .reset_index()
                .rename(columns={"mean": "house_hosp_dist_mean", "max": "house_hosp_dist_max"})
            )
            agg = agg.merge(dist, on=ID_COL, how="left")

        if "HEED_HOSP_YN" in df.columns:
            heed = df["HEED_HOSP_YN"].astype("string").str.upper().eq("Y").astype(int)
            heed_df = pd.DataFrame({ID_COL: df[ID_COL], "_heed": heed})
            heed_agg = heed_df.groupby(ID_COL)["_heed"].agg(["mean", "max"]).reset_index().rename(
                columns={"mean": "heed_hosp_rate", "max": "has_heed_hosp"}
            )
            agg = agg.merge(heed_agg, on=ID_COL, how="left")

        for source, prefix in [
            ("HOSP_CODE", "hospital"),
            ("CHME_LICE_NO", "doctor"),
            ("DSAS_NAME", "disease"),
            ("CAUS_CODE", "cause"),
            ("CAUS_CODE_DTAL", "cause_dtal"),
            ("ACCI_HOSP_ADDR", "addr"),
        ]:
            if source in key_map:
                agg = agg.merge(self._group_nunique(df, key_map[source], f"{prefix}_nunique"), on=ID_COL, how="left")
                if prefix != "cause_dtal":
                    agg = agg.merge(
                        self._group_main_ratio(df, key_map[source], f"main_{prefix}_claim_ratio"),
                        on=ID_COL,
                        how="left",
                    )

        for prefix in ["hospital", "doctor", "disease", "cause", "addr"]:
            nunique_col = f"{prefix}_nunique"
            if {"claim_count", nunique_col}.issubset(agg.columns):
                agg[f"{prefix}_per_claim"] = self._safe_divide(agg[nunique_col], agg["claim_count"])

        for source, prefix in [
            ("HOSP_CODE", "hospital_freq"),
            ("CHME_LICE_NO", "doctor_freq"),
            ("DSAS_NAME", "disease_freq"),
            ("CAUS_CODE", "cause_freq"),
            ("ACCI_HOSP_ADDR", "hospital_area_freq"),
        ]:
            freq_col = f"__{source}_FREQ"
            if freq_col in df.columns:
                freq_agg = (
                    df.groupby(ID_COL)[freq_col]
                    .agg(["mean", "max"])
                    .reset_index()
                    .rename(columns={"mean": f"{prefix}_mean", "max": f"{prefix}_max"})
                )
                agg = agg.merge(freq_agg, on=ID_COL, how="left")

        if "RECP_DATE" in df.columns:
            recp_span = df.groupby(ID_COL)["RECP_DATE"].agg(["min", "max"]).reset_index()
            recp_span["claim_active_days"] = (recp_span["max"] - recp_span["min"]).dt.days.fillna(0)
            recp_span["first_claim_year"] = recp_span["min"].dt.year.fillna(0)
            recp_span["first_claim_month"] = recp_span["min"].dt.month.fillna(0)
            recp_span["last_claim_year"] = recp_span["max"].dt.year.fillna(0)
            recp_span["last_claim_month"] = recp_span["max"].dt.month.fillna(0)
            agg = agg.merge(
                recp_span[
                    [
                        ID_COL,
                        "claim_active_days",
                        "first_claim_year",
                        "first_claim_month",
                        "last_claim_year",
                        "last_claim_month",
                    ]
                ],
                on=ID_COL,
                how="left",
            )

            latest = df.groupby(ID_COL)["RECP_DATE"].transform("max")
            recent_flag = ((latest - df["RECP_DATE"]).dt.days <= 90).fillna(False).astype(int)
            recent_df = pd.DataFrame({ID_COL: df[ID_COL], "_recent": recent_flag})
            recent_cnt = recent_df.groupby(ID_COL)["_recent"].sum().rename("recent_claim_cnt").reset_index()
            agg = agg.merge(recent_cnt, on=ID_COL, how="left")

            recp_nunique = df.groupby(ID_COL).agg(
                recp_month_nunique=("recp_month", "nunique"),
                recp_year_nunique=("recp_year", "nunique"),
                weekend_claim_rate=("is_weekend_claim", "mean"),
            ).reset_index()
            agg = agg.merge(recp_nunique, on=ID_COL, how="left")

        if "claim_interval" in df.columns:
            interval = (
                df.groupby(ID_COL)["claim_interval"]
                .agg(["mean", "min"])
                .reset_index()
                .rename(columns={"mean": "claim_interval_mean", "min": "claim_interval_min"})
            )
            agg = agg.merge(interval, on=ID_COL, how="left")

        if "ACCI_DVSN" in key_map:
            agg = agg.merge(self._group_nunique(df, key_map["ACCI_DVSN"], "acci_dvsn_nunique"), on=ID_COL, how="left")
            agg = agg.merge(
                self._group_main_ratio(df, key_map["ACCI_DVSN"], "main_acci_dvsn_ratio"),
                on=ID_COL,
                how="left",
            )
            if {"acci_dvsn_nunique", "claim_count"}.issubset(agg.columns):
                agg["acci_dvsn_per_claim"] = self._safe_divide(agg["acci_dvsn_nunique"], agg["claim_count"])
            train_unique_acci = (
                df.loc[train_claim_mask, key_map["ACCI_DVSN"]].dropna().astype("string").unique().tolist()
            )
            if len(train_unique_acci) <= 10:
                for value in sorted(train_unique_acci):
                    safe_value = self._safe_token(value)
                    ratio_col = f"acci_dvsn_{safe_value}_ratio"
                    flag = df[key_map["ACCI_DVSN"]].astype("string").eq(value).astype(int)
                    ratio_df = pd.DataFrame({ID_COL: df[ID_COL], ratio_col: flag})
                    ratio_agg = ratio_df.groupby(ID_COL)[ratio_col].mean().reset_index()
                    agg = agg.merge(ratio_agg, on=ID_COL, how="left")

        if {"ACCI_DVSN", "DMND_RESN_CODE"}.issubset(key_map):
            agg = agg.merge(
                self._group_nunique(df, "__ACCI_DMND_COMBO", "acci_dmnd_combo_nunique"),
                on=ID_COL,
                how="left",
            )
            agg = agg.merge(
                self._group_main_ratio(df, "__ACCI_DMND_COMBO", "main_acci_dmnd_combo_ratio"),
                on=ID_COL,
                how="left",
            )
            combo_freq = (
                df.groupby(ID_COL)["__ACCI_DMND_COMBO_FREQ"]
                .agg(["mean", "max"])
                .reset_index()
                .rename(
                    columns={
                        "mean": "acci_dmnd_combo_freq_mean",
                        "max": "acci_dmnd_combo_freq_max",
                    }
                )
            )
            agg = agg.merge(combo_freq, on=ID_COL, how="left")

        if "HOSP_SPEC_DVSN" in key_map:
            agg = agg.merge(
                self._group_nunique(df, key_map["HOSP_SPEC_DVSN"], "hosp_spec_nunique"),
                on=ID_COL,
                how="left",
            )
            agg = agg.merge(
                self._group_main_ratio(df, key_map["HOSP_SPEC_DVSN"], "main_hosp_spec_ratio"),
                on=ID_COL,
                how="left",
            )

        if self.include_target_encoded_features:
            te_cols = ["doc_siu_ratio_max", "hosp_siu_ratio_max", "hosp_doc_siu"]
            te_unique = df[[ID_COL] + [c for c in te_cols if c in df.columns]].drop_duplicates(ID_COL)
            agg = agg.merge(te_unique, on=ID_COL, how="left")

        if self.include_leakage_features:
            if "PAYM_AMT" in df.columns:
                pay = (
                    df.groupby(ID_COL)["PAYM_AMT"]
                    .agg(["sum", "mean", "max"])
                    .reset_index()
                    .rename(columns={"sum": "pay_sum", "mean": "pay_mean", "max": "pay_max"})
                )
                agg = agg.merge(pay, on=ID_COL, how="left")
            if {"PAYM_AMT", "DMND_AMT"}.issubset(df.columns):
                df["pay_ratio"] = self._safe_divide(df["PAYM_AMT"], df["DMND_AMT"])
                df["amount_gap"] = df["PAYM_AMT"] - df["DMND_AMT"]
                leak_agg = (
                    df.groupby(ID_COL)[["pay_ratio", "amount_gap"]]
                    .agg(["mean", "max"])
                    .reset_index()
                )
                leak_agg.columns = [ID_COL, "pay_ratio_mean", "pay_ratio_max", "amount_gap_mean", "amount_gap_max"]
                agg = agg.merge(leak_agg, on=ID_COL, how="left")
            if "delay_recp_to_paym" in df.columns:
                pay_delay = (
                    df.groupby(ID_COL)["delay_recp_to_paym"]
                    .agg(["mean", "max"])
                    .reset_index()
                    .rename(columns={"mean": "delay_recp_to_paym_mean", "max": "delay_recp_to_paym_max"})
                )
                agg = agg.merge(pay_delay, on=ID_COL, how="left")

        numeric_cols = [c for c in agg.columns if c != ID_COL]
        agg[numeric_cols] = agg[numeric_cols].replace([np.inf, -np.inf], np.nan)
        report["claim_missing_before"] = int(claim.isna().sum().sum())
        report["claim_missing_after"] = int(agg.isna().sum().sum())
        return agg, report

    def _add_cross_customer_claim_features(self, df: pd.DataFrame) -> pd.DataFrame:
        merged = df.copy()
        required = {"register_year", "register_month", "first_claim_year", "first_claim_month"}
        if required.issubset(merged.columns):
            valid = (
                merged["register_year"].gt(0)
                & merged["register_month"].gt(0)
                & merged["first_claim_year"].gt(0)
                & merged["first_claim_month"].gt(0)
            )
            months = (
                (merged["first_claim_year"] - merged["register_year"]) * 12
                + (merged["first_claim_month"] - merged["register_month"])
            )
            merged["months_from_register_to_first_claim_isna"] = (~valid).astype(int)
            merged["months_from_register_to_first_claim"] = months.where(valid, 0).clip(lower=0)
            merged["claim_started_soon_after_register"] = (
                valid & merged["months_from_register_to_first_claim"].le(3)
            ).astype(int)
        else:
            merged["months_from_register_to_first_claim_isna"] = 1
            merged["months_from_register_to_first_claim"] = 0
            merged["claim_started_soon_after_register"] = 0
        return merged

    def _build_target_encoded_claim_features(
        self,
        df: pd.DataFrame,
        customer_frame: pd.DataFrame,
        target: pd.Series,
        train_mask: pd.Series,
        key_map: dict[str, str],
    ) -> pd.DataFrame:
        if "CHME_LICE_NO" not in key_map and "HOSP_CODE" not in key_map:
            return pd.DataFrame(columns=[ID_COL])

        target_frame = pd.DataFrame(
            {
                ID_COL: customer_frame[ID_COL].values,
                "_target": pd.to_numeric(target, errors="coerce").values,
                "_train": train_mask.values,
            }
        )
        target_frame = target_frame.dropna(subset=["_target"])
        target_frame = target_frame[target_frame["_train"]]

        train_customer_ids = target_frame[ID_COL].tolist()
        if len(train_customer_ids) < self.target_encode_n_splits:
            return pd.DataFrame(columns=[ID_COL])

        train_claim = df[df[ID_COL].isin(train_customer_ids)].copy()
        global_rate = float(target_frame["_target"].mean())

        doc_row = pd.Series(np.nan, index=df.index, dtype=float)
        hosp_row = pd.Series(np.nan, index=df.index, dtype=float)

        customer_ids = np.array(sorted(set(train_customer_ids)))
        folds = np.array_split(customer_ids, self.target_encode_n_splits)

        for fold_ids in folds:
            valid_ids = set(fold_ids.tolist())
            fit_ids = set(customer_ids.tolist()) - valid_ids
            fit_claim = train_claim[train_claim[ID_COL].isin(fit_ids)]
            valid_claim = train_claim[train_claim[ID_COL].isin(valid_ids)]
            if valid_claim.empty:
                continue

            if "CHME_LICE_NO" in key_map:
                mapping = self._fit_smoothed_target_map(
                    fit_claim,
                    key_map["CHME_LICE_NO"],
                    target_frame[[ID_COL, "_target"]],
                    global_rate,
                )
                doc_row.loc[valid_claim.index] = valid_claim[key_map["CHME_LICE_NO"]].map(mapping).fillna(global_rate)
            if "HOSP_CODE" in key_map:
                mapping = self._fit_smoothed_target_map(
                    fit_claim,
                    key_map["HOSP_CODE"],
                    target_frame[[ID_COL, "_target"]],
                    global_rate,
                )
                hosp_row.loc[valid_claim.index] = valid_claim[key_map["HOSP_CODE"]].map(mapping).fillna(global_rate)

        if "CHME_LICE_NO" in key_map:
            doc_full_map = self._fit_smoothed_target_map(
                train_claim,
                key_map["CHME_LICE_NO"],
                target_frame[[ID_COL, "_target"]],
                global_rate,
            )
            doc_row = doc_row.where(doc_row.notna(), df[key_map["CHME_LICE_NO"]].map(doc_full_map).fillna(global_rate))
            df["__doc_siu_row"] = doc_row
        if "HOSP_CODE" in key_map:
            hosp_full_map = self._fit_smoothed_target_map(
                train_claim,
                key_map["HOSP_CODE"],
                target_frame[[ID_COL, "_target"]],
                global_rate,
            )
            hosp_row = hosp_row.where(hosp_row.notna(), df[key_map["HOSP_CODE"]].map(hosp_full_map).fillna(global_rate))
            df["__hosp_siu_row"] = hosp_row

        agg = pd.DataFrame({ID_COL: sorted(df[ID_COL].dropna().unique().tolist())})
        if "__doc_siu_row" in df.columns:
            doc_agg = df.groupby(ID_COL)["__doc_siu_row"].max().rename("doc_siu_ratio_max").reset_index()
            agg = agg.merge(doc_agg, on=ID_COL, how="left")
        if "__hosp_siu_row" in df.columns:
            hosp_agg = df.groupby(ID_COL)["__hosp_siu_row"].max().rename("hosp_siu_ratio_max").reset_index()
            agg = agg.merge(hosp_agg, on=ID_COL, how="left")
        if {"doc_siu_ratio_max", "hosp_siu_ratio_max"}.issubset(agg.columns):
            agg["hosp_doc_siu"] = agg["doc_siu_ratio_max"] * agg["hosp_siu_ratio_max"]
        return agg

    def _fit_smoothed_target_map(
        self,
        claim_part: pd.DataFrame,
        key_col: str,
        target_frame: pd.DataFrame,
        global_rate: float,
    ) -> dict[str, float]:
        joined = claim_part[[ID_COL, key_col]].merge(target_frame, on=ID_COL, how="inner")
        stats = joined.groupby(key_col)["_target"].agg(["sum", "count"])
        alpha = self.target_encode_alpha
        smooth = (stats["sum"] + global_rate * alpha) / (stats["count"] + alpha)
        return smooth.to_dict()

    def _hierarchical_group_median_fill(
        self,
        df: pd.DataFrame,
        train_mask: pd.Series,
        target_col: str,
        group_levels: list[list[str]],
    ) -> pd.Series:
        series = df[target_col].copy()
        train_df = df.loc[train_mask].copy()
        global_median = float(train_df[target_col].median())
        for group_cols in group_levels:
            if not group_cols:
                series = series.fillna(global_median)
                continue
            group_cols = [c for c in group_cols if c in df.columns]
            if not group_cols:
                continue
            med = train_df.groupby(group_cols, dropna=False)[target_col].median()
            keys = self._frame_to_group_keys(df[group_cols])
            fill_values = keys.map(med.to_dict())
            series = series.where(series.notna(), fill_values)
        return series.fillna(global_median)

    def _make_totalprem_bins(self, totalprem: pd.Series, train_mask: pd.Series) -> pd.Series:
        train_series = totalprem.loc[train_mask].dropna()
        if train_series.empty:
            return pd.Series("ALL", index=totalprem.index, dtype="string")
        quantiles = np.unique(train_series.quantile(np.linspace(0, 1, 6)).to_numpy())
        if len(quantiles) < 2:
            return pd.Series("ALL", index=totalprem.index, dtype="string")
        quantiles[0] = -np.inf
        quantiles[-1] = np.inf
        labels = [f"Q{i+1}" for i in range(len(quantiles) - 1)]
        return pd.cut(totalprem, bins=quantiles, labels=labels, include_lowest=True, duplicates="drop").astype("string")

    def _fit_frequency_mapping(self, series: pd.Series) -> dict[str, float]:
        clean = series.dropna().astype("string")
        if clean.empty:
            return {}
        return clean.value_counts(normalize=True).to_dict()

    def _group_nunique(self, df: pd.DataFrame, key_col: str, feature_name: str) -> pd.DataFrame:
        return df.groupby(ID_COL)[key_col].nunique().rename(feature_name).reset_index()

    def _group_main_ratio(self, df: pd.DataFrame, key_col: str, feature_name: str) -> pd.DataFrame:
        def _top_ratio(s: pd.Series) -> float:
            valid = s.dropna().astype("string")
            if valid.empty:
                return 0.0
            return float(valid.value_counts().max() / len(valid))

        return df.groupby(ID_COL)[key_col].agg(_top_ratio).rename(feature_name).reset_index()

    def _cap_excluded_columns(self, df: pd.DataFrame, ohe_dummy_cols: list[str]) -> set[str]:
        excluded = {ID_COL, DIVIDED_SET_COL}
        for col in df.columns:
            if col.endswith("_isna") or col.endswith("_is_99"):
                excluded.add(col)
        excluded.update(ohe_dummy_cols)
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            values = pd.Series(df[col].dropna().unique())
            if not values.empty and values.isin([0, 1]).all():
                excluded.add(col)
        return excluded

    def _scale_target_columns(self):
        return [
            "AGE",
            "RESI_COST",
            "TOTALPREM",
            "MINCRDT",
            "MAXCRDT",
            "CHLD_CNT",
            "LTBN_CHLD_AGE",
            "MAX_PRM",
            "CUST_INCM",
            "RCBASE_HSHD_INCM",
            "JPBASE_HSHD_INCM",
            "payment_active_months",
            "payment_active_years",
            "premium_income_ratio",
            "max_prm_totalprem_ratio",
            "resi_cost_income_ratio",
            "household_income_gap",
            "credit_gap",
            "credit_mean",
            "CUST_INCM_log",
            "TOTALPREM_log",
            "MAX_PRM_log",
            "RESI_COST_log",
            "RCBASE_HSHD_INCM_log",
            "JPBASE_HSHD_INCM_log",
            "claim_count",
            "claim_count_log",
            "hosp_days_mean",
            "hosp_days_sum",
            "hosp_days_max",
            "dmnd_sum",
            "dmnd_mean",
            "dmnd_max",
            "dmnd_std",
            "dmnd_sum_log",
            "claim_interval_mean",
            "claim_interval_min",
            "claim_active_days",
            "months_from_register_to_first_claim",
        ]

    def _created_customer_features(self):
        return [
            "AGE_isna",
            "CHLD_CNT_isna",
            "LTBN_CHLD_AGE_isna",
            "CUST_INCM_isna",
            "TOTALPREM_isna",
            "MAX_PRM_isna",
            "MAXCRDT_isna",
            "MINCRDT_isna",
            "JPBASE_HSHD_INCM_isna",
            "RCBASE_HSHD_INCM_isna",
            "RESI_COST_isna",
            "RESI_TYPE_CODE_isna",
            "CTPR_isna",
            "OCCP_GRP_1_isna",
            "WEDD_YN_isna",
            "MATE_OCCP_GRP_1_isna",
            "CUST_RGST_isna",
            "MAX_PAYM_YM_isna",
            "MINCRDT_is_99",
            "MAXCRDT_is_99",
            "AGE_GROUP",
            "LTBN_CHLD_AGE_GROUP",
            "has_child",
            "payment_active_months",
            "payment_active_years",
            "register_year",
            "register_month",
            "CUST_INCM_log",
            "TOTALPREM_log",
            "MAX_PRM_log",
            "RESI_COST_log",
            "RCBASE_HSHD_INCM_log",
            "JPBASE_HSHD_INCM_log",
            "premium_income_ratio",
            "max_prm_totalprem_ratio",
            "resi_cost_income_ratio",
            "household_income_gap",
            "credit_gap",
            "credit_mean",
        ]

    def _created_claim_features(self, columns) -> list[str]:
        preferred = [
            "claim_count",
            "claim_count_log",
            "policy_nunique",
            "claim_per_policy",
            "hosp_days_mean",
            "hosp_days_sum",
            "hosp_days_max",
            "dmnd_sum",
            "dmnd_mean",
            "dmnd_max",
            "dmnd_std",
            "dmnd_sum_log",
            "house_hosp_dist_mean",
            "house_hosp_dist_max",
            "heed_hosp_rate",
            "has_heed_hosp",
            "hospital_nunique",
            "doctor_nunique",
            "disease_nunique",
            "cause_nunique",
            "cause_dtal_nunique",
            "addr_nunique",
            "main_hospital_claim_ratio",
            "main_doctor_claim_ratio",
            "main_disease_claim_ratio",
            "main_cause_claim_ratio",
            "main_addr_claim_ratio",
            "hospital_per_claim",
            "doctor_per_claim",
            "disease_per_claim",
            "cause_per_claim",
            "addr_per_claim",
            "hospital_freq_mean",
            "hospital_freq_max",
            "doctor_freq_mean",
            "doctor_freq_max",
            "disease_freq_mean",
            "disease_freq_max",
            "cause_freq_mean",
            "cause_freq_max",
            "hospital_area_freq_mean",
            "hospital_area_freq_max",
            "delay_origin_to_recp_mean",
            "delay_origin_to_recp_max",
            "delay_resn_to_recp_mean",
            "delay_resn_to_recp_max",
            "claim_interval_mean",
            "claim_interval_min",
            "claim_active_days",
            "recent_claim_cnt",
            "recp_month_nunique",
            "recp_year_nunique",
            "weekend_claim_rate",
            "first_claim_year",
            "first_claim_month",
            "acci_dvsn_nunique",
            "main_acci_dvsn_ratio",
            "acci_dvsn_per_claim",
            "acci_dmnd_combo_nunique",
            "main_acci_dmnd_combo_ratio",
            "acci_dmnd_combo_freq_mean",
            "acci_dmnd_combo_freq_max",
            "hosp_spec_nunique",
            "main_hosp_spec_ratio",
            "months_from_register_to_first_claim",
            "months_from_register_to_first_claim_isna",
            "claim_started_soon_after_register",
            "doc_siu_ratio_max",
            "hosp_siu_ratio_max",
            "hosp_doc_siu",
        ]
        return [col for col in preferred if col in columns]

    def _excluded_variables_and_reasons(self):
        return [
            "FP_CAREER 및 파생변수는 명시 요구에 따라 완전 제외했습니다.",
            "COUNT_TRMT_ITEM, NON_PAY, SELF_CHAM, PATT_CHRG_TOTA 등 결측률 과다 CLAIM 원본 변수는 v3 기본 전략에서 제거했습니다.",
            "PAYM_AMT, PAYM_DATE, RESL_CD1, RESL_NM1, PMMI_DLNG_YN, CRNT_PROG_DVSN, delay_recp_to_paym, pay_ratio, amount_gap은 누수 위험으로 safe 버전에서 제외했습니다.",
            "DOC_SIU_RATIO, HOSP_SIU_RATIO, HOSP_DOC_SIU는 include_target_encoded_features=True 일 때만 KFold 내부 계산으로 생성합니다.",
            "last_payment_year, last_payment_month, indicator 조합 변수, 분위수 기반 조건형 flag는 중복/시점효과/해석 약화 우려로 제외했습니다.",
        ]

    def _target_encoding_leakage_prevention_text(self):
        return (
            "train 고객만 사용해 의사/병원 사기비율을 계산하고, KFold로 train 고객을 나눠 "
            "각 validation fold에는 fit fold에서 만든 mapping만 적용합니다. "
            "test 고객에는 전체 train 고객으로 fit한 mapping만 적용하며, "
            f"smoothing(alpha={self.target_encode_alpha})으로 표본 수가 적은 의사/병원의 비율 폭주를 완화합니다."
        )

    def _print_report(self, report: dict):
        print("\n" + "=" * 90)
        print("[member_c_strategy_3] v3 전처리 리포트")
        print("=" * 90)
        ordered_items = [
            ("1. 삭제한 CUST_DATA 컬럼 목록", report.get("deleted_customer_columns", [])),
            ("2. 삭제한 CLAIM_DATA 결측치 과다 컬럼 목록", report.get("deleted_claim_high_missing_columns", [])),
            ("3. 삭제한 DATE 원본 컬럼 목록", report.get("deleted_customer_date_columns", []) + report.get("deleted_claim_date_columns", [])),
            ("4. 생성한 CUST_DATA 파생변수 목록", report.get("created_customer_features", [])),
            ("5. 생성한 CLAIM_DATA 파생변수 목록", report.get("created_claim_features", [])),
            ("6. 생성하지 않은 변수와 이유", report.get("excluded_variables_and_reasons", [])),
            ("7. include_target_encoded_features 여부", report.get("include_target_encoded_features")),
            ("8. include_leakage_features 여부", report.get("include_leakage_features")),
            ("9. target encoding 변수 누수 방지 방식 설명", report.get("target_encoding_leakage_prevention")),
            ("10. 결측치 처리 전후 개수", {
                "customer_before": report.get("customer_missing_before"),
                "customer_after": report.get("customer_missing_after"),
                "claim_before": report.get("claim_missing_before"),
                "claim_after": report.get("claim_missing_after"),
                "merged_before_final_fill": report.get("missing_before_final_fill"),
                "merged_after_final_fill": report.get("missing_after_final_fill"),
            }),
            ("11. 남은 결측치 개수", report.get("remaining_missing_total")),
            ("12. 남은 문자열 컬럼 목록", report.get("remaining_string_columns", [])),
            ("13. 분산이 0에 가까운 컬럼 목록", report.get("near_zero_variance_columns", [])),
            ("14. 결측률이 높은 컬럼 목록", report.get("high_missing_rate_columns", {})),
            ("15. StandardScaler 적용 여부", report.get("standard_scaler_applied")),
            ("16. 최종 관리용 데이터 shape", report.get("final_managed_shape")),
            ("17. 최종 X_train, y_train, X_test shape", {
                "X_train": report.get("x_train_shape"),
                "y_train": report.get("y_train_shape"),
                "X_test": report.get("x_test_shape"),
            }),
            ("18. 타깃 분포", report.get("target_distribution")),
            ("19. 불균형 비율", report.get("imbalance_ratio")),
            ("20. 최종 저장 파일명", report.get("final_save_file_name")),
            ("추가. CLAIM 삭제 전 결측률 리포트", report.get("claim_missing_rate_before_drop", {})),
        ]
        for title, value in ordered_items:
            if isinstance(value, (dict, list)):
                rendered = json.dumps(value, ensure_ascii=False, indent=2)
            else:
                rendered = str(value)
            print(title)
            print(rendered)
            print("-" * 90)

    def _mode_from_train(self, series: pd.Series, default: str) -> str:
        mode = series.dropna().astype("string").mode()
        if mode.empty:
            return default
        return str(mode.iloc[0])

    def _parse_year_month(self, series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        as_string = numeric.astype("Int64").astype("string")
        valid = as_string.str.len().eq(6)
        return pd.to_datetime(as_string.where(valid), format="%Y%m", errors="coerce")

    def _parse_yyyymmdd(self, series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        as_string = numeric.astype("Int64").astype("string")
        valid = as_string.str.len().eq(8)
        return pd.to_datetime(as_string.where(valid), format="%Y%m%d", errors="coerce")

    def _age_group(self, series: pd.Series) -> pd.Series:
        return pd.cut(
            series,
            bins=[0, 20, 30, 40, 50, 60, 70, 120],
            labels=["10s", "20s", "30s", "40s", "50s", "60s", "70plus"],
            include_lowest=True,
        ).astype("string")

    def _child_age_group(self, series: pd.Series) -> pd.Series:
        filled = series.fillna(0)
        return pd.cut(
            filled,
            bins=[-1, 0, 7, 13, 19, np.inf],
            labels=["none", "infant", "child", "teen", "adult_child"],
            include_lowest=True,
        ).astype("string")

    def _month_diff(self, start: pd.Series, end: pd.Series) -> pd.Series:
        valid = start.notna() & end.notna()
        months = (end.dt.year - start.dt.year) * 12 + (end.dt.month - start.dt.month)
        return months.where(valid)

    def _canonical_category(self, series: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series):
            return series.astype("Int64").astype("string")
        numeric = pd.to_numeric(series, errors="coerce")
        non_null = int(series.notna().sum())
        numeric_non_null = int(numeric.notna().sum())
        if non_null > 0 and numeric_non_null / non_null >= 0.8:
            return numeric.astype("Int64").astype("string")
        return series.astype("string").str.strip()

    def _coerce_numeric(self, df: pd.DataFrame, cols):
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    def _frame_to_group_keys(self, frame: pd.DataFrame) -> pd.Series:
        if frame.shape[1] == 1:
            return frame.iloc[:, 0]
        return pd.Series(list(map(tuple, frame.itertuples(index=False, name=None))), index=frame.index)

    def _safe_divide(self, numerator, denominator):
        if isinstance(denominator, (int, float)):
            if denominator == 0:
                return 0
            return numerator / denominator
        denominator = denominator.replace(0, np.nan)
        result = numerator / denominator
        if isinstance(result, pd.Series):
            return result.replace([np.inf, -np.inf], np.nan)
        return result

    def _safe_token(self, value: str) -> str:
        token = re.sub(r"[^0-9A-Za-z가-힣]+", "_", str(value)).strip("_")
        return token or "unknown"
