from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import DIVIDED_SET_COL, ID_COL
from src.pipeline.data_loader import normalize_target
from src.preprocessing.components.encoder import OneHotEncoder
from src.preprocessing.components.missing_value import MedianImputer, ZeroImputer
from src.preprocessing.components.outlier import QuantileCapper


class MemberCV1FeaturePipeline:
    """member_c v1 전략의 전처리 흐름을 담당한다."""

    def run(
        self,
        strategy,
        X: pd.DataFrame,
        y: pd.Series,
        claim_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if claim_df is None:
            raise ValueError("member_c_strategy 는 claim_df가 필요합니다.")

        cust = X.copy()
        claim = claim_df.copy()

        cust = strategy._prepare_customer_features(cust)
        claim_agg = strategy._build_claim_aggregates(claim)
        cust = cust.merge(claim_agg, on=ID_COL, how="left")

        claim_cols = [col for col in claim_agg.columns if col != ID_COL]
        cust = ZeroImputer(cols=claim_cols).transform(cust)

        cat_cols = strategy._existing_columns(
            cust,
            [
                "SEX",
                "RESI_TYPE_CODE",
                "CTPR",
                "OCCP_GRP_1",
                "WEDD_YN",
                "MATE_OCCP_GRP_1",
                "AGE_GROUP",
                "LTBN_CHLD_AGE_GROUP",
            ],
        )
        if cat_cols:
            encoder = OneHotEncoder(cols=cat_cols)
            encoder.fit(cust)
            cust = encoder.transform(cust)

        numeric_cols = [
            col
            for col in cust.columns
            if col not in {"CUST_ID", "DIVIDED_SET", "FP_CAREER", "fp_career_flag"}
            and pd.api.types.is_numeric_dtype(cust[col])
        ]
        median_imputer = MedianImputer(cols=numeric_cols)
        median_imputer.fit(cust)
        cust = median_imputer.transform(cust)

        cap_cols = [
            col for col in numeric_cols
            if not col.endswith("_isna")
            and not col.endswith("_freq")
            and not col.startswith("DIVIDED_SET")
        ]
        capper = QuantileCapper(cols=cap_cols, lower=0.01, upper=0.99)
        capper.fit(cust)
        cust = capper.transform(cust)

        bool_cols = cust.select_dtypes(include="bool").columns.tolist()
        if bool_cols:
            cust[bool_cols] = cust[bool_cols].astype(int)

        non_numeric_cols = cust.select_dtypes(exclude="number").columns.tolist()
        if non_numeric_cols:
            cust = cust.drop(columns=non_numeric_cols)

        cust = cust.drop(
            columns=strategy._existing_columns(
                cust, ["DIVIDED_SET", "FP_CAREER", "fp_career_flag"]
            ),
            errors="ignore",
        )
        return cust


class MemberCV2FeaturePipeline:
    """member_c v2 전략의 전처리 흐름을 담당한다."""

    def run(
        self,
        strategy,
        X: pd.DataFrame,
        y: pd.Series,
        claim_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if claim_df is None:
            raise ValueError("member_c_strategy_2 는 claim_df가 필요합니다.")

        cust = X.copy()
        claim = claim_df.copy()
        y = y.reindex(cust.index)

        divided = cust["DIVIDED_SET"].copy() if "DIVIDED_SET" in cust.columns else None

        cust = strategy._prepare_customer_features(cust, y)
        claim_agg = strategy._build_claim_aggregates(claim)
        cust = cust.merge(claim_agg, on=ID_COL, how="left")

        claim_cols = [c for c in claim_agg.columns if c != ID_COL]
        cust = ZeroImputer(cols=claim_cols).transform(cust)

        cat_cols = strategy._existing_columns(
            cust,
            [
                "SEX",
                "RESI_TYPE_CODE",
                "CTPR",
                "OCCP_GRP_1",
                "WEDD_YN",
                "MATE_OCCP_GRP_1",
                "AGE_GROUP",
                "LTBN_CHLD_AGE_GROUP",
            ],
        )
        ohe_dummy_cols: list[str] = []
        if cat_cols:
            enc = OneHotEncoder(cols=cat_cols)
            enc.fit(cust)
            ohe_dummy_cols = list(enc.dummy_cols_ or [])
            cust = enc.transform(cust)

        if strategy.include_target_encoded_features:
            cust = strategy._attach_kfold_target_encoding(
                cust, claim, y, divided, ohe_dummy_cols
            )

        numeric_cols = [
            c
            for c in cust.columns
            if c not in {ID_COL} and pd.api.types.is_numeric_dtype(cust[c])
        ]
        med_imp = MedianImputer(cols=numeric_cols)
        med_imp.fit(cust)
        cust = med_imp.transform(cust)

        cap_exclude = strategy._cap_excluded_columns(cust.columns, ohe_dummy_cols)
        cap_cols = [c for c in numeric_cols if c not in cap_exclude]
        capper = QuantileCapper(cols=cap_cols, lower=0.01, upper=0.99)
        capper.fit(cust)
        cust = capper.transform(cust)

        if strategy.scale_numeric:
            cust = strategy._apply_standard_scaler(cust, ohe_dummy_cols)

        bool_cols = cust.select_dtypes(include="bool").columns.tolist()
        if bool_cols:
            cust[bool_cols] = cust[bool_cols].astype(int)

        non_num = cust.select_dtypes(exclude="number").columns.tolist()
        if non_num:
            cust = cust.drop(columns=non_num)

        drop_meta = strategy._existing_columns(cust, ["DIVIDED_SET", "FP_CAREER"])
        if drop_meta:
            cust = cust.drop(columns=drop_meta, errors="ignore")
        return cust


class MemberCV3FeaturePipeline:
    """member_c v3 safe 전략의 전처리 흐름을 담당한다."""

    def run(
        self,
        strategy,
        X: pd.DataFrame,
        y: pd.Series,
        claim_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
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
            "include_target_encoded_features": strategy.include_target_encoded_features,
            "include_leakage_features": strategy.include_leakage_features,
            "scale_numeric": strategy.scale_numeric,
            "deleted_customer_columns": [c for c in strategy.CUSTOMER_DROP_COLS if c in cust.columns],
            "deleted_customer_date_columns": [c for c in strategy.CUSTOMER_DATE_COLS if c in cust.columns],
        }

        cust_processed, cust_report = strategy._prepare_customer_data(cust, train_mask)
        claim_agg, claim_report = strategy._prepare_claim_data(
            claim, cust_processed, target, train_mask, train_customer_ids
        )
        report.update(cust_report)
        report.update(claim_report)

        merged = cust_processed.merge(claim_agg, on=ID_COL, how="left")
        claim_cols = [c for c in claim_agg.columns if c != ID_COL]
        if claim_cols:
            merged = ZeroImputer(cols=claim_cols).transform(merged)

        merged = strategy._add_cross_customer_claim_features(merged)

        report["missing_before_final_fill"] = int(merged.isna().sum().sum())

        ohe_cols = [c for c in strategy.CUSTOMER_OHE_COLS if c in merged.columns]
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

        cap_excluded = strategy._cap_excluded_columns(merged, ohe_dummy_cols)
        cap_cols = [c for c in numeric_cols if c not in cap_excluded]
        capper = QuantileCapper(cols=cap_cols, lower=0.01, upper=0.99)
        capper.fit(merged.loc[train_mask, cap_cols])
        merged = capper.transform(merged)

        if strategy.scale_numeric:
            scale_cols = [c for c in strategy._scale_target_columns() if c in merged.columns]
            scale_cols = [c for c in scale_cols if c not in cap_excluded]
            if scale_cols:
                scaler = StandardScaler()
                scaler.fit(merged.loc[train_mask, scale_cols])
                merged[scale_cols] = scaler.transform(merged[scale_cols])
        report["standard_scaler_applied"] = strategy.scale_numeric

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

        report["created_customer_features"] = strategy._created_customer_features()
        report["created_claim_features"] = strategy._created_claim_features(merged.columns)
        report["excluded_variables_and_reasons"] = strategy._excluded_variables_and_reasons()
        report["target_encoding_leakage_prevention"] = strategy._target_encoding_leakage_prevention_text()
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
        report["final_save_file_name"] = f"{strategy.__class__.__module__.split('.')[-1]}_preprocessed.csv"

        strategy._print_report(report)
        return merged


class MemberCV4FeaturePipeline(MemberCV3FeaturePipeline):
    """member_c v4 전략도 v3와 같은 전처리 흐름을 사용한다."""

