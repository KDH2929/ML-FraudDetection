"""
member_c 전처리 전략 v2: 고객–청구 집계, 결측 근거 기반 보완, 누수/스케일 옵션 분리.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from src.preprocessing.components.encoder import OneHotEncoder
from src.preprocessing.components.features.member_c_features import MemberCV2FeaturePipeline
from src.preprocessing.components.missing_value import MedianImputer, ZeroImputer
from src.preprocessing.components.outlier import QuantileCapper
from src.preprocessing.strategies.base_strategy import BaseStrategy
from src.pipeline.data_loader import normalize_target


def split_train_predict(df: pd.DataFrame, target_col: str = "SIU_CUST_YN"):
    """
    DIVIDED_SET이 없을 때: 라벨 유무로 학습/예측 행을 나눈다.
    train_df = 타깃이 있는 행, test_df = 타깃 결측(예측 대상) 행.
    """
    if target_col not in df.columns:
        raise ValueError(f"{target_col} 컬럼이 필요합니다.")
    yraw = df[target_col]
    ynorm = normalize_target(yraw)
    labeled = ynorm.notna()
    train_df = df.loc[labeled].copy()
    test_df = df.loc[~labeled].copy()
    return train_df, test_df


class MemberCStrategy2(BaseStrategy):
    def __init__(
        self,
        include_leakage_features: bool = False,
        include_target_encoded_features: bool = False,
        scale_numeric: bool = False,
        target_encode_n_splits: int = 5,
    ):
        self.include_leakage_features = include_leakage_features
        self.include_target_encoded_features = include_target_encoded_features
        self.scale_numeric = scale_numeric
        self.target_encode_n_splits = int(target_encode_n_splits)

    def get_strategy_name(self) -> str:
        parts = ["C2", "safe"]
        if self.include_leakage_features:
            parts.append("leakage")
        if self.include_target_encoded_features:
            parts.append(f"te{self.target_encode_n_splits}fold")
        if self.scale_numeric:
            parts.append("scaled")
        return " ".join(parts)

    def preprocess(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        claim_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        pipeline = MemberCV2FeaturePipeline()
        return pipeline.run(self, X, y, claim_df)

        if claim_df is None:
            raise ValueError("member_c_strategy_2 는 claim_df가 필요합니다.")

        cust = X.copy()
        claim = claim_df.copy()
        y = y.reindex(cust.index)

        divided = cust["DIVIDED_SET"].copy() if "DIVIDED_SET" in cust.columns else None

        cust = self._prepare_customer_features(cust, y)
        claim_agg = self._build_claim_aggregates(claim)
        cust = cust.merge(claim_agg, on="CUST_ID", how="left")

        claim_cols = [c for c in claim_agg.columns if c != "CUST_ID"]
        cust = ZeroImputer(cols=claim_cols).transform(cust)

        cat_cols = self._existing_columns(
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

        if self.include_target_encoded_features:
            cust = self._attach_kfold_target_encoding(
                cust, claim, y, divided, ohe_dummy_cols
            )

        numeric_cols = [
            c
            for c in cust.columns
            if c not in {"CUST_ID"} and pd.api.types.is_numeric_dtype(cust[c])
        ]
        _med_imp = MedianImputer(cols=numeric_cols)
        _med_imp.fit(cust)
        cust = _med_imp.transform(cust)

        cap_exclude = self._cap_excluded_columns(cust.columns, ohe_dummy_cols)
        cap_cols = [c for c in numeric_cols if c not in cap_exclude]
        capper = QuantileCapper(cols=cap_cols, lower=0.01, upper=0.99)
        capper.fit(cust)
        cust = capper.transform(cust)

        if self.scale_numeric:
            cust = self._apply_standard_scaler(cust, ohe_dummy_cols)

        bool_cols = cust.select_dtypes(include="bool").columns.tolist()
        if bool_cols:
            cust[bool_cols] = cust[bool_cols].astype(int)

        non_num = cust.select_dtypes(exclude="number").columns.tolist()
        if non_num:
            cust = cust.drop(columns=non_num)

        drop_meta = self._existing_columns(cust, ["DIVIDED_SET", "FP_CAREER"])
        if drop_meta:
            cust = cust.drop(columns=drop_meta, errors="ignore")

        return cust

    # --- customer -----------------------------------------------------------------

    def _prepare_customer_features(self, df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        cust = df.copy()

        self._coerce_numeric(
            cust,
            [
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
            ],
        )

        self._add_missing_indicators(
            cust,
            {
                "CHLD_CNT": "CHLD_CNT_isna",
                "LTBN_CHLD_AGE": "LTBN_CHLD_AGE_isna",
                "CUST_INCM": "CUST_INCM_isna",
                "TOTALPREM": "TOTALPREM_isna",
                "MAX_PRM": "MAX_PRM_isna",
                "MAXCRDT": "MAXCRDT_isna",
                "MINCRDT": "MINCRDT_isna",
                "JPBASE_HSHD_INCM": "JPBASE_HSHD_INCM_isna",
                "RCBASE_HSHD_INCM": "RCBASE_HSHD_INCM_isna",
            },
        )

        if "AGE" in cust.columns:
            cust["AGE_GROUP"] = pd.cut(
                cust["AGE"],
                bins=[0, 20, 30, 40, 50, 60, 70, 120],
                labels=["10s", "20s", "30s", "40s", "50s", "60s", "70plus"],
                include_lowest=True,
            ).astype("string")

        ltbn_fill = cust["LTBN_CHLD_AGE"].fillna(0) if "LTBN_CHLD_AGE" in cust.columns else None
        if ltbn_fill is not None:
            cust["LTBN_CHLD_AGE_GROUP"] = pd.cut(
                ltbn_fill,
                bins=[-1, 0, 7, 13, 19, 120],
                labels=["none", "infant", "child", "teen", "adult_child"],
                include_lowest=True,
            ).astype("string")

        for col in self._existing_columns(cust, ["OCCP_GRP_1", "AGE_GROUP", "CTPR", "LTBN_CHLD_AGE_GROUP"]):
            cust[col] = cust[col].astype("string")
        if "AGE_GROUP" in cust.columns:
            cust["AGE_GROUP"] = cust["AGE_GROUP"].fillna("Unknown")
        if "LTBN_CHLD_AGE_GROUP" in cust.columns:
            cust["LTBN_CHLD_AGE_GROUP"] = cust["LTBN_CHLD_AGE_GROUP"].fillna("none")

        for col in self._existing_columns(cust, ["SEX", "OCCP_GRP_1", "RESI_TYPE_CODE", "CTPR", "MATE_OCCP_GRP_1"]):
            m = cust[col].mode(dropna=True)
            cust[col] = cust[col].fillna(m.iloc[0] if len(m) else "UNK")
        if "WEDD_YN" in cust.columns:
            cust["WEDD_YN"] = cust["WEDD_YN"].astype("string").fillna("N")

        self._impute_customer_numerics(cust)

        cust = cust.drop(
            columns=self._existing_columns(cust, ["FP_CAREER", "OCCP_GRP_2", "MATE_OCCP_GRP_2"]),
            errors="ignore",
        )

        self._coerce_year_month(cust, ["CUST_RGST", "MAX_PAYM_YM"])
        if {"CUST_RGST", "MAX_PAYM_YM"}.issubset(cust.columns):
            cust["payment_active_months"] = (
                (cust["MAX_PAYM_YM"] - cust["CUST_RGST"]).dt.days.div(30.44)
            ).fillna(0)
        cust = cust.drop(
            columns=self._existing_columns(cust, ["CUST_RGST", "MAX_PAYM_YM"]),
            errors="ignore",
        )

        for src, tgt in [
            ("RESI_COST", "RESI_COST_log"),
            ("TOTALPREM", "TOTALPREM_log"),
            ("MAX_PRM", "MAX_PRM_log"),
            ("CUST_INCM", "CUST_INCM_log"),
            ("RCBASE_HSHD_INCM", "RCBASE_HSHD_INCM_log"),
            ("JPBASE_HSHD_INCM", "JPBASE_HSHD_INCM_log"),
        ]:
            self._add_log_feature(cust, src, tgt)

        if {"TOTALPREM", "CUST_INCM"}.issubset(cust.columns):
            cust["premium_income_ratio"] = self._safe_divide(cust["TOTALPREM"], cust["CUST_INCM"])
        if {"MAX_PRM", "TOTALPREM"}.issubset(cust.columns):
            cust["max_prm_totalprem_ratio"] = self._safe_divide(cust["MAX_PRM"], cust["TOTALPREM"])
        if {"MAXCRDT", "MINCRDT"}.issubset(cust.columns):
            cust["credit_gap"] = cust["MAXCRDT"].fillna(0) - cust["MINCRDT"].fillna(0)
        if "CHLD_CNT" in cust.columns:
            cust["has_child"] = cust["CHLD_CNT"].fillna(0).gt(0).astype(int)

        return cust

    def _impute_customer_numerics(self, cust: pd.DataFrame) -> None:
        if "CHLD_CNT" in cust.columns:
            cust["CHLD_CNT"] = cust["CHLD_CNT"].fillna(0)
        if "LTBN_CHLD_AGE" in cust.columns:
            cust["LTBN_CHLD_AGE"] = cust["LTBN_CHLD_AGE"].fillna(0)

        def med_chain(col: str, levels: list) -> None:
            s = cust[col].copy()
            for group_cols in levels:
                if not group_cols:
                    s = s.fillna(s.median())
                    continue
                if all(c in cust.columns for c in group_cols):
                    gmed = cust.groupby(list(group_cols), dropna=False)[col].transform(lambda x: x.median())
                    s = s.where(s.notna(), gmed)
            cust[col] = s.fillna(s.median())

        if "CUST_INCM" in cust.columns:
            med_chain(
                "CUST_INCM",
                [
                    ["OCCP_GRP_1", "AGE_GROUP"],
                    ["OCCP_GRP_1"],
                    ["AGE_GROUP"],
                    [],
                ],
            )
        if "JPBASE_HSHD_INCM" in cust.columns:
            med_chain("JPBASE_HSHD_INCM", [["OCCP_GRP_1"], ["AGE_GROUP"], []])
        if "RCBASE_HSHD_INCM" in cust.columns:
            med_chain("RCBASE_HSHD_INCM", [["CTPR"], ["OCCP_GRP_1"], []])
        if "TOTALPREM" in cust.columns:
            med_chain("TOTALPREM", [["AGE_GROUP", "OCCP_GRP_1"], ["OCCP_GRP_1"], []])
            cust["TOTALPREM"] = cust["TOTALPREM"].replace(0, np.nan)
            med_chain("TOTALPREM", [["AGE_GROUP", "OCCP_GRP_1"], ["OCCP_GRP_1"], []])

        if "MAX_PRM" in cust.columns and "TOTALPREM" in cust.columns:
            prem = cust["TOTALPREM"].replace(0, np.nan)
            try:
                bins = pd.qcut(prem, q=5, duplicates="drop")
            except (ValueError, TypeError):
                bins = pd.Series(index=cust.index, dtype="object")
            tmp = cust.copy()
            tmp["_prem_bin"] = bins
            gmed = tmp.groupby("_prem_bin", dropna=False)["MAX_PRM"].transform(lambda x: x.median())
            s = cust["MAX_PRM"].copy()
            s = s.where(s.notna(), gmed)
            g2 = tmp.groupby(["AGE_GROUP", "OCCP_GRP_1"], dropna=False)["MAX_PRM"].transform(lambda x: x.median())
            s = s.where(s.notna(), g2)
            cust["MAX_PRM"] = s.fillna(s.median())

        for colmn in ["MINCRDT", "MAXCRDT"]:
            if colmn in cust.columns:
                med_chain(colmn, [["OCCP_GRP_1"], ["AGE_GROUP"], []])

    # --- claim aggregates -----------------------------------------------------------

    def _build_claim_aggregates(self, claim: pd.DataFrame) -> pd.DataFrame:
        df = claim.copy()
        if "CUST_ID" not in df.columns:
            raise ValueError("CLAIM_DATA에는 CUST_ID가 필요합니다.")

        num_safe = [
            "DMND_AMT",
            "NON_PAY",
            "SELF_CHAM",
            "PATT_CHRG_TOTA",
            "NON_PAY_RATIO",
            "VLID_HOSP_OTDA",
            "COUNT_TRMT_ITEM",
        ]
        self._coerce_numeric(df, num_safe)
        if self.include_leakage_features and "PAYM_AMT" in df.columns:
            self._coerce_numeric(df, ["PAYM_AMT"])

        date_cols = [
            "RECP_DATE",
            "ORIG_RESN_DATE",
            "RESN_DATE",
            "HOSP_OTPA_STDT",
            "HOSP_OTPA_ENDT",
            "PAYM_DATE",
        ]
        self._coerce_dates(df, date_cols)

        for col in self._existing_columns(
            df,
            ["HOSP_CODE", "CHME_LICE_NO", "DSAS_NAME", "CAUS_CODE", "CAUS_CODE_DTAL", "ACCI_HOSP_ADDR"],
        ):
            self._add_global_frequency(df, col, f"{col}_freq")

        if {"ORIG_RESN_DATE", "RECP_DATE"}.issubset(df.columns):
            df["delay_origin_to_recp"] = (
                (df["RECP_DATE"] - df["ORIG_RESN_DATE"]).dt.days.clip(lower=0)
            )
        if {"RESN_DATE", "RECP_DATE"}.issubset(df.columns):
            df["delay_resn_to_recp"] = (df["RECP_DATE"] - df["RESN_DATE"]).dt.days.clip(lower=0)
        if self.include_leakage_features and {"PAYM_DATE", "RECP_DATE"}.issubset(df.columns):
            df["delay_recp_to_paym"] = (df["PAYM_DATE"] - df["RECP_DATE"]).dt.days.clip(lower=0)
        if {"HOSP_OTPA_STDT", "HOSP_OTPA_ENDT"}.issubset(df.columns):
            df["hosp_days_calc"] = (
                (df["HOSP_OTPA_ENDT"] - df["HOSP_OTPA_STDT"]).dt.days.clip(lower=0)
            )

        if "RECP_DATE" in df.columns:
            df["recp_month"] = df["RECP_DATE"].dt.month
            df["recp_year"] = df["RECP_DATE"].dt.year
            df["is_weekend_claim"] = df["RECP_DATE"].dt.weekday.isin([5, 6]).astype(int)

        sorted_c = df.sort_values(["CUST_ID", "RECP_DATE"]).copy() if "RECP_DATE" in df.columns else df.copy()
        if "RECP_DATE" in df.columns:
            sorted_c["prev_recp_date"] = sorted_c.groupby("CUST_ID")["RECP_DATE"].shift(1)
            sorted_c["claim_interval"] = (
                sorted_c["RECP_DATE"] - sorted_c["prev_recp_date"]
            ).dt.days.clip(lower=0)
            df = sorted_c

        if self.include_leakage_features and {"PAYM_AMT", "DMND_AMT"}.issubset(df.columns):
            df["pay_ratio"] = self._safe_divide(df["PAYM_AMT"], df["DMND_AMT"])
            df["amount_gap"] = df["PAYM_AMT"].fillna(0) - df["DMND_AMT"].fillna(0)

        agg = pd.DataFrame({"CUST_ID": df["CUST_ID"].drop_duplicates().sort_values().values})
        agg = self._merge_feature(agg, self._claim_count_features(df))
        for base, prefix, ratio in [
            ("HOSP_CODE", "hospital", "main_hospital_claim_ratio"),
            ("CHME_LICE_NO", "doctor", "main_doctor_claim_ratio"),
            ("DSAS_NAME", "disease", "main_disease_claim_ratio"),
            ("CAUS_CODE", "cause", "main_cause_claim_ratio"),
            ("CAUS_CODE_DTAL", "cause_dtal", "main_cause_dtal_claim_ratio"),
            ("ACCI_HOSP_ADDR", "addr", "main_addr_claim_ratio"),
        ]:
            agg = self._merge_feature(agg, self._categorical_group_summary(df, base, prefix, ratio))

        if {"claim_cnt", "hospital_nunique"}.issubset(agg.columns):
            agg["same_hospital_repeat_ratio"] = self._safe_divide(
                agg["claim_cnt"] - agg["hospital_nunique"],
                agg["claim_cnt"],
            )
        for conc_col, ratio_col in [
            ("hospital_concentration", "main_hospital_claim_ratio"),
            ("doctor_concentration", "main_doctor_claim_ratio"),
            ("disease_concentration", "main_disease_claim_ratio"),
            ("cause_concentration", "main_cause_claim_ratio"),
        ]:
            if ratio_col in agg.columns:
                agg[conc_col] = agg[ratio_col]

        for src, prefix in [
            ("HOSP_CODE_freq", "hospital_freq"),
            ("CHME_LICE_NO_freq", "doctor_freq"),
            ("DSAS_NAME_freq", "disease_freq"),
            ("CAUS_CODE_freq", "cause_freq"),
            ("ACCI_HOSP_ADDR_freq", "hospital_area_freq"),
        ]:
            agg = self._merge_feature(agg, self._frequency_summary_mean_max(df, src, prefix))

        agg = self._merge_feature(agg, self._dmnd_amount_summary(df))
        agg = self._merge_feature(agg, self._numeric_summary_limited(df, "NON_PAY", "non_pay", {"sum", "mean", "max"}))
        agg = self._merge_feature(agg, self._numeric_summary_limited(df, "SELF_CHAM", "self_cham", {"sum", "mean", "max"}))
        agg = self._merge_feature(agg, self._missing_rate(df, "NON_PAY", "non_pay_isna_rate"))
        agg = self._merge_feature(agg, self._missing_rate(df, "SELF_CHAM", "self_cham_isna_rate"))
        if {"NON_PAY", "DMND_AMT"}.issubset(df.columns):
            nr = self._safe_divide(df["NON_PAY"], df["DMND_AMT"])
            nr_df = pd.DataFrame({"CUST_ID": df["CUST_ID"], "nr": nr})
            agg = self._merge_feature(
                agg,
                nr_df.groupby("CUST_ID")["nr"].mean().rename("non_pay_dmnd_ratio").reset_index(),
            )
        if {"SELF_CHAM", "DMND_AMT"}.issubset(df.columns):
            sr = self._safe_divide(df["SELF_CHAM"], df["DMND_AMT"])
            sr_df = pd.DataFrame({"CUST_ID": df["CUST_ID"], "sr": sr})
            agg = self._merge_feature(
                agg,
                sr_df.groupby("CUST_ID")["sr"].mean().rename("self_cham_dmnd_ratio").reset_index(),
            )

        agg = self._merge_feature(
            agg, self._numeric_summary_limited(df, "VLID_HOSP_OTDA", "valid_hosp_days", {"sum", "mean", "max", "std"})
        )
        if "valid_hosp_days_sum" in agg.columns:
            agg["valid_hosp_days_sum_log"] = np.log1p(agg["valid_hosp_days_sum"].clip(lower=0))
        agg = self._merge_feature(
            agg, self._numeric_summary_limited(df, "hosp_days_calc", "hosp_days_calc", {"mean", "max", "sum"})
        )
        agg = self._merge_feature(agg, self._missing_rate(df, "hosp_days_calc", "hosp_period_isna_rate"))

        agg = self._merge_feature(
            agg,
            self._numeric_summary_limited(
                df, "delay_origin_to_recp", "delay_origin_to_recp", {"mean", "max", "std"}
            ),
        )
        agg = self._merge_feature(
            agg,
            self._numeric_summary_limited(df, "delay_resn_to_recp", "delay_resn_to_recp", {"mean", "max"}),
        )
        if self.include_leakage_features:
            agg = self._merge_feature(
                agg,
                self._numeric_summary_limited(
                    df, "delay_recp_to_paym", "delay_recp_to_paym", {"mean", "max", "std"}
                ),
            )
        agg = self._merge_feature(agg, self._claim_interval_agg(df))

        if "HEED_HOSP_YN" in df.columns:
            agg = self._merge_feature(agg, self._binary_rate(df, "HEED_HOSP_YN", "heed_hosp_rate"))
            has_h = df.assign(_h=df["HEED_HOSP_YN"].astype("string").str.upper().eq("Y").astype(int))
            agg = self._merge_feature(
                agg,
                has_h.groupby("CUST_ID")["_h"].max().rename("has_heed_hosp").reset_index(),
            )

        if "ACCI_DVSN" in df.columns:
            agg = self._merge_feature(agg, self._nunique_feature(df, "ACCI_DVSN", "acci_dvsn_nunique"))
            agg = self._merge_feature(agg, self._main_ratio(df, "ACCI_DVSN", "main_acci_dvsn_ratio"))
            if "claim_cnt" in agg.columns:
                agg["acci_dvsn_per_claim"] = self._safe_divide(agg["acci_dvsn_nunique"], agg["claim_cnt"])

        if "DMND_RESN_CODE" in df.columns:
            agg = self._merge_feature(
                agg, self._nunique_feature(df, "DMND_RESN_CODE", "dmnd_resn_code_nunique")
            )
            agg = self._merge_feature(
                agg, self._main_ratio(df, "DMND_RESN_CODE", "main_dmnd_resn_code_ratio")
            )
            if "claim_cnt" in agg.columns:
                agg["dmnd_resn_code_per_claim"] = self._safe_divide(
                    agg["dmnd_resn_code_nunique"], agg["claim_cnt"]
                )

        if {"ACCI_DVSN", "DMND_RESN_CODE"}.issubset(df.columns):
            df["_acci_dmnd"] = (
                df["ACCI_DVSN"].astype("string") + "||" + df["DMND_RESN_CODE"].astype("string")
            )
            agg = self._merge_feature(agg, self._nunique_feature(df, "_acci_dmnd", "acci_dmnd_combo_nunique"))
            agg = self._merge_feature(agg, self._main_ratio(df, "_acci_dmnd", "main_acci_dmnd_combo_ratio"))

        if self.include_leakage_features:
            if "PAYM_AMT" in df.columns:
                agg = self._merge_feature(
                    agg, self._numeric_summary_limited(df, "PAYM_AMT", "pay", {"sum", "mean", "max", "std"})
                )
            for col, name in [
                ("pay_ratio", "pay_ratio"),
                ("amount_gap", "amount_gap"),
            ]:
                if col in df.columns:
                    agg = self._merge_feature(
                        agg, self._numeric_summary_limited(df, col, name, {"mean", "max", "std"})
                    )

        if {"claim_cnt", "policy_nunique"}.issubset(agg.columns):
            agg["claim_per_policy"] = self._safe_divide(agg["claim_cnt"], agg["policy_nunique"])
            agg["policy_per_claim"] = self._safe_divide(agg["policy_nunique"], agg["claim_cnt"])
        if {"claim_cnt", "claim_active_days"}.issubset(agg.columns):
            years = (agg["claim_active_days"] / 365.25).replace(0, np.nan)
            agg["claim_per_year"] = self._safe_divide(agg["claim_cnt"], years)
        if {"claim_cnt", "valid_hosp_days_sum"}.issubset(agg.columns):
            agg["valid_days_per_claim"] = self._safe_divide(agg["valid_hosp_days_sum"], agg["claim_cnt"])
        if {"dmnd_sum", "valid_hosp_days_sum"}.issubset(agg.columns):
            agg["dmnd_per_valid_day"] = self._safe_divide(agg["dmnd_sum"], agg["valid_hosp_days_sum"])

        num_cols = [c for c in agg.columns if c != "CUST_ID"]
        agg[num_cols] = agg[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        return agg

    def _claim_count_features(self, claim: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame({"CUST_ID": claim["CUST_ID"].drop_duplicates().sort_values().values})
        features = features.merge(
            claim.groupby("CUST_ID").size().rename("claim_cnt").reset_index(),
            on="CUST_ID",
            how="left",
        )
        if "POLY_NO" in claim.columns:
            features = features.merge(
                claim.groupby("CUST_ID")["POLY_NO"].nunique().rename("policy_nunique").reset_index(),
                on="CUST_ID",
                how="left",
            )
        if "RECP_DATE" in claim.columns:
            span = claim.groupby("CUST_ID")["RECP_DATE"].agg(["min", "max"]).reset_index()
            span["claim_active_days"] = (span["max"] - span["min"]).dt.days.fillna(0)
            features = features.merge(span[["CUST_ID", "claim_active_days"]], on="CUST_ID", how="left")
            rc = claim.sort_values(["CUST_ID", "RECP_DATE"]).copy()
            latest = rc.groupby("CUST_ID")["RECP_DATE"].transform("max")
            rc["recent_claim_flag"] = ((latest - rc["RECP_DATE"]).dt.days.le(90)).fillna(False).astype(int)
            features = features.merge(
                rc.groupby("CUST_ID")["recent_claim_flag"]
                .sum()
                .rename("recent_claim_cnt")
                .reset_index(),
                on="CUST_ID",
                how="left",
            )
            features = features.merge(
                claim.groupby("CUST_ID")["recp_month"].nunique().rename("recp_month_nunique").reset_index(),
                on="CUST_ID",
                how="left",
            )
            features = features.merge(
                claim.groupby("CUST_ID")["recp_year"].nunique().rename("recp_year_nunique").reset_index(),
                on="CUST_ID",
                how="left",
            )
            features = features.merge(
                claim.groupby("CUST_ID")["is_weekend_claim"].mean().rename("weekend_claim_rate").reset_index(),
                on="CUST_ID",
                how="left",
            )
        return features

    def _categorical_group_summary(
        self, claim: pd.DataFrame, base_col: str, prefix: str, ratio_name: str
    ) -> pd.DataFrame:
        if base_col not in claim.columns:
            return pd.DataFrame(columns=["CUST_ID"])
        base = pd.DataFrame({"CUST_ID": claim["CUST_ID"].drop_duplicates().sort_values().values})
        base = base.merge(
            claim.groupby("CUST_ID")[base_col].nunique().rename(f"{prefix}_nunique").reset_index(),
            on="CUST_ID",
            how="left",
        )
        base = base.merge(self._main_ratio(claim, base_col, ratio_name), on="CUST_ID", how="left")
        return base

    def _frequency_summary_mean_max(self, claim: pd.DataFrame, source_col: str, prefix: str) -> pd.DataFrame:
        if source_col not in claim.columns:
            return pd.DataFrame(columns=["CUST_ID"])
        return (
            claim.groupby("CUST_ID")[source_col]
            .agg(["mean", "max"])
            .reset_index()
            .rename(columns={"mean": f"{prefix}_mean", "max": f"{prefix}_max"})
        )

    def _dmnd_amount_summary(self, claim: pd.DataFrame) -> pd.DataFrame:
        if "DMND_AMT" not in claim.columns:
            return pd.DataFrame(columns=["CUST_ID"])
        g = claim.groupby("CUST_ID")["DMND_AMT"].agg(["sum", "mean", "max", "std"]).reset_index()
        g = g.rename(
            columns={
                "sum": "dmnd_sum",
                "mean": "dmnd_mean",
                "max": "dmnd_max",
                "std": "dmnd_std",
            }
        )
        g["dmnd_sum_log"] = np.log1p(g["dmnd_sum"].clip(lower=0))
        g["dmnd_mean_log"] = np.log1p(g["dmnd_mean"].clip(lower=0))
        g["dmnd_max_log"] = np.log1p(g["dmnd_max"].clip(lower=0))
        return g

    def _numeric_summary_limited(
        self, claim: pd.DataFrame, source_col: str, prefix: str, stats: set
    ) -> pd.DataFrame:
        if source_col not in claim.columns:
            return pd.DataFrame(columns=["CUST_ID"])
        agg_dict = {s: s for s in stats if s in {"sum", "mean", "max", "min", "std", "median"}}
        if not agg_dict:
            return pd.DataFrame(columns=["CUST_ID"])
        g = claim.groupby("CUST_ID")[source_col].agg(list(agg_dict.keys())).reset_index()
        rename = {k: f"{prefix}_{k}" for k in agg_dict}
        return g.rename(columns=rename)

    def _claim_interval_agg(self, claim: pd.DataFrame) -> pd.DataFrame:
        if "claim_interval" not in claim.columns:
            return pd.DataFrame(columns=["CUST_ID"])
        return (
            claim.groupby("CUST_ID")["claim_interval"]
            .agg(["mean", "min", "max", "std"])
            .reset_index()
            .rename(
                columns={
                    "mean": "claim_interval_mean",
                    "min": "claim_interval_min",
                    "max": "claim_interval_max",
                    "std": "claim_interval_std",
                }
            )
        )

    def _main_ratio(self, claim: pd.DataFrame, source_col: str, feature_name: str) -> pd.DataFrame:
        if source_col not in claim.columns:
            return pd.DataFrame(columns=["CUST_ID"])

        def top_share(s: pd.Series) -> float:
            if len(s) == 0:
                return 0.0
            return float(s.value_counts(dropna=False).max() / len(s))

        agg_series = claim.groupby("CUST_ID")[source_col].agg(top_share)
        return agg_series.rename(feature_name).reset_index()

    def _binary_rate(self, claim: pd.DataFrame, source_col: str, feature_name: str) -> pd.DataFrame:
        if source_col not in claim.columns:
            return pd.DataFrame(columns=["CUST_ID"])
        r = claim.groupby("CUST_ID")[source_col].apply(
            lambda x: x.astype("string").str.upper().eq("Y").mean()
        )
        return r.rename(feature_name).reset_index()

    def _nunique_feature(self, claim: pd.DataFrame, source_col: str, feature_name: str) -> pd.DataFrame:
        if source_col not in claim.columns:
            return pd.DataFrame(columns=["CUST_ID"])
        return claim.groupby("CUST_ID")[source_col].nunique().rename(feature_name).reset_index()

    def _missing_rate(self, claim: pd.DataFrame, source_col: str, feature_name: str) -> pd.DataFrame:
        if source_col not in claim.columns:
            return pd.DataFrame(columns=["CUST_ID"])
        return claim.groupby("CUST_ID")[source_col].apply(lambda x: x.isna().mean()).rename(feature_name).reset_index()

    def _merge_feature(self, base: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
        if feat.empty:
            return base
        return base.merge(feat, on="CUST_ID", how="left")

    # --- K-fold target encoding (DOC/HOSP SIU) --------------------------------------

    def _attach_kfold_target_encoding(
        self,
        cust: pd.DataFrame,
        claim: pd.DataFrame,
        y: pd.Series,
        divided: pd.Series | None,
        _ohe_dummy_cols: list[str],
    ) -> pd.DataFrame:
        if "CHME_LICE_NO" not in claim.columns and "HOSP_CODE" not in claim.columns:
            return cust

        y_int = pd.to_numeric(y, errors="coerce")
        cy = pd.DataFrame({"CUST_ID": cust["CUST_ID"].values, "y_te": y_int.values})
        mask = cy["y_te"].notna()
        if divided is not None:
            d = divided.astype("string").str.strip()
            mask &= d.eq("1") | d.eq("1.0")
        train_ids = cy.loc[mask, "CUST_ID"].unique()
        if len(train_ids) < self.target_encode_n_splits:
            return cust

        global_mean = float(cy.loc[mask, "y_te"].mean())
        folds = KFold(n_splits=self.target_encode_n_splits, shuffle=True, random_state=42)
        doc_oof = pd.Series(np.nan, index=claim.index, dtype=float)
        hosp_oof = pd.Series(np.nan, index=claim.index, dtype=float)
        combo_oof = pd.Series(np.nan, index=claim.index, dtype=float)

        for tr, va in folds.split(train_ids):
            tr_set = set(train_ids[tr])
            va_set = set(train_ids[va])
            tr_claim = claim[claim["CUST_ID"].isin(tr_set)]
            va_claim = claim[claim["CUST_ID"].isin(va_set)]
            if va_claim.empty:
                continue

            if "CHME_LICE_NO" in claim.columns:
                m = self._group_target_mean(tr_claim, "CHME_LICE_NO", cy)
                enc = self._map_encoding(va_claim["CHME_LICE_NO"], m, global_mean)
                doc_oof.loc[va_claim.index] = enc.to_numpy()
            if "HOSP_CODE" in claim.columns:
                m = self._group_target_mean(tr_claim, "HOSP_CODE", cy)
                enc = self._map_encoding(va_claim["HOSP_CODE"], m, global_mean)
                hosp_oof.loc[va_claim.index] = enc.to_numpy()
            if {"CHME_LICE_NO", "HOSP_CODE"}.issubset(claim.columns):
                tr_k = tr_claim.assign(
                    _k=tr_claim["HOSP_CODE"].astype("string") + "||" + tr_claim["CHME_LICE_NO"].astype("string")
                )
                va_k = va_claim.assign(
                    _k=va_claim["HOSP_CODE"].astype("string") + "||" + va_claim["CHME_LICE_NO"].astype("string")
                )
                m = self._group_target_mean(tr_k, "_k", cy)
                enc = self._map_encoding(va_k["_k"], m, global_mean)
                combo_oof.loc[va_claim.index] = enc.to_numpy()

        tr_claim_all = claim[claim["CUST_ID"].isin(train_ids)]
        doc_full = (
            self._group_target_mean(tr_claim_all, "CHME_LICE_NO", cy)
            if "CHME_LICE_NO" in claim.columns
            else {}
        )
        hosp_full = (
            self._group_target_mean(tr_claim_all, "HOSP_CODE", cy) if "HOSP_CODE" in claim.columns else {}
        )
        if {"CHME_LICE_NO", "HOSP_CODE"}.issubset(claim.columns) and not tr_claim_all.empty:
            ck = tr_claim_all.assign(
                _k=tr_claim_all["HOSP_CODE"].astype("string") + "||" + tr_claim_all["CHME_LICE_NO"].astype("string")
            )
            combo_full = self._group_target_mean(ck, "_k", cy)
        else:
            combo_full = {}

        out = cust.copy()
        if "CHME_LICE_NO" in claim.columns:
            doc_fill = doc_oof.where(
                doc_oof.notna(), self._map_encoding(claim["CHME_LICE_NO"], doc_full, global_mean)
            )
            doc_feat = (
                pd.DataFrame({"CUST_ID": claim["CUST_ID"].to_numpy(), "_v": doc_fill.to_numpy()})
                .groupby("CUST_ID")["_v"]
                .mean()
                .rename("DOC_SIU_RATIO")
                .reset_index()
            )
            out = out.merge(doc_feat, on="CUST_ID", how="left")
        if "HOSP_CODE" in claim.columns:
            hosp_fill = hosp_oof.where(
                hosp_oof.notna(), self._map_encoding(claim["HOSP_CODE"], hosp_full, global_mean)
            )
            hosp_feat = (
                pd.DataFrame({"CUST_ID": claim["CUST_ID"].to_numpy(), "_v": hosp_fill.to_numpy()})
                .groupby("CUST_ID")["_v"]
                .mean()
                .rename("HOSP_SIU_RATIO")
                .reset_index()
            )
            out = out.merge(hosp_feat, on="CUST_ID", how="left")
        if {"CHME_LICE_NO", "HOSP_CODE"}.issubset(claim.columns):
            ck_all = claim.assign(
                _k=claim["HOSP_CODE"].astype("string") + "||" + claim["CHME_LICE_NO"].astype("string")
            )
            combo_fill = combo_oof.where(
                combo_oof.notna(), self._map_encoding(ck_all["_k"], combo_full, global_mean)
            )
            combo_feat = (
                pd.DataFrame({"CUST_ID": claim["CUST_ID"].to_numpy(), "_v": combo_fill.to_numpy()})
                .groupby("CUST_ID")["_v"]
                .mean()
                .rename("HOSP_DOC_SIU")
                .reset_index()
            )
            out = out.merge(combo_feat, on="CUST_ID", how="left")

        return out

    @staticmethod
    def _group_target_mean(claim_part: pd.DataFrame, col: str, cy: pd.DataFrame) -> dict:
        if claim_part.empty or col not in claim_part.columns:
            return {}
        m = claim_part[["CUST_ID", col]].merge(cy, on="CUST_ID", how="left")
        raw = m.groupby(col, dropna=False)["y_te"].mean().to_dict()
        return {str(k): v for k, v in raw.items()}

    @staticmethod
    def _map_encoding(series: pd.Series, mapping: dict, default: float) -> pd.Series:
        if not mapping:
            return pd.Series(default, index=series.index, dtype=float)
        return series.astype("string").map(mapping).astype(float).fillna(default)

    # --- scaling / cap helpers ------------------------------------------------------

    def _cap_excluded_columns(self, columns: pd.Index, ohe_dummy_cols: list[str]) -> set[str]:
        ex: set[str] = set()
        for c in columns:
            if c.endswith("_isna"):
                ex.add(c)
        ex.update(ohe_dummy_cols)
        return ex

    def _apply_standard_scaler(self, cust: pd.DataFrame, ohe_dummy_cols: list[str]) -> pd.DataFrame:
        prefer = {
            "RESI_COST",
            "TOTALPREM",
            "MAX_PRM",
            "CUST_INCM",
            "RCBASE_HSHD_INCM",
            "JPBASE_HSHD_INCM",
            "RESI_COST_log",
            "TOTALPREM_log",
            "MAX_PRM_log",
            "CUST_INCM_log",
            "RCBASE_HSHD_INCM_log",
            "JPBASE_HSHD_INCM_log",
        }
        num = [
            c
            for c in cust.columns
            if c != "CUST_ID" and pd.api.types.is_numeric_dtype(cust[c])
        ]
        scale_cols = []
        for c in num:
            if c in self._cap_excluded_columns(cust.columns, ohe_dummy_cols):
                continue
            if cust[c].nunique(dropna=True) <= 2:
                continue
            if c in prefer or c.startswith("dmnd_") or c.startswith("non_pay_") or c.startswith("self_cham_"):
                scale_cols.append(c)
        scale_cols = sorted(set(scale_cols))
        if not scale_cols:
            return cust
        scaler = StandardScaler()
        arr = scaler.fit_transform(cust[scale_cols].astype(float))
        cust[scale_cols] = arr
        return cust

    # --- small utils ----------------------------------------------------------------

    def _coerce_numeric(self, df: pd.DataFrame, cols):
        for col in self._existing_columns(df, cols):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    def _coerce_dates(self, df: pd.DataFrame, cols):
        for col in self._existing_columns(df, cols):
            df[col] = pd.to_datetime(
                df[col].astype("Int64").astype("string"),
                format="%Y%m%d",
                errors="coerce",
            )

    def _coerce_year_month(self, df: pd.DataFrame, cols):
        for col in self._existing_columns(df, cols):
            df[col] = pd.to_datetime(
                df[col].astype("Int64").astype("string"),
                format="%Y%m",
                errors="coerce",
            )

    def _add_missing_indicators(self, df: pd.DataFrame, mapping: dict):
        for source_col, target_col in mapping.items():
            if source_col in df.columns:
                df[target_col] = df[source_col].isna().astype(int)

    def _add_log_feature(self, df: pd.DataFrame, source_col: str, target_col: str):
        if source_col in df.columns:
            df[target_col] = np.log1p(pd.to_numeric(df[source_col], errors="coerce").clip(lower=0))

    def _add_global_frequency(self, df: pd.DataFrame, source_col: str, target_col: str):
        if source_col not in df.columns:
            return
        freq = df[source_col].astype("string").value_counts(normalize=True, dropna=False)
        df[target_col] = df[source_col].astype("string").map(freq).fillna(0)

    def _existing_columns(self, df: pd.DataFrame, cols):
        return [c for c in cols if c in df.columns]

    def _safe_divide(self, numerator, denominator):
        if isinstance(denominator, (int, float)):
            if denominator == 0:
                return 0 * numerator
            return numerator / denominator
        d = denominator.replace(0, np.nan)
        r = numerator / d
        if isinstance(r, pd.Series):
            return r.replace([np.inf, -np.inf], np.nan)
        return r
