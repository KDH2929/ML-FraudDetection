from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ID_COL
from src.preprocessing.strategies.member_c_strategy_3 import MemberCStrategy3


class MemberCStrategy4(MemberCStrategy3):
    """
    member_c v4 전처리 전략

    safe한 v3 구조를 유지하면서 v1에서 효과가 컸던 지연/변동성/반복성 변수만 선별 복원한다.
    v3.1, v3.2 실험에서 실제로 기여한 변수군만 남겨 최종 후보로 정리한 버전이다.
    """

    def get_strategy_name(self) -> str:
        base = super().get_strategy_name()
        return base.replace("C3", "C4")

    def _prepare_claim_data(
        self,
        claim: pd.DataFrame,
        cust_processed: pd.DataFrame,
        target: pd.Series,
        train_mask: pd.Series,
        train_customer_ids: set,
    ):
        agg, report = super()._prepare_claim_data(
            claim=claim,
            cust_processed=cust_processed,
            target=target,
            train_mask=train_mask,
            train_customer_ids=train_customer_ids,
        )

        df = claim.copy()
        drop_cols = [c for c in self.HIGH_MISSING_CLAIM_COLS if c in df.columns]
        df = df.drop(columns=drop_cols, errors="ignore")

        self._coerce_numeric(
            df,
            [
                "DMND_AMT",
                "VLID_HOSP_OTDA",
                "HOSP_CODE",
                "CHME_LICE_NO",
                "CAUS_CODE",
            ],
        )

        for col in self.CLAIM_DATE_COLS:
            if col in df.columns:
                df[col] = self._parse_yyyymmdd(df[col])

        if {"RECP_DATE", "ORIG_RESN_DATE"}.issubset(df.columns):
            df["delay_origin_to_recp"] = (df["RECP_DATE"] - df["ORIG_RESN_DATE"]).dt.days.clip(lower=0)
        if {"RECP_DATE", "RESN_DATE"}.issubset(df.columns):
            df["delay_resn_to_recp"] = (df["RECP_DATE"] - df["RESN_DATE"]).dt.days.clip(lower=0)
        if {"HOSP_OTPA_STDT", "HOSP_OTPA_ENDT"}.issubset(df.columns):
            df["hosp_days_calc"] = (df["HOSP_OTPA_ENDT"] - df["HOSP_OTPA_STDT"]).dt.days.clip(lower=0)
        if "VLID_HOSP_OTDA" in df.columns:
            df["vlid_hosp_otda_log"] = np.log1p(df["VLID_HOSP_OTDA"].clip(lower=0))

        train_claim_mask = df[ID_COL].isin(train_customer_ids)

        key_map = {}
        for col in ["HOSP_CODE", "CHME_LICE_NO"]:
            if col in df.columns:
                key_col = f"__{col}_KEY"
                df[key_col] = self._canonical_category(df[col])
                key_map[col] = key_col

        for source in ["HOSP_CODE", "CHME_LICE_NO"]:
            if source in key_map:
                mapping = self._fit_frequency_mapping(df.loc[train_claim_mask, key_map[source]])
                df[f"__{source}_FREQ"] = df[key_map[source]].map(mapping).fillna(0.0)

        extra_frames: list[pd.DataFrame] = []

        for source, prefix in [
            ("HOSP_CODE", "hospital_freq"),
            ("CHME_LICE_NO", "doctor_freq"),
        ]:
            freq_col = f"__{source}_FREQ"
            if freq_col in df.columns:
                freq_min = (
                    df.groupby(ID_COL)[freq_col]
                    .min()
                    .rename(f"{prefix}_min")
                    .reset_index()
                )
                extra_frames.append(freq_min)

        for source, prefix in [
            ("delay_origin_to_recp", "delay_origin_to_recp"),
            ("delay_resn_to_recp", "delay_resn_to_recp"),
        ]:
            if source in df.columns:
                delay_stats = (
                    df.groupby(ID_COL)[source]
                    .agg(["min", "median", "max"])
                    .reset_index()
                    .rename(
                        columns={
                            "min": f"{prefix}_min",
                            "median": f"{prefix}_median",
                            "max": f"{prefix}_max",
                        }
                    )
                )
                extra_frames.append(delay_stats)

        if "hosp_days_calc" in df.columns:
            hosp_stats = (
                df.groupby(ID_COL)["hosp_days_calc"]
                .agg(["median", "std"])
                .reset_index()
                .rename(columns={"median": "hosp_days_calc_median", "std": "hosp_days_calc_std"})
            )
            extra_frames.append(hosp_stats)

        if "vlid_hosp_otda_log" in df.columns:
            valid_log_mean = (
                df.groupby(ID_COL)["vlid_hosp_otda_log"]
                .mean()
                .rename("valid_hosp_days_log_mean")
                .reset_index()
            )
            extra_frames.append(valid_log_mean)

        if "DMND_AMT" in df.columns:
            dmnd_cv = (
                df.groupby(ID_COL)["DMND_AMT"]
                .agg(["mean", "std"])
                .reset_index()
            )
            dmnd_cv["dmnd_cv"] = self._safe_divide(dmnd_cv["std"], dmnd_cv["mean"])
            extra_frames.append(dmnd_cv[[ID_COL, "dmnd_cv"]])

        for frame in extra_frames:
            agg = agg.merge(frame, on=ID_COL, how="left")

        if {"claim_count", "claim_active_days"}.issubset(agg.columns):
            active_years = (agg["claim_active_days"] / 365.25).replace(0, np.nan)
            agg["claim_per_year"] = self._safe_divide(agg["claim_count"], active_years)

        if {"dmnd_sum", "hosp_days_sum"}.issubset(agg.columns):
            agg["dmnd_per_valid_day"] = self._safe_divide(agg["dmnd_sum"], agg["hosp_days_sum"])

        numeric_cols = [c for c in agg.columns if c != ID_COL]
        agg[numeric_cols] = agg[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        return agg, report

    def _created_claim_features(self, columns) -> list[str]:
        base = set(super()._created_claim_features(columns))
        extra = {
            "hospital_freq_min",
            "doctor_freq_min",
            "claim_per_year",
            "delay_origin_to_recp_min",
            "delay_origin_to_recp_median",
            "delay_origin_to_recp_max",
            "delay_resn_to_recp_min",
            "delay_resn_to_recp_median",
            "delay_resn_to_recp_max",
            "hosp_days_calc_median",
            "hosp_days_calc_std",
            "dmnd_cv",
            "valid_hosp_days_log_mean",
            "dmnd_per_valid_day",
        }
        return [col for col in sorted(base | extra) if col in columns]
