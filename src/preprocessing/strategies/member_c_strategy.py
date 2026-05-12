import numpy as np
import pandas as pd

from src.preprocessing.components.encoder import OneHotEncoder
from src.preprocessing.components.missing_value import MedianImputer, ZeroImputer
from src.preprocessing.components.outlier import QuantileCapper
from src.preprocessing.strategies.base_strategy import BaseStrategy


class MemberCStrategy(BaseStrategy):
    """
    member_c 첫 번째 전략

    설계 원칙
    1. 최종 예측 단위가 고객이므로 CLAIM_DATA는 반드시 고객 단위로 집계한다.
    2. 고카디널리티 원본값은 직접 원-핫하지 않고, 개수/집중도/빈도 통계로 요약한다.
    3. 결측치는 삭제하지 않고 indicator를 남긴 뒤 수치형은 중앙값 또는 0으로 보완한다.
    4. 지급금액/지급일/심사결과처럼 사후 정보 성격이 강한 변수는 옵션으로 분리한다.
    """

    def __init__(self, include_leakage_features: bool = False):
        # 기본값은 False로 두어, 사후 정보가 들어갈 수 있는 집계는 제외한다.
        self.include_leakage_features = include_leakage_features

    def get_strategy_name(self) -> str:
        suffix = "with_leakage" if self.include_leakage_features else "safe"
        return f"C: customer-claim aggregate ({suffix})"

    def preprocess(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        claim_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        if claim_df is None:
            raise ValueError("member_c_strategy 는 claim_df가 필요합니다.")

        cust = X.copy()
        claim = claim_df.copy()

        # 1) 고객 원천 데이터에서 고객 자체 특징을 먼저 만든다.
        #    고객 단위 특징은 청구 이력 유무와 관계없이 항상 존재하므로 베이스 테이블 역할을 한다.
        cust = self._prepare_customer_features(cust)

        # 2) 청구 원천 데이터는 고객 단위로 요약해 붙인다.
        #    개별 청구 row를 그대로 쓰면 고객 단위 예측과 맞지 않기 때문에 반드시 집계가 필요하다.
        claim_agg = self._build_claim_aggregates(claim)
        cust = cust.merge(claim_agg, on="CUST_ID", how="left")

        # 3) 청구 이력이 없는 고객도 모델 입력에 포함되어야 하므로
        #    청구 집계 컬럼은 0으로 채운다.
        claim_cols = [col for col in claim_agg.columns if col != "CUST_ID"]
        zero_imputer = ZeroImputer(cols=claim_cols)
        cust = zero_imputer.transform(cust)

        # 4) 낮은 카디널리티 범주형만 원-핫 인코딩한다.
        #    RESI_TYPE_CODE, CTPR, OCCP_GRP_1 정도는 차원이 폭발하지 않으면서 의미를 유지할 수 있다.
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
        if cat_cols:
            encoder = OneHotEncoder(cols=cat_cols)
            encoder.fit(cust)
            cust = encoder.transform(cust)

        # 5) 최종 CSV는 모두 숫자형이어야 하므로 수치형 결측은 중앙값으로 정리한다.
        #    고객 원천값에서 생기는 NaN은 "값이 비어 있었다"는 indicator를 이미 따로 만들었기 때문에
        #    모델 입력용 실제 값은 중앙값으로 보강하는 편이 안정적이다.
        numeric_cols = [
            col
            for col in cust.columns
            if col not in {"CUST_ID", "DIVIDED_SET"}
            and pd.api.types.is_numeric_dtype(cust[col])
        ]
        median_imputer = MedianImputer(cols=numeric_cols)
        median_imputer.fit(cust)
        cust = median_imputer.transform(cust)

        # 6) 금액/횟수 집계는 분포가 매우 치우쳐 있으므로 상하위 극단값을 완만하게 잘라준다.
        #    삭제가 아니라 cap 처리로 두어 정보 손실을 최소화한다.
        cap_cols = [
            col for col in numeric_cols
            if not col.endswith("_isna")
            and not col.endswith("_freq")
            and not col.startswith("DIVIDED_SET")
        ]
        capper = QuantileCapper(cols=cap_cols, lower=0.01, upper=0.99)
        capper.fit(cust)
        cust = capper.transform(cust)

        # 7) 혹시 남아 있을 수 있는 bool 타입은 명시적으로 0/1 정수형으로 바꾼다.
        bool_cols = cust.select_dtypes(include="bool").columns.tolist()
        if bool_cols:
            cust[bool_cols] = cust[bool_cols].astype(int)

        # 8) 마지막 안전장치:
        #    문자열 컬럼은 모델 입력에 들어가면 안 되므로, 사용하지 않은 원본 텍스트 컬럼은 제거한다.
        non_numeric_cols = cust.select_dtypes(exclude="number").columns.tolist()
        if non_numeric_cols:
            cust = cust.drop(columns=non_numeric_cols)

        return cust

    def _prepare_customer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        cust = df.copy()

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
        self._coerce_numeric(cust, numeric_cols)

        # 결측 자체가 신호일 수 있으므로 indicator를 먼저 만든다.
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
            },
        )

        # 낮은 카디널리티 범주형은 Unknown으로 통일해 결측을 정보로 남긴다.
        categorical_fill_cols = [
            "SEX",
            "RESI_TYPE_CODE",
            "CTPR",
            "OCCP_GRP_1",
            "WEDD_YN",
            "MATE_OCCP_GRP_1",
        ]
        for col in self._existing_columns(cust, categorical_fill_cols):
            cust[col] = cust[col].astype("string").fillna("Unknown")

        # FP_CAREER는 Y/N 성격이라 별도 flag로 단순화하는 편이 해석과 안정성에 유리하다.
        if "FP_CAREER" in cust.columns:
            cust["fp_career_flag"] = cust["FP_CAREER"].astype("string").str.upper().eq("Y").astype(int)

        # 연속형 나이는 그대로도 쓰되, 생애주기 성격을 보강하기 위해 그룹 변수도 함께 만든다.
        if "AGE" in cust.columns:
            cust["AGE_GROUP"] = pd.cut(
                cust["AGE"],
                bins=[0, 20, 30, 40, 50, 60, 70, 120],
                labels=["10s", "20s", "30s", "40s", "50s", "60s", "70plus"],
                include_lowest=True,
            ).astype("string").fillna("Unknown")

        if "LTBN_CHLD_AGE" in cust.columns:
            cust["LTBN_CHLD_AGE_GROUP"] = pd.cut(
                cust["LTBN_CHLD_AGE"],
                bins=[-1, 0, 7, 13, 19, 120],
                labels=["none", "infant", "child", "teen", "adult_child"],
                include_lowest=True,
            ).astype("string").fillna("Unknown")

        if "CHLD_CNT" in cust.columns:
            cust["has_child"] = cust["CHLD_CNT"].fillna(0).gt(0).astype(int)

        # 고객 가입 시점과 마지막 납입월은 장기 유지 여부를 보는 데 도움된다.
        self._coerce_year_month(cust, ["CUST_RGST", "MAX_PAYM_YM"])
        if {"CUST_RGST", "MAX_PAYM_YM"}.issubset(cust.columns):
            cust["payment_active_months"] = (
                (cust["MAX_PAYM_YM"] - cust["CUST_RGST"]).dt.days.div(30.44)
            ).fillna(0)
            cust["register_year"] = cust["CUST_RGST"].dt.year.fillna(0)
            cust["register_month"] = cust["CUST_RGST"].dt.month.fillna(0)

        # 금액형은 원본과 로그형을 같이 둔다.
        # 원본은 절대 규모, 로그형은 극단값 완화 후 상대 차이를 표현하는 데 유리하다.
        self._add_log_feature(cust, "CUST_INCM", "CUST_INCM_log")
        self._add_log_feature(cust, "TOTALPREM", "TOTALPREM_log")
        self._add_log_feature(cust, "MAX_PRM", "MAX_PRM_log")
        self._add_log_feature(cust, "RESI_COST", "RESI_COST_log")

        if {"TOTALPREM", "CUST_INCM"}.issubset(cust.columns):
            cust["premium_income_ratio"] = self._safe_divide(cust["TOTALPREM"], cust["CUST_INCM"])
        if {"MAX_PRM", "TOTALPREM"}.issubset(cust.columns):
            cust["max_prm_totalprem_ratio"] = self._safe_divide(cust["MAX_PRM"], cust["TOTALPREM"])
        if {"MAXCRDT", "MINCRDT"}.issubset(cust.columns):
            cust["credit_gap"] = cust["MAXCRDT"].fillna(0) - cust["MINCRDT"].fillna(0)

        # 모델 입력에는 쓰지 않을 원본 텍스트/중복 정보 컬럼은 정리한다.
        # OCCP_GRP_2, MATE_OCCP_GRP_2 는 텍스트 상태 그대로 두기보다 1차 그룹을 활용하는 편이 간결하다.
        cust = cust.drop(
            columns=self._existing_columns(
                cust,
                ["FP_CAREER", "OCCP_GRP_2", "MATE_OCCP_GRP_2", "CUST_RGST", "MAX_PAYM_YM"],
            ),
            errors="ignore",
        )
        return cust

    def _build_claim_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
        claim = df.copy()

        if "CUST_ID" not in claim.columns:
            raise ValueError("CLAIM_DATA에는 CUST_ID가 반드시 있어야 합니다.")

        # 원본 CLAIM_DATA에서 실제로 존재하는 수치형 컬럼만 숫자로 바꾼다.
        self._coerce_numeric(
            claim,
            [
                "DMND_AMT",
                "PAYM_AMT",
                "NON_PAY",
                "SELF_CHAM",
                "PATT_CHRG_TOTA",
                "NON_PAY_RATIO",
                "VLID_HOSP_OTDA",
                "COUNT_TRMT_ITEM",
            ],
        )

        # 날짜형은 지연일/간격/활동기간 파생변수의 기반이 되므로 먼저 변환한다.
        self._coerce_dates(
            claim,
            [
                "RECP_DATE",
                "ORIG_RESN_DATE",
                "RESN_DATE",
                "HOSP_OTPA_STDT",
                "HOSP_OTPA_ENDT",
                "PAYM_DATE",
            ],
        )

        # 고유값이 많은 코드류는 직접 인코딩하지 않고, 전역 등장 빈도로 수치화한다.
        self._add_global_frequency(
            claim,
            "HOSP_CODE",
            "HOSP_CODE_freq",
        )
        self._add_global_frequency(
            claim,
            "CHME_LICE_NO",
            "CHME_LICE_NO_freq",
        )
        self._add_global_frequency(
            claim,
            "DSAS_NAME",
            "DSAS_NAME_freq",
        )
        self._add_global_frequency(
            claim,
            "CAUS_CODE",
            "CAUS_CODE_freq",
        )
        self._add_global_frequency(
            claim,
            "CAUS_CODE_DTAL",
            "CAUS_CODE_DTAL_freq",
        )
        self._add_global_frequency(
            claim,
            "RESL_CD1",
            "RESL_CD1_freq",
        )
        self._add_global_frequency(
            claim,
            "ACCI_HOSP_ADDR",
            "ACCI_HOSP_ADDR_freq",
        )

        # 질병명 길이는 텍스트 원본을 직접 쓰지 않으면서도 정보량을 일부 보존할 수 있다.
        if "DSAS_NAME" in claim.columns:
            claim["disease_name_len"] = claim["DSAS_NAME"].astype("string").str.len()

        # 날짜 차이는 청구 패턴을 요약하는 데 유용하지만, 미래 시점 정보가 포함된 변수는 옵션으로 다룬다.
        if {"ORIG_RESN_DATE", "RECP_DATE"}.issubset(claim.columns):
            claim["delay_origin_to_recp"] = (claim["RECP_DATE"] - claim["ORIG_RESN_DATE"]).dt.days.clip(lower=0)
        if {"RESN_DATE", "RECP_DATE"}.issubset(claim.columns):
            claim["delay_resn_to_recp"] = (claim["RECP_DATE"] - claim["RESN_DATE"]).dt.days.clip(lower=0)
        if {"PAYM_DATE", "RECP_DATE"}.issubset(claim.columns):
            claim["delay_recp_to_paym"] = (claim["PAYM_DATE"] - claim["RECP_DATE"]).dt.days.clip(lower=0)
        if {"HOSP_OTPA_STDT", "HOSP_OTPA_ENDT"}.issubset(claim.columns):
            claim["hosp_days_calc"] = (claim["HOSP_OTPA_ENDT"] - claim["HOSP_OTPA_STDT"]).dt.days.clip(lower=0)

        # 금액/건수처럼 오른쪽 꼬리가 긴 변수는 로그형을 함께 둔다.
        self._add_log_feature(claim, "DMND_AMT", "DMND_AMT_log")
        self._add_log_feature(claim, "PAYM_AMT", "PAYM_AMT_log")
        self._add_log_feature(claim, "VLID_HOSP_OTDA", "valid_hosp_days_log")

        # 지급 관련 비율은 누수 가능성이 있으므로 기본 전략에서는 만들지 않는다.
        if self.include_leakage_features and {"PAYM_AMT", "DMND_AMT"}.issubset(claim.columns):
            claim["pay_ratio"] = self._safe_divide(claim["PAYM_AMT"], claim["DMND_AMT"])
            claim["amount_gap"] = claim["PAYM_AMT"].fillna(0) - claim["DMND_AMT"].fillna(0)
            self._add_log_feature(claim, "amount_gap", "amount_gap_log")

        if {"NON_PAY", "DMND_AMT"}.issubset(claim.columns):
            claim["non_pay_dmnd_ratio"] = self._safe_divide(claim["NON_PAY"], claim["DMND_AMT"])
        if {"SELF_CHAM", "DMND_AMT"}.issubset(claim.columns):
            claim["self_cham_dmnd_ratio"] = self._safe_divide(claim["SELF_CHAM"], claim["DMND_AMT"])

        # 주말 청구 여부는 단순한 날짜 정보에서 얻을 수 있는 패턴 변수다.
        if "RECP_DATE" in claim.columns:
            claim["recp_year"] = claim["RECP_DATE"].dt.year
            claim["recp_month"] = claim["RECP_DATE"].dt.month
            claim["is_weekend_claim"] = claim["RECP_DATE"].dt.weekday.isin([5, 6]).astype(int)

        agg = pd.DataFrame({"CUST_ID": claim["CUST_ID"].drop_duplicates().sort_values().values})

        # 가장 기본이 되는 청구 건수/계약 수/활동 기간 계열
        agg = self._merge_feature(agg, self._claim_count_features(claim))

        # 고카디널리티 컬럼은 개수 + 집중도 + 빈도 통계로 요약
        agg = self._merge_feature(
            agg,
            self._main_ratio(claim, "POLY_NO", "main_policy_claim_ratio"),
        )
        agg = self._merge_feature(
            agg,
            self._categorical_group_summary(
                claim,
                base_col="HOSP_CODE",
                prefix="hospital",
                ratio_name="main_hospital_claim_ratio",
            ),
        )
        agg = self._merge_feature(
            agg,
            self._categorical_group_summary(
                claim,
                base_col="CHME_LICE_NO",
                prefix="doctor",
                ratio_name="main_doctor_claim_ratio",
            ),
        )
        agg = self._merge_feature(
            agg,
            self._categorical_group_summary(
                claim,
                base_col="DSAS_NAME",
                prefix="disease",
                ratio_name="main_disease_claim_ratio",
            ),
        )
        agg = self._merge_feature(
            agg,
            self._categorical_group_summary(
                claim,
                base_col="CAUS_CODE",
                prefix="cause",
                ratio_name="main_cause_claim_ratio",
            ),
        )
        agg = self._merge_feature(
            agg,
            self._categorical_group_summary(
                claim,
                base_col="CAUS_CODE_DTAL",
                prefix="cause_dtal",
                ratio_name="main_cause_dtal_claim_ratio",
            ),
        )
        agg = self._merge_feature(
            agg,
            self._categorical_group_summary(
                claim,
                base_col="ACCI_HOSP_ADDR",
                prefix="addr",
                ratio_name="main_addr_claim_ratio",
            ),
        )

        # 특정 코드에 대한 집중도가 높으면 반복 패턴 신호가 될 수 있다.
        if {"claim_cnt", "hospital_nunique", "main_hospital_claim_ratio"}.issubset(agg.columns):
            agg["same_hospital_repeat_ratio"] = self._safe_divide(
                agg["claim_cnt"] - agg["hospital_nunique"],
                agg["claim_cnt"],
            )
            agg["hospital_concentration"] = agg["main_hospital_claim_ratio"]
        if {"claim_cnt", "main_doctor_claim_ratio"}.issubset(agg.columns):
            agg["doctor_concentration"] = agg["main_doctor_claim_ratio"]
        if {"claim_cnt", "main_disease_claim_ratio"}.issubset(agg.columns):
            agg["disease_concentration"] = agg["main_disease_claim_ratio"]
        if {"claim_cnt", "main_cause_claim_ratio"}.issubset(agg.columns):
            agg["cause_concentration"] = agg["main_cause_claim_ratio"]

        # 고카디널리티 원본값의 전역 빈도를 고객 단위로 요약하면
        # "흔한 병원만 다니는지", "희귀 코드가 많은지" 같은 차이를 숫자로 표현할 수 있다.
        agg = self._merge_feature(
            agg,
            self._frequency_summary(
                claim,
                source_col="HOSP_CODE_freq",
                prefix="hospital_freq",
            ),
        )
        agg = self._merge_feature(
            agg,
            self._frequency_summary(
                claim,
                source_col="CHME_LICE_NO_freq",
                prefix="doctor_freq",
            ),
        )
        agg = self._merge_feature(
            agg,
            self._frequency_summary(
                claim,
                source_col="DSAS_NAME_freq",
                prefix="disease_freq",
            ),
        )
        agg = self._merge_feature(
            agg,
            self._frequency_summary(
                claim,
                source_col="CAUS_CODE_freq",
                prefix="cause_freq",
            ),
        )
        agg = self._merge_feature(
            agg,
            self._frequency_summary(
                claim,
                source_col="ACCI_HOSP_ADDR_freq",
                prefix="hospital_area_freq",
            ),
        )

        # 청구 금액 집계
        agg = self._merge_feature(
            agg,
            self._numeric_summary(claim, "DMND_AMT", "dmnd"),
        )
        agg = self._merge_feature(
            agg,
            self._numeric_summary(claim, "DMND_AMT_log", "dmnd_log"),
        )

        # 지급 관련 변수는 누수 옵션에서만 포함한다.
        if self.include_leakage_features:
            agg = self._merge_feature(
                agg,
                self._numeric_summary(claim, "PAYM_AMT", "pay"),
            )
            agg = self._merge_feature(
                agg,
                self._numeric_summary(claim, "PAYM_AMT_log", "pay_log"),
            )
            agg = self._merge_feature(
                agg,
                self._numeric_summary(claim, "pay_ratio", "pay_ratio"),
            )
            agg = self._merge_feature(
                agg,
                self._numeric_summary(claim, "amount_gap", "amount_gap"),
            )

        # 비급여 / 본인부담금 계열
        agg = self._merge_feature(agg, self._numeric_summary(claim, "NON_PAY", "non_pay"))
        agg = self._merge_feature(agg, self._numeric_summary(claim, "SELF_CHAM", "self_cham"))
        agg = self._merge_feature(agg, self._numeric_summary(claim, "PATT_CHRG_TOTA", "patt_chrg_tota"))
        agg = self._merge_feature(agg, self._numeric_summary(claim, "non_pay_dmnd_ratio", "non_pay_ratio"))
        agg = self._merge_feature(agg, self._numeric_summary(claim, "self_cham_dmnd_ratio", "self_cham_ratio"))
        agg = self._merge_feature(agg, self._missing_rate(claim, "NON_PAY", "non_pay_isna_rate"))
        agg = self._merge_feature(agg, self._missing_rate(claim, "SELF_CHAM", "self_cham_isna_rate"))

        # 입원/통원 관련
        agg = self._merge_feature(agg, self._numeric_summary(claim, "VLID_HOSP_OTDA", "valid_hosp_days"))
        agg = self._merge_feature(agg, self._numeric_summary(claim, "valid_hosp_days_log", "valid_hosp_days_log"))
        agg = self._merge_feature(agg, self._numeric_summary(claim, "hosp_days_calc", "hosp_days_calc"))
        agg = self._merge_feature(agg, self._missing_rate(claim, "hosp_days_calc", "hosp_period_isna_rate"))

        # 날짜/간격 관련
        agg = self._merge_feature(agg, self._numeric_summary(claim, "delay_origin_to_recp", "delay_origin_to_recp"))
        agg = self._merge_feature(agg, self._numeric_summary(claim, "delay_resn_to_recp", "delay_resn_to_recp"))
        if self.include_leakage_features:
            agg = self._merge_feature(agg, self._numeric_summary(claim, "delay_recp_to_paym", "delay_recp_to_paym"))
        agg = self._merge_feature(agg, self._claim_interval_summary(claim))
        agg = self._merge_feature(agg, self._date_span_features(claim))

        # 심사/진행 결과 관련도 누수 옵션에서만 포함한다.
        if self.include_leakage_features:
            agg = self._merge_feature(agg, self._binary_rate(claim, "PMMI_DLNG_YN", "pmmi_rate"))
            agg = self._merge_feature(agg, self._binary_rate(claim, "CHANG_FP_YN", "chang_fp_rate"))
            agg = self._merge_feature(agg, self._nunique_feature(claim, "CRNT_PROG_DVSN", "crnt_prog_nunique"))
            agg = self._merge_feature(agg, self._nunique_feature(claim, "RESL_CD1", "resl_cd_nunique"))
            agg = self._merge_feature(agg, self._nunique_feature(claim, "RESL_NM1", "resl_nm_nunique"))
            agg = self._merge_feature(
                agg,
                self._main_ratio(claim, "RESL_CD1", "main_resl_cd_ratio"),
            )
            agg = self._merge_feature(
                agg,
                self._main_ratio(claim, "RESL_NM1", "main_resl_nm_ratio"),
            )

        # 추가 비율형 파생변수
        if {"claim_cnt", "policy_nunique"}.issubset(agg.columns):
            agg["claim_per_policy"] = self._safe_divide(agg["claim_cnt"], agg["policy_nunique"])
            agg["policy_per_claim"] = self._safe_divide(agg["policy_nunique"], agg["claim_cnt"])
        if {"claim_cnt", "claim_active_days"}.issubset(agg.columns):
            active_years = (agg["claim_active_days"] / 365.25).replace(0, np.nan)
            agg["claim_per_year"] = self._safe_divide(agg["claim_cnt"], active_years)
        if {"claim_cnt", "hospital_nunique"}.issubset(agg.columns):
            agg["hospital_per_claim"] = self._safe_divide(agg["hospital_nunique"], agg["claim_cnt"])
        if {"claim_cnt", "doctor_nunique"}.issubset(agg.columns):
            agg["doctor_per_claim"] = self._safe_divide(agg["doctor_nunique"], agg["claim_cnt"])
        if {"claim_cnt", "disease_nunique"}.issubset(agg.columns):
            agg["disease_per_claim"] = self._safe_divide(agg["disease_nunique"], agg["claim_cnt"])
        if {"claim_cnt", "cause_nunique"}.issubset(agg.columns):
            agg["cause_per_claim"] = self._safe_divide(agg["cause_nunique"], agg["claim_cnt"])
        if {"claim_cnt", "addr_nunique"}.issubset(agg.columns):
            agg["addr_per_claim"] = self._safe_divide(agg["addr_nunique"], agg["claim_cnt"])
        if {"claim_cnt", "valid_hosp_days_sum"}.issubset(agg.columns):
            agg["valid_days_per_claim"] = self._safe_divide(agg["valid_hosp_days_sum"], agg["claim_cnt"])
        if {"dmnd_sum", "valid_hosp_days_sum"}.issubset(agg.columns):
            agg["dmnd_per_valid_day"] = self._safe_divide(agg["dmnd_sum"], agg["valid_hosp_days_sum"])
        if self.include_leakage_features and {"pay_sum", "valid_hosp_days_sum"}.issubset(agg.columns):
            agg["pay_per_valid_day"] = self._safe_divide(agg["pay_sum"], agg["valid_hosp_days_sum"])

        # 치우친 고객 단위 집계는 로그형을 한 번 더 추가한다.
        for source_col, target_col in [
            ("claim_cnt", "claim_cnt_log"),
            ("dmnd_sum", "dmnd_sum_log"),
            ("dmnd_mean", "dmnd_mean_log"),
            ("dmnd_max", "dmnd_max_log"),
            ("valid_hosp_days_sum", "valid_hosp_days_sum_log"),
        ]:
            if source_col in agg.columns:
                agg[target_col] = np.log1p(agg[source_col].clip(lower=0))
        if self.include_leakage_features:
            for source_col, target_col in [
                ("pay_sum", "pay_sum_log"),
                ("pay_mean", "pay_mean_log"),
                ("pay_max", "pay_max_log"),
            ]:
                if source_col in agg.columns:
                    agg[target_col] = np.log1p(agg[source_col].clip(lower=0))

        # 최종 집계 테이블은 숫자형으로 통일한다.
        numeric_cols = [col for col in agg.columns if col != "CUST_ID"]
        agg[numeric_cols] = agg[numeric_cols].replace([np.inf, -np.inf], np.nan)
        agg[numeric_cols] = agg[numeric_cols].fillna(0)
        return agg

    def _claim_count_features(self, claim: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame({"CUST_ID": claim["CUST_ID"].drop_duplicates().sort_values().values})

        claim_cnt = claim.groupby("CUST_ID").size().rename("claim_cnt").reset_index()
        features = features.merge(claim_cnt, on="CUST_ID", how="left")

        if "POLY_NO" in claim.columns:
            policy_nunique = claim.groupby("CUST_ID")["POLY_NO"].nunique().rename("policy_nunique").reset_index()
            features = features.merge(policy_nunique, on="CUST_ID", how="left")

        if "RECP_DATE" in claim.columns:
            recp_span = claim.groupby("CUST_ID")["RECP_DATE"].agg(["min", "max"]).reset_index()
            recp_span["claim_active_days"] = (recp_span["max"] - recp_span["min"]).dt.days.fillna(0)
            features = features.merge(recp_span[["CUST_ID", "claim_active_days"]], on="CUST_ID", how="left")

            recent_claim = claim.sort_values(["CUST_ID", "RECP_DATE"]).copy()
            latest_recp = recent_claim.groupby("CUST_ID")["RECP_DATE"].transform("max")
            recent_claim["recent_claim_flag"] = (
                (latest_recp - recent_claim["RECP_DATE"]).dt.days.le(90)
            ).fillna(False).astype(int)
            recent_claim_cnt = recent_claim.groupby("CUST_ID")["recent_claim_flag"].sum().rename("recent_claim_cnt").reset_index()
            features = features.merge(recent_claim_cnt, on="CUST_ID", how="left")

            recp_month_nunique = claim.groupby("CUST_ID")["recp_month"].nunique().rename("recp_month_nunique").reset_index()
            recp_year_nunique = claim.groupby("CUST_ID")["recp_year"].nunique().rename("recp_year_nunique").reset_index()
            weekend_claim_rate = claim.groupby("CUST_ID")["is_weekend_claim"].mean().rename("weekend_claim_rate").reset_index()
            features = features.merge(recp_month_nunique, on="CUST_ID", how="left")
            features = features.merge(recp_year_nunique, on="CUST_ID", how="left")
            features = features.merge(weekend_claim_rate, on="CUST_ID", how="left")

        return features

    def _categorical_group_summary(
        self,
        claim: pd.DataFrame,
        base_col: str,
        prefix: str,
        ratio_name: str,
    ) -> pd.DataFrame:
        if base_col not in claim.columns:
            return pd.DataFrame(columns=["CUST_ID"])

        grouped = claim.groupby("CUST_ID")[base_col]
        summary = pd.DataFrame({"CUST_ID": claim["CUST_ID"].drop_duplicates().sort_values().values})
        summary = summary.merge(grouped.nunique().rename(f"{prefix}_nunique").reset_index(), on="CUST_ID", how="left")
        summary = summary.merge(self._main_ratio(claim, base_col, ratio_name), on="CUST_ID", how="left")
        return summary

    def _frequency_summary(self, claim: pd.DataFrame, source_col: str, prefix: str) -> pd.DataFrame:
        if source_col not in claim.columns:
            return pd.DataFrame(columns=["CUST_ID"])
        return (
            claim.groupby("CUST_ID")[source_col]
            .agg(["mean", "max", "min"])
            .reset_index()
            .rename(
                columns={
                    "mean": f"{prefix}_mean",
                    "max": f"{prefix}_max",
                    "min": f"{prefix}_min",
                }
            )
        )

    def _numeric_summary(self, claim: pd.DataFrame, source_col: str, prefix: str) -> pd.DataFrame:
        if source_col not in claim.columns:
            return pd.DataFrame(columns=["CUST_ID"])

        summary = (
            claim.groupby("CUST_ID")[source_col]
            .agg(["sum", "mean", "median", "max", "min", "std"])
            .reset_index()
            .rename(
                columns={
                    "sum": f"{prefix}_sum",
                    "mean": f"{prefix}_mean",
                    "median": f"{prefix}_median",
                    "max": f"{prefix}_max",
                    "min": f"{prefix}_min",
                    "std": f"{prefix}_std",
                }
            )
        )
        if f"{prefix}_std" in summary.columns and f"{prefix}_mean" in summary.columns:
            summary[f"{prefix}_cv"] = self._safe_divide(summary[f"{prefix}_std"], summary[f"{prefix}_mean"])
        return summary

    def _claim_interval_summary(self, claim: pd.DataFrame) -> pd.DataFrame:
        if "RECP_DATE" not in claim.columns:
            return pd.DataFrame(columns=["CUST_ID"])

        sorted_claim = claim.sort_values(["CUST_ID", "RECP_DATE"]).copy()
        sorted_claim["prev_recp_date"] = sorted_claim.groupby("CUST_ID")["RECP_DATE"].shift(1)
        sorted_claim["claim_interval"] = (
            sorted_claim["RECP_DATE"] - sorted_claim["prev_recp_date"]
        ).dt.days.clip(lower=0)

        return (
            sorted_claim.groupby("CUST_ID")["claim_interval"]
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

    def _date_span_features(self, claim: pd.DataFrame) -> pd.DataFrame:
        if "RECP_DATE" not in claim.columns:
            return pd.DataFrame(columns=["CUST_ID"])

        grouped = claim.groupby("CUST_ID")["RECP_DATE"]
        span = grouped.agg(["min", "max"]).reset_index()
        span["first_claim_year"] = span["min"].dt.year.fillna(0)
        span["last_claim_year"] = span["max"].dt.year.fillna(0)
        span["first_claim_month"] = span["min"].dt.month.fillna(0)
        span["last_claim_month"] = span["max"].dt.month.fillna(0)
        return span[
            [
                "CUST_ID",
                "first_claim_year",
                "last_claim_year",
                "first_claim_month",
                "last_claim_month",
            ]
        ]

    def _main_ratio(self, claim: pd.DataFrame, source_col: str, feature_name: str) -> pd.DataFrame:
        if source_col not in claim.columns:
            return pd.DataFrame(columns=["CUST_ID"])

        top_share = claim.groupby("CUST_ID")[source_col].apply(
            lambda x: x.value_counts(dropna=False).max() / len(x) if len(x) > 0 else 0
        )
        return top_share.rename(feature_name).reset_index()

    def _binary_rate(self, claim: pd.DataFrame, source_col: str, feature_name: str) -> pd.DataFrame:
        if source_col not in claim.columns:
            return pd.DataFrame(columns=["CUST_ID"])
        rate = claim.groupby("CUST_ID")[source_col].apply(
            lambda x: x.astype("string").str.upper().eq("Y").mean()
        )
        return rate.rename(feature_name).reset_index()

    def _nunique_feature(self, claim: pd.DataFrame, source_col: str, feature_name: str) -> pd.DataFrame:
        if source_col not in claim.columns:
            return pd.DataFrame(columns=["CUST_ID"])
        return claim.groupby("CUST_ID")[source_col].nunique().rename(feature_name).reset_index()

    def _missing_rate(self, claim: pd.DataFrame, source_col: str, feature_name: str) -> pd.DataFrame:
        if source_col not in claim.columns:
            return pd.DataFrame(columns=["CUST_ID"])
        rate = claim.groupby("CUST_ID")[source_col].apply(lambda x: x.isna().mean())
        return rate.rename(feature_name).reset_index()

    def _merge_feature(self, base: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
        if feature_df.empty:
            return base
        return base.merge(feature_df, on="CUST_ID", how="left")

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

    def _add_missing_indicators(self, df: pd.DataFrame, mapping):
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
        return [col for col in cols if col in df.columns]

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
