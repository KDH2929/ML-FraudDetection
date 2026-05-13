from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

CUST_DATA_PATH = BASE_DIR / "data" / "raw" / "CUST_DATA.csv"
CLAIM_DATA_PATH = BASE_DIR / "data" / "raw" / "CLAIM_DATA.csv"
RAW_DATA_ENCODING = "utf-16"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

TARGET_COL = "SIU_CUST_YN"
ID_COL = "CUST_ID"
DIVIDED_SET_COL = "DIVIDED_SET"
# 모델 입력에서 제외할 식별/분할 컬럼 (DIVIDED_SET은 전처리 CSV에서 제거됨)
DROP_COLS = [ID_COL]

RANDOM_STATE = 42
TEST_SIZE = 0.3

CAT_COLS = ["SEX", "RESI_TYPE_CODE", "CTPR", "OCCP_GRP_1", "WEDD_YN", "MATE_OCCP_GRP_1"]
NUM_COLS = ["RESI_COST", "TOTALPREM", "MAX_PRM", "CUST_INCM", "RCBASE_HSHD_INCM", "JPBASE_HSHD_INCM"]

MODEL_PARAMS = {
    "lgbm": {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1,
    },
    "rf": {
        "n_estimators": 300,
        "max_depth": 10,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
    "logistic": {
        "max_iter": 1000,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
}

# 전략별 최적 threshold (F1 maximization)
STRATEGY_THRESHOLDS = {
    "member_b_strategy_3": 0.3819,  # v3 optimized threshold
    "member_c_strategy_4": 0.4642899201825889,
}
