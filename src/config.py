from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "original_data.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

TARGET_COL = "SIU_CUST_YN"
ID_COL = "CUST_ID"
DROP_COLS = ["CUST_ID", "DIVIDED_SET"]

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
