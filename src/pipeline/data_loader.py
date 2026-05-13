import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CLAIM_DATA_PATH,
    CUST_DATA_PATH,
    DIVIDED_SET_COL,
    DROP_COLS,
    RANDOM_STATE,
    RAW_DATA_ENCODING,
    TARGET_COL,
    TEST_SIZE,
)


def _read_csv(path, encoding=None) -> pd.DataFrame:
    if encoding:
        return pd.read_csv(path, encoding=encoding)
    return pd.read_csv(path)


def load_customer_data() -> pd.DataFrame:
    """현재 프로젝트 구조에 맞는 고객 원천 데이터를 불러온다."""
    return _read_csv(CUST_DATA_PATH, encoding=RAW_DATA_ENCODING)


def load_claim_data() -> pd.DataFrame:
    """현재 프로젝트 구조에 맞는 청구 원천 데이터를 불러온다."""
    return _read_csv(CLAIM_DATA_PATH, encoding=RAW_DATA_ENCODING)


def normalize_target(y: pd.Series) -> pd.Series:
    # 원천/가공 CSV 어디서 읽어와도 동일하게 0/1 라벨로 맞춘다.
    normalized = y.astype("string").str.strip().str.upper()
    normalized = normalized.replace({"1.0": "1", "0.0": "0"})
    normalized = normalized.map({"Y": 1, "N": 0, "1": 1, "0": 0})
    return normalized.astype("Int64")


def load_and_split(processed_csv: str):
    """전처리된 CSV에서 학습 가능한 라벨 데이터만 골라 train/test로 나눈다."""
    df = _read_csv(processed_csv)
    if TARGET_COL not in df.columns:
        raise ValueError(f"전처리 CSV에는 타깃 컬럼이 반드시 있어야 합니다: {TARGET_COL}")

    y = normalize_target(df[TARGET_COL])
    labeled_mask = y.notna()
    if DIVIDED_SET_COL in df.columns:
        # 평가용 unlabeled 데이터는 제외하고 학습 대상만 사용한다.
        # CSV에서 1.0 등으로 읽히면 문자열 "1" 비교와 어긋나 0행이 될 수 있어 숫자 비교로 통일한다.
        ds = pd.to_numeric(df[DIVIDED_SET_COL], errors="coerce")
        labeled_mask &= ds == 1

    df = df.loc[labeled_mask].copy()
    y = y.loc[labeled_mask].astype(int)
    if df.empty:
        raise ValueError("전처리 CSV에서 학습 가능한 라벨 행을 찾지 못했습니다.")

    X = df.drop(columns=[TARGET_COL] + [c for c in DROP_COLS if c in df.columns])
    train_X, test_X, train_y, test_y = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return train_X, test_X, train_y, test_y
