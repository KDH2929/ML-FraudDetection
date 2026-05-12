import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import TARGET_COL, DROP_COLS, RANDOM_STATE, TEST_SIZE


def load_and_split(processed_csv: str):
    """전처리된 CSV를 로드하고 고정 조건으로 train/test split 수행."""
    df = pd.read_csv(processed_csv)
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL] + [c for c in DROP_COLS if c in df.columns])
    train_X, test_X, train_y, test_y = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return train_X, test_X, train_y, test_y
