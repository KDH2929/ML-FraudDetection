import joblib
import pandas as pd
from pathlib import Path


def save_pipeline(pipeline, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_pipeline(path):
    return joblib.load(path)


def save_result(df: pd.DataFrame, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
