import pandas as pd


def train(pipeline, train_X: pd.DataFrame, train_y: pd.Series):
    """파이프라인을 학습 데이터로 학습시키고 fitted pipeline 반환."""
    pipeline.fit(train_X, train_y)
    return pipeline
