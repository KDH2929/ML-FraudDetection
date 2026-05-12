from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline
from src.preprocessing.strategies.base_strategy import BaseStrategy
from src.optimization.feature_selector import FeatureSelector


def build(strategy: BaseStrategy, model, sampler=None, selector: FeatureSelector = None):
    """전략 객체와 모델을 받아 실행 가능한 파이프라인 반환.

    전처리(strategy)는 파이프라인 밖에서 미리 수행된 데이터를 받는다고 가정.
    sampler와 selector는 선택적으로 삽입된다.
    """
    steps = []
    if selector is not None:
        steps.append(("selector", selector))
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("model", model))

    if sampler is not None:
        return ImbPipeline(steps=steps)
    return Pipeline(steps=steps)
