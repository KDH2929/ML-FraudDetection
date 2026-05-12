import importlib
import inspect
from pathlib import Path

from src.preprocessing.strategies.base_strategy import BaseStrategy


STRATEGY_DIR = Path(__file__).resolve().parent / "strategies"


def _module_stem_to_strategy_id(stem: str) -> str:
    # 기존 member_a_strategy.py 형태는 member_a로 짧게 매핑한다.
    if stem.endswith("_strategy") and stem.count("_") == 2:
        return stem[: -len("_strategy")]
    return stem


def _discover_strategy_modules():
    modules = []
    # member_c_strategy_1.py 같은 수동 버전 파일도 자동으로 인식한다.
    for path in sorted(STRATEGY_DIR.glob("member*_strategy*.py")):
        if path.name.startswith("__"):
            continue
        modules.append(path.stem)
    return modules


def _load_strategy_class(module_stem: str):
    module_name = f"src.preprocessing.strategies.{module_stem}"
    module = importlib.import_module(module_name)
    classes = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj is BaseStrategy:
            continue
        if not issubclass(obj, BaseStrategy):
            continue
        if obj.__module__ != module.__name__:
            continue
        classes.append(obj)

    if not classes:
        raise ValueError(f"전략 클래스가 없는 모듈입니다: {module_name}")
    if len(classes) > 1:
        class_names = [cls.__name__ for cls in classes]
        raise ValueError(
            f"{module_name} 에는 전략 클래스가 정확히 1개만 있어야 합니다. 현재: {class_names}"
        )
    return classes[0]


def _build_registry():
    registry = {}
    for module_stem in _discover_strategy_modules():
        strategy_id = _module_stem_to_strategy_id(module_stem)
        registry[strategy_id] = _load_strategy_class(module_stem)
    return registry


STRATEGY_REGISTRY = _build_registry()


def list_strategies():
    return list(STRATEGY_REGISTRY.keys())


def get_strategy(strategy_id: str) -> BaseStrategy:
    key = strategy_id.lower()
    if key not in STRATEGY_REGISTRY:
        raise ValueError(
            f"알 수 없는 전략입니다: {strategy_id}. 사용 가능: {list(STRATEGY_REGISTRY.keys())}"
        )
    return STRATEGY_REGISTRY[key]()
