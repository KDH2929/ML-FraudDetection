from src.preprocessing.strategies.base_strategy import BaseStrategy
from src.preprocessing.strategies.member_a_strategy import MemberAStrategy
from src.preprocessing.strategies.member_b_strategy import MemberBStrategy
from src.preprocessing.strategies.member_c_strategy import MemberCStrategy

STRATEGY_REGISTRY = {
    "member_a": MemberAStrategy,
    "member_b": MemberBStrategy,
    "member_c": MemberCStrategy,
}


def get_strategy(member: str) -> BaseStrategy:
    key = member.lower()
    if key not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown member: {member}. Available: {list(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[key]()
