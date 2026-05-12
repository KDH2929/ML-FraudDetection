def get_param_grid(model_name: str) -> dict:
    """하이퍼파라미터 탐색 범위를 모델별로 반환"""
    grids = {
        "lgbm": {
            "n_estimators": [300, 500, 700],
            "learning_rate": [0.01, 0.05, 0.1],
            "num_leaves": [15, 31, 63],
            "min_child_samples": [10, 20, 50],
        },
        "rf": {
            "n_estimators": [100, 300, 500],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5, 10],
        },
        "logistic": {
            "C": [0.01, 0.1, 1.0, 10.0],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear"],
        },
    }
    key = model_name.lower()
    if key not in grids:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(grids.keys())}")
    return grids[key]
