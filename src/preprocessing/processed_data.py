import pandas as pd

from src.config import DIVIDED_SET_COL, ID_COL, TARGET_COL
from src.pipeline.data_loader import load_claim_data, load_customer_data, normalize_target
from src.preprocessing.preprocessor_factory import STRATEGY_REGISTRY, get_strategy
from src.project_paths import processed_csv_path


def _restore_required_columns(raw_X: pd.DataFrame, processed_X: pd.DataFrame) -> pd.DataFrame:
    # 전략 구현에서 빠뜨리기 쉬운 기본 식별 컬럼은 공통부에서 다시 보강한다.
    restored = processed_X.copy()
    for col in [ID_COL]:
        if col in raw_X.columns and col not in restored.columns:
            restored[col] = raw_X[col].values
    return restored


def _coerce_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    coerced = df.copy()
    for col in [ID_COL]:
        if col in coerced.columns:
            coerced[col] = pd.to_numeric(coerced[col], errors="raise")
    return coerced


def _validate_processed_dataframe(member: str, df: pd.DataFrame):
    missing_required = [col for col in [ID_COL, TARGET_COL] if col not in df.columns]
    if missing_required:
        raise ValueError(f"{member} 전처리 데이터에 필수 컬럼이 없습니다: {missing_required}")

    feature_cols = [col for col in df.columns if col != TARGET_COL]
    feature_missing = df[feature_cols].isna().sum()
    feature_missing = feature_missing[feature_missing > 0]
    if not feature_missing.empty:
        raise ValueError(
            f"{member} 전처리 데이터에 결측치가 남아 있습니다: {feature_missing.to_dict()}"
        )

    non_numeric = df[feature_cols].select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise ValueError(
            f"{member} 전처리 데이터에 숫자가 아닌 feature 컬럼이 남아 있습니다: {non_numeric}"
        )


def build_processed_dataframe(member: str, strategy_kwargs: dict | None = None) -> pd.DataFrame:
    # 전략은 X만 가공하고, 공통부가 메타 컬럼/타깃을 붙여 최종 CSV 형태를 맞춘다.
    strategy = get_strategy(member, **(strategy_kwargs or {}))
    cust_df = load_customer_data()
    claim_df = load_claim_data()

    y = normalize_target(cust_df[TARGET_COL])
    raw_X = cust_df.drop(columns=[TARGET_COL]).copy()
    processed_X = strategy.preprocess(raw_X.copy(), y.copy(), claim_df=claim_df.copy())

    if not isinstance(processed_X, pd.DataFrame):
        raise TypeError(f"{member} 전략은 pandas DataFrame을 반환해야 합니다.")
    if len(processed_X) != len(raw_X):
        raise ValueError(
            f"{member} 전략이 행 개수를 바꿨습니다. {len(raw_X)} -> {len(processed_X)}"
        )

    processed_X = _restore_required_columns(raw_X, processed_X)
    processed_X = _coerce_metadata_columns(processed_X)
    processed_df = processed_X.copy()
    processed_df[TARGET_COL] = y
    ordered_cols = [col for col in [ID_COL, TARGET_COL] if col in processed_df.columns]
    if DIVIDED_SET_COL in processed_df.columns:
        ordered_cols.append(DIVIDED_SET_COL)
    ordered_cols += [col for col in processed_df.columns if col not in ordered_cols]
    processed_df = processed_df[ordered_cols]

    _validate_processed_dataframe(member, processed_df)
    return processed_df


def ensure_processed_csv(member: str, force: bool = False, strategy_kwargs: dict | None = None):
    processed_path = processed_csv_path(member)
    if processed_path.exists() and not force:
        return processed_path

    try:
        processed_df = build_processed_dataframe(member, strategy_kwargs=strategy_kwargs)
    except NotImplementedError:
        print(f"[SKIP] {member} 전략은 아직 구현되지 않았습니다.")
        return None

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(processed_path, index=False)
    print(f"[OK] 전처리 CSV 생성 완료: {processed_path}")
    return processed_path


def ensure_processed_csvs(members=None, force: bool = False):
    members = members or list(STRATEGY_REGISTRY.keys())
    prepared = {}
    for member in members:
        prepared[member] = ensure_processed_csv(member, force=force)
    return prepared
