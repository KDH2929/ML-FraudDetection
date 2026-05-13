"""
전처리 CSV 기준, 특징(열) 부분집합을 **완전탐색**하여 CV 점수가 가장 좋은 조합을 찾는다.

- 기본: 사전에 열 개수를 `--max-exhaustive-p` 이하로 줄인 뒤, 비어 있지 않은 모든 부분집합을
  `2^p - 1`번 평가한다(완전탐색). p가 크면 지수 폭발하므로 **상한·사전 축소**가 필수다.
- `--mode fixed_k`: `C(n, k)`개의 k개 조합만 전수 평가.

사용 예:
  python run_feature_subset_exhaustive.py --strategy member_a_strategy_5
  python run_feature_subset_exhaustive.py --csv data/processed/member_a_strategy_5_preprocessed.csv --max-exhaustive-p 12
  python run_feature_subset_exhaustive.py --strategy member_a_strategy_5 --mode fixed_k --subset-size 8 --preselect-top 16

주의: 학습용으로만 쓰고, 선택 결과를 test에 그대로 적용할 때는 별도 hold-out에서 검증하는 것이 안전하다.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.config import (
    ARTIFACTS_DIR,
    DIVIDED_SET_COL,
    DROP_COLS,
    ID_COL,
    RANDOM_STATE,
    TARGET_COL,
)
from src.pipeline.data_loader import _read_csv, normalize_target
from src.project_paths import processed_csv_path, strategy_artifact_dir


def _load_xy_from_processed(
    csv_path: Path,
    *,
    drop_divided_set: bool,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    df = _read_csv(str(csv_path))
    if TARGET_COL not in df.columns:
        raise ValueError(f"CSV에 {TARGET_COL} 컬럼이 필요합니다: {csv_path}")

    y = normalize_target(df[TARGET_COL])
    labeled = y.notna()
    if DIVIDED_SET_COL in df.columns:
        ds = pd.to_numeric(df[DIVIDED_SET_COL], errors="coerce")
        labeled &= ds == 1

    df = df.loc[labeled].copy()
    y = y.loc[labeled].astype(int)
    if df.empty:
        raise ValueError("라벨·DIVIDED_SET==1 조건을 만족하는 행이 없습니다.")

    drop_x = [TARGET_COL] + [c for c in DROP_COLS if c in df.columns]
    if drop_divided_set and DIVIDED_SET_COL in df.columns:
        drop_x.append(DIVIDED_SET_COL)

    X = df.drop(columns=drop_x, errors="ignore")
    non_num = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_num:
        raise ValueError(f"수치가 아닌 특징 열이 남아 있습니다: {non_num[:10]}...")

    feat_names = list(X.columns)
    return X, y, feat_names


def _preselect_top_importance(
    X: pd.DataFrame,
    y: pd.Series,
    top: int,
    random_state: int,
) -> tuple[pd.DataFrame, list[str]]:
    if top >= X.shape[1]:
        return X, list(X.columns)
    rf = RandomForestClassifier(
        n_estimators=80,
        max_depth=12,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X.to_numpy(dtype=np.float64, copy=False), y.to_numpy())
    imp = rf.feature_importances_
    order = np.argsort(imp)[::-1][:top]
    cols = [X.columns[i] for i in order]
    return X[cols], cols


def _make_clf(n_estimators: int, random_state: int) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.08,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=1,
        verbose=-1,
    )


def _cv_score_one(
    col_idx: tuple[int, ...],
    X_np: np.ndarray,
    y_np: np.ndarray,
    n_estimators: int,
    cv_splits: int,
    random_state: int,
) -> float:
    if len(col_idx) == 0:
        return -1.0
    Xs = X_np[:, col_idx]
    clf = _make_clf(n_estimators, random_state)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scorer = make_scorer(f1_score, pos_label=1, zero_division=0)
    try:
        scores = cross_val_score(clf, Xs, y_np, cv=cv, scoring=scorer, n_jobs=1)
        return float(np.mean(scores))
    except Exception:
        return -1.0


def _exhaustive_all_subsets(
    p: int,
    X_np: np.ndarray,
    y_np: np.ndarray,
    min_features: int,
    max_subset_size: int | None,
    n_estimators: int,
    cv_splits: int,
    random_state: int,
    n_jobs: int,
    max_subsets: int,
) -> tuple[float, int, tuple[int, ...]]:
    total_masks = (1 << p) - 1
    if total_masks > max_subsets:
        raise SystemExit(
            f"부분집합 수 {total_masks} > --max-subsets {max_subsets}. "
            f"--max-exhaustive-p 를 줄이거나 --preselect-top 으로 p를 낮추거나 --max-subsets 을 올리세요."
        )

    def job(mask: int) -> tuple[float, int]:
        k = mask.bit_count()
        if k < min_features or (max_subset_size is not None and k > max_subset_size):
            return -1.0, mask
        idx = tuple(i for i in range(p) if (mask >> i) & 1)
        s = _cv_score_one(idx, X_np, y_np, n_estimators, cv_splits, random_state)
        return s, mask

    masks = list(range(1, 1 << p))
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(job)(m) for m in masks
    )
    best_s, best_mask = max(results, key=lambda t: t[0])
    idx = tuple(i for i in range(p) if (best_mask >> i) & 1) if best_s > -0.5 else tuple()
    return best_s, best_mask, idx


def _exhaustive_fixed_k(
    p: int,
    k: int,
    X_np: np.ndarray,
    y_np: np.ndarray,
    n_estimators: int,
    cv_splits: int,
    random_state: int,
    n_jobs: int,
    max_subsets: int,
) -> tuple[float, tuple[int, ...]]:
    combs = list(combinations(range(p), k))
    if len(combs) > max_subsets:
        raise SystemExit(
            f"C({p},{k})={len(combs)} > --max-subsets {max_subsets}. "
            "n·k를 줄이거나 --max-subsets 을 올리세요."
        )

    def job(idx: tuple[int, ...]) -> tuple[float, tuple[int, ...]]:
        s = _cv_score_one(idx, X_np, y_np, n_estimators, cv_splits, random_state)
        return s, idx

    results = Parallel(n_jobs=n_jobs, verbose=5)(delayed(job)(c) for c in combs)
    best_s, best_idx = max(results, key=lambda t: t[0])
    return best_s, best_idx


def main() -> None:
    ap = argparse.ArgumentParser(description="특징 부분집합 완전탐색 (CV F1 class1)")
    ap.add_argument("--strategy", type=str, default=None, help="예: member_a_strategy_5 → processed CSV 경로")
    ap.add_argument("--csv", type=str, default=None, help="전처리 CSV 직접 지정 (strategy 보다 우선)")
    ap.add_argument("--drop-divided-set", action="store_true", default=True, help="DIVIDED_SET 열을 특징에서 제외 (기본 True)")
    ap.add_argument("--keep-divided-set", action="store_true", help="DIVIDED_SET 을 특징에 남김")
    ap.add_argument("--max-exhaustive-p", type=int, default=14, help="완전탐색에 쓸 최대 특징 개수 p (초과 시 RF로 preselect)")
    ap.add_argument("--preselect-top", type=int, default=None, help="RF 중요도 상위 n개만 남김 (기본: max-exhaustive-p와 동일)")
    ap.add_argument("--mode", choices=["all_subsets", "fixed_k"], default="all_subsets")
    ap.add_argument("--subset-size", type=int, default=None, help="fixed_k 모드에서만 사용: 부분집합 크기 k")
    ap.add_argument("--min-features", type=int, default=3, help="all_subsets: 최소 포함 특징 수")
    ap.add_argument("--max-subset-size", type=int, default=None, help="all_subsets: 최대 포함 특징 수 (미설정이면 제한 없음)")
    ap.add_argument("--cv", type=int, default=3, help="StratifiedKFold 분할 수")
    ap.add_argument("--lgbm-rounds", type=int, default=120, help="부분집합 평가용 LGBM n_estimators (작을수록 빠름)")
    ap.add_argument("--n-jobs", type=int, default=-1, help="joblib 병렬 작업 수")
    ap.add_argument("--max-subsets", type=int, default=200_000, help="평가할 부분집합/조합 개수 상한 (안전장치)")
    ap.add_argument("--random-state", type=int, default=RANDOM_STATE)
    ap.add_argument(
        "--output-json",
        type=str,
        default=None,
        help=f"결과 JSON 경로 (기본: {ARTIFACTS_DIR}/feature_subset_exhaustive_<stem>.json)",
    )
    args = ap.parse_args()

    drop_ds = not args.keep_divided_set if args.keep_divided_set else args.drop_divided_set
    if args.keep_divided_set:
        drop_ds = False

    if args.csv:
        csv_path = Path(args.csv)
    elif args.strategy:
        csv_path = processed_csv_path(args.strategy)
    else:
        ap.error("--strategy 또는 --csv 중 하나는 필요합니다.")

    if not csv_path.is_file():
        raise SystemExit(f"CSV가 없습니다: {csv_path}\n전처리를 먼저 생성하세요.")

    pre_top = args.preselect_top if args.preselect_top is not None else args.max_exhaustive_p

    X, y, feat_names = _load_xy_from_processed(csv_path, drop_divided_set=drop_ds)
    X, cols_pre = _preselect_top_importance(X, y, max(pre_top, args.max_exhaustive_p), args.random_state)
    p0 = len(cols_pre)
    if p0 > args.max_exhaustive_p:
        X = X[cols_pre[: args.max_exhaustive_p]]
        cols_pre = list(X.columns)
    p = len(cols_pre)

    if args.mode == "fixed_k":
        if args.subset_size is None:
            ap.error("--mode fixed_k 일 때는 --subset-size k 가 필요합니다.")
        k = args.subset_size
        if k < 1 or k > p:
            raise SystemExit(f"--subset-size k={k} 가 특징 수 p={p} 와 맞지 않습니다.")
        from math import comb

        ncomb = comb(p, k)
        print(f"[info] p={p}, mode=fixed_k, k={k}, combinations={ncomb}")
        X_np = X.to_numpy(dtype=np.float64, copy=False)
        y_np = y.to_numpy()
        best_s, best_idx = _exhaustive_fixed_k(
            p,
            k,
            X_np,
            y_np,
            args.lgbm_rounds,
            args.cv,
            args.random_state,
            args.n_jobs,
            args.max_subsets,
        )
        best_names = [cols_pre[i] for i in best_idx]
        best_mask = None
    else:
        total = (1 << p) - 1
        print(f"[info] p={p}, mode=all_subsets, non-empty masks={total} (min_features={args.min_features})")
        X_np = X.to_numpy(dtype=np.float64, copy=False)
        y_np = y.to_numpy()
        best_s, best_mask, best_idx = _exhaustive_all_subsets(
            p,
            X_np,
            y_np,
            args.min_features,
            args.max_subset_size,
            args.lgbm_rounds,
            args.cv,
            args.random_state,
            args.n_jobs,
            args.max_subsets,
        )
        best_names = [cols_pre[i] for i in best_idx]

    out = {
        "csv": str(csv_path.resolve()),
        "mode": args.mode,
        "p_features_searched": p,
        "column_names_searched": cols_pre,
        "best_cv_f1_class1_mean": best_s,
        "best_feature_subset": best_names,
        "best_subset_size": len(best_names),
        "cv_folds": args.cv,
        "lgbm_n_estimators": args.lgbm_rounds,
        "random_state": args.random_state,
    }
    if args.mode == "all_subsets" and best_mask is not None:
        out["best_mask"] = int(best_mask)

    stem = csv_path.stem.replace("_preprocessed", "")
    if args.strategy:
        default_out = strategy_artifact_dir(args.strategy) / "feature_subset_exhaustive.json"
    else:
        default_out = ARTIFACTS_DIR / f"feature_subset_exhaustive_{stem}.json"
    out_path = Path(args.output_json) if args.output_json else default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps({k: out[k] for k in ("best_cv_f1_class1_mean", "best_subset_size", "best_feature_subset")}, ensure_ascii=False, indent=2))
    print(f"[OK] 결과 저장: {out_path.resolve()}")


if __name__ == "__main__":
    main()
