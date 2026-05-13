"""
[DEPRECATED · 미사용] 팀 기준 Member A 논문 파생 정리는 **member_a_strategy_4 (A4)** 만 사용한다.

Member A 전처리 v5 — 기본은 **A4 + `paper_*` 다중공선성 드롭만** (지표가 잘 나오는 얇은 확장).

선택적으로 `advanced_feature_selection=True` 를 주면, A4+dedup 뒤에
상수·상관·VIF, f/chi2 필터, RF `SelectFromModel`, LR-RFE 를 **실험용**으로 적용한다.
전체 CSV용 `preprocess` 에서 y로 선택기를 맞추면 유의 신호를 잘라 **F1이 떨어질 수 있어**
기본값은 끔.

구성 (advanced_feature_selection=True 일 때만)
- **정제**: 상수·초고상관, VIF 반복 축소
- **필터**: `f_classif` `SelectKBest` + 비음수 변환 후 `chi2` `SelectKBest` → 합집합
- **임베디드**: `SelectFromModel(RandomForest)` threshold=mean
- **래퍼**: `RFE(LogisticRegression)`

주의
- **FP_CAREER** 는 `member_a_strategy` 의 `_UNUSED_CUST_FOR_A_PIPELINE` 에 포함되어 A 계통 전부에서 제거됨.
- SMOTE 등 행 수 변경은 `preprocess` 에 넣지 않음.

튜닝: `python -m src.optimization.hyperparameter_tuner --strategy member_a_strategy_5`

실험 CLI ID: member_a_strategy_5
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, chi2, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.config import DIVIDED_SET_COL, ID_COL
from src.optimization.feature_selector import CorrelationRemover, ImportanceSelector
from src.preprocessing.strategies.base_strategy import BaseStrategy
from src.preprocessing.strategies.book.book_survey_paper_dedup_strategy import (
    _PAPER_DROP_FOR_MULTICOLLINEARITY,
)
from .member_a_strategy_4 import MemberA4Strategy


def _drop_paper_multicollinearity(df: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in _PAPER_DROP_FOR_MULTICOLLINEARITY if c in df.columns]
    if drop:
        return df.drop(columns=drop, errors="ignore")
    return df


def _meta_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in (ID_COL, DIVIDED_SET_COL) if c in df.columns]


def _as_finite_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    X = df[cols].to_numpy(dtype=np.float64, copy=True)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def _drop_near_constant(names: list[str], X: np.ndarray, eps: float = 1e-12) -> tuple[list[str], np.ndarray]:
    if X.size == 0:
        return [], X
    v = X.var(axis=0)
    keep = v > eps
    names_f = [n for n, k in zip(names, keep) if k]
    Xf = X[:, keep] if keep.any() else X[:, :0]
    return names_f, Xf


def _corr_drop(names: list[str], X: np.ndarray, thr: float) -> tuple[list[str], np.ndarray]:
    p = len(names)
    if p < 2:
        return names, X
    cm = np.corrcoef(X.T)
    drop_idx: set[int] = set()
    for i in range(p):
        for j in range(i + 1, p):
            r = cm[i, j]
            if np.isfinite(r) and abs(r) >= thr:
                drop_idx.add(j)
    if not drop_idx:
        return names, X
    keep = [k for k in range(p) if k not in drop_idx]
    names_f = [names[k] for k in keep]
    return names_f, X[:, keep]


def _vif_one_pass(X: np.ndarray) -> np.ndarray:
    """각 열 j에 대해 VIF_j ≈ 1/(1-R²), R²는 다른 열로 선형회귀."""
    n, p = X.shape
    out = np.ones(p, dtype=np.float64)
    if p < 2 or n < p + 2:
        return out
    for j in range(p):
        mask = np.ones(p, dtype=bool)
        mask[j] = False
        ycol = X[:, j]
        Xo = X[:, mask]
        if Xo.size == 0:
            continue
        try:
            beta, *_ = np.linalg.lstsq(Xo, ycol, rcond=None)
            pred = Xo @ beta
            ss_res = float(np.sum((ycol - pred) ** 2))
            ss_tot = float(np.sum((ycol - np.mean(ycol)) ** 2))
            r2 = 0.0 if ss_tot < 1e-15 else 1.0 - ss_res / ss_tot
            r2 = min(max(r2, 0.0), 1.0 - 1e-15)
            out[j] = 1.0 / (1.0 - r2)
        except np.linalg.LinAlgError:
            out[j] = np.inf
    return out


def _vif_prune(
    names: list[str],
    X: np.ndarray,
    max_vif: float,
    max_iter: int,
) -> tuple[list[str], np.ndarray]:
    names = list(names)
    Xw = np.array(X, dtype=np.float64, copy=True)
    for _ in range(max_iter):
        if len(names) < 3:
            break
        vifs = _vif_one_pass(Xw)
        mx = float(np.nanmax(vifs))
        if not np.isfinite(mx) or mx <= max_vif:
            break
        j = int(np.nanargmax(vifs))
        keep = [i for i in range(len(names)) if i != j]
        names = [names[i] for i in keep]
        Xw = Xw[:, keep]
    return names, Xw


def _y_int_binary(y: pd.Series) -> np.ndarray:
    s = pd.to_numeric(y, errors="coerce").fillna(0).astype(int).to_numpy()
    return np.clip(s, 0, 1)


@dataclass
class _A5SelectionState:
    meta_cols: list[str]
    feature_cols: list[str]


def _fit_feature_selection(
    train_df: pd.DataFrame,
    y: pd.Series,
    *,
    corr_thr: float,
    max_vif: float,
    max_vif_iter: int,
    rfe_max_features: int,
    use_rfe: bool,
) -> _A5SelectionState:
    meta = _meta_columns(train_df)
    feat_names = [c for c in train_df.columns if c not in meta]
    if not feat_names:
        return _A5SelectionState(meta_cols=meta, feature_cols=[])

    X = _as_finite_matrix(train_df, feat_names)
    feat_names, X = _drop_near_constant(feat_names, X)
    if X.size == 0 or len(feat_names) == 0:
        return _A5SelectionState(meta_cols=meta, feature_cols=[])

    feat_names, X = _corr_drop(feat_names, X, corr_thr)
    if len(feat_names) == 0:
        return _A5SelectionState(meta_cols=meta, feature_cols=[])

    feat_names, X = _vif_prune(feat_names, X, max_vif=max_vif, max_iter=max_vif_iter)
    if len(feat_names) == 0:
        return _A5SelectionState(meta_cols=meta, feature_cols=[])

    y_arr = _y_int_binary(y.reindex(train_df.index))
    if len(np.unique(y_arr)) < 2:
        return _A5SelectionState(meta_cols=meta, feature_cols=feat_names)

    p = X.shape[1]
    k_f = max(15, min(p, max(20, (2 * p) // 3)))
    try:
        skb_f = SelectKBest(score_func=f_classif, k=min(k_f, p))
        skb_f.fit(X, y_arr)
        sup_f = skb_f.get_support(indices=True)
        set_f = set(np.array(feat_names)[sup_f].tolist())
    except Exception:
        set_f = set(feat_names)

    Xmin = X.min(axis=0, keepdims=True)
    Xpos = X - Xmin + 1e-8
    k_c = max(10, min(p, max(15, p // 2)))
    try:
        skb_c = SelectKBest(score_func=chi2, k=min(k_c, p))
        skb_c.fit(Xpos, y_arr)
        sup_c = skb_c.get_support(indices=True)
        set_c = set(np.array(feat_names)[sup_c].tolist())
    except Exception:
        set_c = set()

    union = list(set_f | set_c)
    if not union:
        union = feat_names[: min(p, max(25, p // 2))]

    X_u = _as_finite_matrix(train_df, union)
    union = [c for c in union if c in train_df.columns]

    try:
        rf = RandomForestClassifier(
            n_estimators=80,
            max_depth=12,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        sfm = SelectFromModel(rf, threshold="mean", prefit=False)
        sfm.fit(X_u, y_arr)
        sup_e = sfm.get_support()
        emb_names = [union[i] for i, s in enumerate(sup_e) if s]
        if len(emb_names) < 10:
            emb_names = union
        X_e = _as_finite_matrix(train_df, emb_names)
    except Exception:
        emb_names = union
        X_e = X_u

    n_e = len(emb_names)
    n_target = max(20, min(rfe_max_features, max(25, n_e // 2)))
    n_target = min(n_target, n_e)
    step = max(1, n_e // 45)

    if use_rfe and n_e > n_target:
        try:
            est = LogisticRegression(
                max_iter=2500,
                class_weight="balanced",
                random_state=42,
                solver="lbfgs",
            )
            rfe = RFE(estimator=est, n_features_to_select=n_target, step=step)
            rfe.fit(X_e, y_arr)
            final = [emb_names[i] for i, s in enumerate(rfe.support_) if s]
        except Exception:
            final = emb_names[:n_target]
    else:
        final = emb_names[:n_target] if n_e > n_target else emb_names

    return _A5SelectionState(meta_cols=meta, feature_cols=final)


def _apply_feature_selection(df: pd.DataFrame, state: _A5SelectionState) -> pd.DataFrame:
    cols = state.meta_cols + state.feature_cols
    return df.reindex(columns=cols, fill_value=0.0)


def _fit_lgbm_top_k(
    train_df: pd.DataFrame,
    y: pd.Series,
    *,
    top_k: int,
    corr_threshold: float,
) -> _A5SelectionState:
    """CorrelationRemover → LightGBM ImportanceSelector(top_k). train 행에만 fit."""
    meta = _meta_columns(train_df)
    feat_names = [c for c in train_df.columns if c not in meta]
    if not feat_names:
        return _A5SelectionState(meta_cols=meta, feature_cols=[])

    X_feats = (
        train_df[feat_names]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    corr_rem = CorrelationRemover(threshold=corr_threshold)
    corr_rem.fit(X_feats)
    feat_names = corr_rem.selected_cols_
    X_feats = X_feats[feat_names]

    y_arr = _y_int_binary(y.reindex(train_df.index))
    k = min(top_k, len(feat_names))
    if len(np.unique(y_arr)) < 2 or k == 0:
        return _A5SelectionState(meta_cols=meta, feature_cols=feat_names[:k])

    selector = ImportanceSelector(top_k=k)
    selector.fit(X_feats, y_arr)
    return _A5SelectionState(meta_cols=meta, feature_cols=selector.selected_cols_)


class MemberA5Strategy(BaseStrategy):
    """
    기본: A4 + paper dedup (A4 대비 초고상관 paper_* 만 정리).
    top_k 설정 시: CorrelationRemover + LightGBM top-k 피처 선택 (B v3 방식).
    skip_paper_drop=True: paper_* 다중공선성 제거를 건너뜀 → A4 출력에 바로 top_k 적용.
    advanced_feature_selection=True: VIF·필터·RFE 실험 파이프라인 (성능 하락 가능).

    실험 조합:
      A5 기본          : skip_paper_drop=False, top_k=None
      A4 + top_k       : skip_paper_drop=True,  top_k=100
      A5 + top_k       : skip_paper_drop=False, top_k=100
    """

    def __init__(
        self,
        *,
        advanced_feature_selection: bool = False,
        corr_threshold: float = 0.995,
        max_vif: float = 15.0,
        max_vif_iter: int = 28,
        rfe_max_features: int = 120,
        use_rfe: bool = True,
        top_k: int | None = None,
        lgbm_corr_threshold: float = 0.95,
        skip_paper_drop: bool = False,
    ) -> None:
        self._base = MemberA4Strategy()
        self._advanced = advanced_feature_selection
        self._corr_threshold = corr_threshold
        self._max_vif = max_vif
        self._max_vif_iter = max_vif_iter
        self._rfe_max_features = rfe_max_features
        self._use_rfe = use_rfe
        self._top_k = top_k
        self._lgbm_corr_threshold = lgbm_corr_threshold
        self._skip_paper_drop = skip_paper_drop

    def get_strategy_name(self) -> str:
        parts = ["A5: A4"]
        if not self._skip_paper_drop:
            parts.append("paper_* 다중공선성 축소")
        if self._top_k is not None:
            parts.append(f"LightGBM top-{self._top_k}")
        if self._advanced:
            parts.append("정제·필터·RFE")
        return " + ".join(parts)

    def preprocess(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        claim_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        out = self._base.preprocess(X, y, claim_df=claim_df)
        if not self._skip_paper_drop:
            out = _drop_paper_multicollinearity(out)
        if self._top_k is not None:
            state = _fit_lgbm_top_k(
                out, y, top_k=self._top_k, corr_threshold=self._lgbm_corr_threshold
            )
            out = _apply_feature_selection(out, state)
        if not self._advanced:
            return out
        state = _fit_feature_selection(
            out,
            y,
            corr_thr=self._corr_threshold,
            max_vif=self._max_vif,
            max_vif_iter=self._max_vif_iter,
            rfe_max_features=self._rfe_max_features,
            use_rfe=self._use_rfe,
        )
        return _apply_feature_selection(out, state)

    def preprocess_train_test(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        claim_df: pd.DataFrame = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        tr, te = self._base.preprocess_train_test(
            X_train, y_train, X_test, claim_df=claim_df
        )
        if not self._skip_paper_drop:
            tr = _drop_paper_multicollinearity(tr)
            te = _drop_paper_multicollinearity(te)
            te = te.reindex(columns=tr.columns, fill_value=0.0)
        if self._top_k is not None:
            state = _fit_lgbm_top_k(
                tr, y_train, top_k=self._top_k, corr_threshold=self._lgbm_corr_threshold
            )
            tr = _apply_feature_selection(tr, state)
            te = _apply_feature_selection(te, state)
            te = te.reindex(columns=tr.columns, fill_value=0.0)
        if not self._advanced:
            return tr, te
        state = _fit_feature_selection(
            tr,
            y_train,
            corr_thr=self._corr_threshold,
            max_vif=self._max_vif,
            max_vif_iter=self._max_vif_iter,
            rfe_max_features=self._rfe_max_features,
            use_rfe=self._use_rfe,
        )
        tr = _apply_feature_selection(tr, state)
        te = _apply_feature_selection(te, state)
        te = te.reindex(columns=tr.columns, fill_value=0.0)
        return tr, te
