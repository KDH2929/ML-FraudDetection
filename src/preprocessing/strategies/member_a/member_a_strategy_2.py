"""
Member A 전처리 v2 — v1 위에 KMeans 군집 파생을 얹은 확장.

v1(`MemberAStrategy`)과 동일한 CLAIM 요약 merge·결측·타깃인코딩·IQR·Robust 흐름을 유지한 뒤,
**통계 적합은 DIVIDED_SET==1 행만** 사용해 라벨·평가구간 정보가 군집에 직접 들어가지 않게 한다.
(`DIVIDED_SET` 컬럼이 없으면 전체 행으로 적합 — 노트북/실험용)

추가 열
- `ma2_kmeans_cluster`, `ma2_kmeans_dist`: 수치형 변수를 StandardScaler 후 KMeans 적합.
  군집 레이블과 중심까지 거리를 파생변수로 추가.

※ PCA 차원 압축은 v4(member_a_strategy_4)에서 다중공선성 제거와 함께 적용.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.config import CAT_COLS, DIVIDED_SET_COL, ID_COL, NUM_COLS
from src.preprocessing.components.encoder import TargetEncoder
from src.preprocessing.components.missing_value import GroupMeanImputer, MedianImputer, ModeImputer
from src.preprocessing.components.outlier import IQRCapper
from src.preprocessing.components.scaler import RobustScalerWrapper
from src.preprocessing.strategies.base_strategy import BaseStrategy
from src.preprocessing.stage_cache import (
    combined_fingerprint,
    dataframe_fingerprint,
    file_fingerprint,
    load_or_build_stage,
)
from .member_a_strategy import (
    _add_key_missing_indicators,
    _drop_unused_cust_columns,
    _fill_residual_missing_for_ml,
    _load_or_build_member_a_raw_features,
    _merge_claim_features,
    _y_to_float01,
)

_RANDOM_STATE = 42
_STRATEGY_ID = "member_a_strategy_2"
_MEMBER_A_DIR = Path(__file__).resolve().parent
_A2_SOURCE_FILES = (
    _MEMBER_A_DIR / "member_a_strategy.py",
    _MEMBER_A_DIR / "member_a_strategy_2.py",
)


def _source_fingerprint(paths: tuple[Path, ...]) -> str:
    return combined_fingerprint(file_fingerprint(p) for p in paths)


def _fit_rows_mask(X: pd.DataFrame) -> np.ndarray:
    if DIVIDED_SET_COL not in X.columns:
        return np.ones(len(X), dtype=bool)
    ds = pd.to_numeric(X[DIVIDED_SET_COL], errors="coerce")
    return (ds == 1).fillna(False).to_numpy()


def _add_cluster_pca(
    X_out: pd.DataFrame,
    *,
    n_clusters: int = 5,
    random_state: int = _RANDOM_STATE,
) -> pd.DataFrame:
    meta = {ID_COL, DIVIDED_SET_COL}
    num_candidates = [
        c
        for c in X_out.columns
        if c not in meta and pd.api.types.is_numeric_dtype(X_out[c])
    ]
    if not num_candidates:
        return X_out

    mat = (
        X_out[num_candidates]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )

    mask = _fit_rows_mask(X_out)
    n_fit = int(mask.sum())
    if n_fit < max(n_clusters, 2) or mat.shape[1] == 0:
        return X_out

    scaler = StandardScaler()
    scaler.fit(mat[mask])
    Z_all = scaler.transform(mat)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(Z_all[mask])
    labels = kmeans.predict(Z_all)
    cc = kmeans.cluster_centers_
    dist = np.linalg.norm(Z_all - cc[labels], axis=1)

    out = X_out.copy()
    out["ma2_kmeans_cluster"] = labels.astype(np.float64)
    out["ma2_kmeans_dist"] = dist.astype(np.float64)
    return out


def _load_or_build_cluster_pca(
    X_out: pd.DataFrame,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    if not use_cache:
        return _add_cluster_pca(X_out)

    fp = combined_fingerprint(
        (
            "a2_kmeans_v2",
            dataframe_fingerprint(X_out),
            _source_fingerprint(_A2_SOURCE_FILES),
            _RANDOM_STATE,
        )
    )
    return load_or_build_stage(
        strategy_id=_STRATEGY_ID,
        stage_name="a2_cluster_pca_features",
        fingerprint=fp,
        builder=lambda: _add_cluster_pca(X_out),
    )


class MemberA2Strategy(BaseStrategy):

    def get_strategy_name(self) -> str:
        return "A2: v1 + KMeans 군집 (DIVIDED_SET=1 fit, PCA는 v4로 이동)"

    def preprocess(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        claim_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        X_out = _load_or_build_member_a_raw_features(X, claim_df)

        if "RESI_COST" in X_out.columns:
            X_out["RESI_COST"] = X_out["RESI_COST"].replace(0, pd.NA)

        for c in NUM_COLS:
            if c in X_out.columns:
                X_out[c] = pd.to_numeric(X_out[c], errors="coerce")

        if all(c in X_out.columns for c in ("OCCP_GRP_1", "AGE", "CUST_INCM")):
            gim = GroupMeanImputer()
            gim.fit(X_out, y)
            X_out = gim.transform(X_out)

        num_cols = [
            c
            for c in X_out.columns
            if c != ID_COL and pd.api.types.is_numeric_dtype(X_out[c])
        ]
        if num_cols:
            med = MedianImputer(cols=num_cols)
            med.fit(X_out, y)
            X_out = med.transform(X_out)
            X_out[num_cols] = X_out[num_cols].apply(pd.to_numeric, errors="coerce")
            med2 = X_out[num_cols].median()
            X_out[num_cols] = X_out[num_cols].fillna(med2).fillna(0.0).astype(np.float64)

        X_out = _load_or_build_cluster_pca(X_out)

        cat_cols = [c for c in CAT_COLS if c in X_out.columns]
        if cat_cols:
            mode_imp = ModeImputer(cols=cat_cols)
            mode_imp.fit(X_out, y)
            X_out = mode_imp.transform(X_out)

        if cat_cols and y is not None and len(y) == len(X_out):
            y01 = _y_to_float01(y).reindex(X_out.index)
            te = TargetEncoder(cols=cat_cols)
            te.fit(X_out, y01)
            X_out = te.transform(X_out)

        iqr_cols = [c for c in NUM_COLS if c in X_out.columns]
        if iqr_cols:
            cap = IQRCapper(cols=iqr_cols)
            cap.fit(X_out, y)
            X_out = cap.transform(X_out)

        scale_cols = [c for c in NUM_COLS if c in X_out.columns]
        if scale_cols:
            sc = RobustScalerWrapper(cols=scale_cols)
            sc.fit(X_out, y)
            X_out = sc.transform(X_out)

        X_out = _fill_residual_missing_for_ml(X_out)
        return X_out

    def preprocess_train_test(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        claim_df: pd.DataFrame = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """train에만 통계 적합 후 test 적용. KMeans·PCA는 train 행만으로 fit."""
        tr = _merge_claim_features(X_train, claim_df)
        te = _merge_claim_features(X_test, claim_df)
        tr = _drop_unused_cust_columns(tr)
        te = _drop_unused_cust_columns(te)
        tr = _add_key_missing_indicators(tr)
        te = _add_key_missing_indicators(te)

        for Xs in (tr, te):
            if "RESI_COST" in Xs.columns:
                Xs["RESI_COST"] = Xs["RESI_COST"].replace(0, pd.NA)
            for c in NUM_COLS:
                if c in Xs.columns:
                    Xs[c] = pd.to_numeric(Xs[c], errors="coerce")

        if all(c in tr.columns for c in ("OCCP_GRP_1", "AGE", "CUST_INCM")):
            gim = GroupMeanImputer()
            gim.fit(tr, y_train)
            tr = gim.transform(tr)
            te = gim.transform(te)

        num_cols = [
            c
            for c in tr.columns
            if c != ID_COL and pd.api.types.is_numeric_dtype(tr[c])
        ]
        if num_cols:
            med = MedianImputer(cols=num_cols)
            med.fit(tr, y_train)
            tr = med.transform(tr)
            te = med.transform(te)
            tr[num_cols] = tr[num_cols].apply(pd.to_numeric, errors="coerce")
            med2 = tr[num_cols].median()
            tr[num_cols] = tr[num_cols].fillna(med2).fillna(0.0).astype(np.float64)
            te[num_cols] = te[num_cols].apply(pd.to_numeric, errors="coerce")
            te[num_cols] = te[num_cols].fillna(med2).fillna(0.0).astype(np.float64)

        tr, te = _add_cluster_pca_train_test(tr, te)

        cat_cols = [c for c in CAT_COLS if c in tr.columns]
        if cat_cols:
            mode_imp = ModeImputer(cols=cat_cols)
            mode_imp.fit(tr, y_train)
            tr = mode_imp.transform(tr)
            te = mode_imp.transform(te)

        if cat_cols and len(y_train) == len(tr):
            y01 = _y_to_float01(y_train).reindex(tr.index)
            te_enc = TargetEncoder(cols=cat_cols)
            te_enc.fit(tr, y01)
            tr = te_enc.transform(tr)
            te = te_enc.transform(te)

        iqr_cols = [c for c in NUM_COLS if c in tr.columns]
        if iqr_cols:
            cap = IQRCapper(cols=iqr_cols)
            cap.fit(tr, y_train)
            tr = cap.transform(tr)
            te = cap.transform(te)

        scale_cols = [c for c in NUM_COLS if c in tr.columns]
        if scale_cols:
            sc = RobustScalerWrapper(cols=scale_cols)
            sc.fit(tr, y_train)
            tr = sc.transform(tr)
            te = sc.transform(te)

        te = te.reindex(columns=tr.columns, fill_value=0.0)
        tr = _fill_residual_missing_for_ml(tr)
        te = _fill_residual_missing_for_ml(te)
        return tr, te


def _add_cluster_pca_train_test(
    tr: pd.DataFrame,
    te: pd.DataFrame,
    *,
    n_clusters: int = 5,
    random_state: int = _RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = {ID_COL, DIVIDED_SET_COL}
    num_candidates = [
        c
        for c in tr.columns
        if c not in meta and pd.api.types.is_numeric_dtype(tr[c])
    ]
    if not num_candidates:
        return tr, te

    def _mat(df: pd.DataFrame) -> np.ndarray:
        return (
            df[num_candidates]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )

    m_tr = _mat(tr)
    m_te = _mat(te)
    if m_tr.shape[1] == 0 or len(tr) < max(n_clusters, 2):
        return tr, te

    scaler = StandardScaler()
    scaler.fit(m_tr)
    Z_tr = scaler.transform(m_tr)
    Z_te = scaler.transform(m_te)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(Z_tr)
    lb_tr = kmeans.predict(Z_tr)
    lb_te = kmeans.predict(Z_te)
    cc = kmeans.cluster_centers_
    d_tr = np.linalg.norm(Z_tr - cc[lb_tr], axis=1)
    d_te = np.linalg.norm(Z_te - cc[lb_te], axis=1)

    tr = tr.copy()
    te = te.copy()
    tr["ma2_kmeans_cluster"] = lb_tr.astype(np.float64)
    tr["ma2_kmeans_dist"] = d_tr.astype(np.float64)
    te["ma2_kmeans_cluster"] = lb_te.astype(np.float64)
    te["ma2_kmeans_dist"] = d_te.astype(np.float64)
    return tr, te
