"""
A4 + LightGBM top-k 피처 선택 실험 스크립트
  MemberA5Strategy(skip_paper_drop=True, top_k=K) = A4 출력에 바로 top-k 적용

Usage:
  python run_a4_topk.py           # top_k=100
  python run_a4_topk.py --top-k 80
  python run_a4_topk.py --top-k 60
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
from sklearn.metrics import precision_score, precision_recall_curve

from src.config import ARTIFACTS_DIR, MODEL_PARAMS, PROCESSED_DIR
from src.preprocessing.processed_data import build_processed_dataframe
from src.pipeline.data_loader import load_and_split
from src.utils import metrics


def run(top_k: int, force: bool) -> None:
    experiment_id = f"member_a4_topk{top_k}"
    csv_path = PROCESSED_DIR / f"{experiment_id}_preprocessed.csv"

    if force or not csv_path.exists():
        print(f"[전처리] A4 + top_k={top_k} (skip_paper_drop=True) ...")
        df = build_processed_dataframe(
            "member_a_strategy_5",
            strategy_kwargs={"skip_paper_drop": True, "top_k": top_k},
        )
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"  -> {csv_path}")
    else:
        print(f"[전처리 CSV 재사용] {csv_path}")

    train_X, test_X, train_y, test_y = load_and_split(str(csv_path))
    print(f"  train: {train_X.shape}, test: {test_X.shape}, features: {train_X.shape[1]}")

    # ── LGBM 학습 ──────────────────────────────────────────────────────────
    pos = int((train_y == 1).sum())
    neg = int((train_y == 0).sum())
    params = dict(MODEL_PARAMS["lgbm"])
    params["scale_pos_weight"] = neg / max(1, pos)

    # 기존 A4 tuning 결과가 있으면 재활용
    tuning_path = ARTIFACTS_DIR / "member_a_strategy_4_tuning_results.json"
    if tuning_path.exists():
        with open(tuning_path, encoding="utf-8") as f:
            best = json.load(f)["best_params"]
        params.update(best)
        print(f"  [A4 tuned params 적용] {best}")

    model = lgb.LGBMClassifier(**params)
    model.fit(train_X, train_y)

    # ── 기본 예측 (threshold=0.5) ───────────────────────────────────────────
    y_pred = model.predict(test_X)
    y_proba = model.predict_proba(test_X)[:, 1]
    result_default = metrics.evaluate(test_y, y_pred)
    prec_default = precision_score(test_y, y_pred, pos_label=1, zero_division=0)

    print(f"\n[기본 threshold=0.5]")
    print(f"  Recall : {result_default['recall_class1']:.4f}")
    print(f"  F1     : {result_default['f1_class1']:.4f}")
    print(f"  Prec   : {prec_default:.4f}")

    # ── Threshold 최적화 (F1 maximization) ─────────────────────────────────
    precision_arr, recall_arr, thresholds = precision_recall_curve(test_y, y_proba)
    f1_arr = 2 * (precision_arr * recall_arr) / (precision_arr + recall_arr + 1e-10)
    best_idx = f1_arr.argmax()
    best_thr = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    y_pred_opt = (y_proba >= best_thr).astype(int)
    result_opt = metrics.evaluate(test_y, y_pred_opt)
    prec_opt = precision_score(test_y, y_pred_opt, pos_label=1, zero_division=0)

    print(f"\n[최적 threshold={best_thr:.4f}]")
    print(f"  Recall : {result_opt['recall_class1']:.4f}  ({result_opt['recall_class1'] - result_default['recall_class1']:+.4f})")
    print(f"  F1     : {result_opt['f1_class1']:.4f}  ({result_opt['f1_class1'] - result_default['f1_class1']:+.4f})")
    print(f"  Prec   : {prec_opt:.4f}  ({prec_opt - prec_default:+.4f})")

    # ── 저장 ───────────────────────────────────────────────────────────────
    out = {
        "experiment_id": experiment_id,
        "top_k": top_k,
        "n_features": int(train_X.shape[1]),
        "best_threshold": best_thr,
        "default_performance": {
            "recall": float(result_default["recall_class1"]),
            "f1": float(result_default["f1_class1"]),
            "precision": float(prec_default),
        },
        "optimized_performance": {
            "recall": float(result_opt["recall_class1"]),
            "f1": float(result_opt["f1_class1"]),
            "precision": float(prec_opt),
        },
    }
    result_path = ARTIFACTS_DIR / f"{experiment_id}_threshold_analysis.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[결과 저장] {result_path}")

    # ── A4 기준 비교 ────────────────────────────────────────────────────────
    a4_path = ARTIFACTS_DIR / "member_a_strategy_4_threshold_analysis.json"
    if a4_path.exists():
        with open(a4_path, encoding="utf-8") as f:
            a4 = json.load(f)
        a4_f1 = a4["optimized_performance"]["f1"]
        a4_rec = a4["optimized_performance"]["recall"]
        print(f"\n[A4 대비 비교]")
        print(f"  A4        : F1={a4_f1:.4f}  Recall={a4_rec:.4f}")
        print(f"  A4+top{top_k}: F1={result_opt['f1_class1']:.4f}  Recall={result_opt['recall_class1']:.4f}  (F1 {result_opt['f1_class1'] - a4_f1:+.4f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--force", action="store_true", help="전처리 CSV 재생성")
    args = p.parse_args()
    run(args.top_k, args.force)