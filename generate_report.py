"""
최적화 보고서 자동 생성 스크립트

사용법:
    python generate_report.py --strategy member_b_strategy_2 --baseline member_b
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
import pandas as pd

REPORTS_DIR = Path("docs/optimization_reports")
ARTIFACTS_DIR = Path("artifacts")

_META_COLS = frozenset({"CUST_ID", "SIU_CUST_YN", "DIVIDED_SET"})

_CLAIM_FEATURE_MARKERS = (
    "claim_",
    "policy_",
    "hospital_",
    "doctor_",
    "disease_",
    "cause_",
    "addr_",
    "dmnd_",
    "non_pay",
    "self_cham",
    "valid_hosp",
    "hosp_days",
    "weekend_",
    "recp_",
    "delay_",
    "heed_",
    "acci_",
    "same_hospital",
    "main_",
    "patt_",
    "interval",
    "combo",
    "concentration",
    "DOC_SIU",
    "HOSP_SIU",
    "HOSP_DOC",
    "dmnd_resn",
)

_OHE_MARKERS = (
    "SEX_",
    "RESI_TYPE_CODE_",
    "CTPR_",
    "OCCP_GRP_1_",
    "WEDD_YN_",
    "MATE_OCCP_",
    "AGE_GROUP_",
    "LTBN_CHLD_AGE_GROUP_",
)


def _member_c_feature_groups(columns: list[str]) -> dict[str, int]:
    """member_c / member_c_strategy_2 등 C계열 전처리 CSV용 피처 요약."""
    feats = [c for c in columns if c not in _META_COLS]
    isna = [c for c in feats if c.endswith("_isna")]
    rest = [c for c in feats if c not in isna]
    logs = [c for c in rest if "_log" in c]
    rest2 = [c for c in rest if c not in logs]

    def is_claim(name: str) -> bool:
        return any(name.startswith(p) for p in _CLAIM_FEATURE_MARKERS)

    claim = [c for c in rest2 if is_claim(c)]
    rest3 = [c for c in rest2 if c not in claim]

    def is_ohe(name: str) -> bool:
        return any(m in name for m in _OHE_MARKERS)

    ohe = [c for c in rest3 if is_ohe(c)]
    cust_tab = [c for c in rest3 if c not in ohe]

    return {
        "C_isna": len(isna),
        "C_log": len(logs),
        "C_claim_agg": len(claim),
        "C_cust_tab": len(cust_tab),
        "C_onehot": len(ohe),
    }


def load_experiment_result(strategy_id: str):
    """실험 결과 로드"""
    result_path = ARTIFACTS_DIR / f"experiment_result_{strategy_id}.csv"
    if not result_path.exists():
        return None

    df = pd.read_csv(result_path)
    # LightGBM 결과만 추출
    lgbm_result = df[df['model'] == 'lgbm'].iloc[0]
    return {
        'f1_class1': lgbm_result['f1_class1'],
        'recall_class1': lgbm_result['recall_class1'],
        'f1_macro': lgbm_result['f1_macro'],
        'strategy': lgbm_result['strategy'],
    }


def load_processed_csv_info(strategy_id: str):
    """전처리 CSV 정보 로드"""
    csv_path = Path(f"data/processed/{strategy_id}_preprocessed.csv")
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path, nrows=1)
    cols = list(df.columns)
    feature_groups = {
        "DEV": len([c for c in cols if c.startswith("DEV_")]),
        "BURST": len([c for c in cols if c.startswith("BURST_")]),
        "GRAPH": len([c for c in cols if c.startswith("GRAPH_")]),
        "AMT": len([c for c in cols if c.startswith("AMT_")]),
        "TIME": len([c for c in cols if c.startswith("TIME_")]),
        "MED": len([c for c in cols if c.startswith("MED_")]),
    }
    if strategy_id.startswith("member_c"):
        feature_groups = _member_c_feature_groups(cols)
    return {"n_columns": len(cols), "feature_groups": feature_groups}


def generate_report(strategy_id: str, baseline_id: str = None, version: str = None):
    """최적화 보고서 생성"""

    # 데이터 로드
    current_result = load_experiment_result(strategy_id)
    current_csv_info = load_processed_csv_info(strategy_id)

    if not current_result:
        print(f"[ERROR] {strategy_id}의 실험 결과를 찾을 수 없습니다.")
        return None

    baseline_result = None
    baseline_csv_info = None
    if baseline_id:
        baseline_result = load_experiment_result(baseline_id)
        baseline_csv_info = load_processed_csv_info(baseline_id)

    # 버전 자동 감지
    if not version:
        if 'strategy_2' in strategy_id or '_2' in strategy_id:
            version = 'v2'
        elif 'strategy_3' in strategy_id or '_3' in strategy_id:
            version = 'v3'
        else:
            version = 'v1'

    # 보고서 생성
    report = []
    report.append(f"# {strategy_id} ({version}) 최적화 보고서\n")
    report.append(f"**생성 날짜**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**전략 ID**: `{strategy_id}`\n")
    if baseline_id:
        report.append(f"**이전 버전**: `{baseline_id}`\n")
    report.append("\n---\n")

    # 성능 비교
    report.append("\n## 성능 비교\n")
    report.append("\n### 최종 성능 (LightGBM)\n")
    report.append("\n| 지표 | ")
    if baseline_id:
        report.append(f"{baseline_id} | ")
    report.append(f"{strategy_id} |")
    if baseline_id:
        report.append(" 개선량 | 개선률 |")
    report.append("\n|------|")
    if baseline_id:
        report.append("------|")
    report.append("------|")
    if baseline_id:
        report.append("--------|--------|")
    report.append("\n")

    # F1-score
    report.append("| **F1-score** | ")
    if baseline_id and baseline_result:
        report.append(f"{baseline_result['f1_class1']:.4f} | ")
    report.append(f"**{current_result['f1_class1']:.4f}** |")
    if baseline_id and baseline_result:
        improvement = current_result['f1_class1'] - baseline_result['f1_class1']
        improvement_pct = (improvement / baseline_result['f1_class1']) * 100
        report.append(f" {improvement:+.4f} | **{improvement_pct:+.1f}%** |")
    report.append("\n")

    # Recall
    report.append("| **Recall** | ")
    if baseline_id and baseline_result:
        report.append(f"{baseline_result['recall_class1']:.4f} | ")
    report.append(f"**{current_result['recall_class1']:.4f}** |")
    if baseline_id and baseline_result:
        improvement = current_result['recall_class1'] - baseline_result['recall_class1']
        improvement_pct = (improvement / baseline_result['recall_class1']) * 100
        report.append(f" {improvement:+.4f} | **{improvement_pct:+.1f}%** |")
    report.append("\n")

    # F1-macro
    report.append("| **F1-macro** | ")
    if baseline_id and baseline_result:
        report.append(f"{baseline_result['f1_macro']:.4f} | ")
    report.append(f"{current_result['f1_macro']:.4f} |")
    if baseline_id and baseline_result:
        improvement = current_result['f1_macro'] - baseline_result['f1_macro']
        improvement_pct = (improvement / baseline_result['f1_macro']) * 100
        report.append(f" {improvement:+.4f} | {improvement_pct:+.1f}% |")
    report.append("\n")

    # Feature 구성
    if current_csv_info:
        report.append("\n## Feature 구성\n")
        report.append(f"\n**전체 컬럼 수**: {current_csv_info['n_columns']}개\n")

        if baseline_csv_info:
            added = current_csv_info['n_columns'] - baseline_csv_info['n_columns']
            report.append(f"**추가된 Feature**: {added}개\n")

        report.append("\n### Feature 그룹별 개수\n")
        report.append("\n| 그룹 | ")
        if baseline_id and baseline_csv_info:
            report.append(f"{baseline_id} | ")
        report.append(f"{strategy_id} |")
        if baseline_id and baseline_csv_info:
            report.append(" 변화 |")
        report.append("\n|------|")
        if baseline_id and baseline_csv_info:
            report.append("------|")
        report.append("------|")
        if baseline_id and baseline_csv_info:
            report.append("------|")
        report.append("\n")

        gc = current_csv_info["feature_groups"]
        gb = baseline_csv_info["feature_groups"] if baseline_csv_info else {}
        if baseline_id and baseline_csv_info:
            group_keys = sorted(set(gc) | set(gb))
        else:
            group_keys = sorted(gc.keys())

        for group_name in group_keys:
            count = gc.get(group_name, 0)
            baseline_count = gb.get(group_name, 0) if baseline_csv_info else 0
            if not baseline_id and count == 0:
                continue
            if baseline_id and baseline_csv_info and count == 0 and baseline_count == 0:
                continue
            report.append(f"| {group_name} | ")
            if baseline_id and baseline_csv_info:
                report.append(f"{baseline_count}개 | ")
            report.append(f"{count}개 |")
            if baseline_id and baseline_csv_info:
                diff = count - baseline_count
                if diff > 0:
                    report.append(f" +{diff}개 (신규) |")
                elif diff < 0:
                    report.append(f" {diff}개 (제거) |")
                else:
                    report.append(" - |")
            report.append("\n")

    # 실행 방법
    report.append("\n## 실행 방법\n")
    report.append("\n### 전처리만 실행\n")
    report.append(f"```bash\n")
    report.append(f"python run_experiment.py --strategy {strategy_id} --prepare-only\n")
    report.append(f"```\n")

    report.append("\n### 모델 학습 및 평가\n")
    report.append(f"```bash\n")
    report.append(f"# 단일 모델 (LightGBM)\n")
    report.append(f"python run_experiment.py --strategy {strategy_id} --model lgbm\n\n")
    report.append(f"# 전체 모델 비교\n")
    report.append(f"python run_experiment.py --strategy {strategy_id} --model all\n")
    report.append(f"```\n")

    # 파일 저장
    report_filename = f"{strategy_id}_{version}_report.md"
    report_path = REPORTS_DIR / report_filename

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(report)

    print(f"[OK] 보고서 생성 완료: {report_path}")
    return report_path


def update_index():
    """보고서 인덱스의 '보고서 목록' 섹만 갱신하고, README 나머지(사용 방법 등)는 유지한다."""
    reports = sorted(REPORTS_DIR.glob("*_report.md"))

    lines = ["## 보고서 목록\n", "\n"]
    for report_path in reports:
        filename = report_path.stem.replace("_report", "")
        lines.append(f"- [{filename}](./{report_path.name})\n")
    lines.append("\n")
    list_block = "".join(lines)

    index_path = REPORTS_DIR / "README.md"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if index_path.exists():
        text = index_path.read_text(encoding="utf-8")
        marker = "## 보고서 목록"
        start = text.find(marker)
        if start != -1:
            sep = text.find("\n---", start)
            if sep != -1:
                new_text = text[:start] + list_block + text[sep:]
                index_path.write_text(new_text, encoding="utf-8")
                print(f"[OK] 인덱스 업데이트 완료: {index_path}")
                return

    index = [
        "# 최적화 보고서 자동 생성 시스템\n",
        "\n이 폴더는 각 버전별 최적화 보고서를 자동으로 생성하여 보관합니다.\n",
        "\n",
        list_block,
        "\n---\n",
        "\n## 사용 방법\n",
        "\n`python generate_report.py --strategy <전략ID> [--baseline <이전전략>]`\n",
    ]
    with open(index_path, "w", encoding="utf-8") as f:
        f.writelines(index)
    print(f"[OK] 인덱스 생성 완료: {index_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="최적화 보고서 자동 생성")
    parser.add_argument("--strategy", required=True, help="현재 전략 ID (예: member_b_strategy_2)")
    parser.add_argument("--baseline", help="비교 대상 전략 ID (예: member_b)")
    parser.add_argument("--version", help="버전 (예: v2, 자동 감지)")
    args = parser.parse_args()

    generate_report(args.strategy, args.baseline, args.version)
    update_index()
