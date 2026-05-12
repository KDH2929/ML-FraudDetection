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
    return {
        'n_columns': len(df.columns),
        'feature_groups': {
            'DEV': len([c for c in df.columns if c.startswith('DEV_')]),
            'BURST': len([c for c in df.columns if c.startswith('BURST_')]),
            'GRAPH': len([c for c in df.columns if c.startswith('GRAPH_')]),
            'AMT': len([c for c in df.columns if c.startswith('AMT_')]),
            'TIME': len([c for c in df.columns if c.startswith('TIME_')]),
            'MED': len([c for c in df.columns if c.startswith('MED_')]),
        }
    }


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

        for group_name, count in current_csv_info['feature_groups'].items():
            if count > 0:
                report.append(f"| {group_name} | ")
                if baseline_id and baseline_csv_info:
                    baseline_count = baseline_csv_info['feature_groups'].get(group_name, 0)
                    report.append(f"{baseline_count}개 | ")
                report.append(f"{count}개 |")
                if baseline_id and baseline_csv_info:
                    diff = count - baseline_count
                    if diff > 0:
                        report.append(f" +{diff}개 (신규) |")
                    elif diff < 0:
                        report.append(f" {diff}개 (제거) |")
                    else:
                        report.append(f" - |")
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
    """보고서 인덱스 업데이트"""
    reports = sorted(REPORTS_DIR.glob("*_report.md"))

    index = []
    index.append("# 최적화 보고서 목록\n")
    index.append("\n이 폴더는 각 버전별 최적화 보고서를 자동으로 생성하여 보관합니다.\n")
    index.append("\n## 보고서 목록\n")

    for report_path in reports:
        # 파일명에서 정보 추출
        filename = report_path.stem.replace('_report', '')
        index.append(f"\n- [{filename}](./{report_path.name})")

    index_path = REPORTS_DIR / "README.md"
    with open(index_path, 'w', encoding='utf-8') as f:
        f.writelines(index)

    print(f"[OK] 인덱스 업데이트 완료: {index_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="최적화 보고서 자동 생성")
    parser.add_argument("--strategy", required=True, help="현재 전략 ID (예: member_b_strategy_2)")
    parser.add_argument("--baseline", help="비교 대상 전략 ID (예: member_b)")
    parser.add_argument("--version", help="버전 (예: v2, 자동 감지)")
    args = parser.parse_args()

    generate_report(args.strategy, args.baseline, args.version)
    update_index()
