# 최적화 보고서 자동 생성 시스템

이 폴더는 각 버전별 최적화 보고서를 자동으로 생성하여 보관합니다.

## 보고서 목록

- [member_b_strategy_2_v2](./member_b_strategy_2_v2_report.md)
- [member_b_v1](./member_b_v1_report.md)
- [member_c_strategy_2_v2](./member_c_strategy_2_v2_report.md)
- [member_c_strategy_3_v3](./member_c_strategy_3_v3_report.md)
- [member_c_v1](./member_c_v1_report.md)


---

## 사용 방법

### 1. 실험 실행 시 자동 생성 (추천)

```bash
# v2 실험 실행 + 보고서 자동 생성
python run_experiment.py --strategy member_b_strategy_2 --model lgbm --generate-report --baseline member_b
python run_experiment.py --strategy member_c_strategy_2 --model lgbm --generate-report --baseline member_c
python run_experiment.py --strategy member_c_strategy_3 --model lgbm
```

**옵션**:
- `--generate-report`: 실험 완료 후 보고서 자동 생성
- `--baseline`: 비교 대상 전략 (이전 버전)

### 2. 보고서만 생성

```bash
python generate_report.py --strategy member_b_strategy_2 --baseline member_b --version v2
python generate_report.py --strategy member_c_strategy_2 --baseline member_c
```

---

## 팀원 협업 가이드

### 새 버전 개발 시

1. 브랜치 생성 및 전략 구현
2. 실험 실행: `python run_experiment.py --strategy [전략] --model all --generate-report --baseline [이전버전]`
3. 보고서 자동 생성 확인
4. 커밋 및 Push

### 타 팀원 작업 확인 시

1. `git pull` 후 `docs/optimization_reports/` 폴더 확인
2. 최신 보고서에서 성능 개선 내역 확인
3. 필요시 재현: `python run_experiment.py --strategy [전략] --model lgbm`

---

**최종 업데이트**: 2026-05-12
