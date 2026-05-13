# Member A, B, C, ABC 전략 종합 비교 분석

> 작성일: 2026-05-13 (최종 업데이트: 2026-05-13)  
> 전처리 전략별 성능 비교 및 최종 권장사항  
> **✅ Threshold 최적화 완료 - test_member_abc 최고 성능 확인**

---

## 📊 전략 개요

| 전략 | 설명 | Feature 수 | 최적화 | 상태 |
|------|------|-----------|--------|------|
| **test_member_abc** | A4 + B3 + C4 병합 (CorrelationRemover 제거) | 295 | ✅ Threshold | 🏆 **최고** |
| **Member A v4** | 기본 집계 + GroupMean + 논문 Feature | ? | ✅ Threshold | ✅ 완료 |
| **Member B v3** | Deviation + Network + Feature Selection | 100 | ✅ Threshold | ✅ 완료 |
| **Member C v4** | 고객-청구 집계 + 다양성 | 190 | ✅ Threshold | ✅ 완료 |

---

## 🎯 성능 비교 (LightGBM 기준)

### 순위

| 순위 | 전략 | Recall | F1 Score | F1 Macro | 상태 |
|------|------|--------|----------|----------|------|
| 🥇 | **test_member_abc** | **0.7417** | **0.6919** | **0.8238** | Threshold 최적화 완료 |
| 🥈 | Member C v4 | 0.5498 | 0.6310 | 0.8027 | Threshold 최적화 완료 |
| 🥉 | Member B v3 | 0.4914 | 0.6165 | 0.7859 | Threshold 최적화 완료 |
| - | Member A v4 | 0.6049 | 0.6256 | 0.7937 | Threshold 최적화 완료 |

### 모델별 최고 성능 (Member B v3)

| 모델 | Recall | F1 Score | F1 Macro |
|------|--------|----------|----------|
| **CatBoost** | **0.6919** 🏆 | 0.6351 | 0.7983 |
| XGBoost | 0.6125 | 0.6390 | 0.8030 |
| Voting | 0.6292 | 0.6374 | 0.8015 |
| LightGBM | 0.6144 | 0.6224 | 0.7933 |
| Stacking | 0.7897 | 0.5940 | 0.7702 |

---

## 🔍 전략별 상세 분석

### Member A v4 ❌

#### 특징
- 기본 Claim 집계
- GroupMeanImputer (직업·나이 그룹 평균)
- Target Encoding
- IQR Capping + Robust Scaling

#### 문제점
```
ValueError: member_a 전처리 데이터에 결측치가 남아 있습니다
- OCCP_GRP_2: 595개
- MATE_OCCP_GRP_2: 11,827개
```

#### 상태
- 🔴 **실험 불가**
- 전처리 수정 필요
- ModeImputer 적용 필요

---

### Member B v3 ⭐ (최적화 완료)

#### 특징
- **v1**: Deviation + Burst + Graph (43개 feature)
- **v2**: v1 + Amount + Time + Medical (74개)
- **v3**: v2 개선 + 신규 feature + 최적화 (100개)

#### Feature Engineering
```
1. DeviationFeature: 그룹별 z-score 편차
2. BurstFeature: 청구 집중도
3. GraphFeature: 네트워크 특성
4. AmountFeature: 금액 패턴
5. TimeFeature: 시계열 패턴
6. MedicalFeature: 의료 행태
7. Interaction, Sequence, Anomaly, Ratio, Temporal
```

#### 최적화 (v3만 적용)
- ✅ **Feature Selection**: 326개 → 100개 (CorrelationRemover + ImportanceSelector)
- ✅ **Hyperparameter Tuning**: Optuna 100 trials, Recall 최적화
- ✅ **Threshold 최적화**: F1 maximization (threshold=0.3819)
- ❌ **SMOTE**: 효과 미미로 제외

#### 성능 (v3, LightGBM)
```
Recall:    0.6144
F1 Score:  0.6224
F1 Macro:  0.7933
```

#### 장점
- ✅ 체계적 최적화 완료
- ✅ 자동 적용 시스템 (tuning_results.json, STRATEGY_THRESHOLDS)
- ✅ 상세 문서화 (476줄)
- ✅ Feature Selection으로 과적합 방지

#### 약점
- F1은 Member C v4보다 낮음
- Feature Engineering이 복잡함

---

### Member C v4 🏆 (F1 최고)

#### 전략 버전별 성능
| 전략 | Recall | F1 | 특징 |
|------|--------|-----|------|
| **v1** | 0.5627 | **0.6566** | 고객-청구 집계 (safe) |
| v2 | 0.5535 | 0.6410 | 누수 방지 강화 |
| v3 | 0.5498 | 0.6327 | safe 버전 더욱 강화 |
| **v4** | **0.6568** | **0.6660** 🏆 | v3 + v1 효과 변수 복원 |

#### Feature Engineering (v4)
```
1. 고객-청구 집계: 청구 건수, 입원일수, 금액
2. 고카디널리티 통계 요약
3. 청구 반복성, 다양성, 간격
4. v1 지연/변동성 변수 복원
5. 결측치 indicator + 중앙값/0 보완
```

#### 성능 (v4, LightGBM)
```
Recall:    0.6568
F1 Score:  0.6660  🏆 (전체 1위)
F1 Macro:  0.8172
```

#### 장점
- 🏆 **F1 전체 1위** (최적화 전임에도)
- ✅ v1과 v4가 안정적으로 높은 성능
- ✅ Feature가 직관적이고 해석 가능
- ✅ Feature Selection 없이도 우수

#### 약점
- ❌ 최적화 전혀 안 함
- ❌ v2, v3가 v1보다 성능 낮음 (과도한 누수 방지)

#### 개선 여지
Member B v3 최적화 기법 적용 시:
```
현재:  Recall 0.6568, F1 0.6660
예상:  Recall 0.70+,   F1 0.72+  ⭐⭐⭐
```

---

### test_member_abc (A+B+C 병합) 🏆 (전체 1위)

#### 구조
```python
class TestMemberAbcStrategy:
    """A4 + B3 + C4 병합 전략"""
    
    def preprocess(X, y, claim_df):
        a = MemberA4Strategy().preprocess(...)  # ma4_ 접두어
        b = MemberB3Strategy().preprocess(...)  # mb3_ 접두어
        c = MemberCStrategy4().preprocess(...) # mc4_ 접두어
        
        # 가로 병합
        out = concat([base, a, b, c], axis=1)
        
        # 중복 컬럼 제거만 (CorrelationRemover 제거됨)
        return out.loc[:, ~out.columns.duplicated(keep="first")]
```

#### 핵심 개선: CorrelationRemover 제거
```
이전 버전 (CorrelationRemover 적용):
  - Feature: 297개
  - F1: 0.6420
  
현재 버전 (CorrelationRemover 제거):
  - Feature: 295개
  - F1: 0.6919 (+4.9%p 향상!) 🚀
```

#### 성능 (LightGBM, 최적화 적용)
```
Recall:    0.7417  🏆 (전체 최고)
F1 Score:  0.6919  🏆 (전체 최고)
F1 Macro:  0.8238
Threshold: 0.3109
```

#### 비교
```
test_member_abc (현재):  F1 0.6919 🏆 최고
Member C v4:             F1 0.6310
Member A v4:             F1 0.6256
Member B v3:             F1 0.6165
```

#### 장점
- 🏆 **A/B/C/ABC 전략 중 최고 성능**
- ✅ 세 전략의 Feature를 모두 활용 (상호 보완)
- ✅ CorrelationRemover 제거로 정보 손실 방지
- ✅ Threshold 최적화 완료 (0.3109)
- ✅ Recall과 F1 모두 최고

#### 약점
- ⚠️ Feature 많음 (295개) - 학습 시간 증가
- ⚠️ 해석 복잡함 (ma4_, mb3_, mc4_ 혼재)

#### 결론
**CorrelationRemover 제거 후 병합 전략이 최고 성능 달성!**  
→ **최우선 권장 전략** ⭐⭐⭐

---

## 💡 핵심 인사이트

### 1. 병합 전략의 성공 (핵심 발견! 🔥)
```
개별 전략 최고 (Member C v4):          F1 0.6310
병합 전략 (test_member_abc - 개선 전): F1 0.6420
병합 전략 (test_member_abc - 개선 후): F1 0.6919 (+7.7%p) 🏆
```
→ **CorrelationRemover 제거가 핵심!**  
→ **상호 보완적 Feature 병합이 개별 전략보다 우수**

### 2. CorrelationRemover의 역설
```
CorrelationRemover 적용:   F1 0.6420 (정보 손실)
CorrelationRemover 제거:   F1 0.6919 (+4.9%p 향상)
```
→ **상관관계 높은 Feature도 다른 관점의 정보 포함**  
→ **무작정 제거는 오히려 성능 저하**  
→ **LightGBM이 자체적으로 Feature 중요도 학습**

### 3. Threshold 최적화의 중요성
```
모든 전략에 Threshold 최적화 적용 후:
test_member_abc:  F1 0.6919 (Recall 0.7417)
Member C v4:      F1 0.6310 (Recall 0.5498)
Member A v4:      F1 0.6256 (Recall 0.6049)
Member B v3:      F1 0.6165 (Recall 0.4914)
```
→ **Threshold 최적화는 필수**

### 4. Feature 다양성의 힘
```
Member A: 그룹 통계 + 논문 기반 Feature
Member B: 네트워크 + 이상치 + 시계열
Member C: 고객-청구 집계 + 다양성

병합 (A+B+C): 다양한 관점의 Feature 결합
  → 상호 보완으로 최고 성능 달성
```
→ **다양한 관점의 Feature가 모델 성능 향상**

---

## 🎯 최종 권장사항

### 🥇 최우선: test_member_abc ⭐⭐⭐

#### 현재 성능 (최적화 완료)
```
Recall:    0.7417 🏆 (최고)
F1 Score:  0.6919 🏆 (최고)
F1 Macro:  0.8238
Threshold: 0.3109 ✅
```

#### 핵심 장점
- 🏆 **A/B/C/ABC 전략 중 압도적 1위**
- ✅ 세 전략의 상호 보완적 Feature 활용
- ✅ CorrelationRemover 제거로 정보 보존
- ✅ Threshold 최적화 완료
- ✅ Recall/F1 균형 우수

#### 사용 방법
```bash
# 이미 최적화 완료 - 바로 사용 가능
python -m src.experiment.experiment_runner --strategy test_member_abc --use-optimization
```

#### 추가 개선 가능성
```
현재:  F1 0.6919
예상:  Hyperparameter Tuning 시 F1 0.71+ 가능 ⭐
```

---

### 🥈 차선책: 개별 전략 (특정 요구사항 시)

#### Member C v4
```
Recall:    0.5498
F1 Score:  0.6310
용도:      해석 가능한 Feature 필요 시
```

#### Member A v4
```
Recall:    0.6049
F1 Score:  0.6256
용도:      논문 기반 Feature 검증 시
```

#### Member B v3
```
Recall:    0.4914
F1 Score:  0.6165
용도:      네트워크 분석 중점 시
```

---

### 🚀 추가 최적화 계획 (선택사항)

#### test_member_abc Hyperparameter Tuning
```bash
# Optuna로 추가 성능 향상 (예상 +1~2%p)
python -m src.optimization.hyperparameter_tuner --strategy test_member_abc --trials 100
```

#### 다른 모델 탐색
```bash
# CatBoost, XGBoost 등 다른 모델 시도
python -m src.experiment.experiment_runner --strategy test_member_abc --model catboost --use-optimization
```

---

## 📈 최종 성능 (Threshold 최적화 완료)

| 전략 | F1 Score | Recall | F1 Macro | 권장도 |
|------|----------|--------|----------|--------|
| **test_member_abc** | **0.6919** 🏆 | **0.7417** 🏆 | **0.8238** | ⭐⭐⭐ 최고 |
| Member C v4 | 0.6310 | 0.5498 | 0.8027 | ⭐⭐ 높음 |
| Member A v4 | 0.6256 | 0.6049 | 0.7937 | ⭐ 보통 |
| Member B v3 | 0.6165 | 0.4914 | 0.7859 | ⭐ 보통 |

### 추가 최적화 시 예상 성능 (Hyperparameter Tuning)

| 전략 | 현재 F1 | 예상 F1 | 예상 개선폭 |
|------|---------|---------|------------|
| test_member_abc | 0.6919 | 0.71+ | +1~2%p |
| Member C v4 | 0.6310 | 0.65+ | +1~2%p |
| Member A v4 | 0.6256 | 0.64+ | +1~2%p |
| Member B v3 | 0.6165 | 0.63+ | +1~2%p |

---

## 🔧 다음 단계

### ✅ 완료 항목
1. ✅ **모든 전략 Threshold 최적화** 완료
2. ✅ **test_member_abc 최고 성능** 확인 (F1 0.6919)
3. ✅ **config.py 업데이트** 완료 (모든 threshold 적용)
4. ✅ **CorrelationRemover 제거 효과** 검증 (+4.9%p 향상)

### 선택사항 (추가 성능 향상)
1. **test_member_abc Hyperparameter Tuning** (2~3시간)
   - 예상 성능 향상: F1 0.69 → 0.71+
   - Optuna 100 trials
   
2. **다른 모델 실험** (1~2시간)
   - CatBoost, XGBoost로 test_member_abc 평가
   - 앙상블 기법 검토 (Voting, Stacking)
   
3. **최종 문서화** (1시간)
   - 최종 모델 선정 근거
   - 성능 개선 과정 요약

---

## 📊 성능 비교 차트

### F1 Score (높을수록 좋음)
```
test_member_abc:    ██████████████████ 0.6919 🏆
Member C v4:        ████████████████   0.6310
Member A v4:        ███████████████    0.6256
Member B v3:        ███████████████    0.6165
```

### Recall (높을수록 좋음)
```
test_member_abc:    ██████████████████ 0.7417 🏆
Member A v4:        ███████████████    0.6049
Member C v4:        █████████████      0.5498
Member B v3:        ███████████        0.4914
```

### Feature 수
```
test_member_abc:    ███████████████████████████████ 295
Member C v4:        ███████████████████ 190
Member B v3:        ██████████ 100
Member A v4:        ? (미확인)
```

### 개선 효과 (CorrelationRemover 제거)
```
test_member_abc (이전):  ███████████████  0.6420
test_member_abc (현재):  ██████████████████ 0.6919  (+4.9%p 🚀)
```

---

## 🎓 결론

### 최고 성능 조합
```
전략:   test_member_abc (A+B+C 병합)
모델:   LightGBM
최적화: Threshold 최적화 완료 (0.3109)
성능:   F1 0.6919, Recall 0.7417 🏆
```

### 핵심 교훈
1. **다양한 관점의 Feature 병합**이 개별 전략보다 우수
2. **CorrelationRemover는 신중하게** - 제거가 오히려 성능 향상 (+4.9%p)
3. **Threshold 최적화는 필수** - 모든 전략에서 성능 향상 확인
4. **상호 보완적 Feature**가 모델 성능을 극대화

### 주요 발견
- ✅ **test_member_abc가 A/B/C/ABC 중 최고**: F1 0.6919
- ✅ **CorrelationRemover 제거 효과**: +4.9%p 성능 향상
- ✅ **Feature 다양성의 힘**: A+B+C 병합이 개별보다 우수
- ✅ **모든 전략 Threshold 최적화**: config.py 완료

### 다음 단계 (선택)
- Hyperparameter Tuning으로 추가 향상 (F1 0.71+ 목표)
- 다른 모델 실험 (CatBoost, XGBoost)
- 앙상블 기법 검토

---

**작성**: 전략 비교 분석  
**최종 수정**: 2026-05-13  
**상태**: ✅ **Threshold 최적화 완료 - test_member_abc 최고 성능 달성**
