# Member A, B, C, ABC 전략 종합 비교 분석

> 작성일: 2026-05-13  
> 전처리 전략별 성능 비교 및 최종 권장사항

---

## 📊 전략 개요

| 전략 | 설명 | Feature 수 | 최적화 |
|------|------|-----------|--------|
| **Member A v4** | 기본 집계 + GroupMean | ? | ❌ |
| **Member B v3** | Deviation + Network + Feature Selection | 100 | ✅ |
| **Member C v4** | 고객-청구 집계 + 다양성 | 190 | ❌ |
| **test_member_abc** | A4 + B3 + C4 병합 | 297 | ⚠️ Threshold만 |

---

## 🎯 성능 비교 (LightGBM 기준)

### 순위

| 순위 | 전략 | Recall | F1 Score | F1 Macro | 상태 |
|------|------|--------|----------|----------|------|
| 🥇 | **Member C v4** | **0.6568** | **0.6660** | **0.8172** | 최적화 전 |
| 🥈 | test_member_abc | 0.5443 | 0.6420 | 0.8066 | Threshold 적용 |
| 🥉 | Member B v3 | 0.6144 | 0.6224 | 0.7933 | 최적화 완료 |
| - | Member A v4 | ❌ | ❌ | ❌ | 결측치 오류 |

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

### test_member_abc (A+B+C 병합) ⚠️

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
        
        # CorrelationRemover (threshold=0.98)
        return drop_high_correlation_features(out)
```

#### Feature 축소
```
A4 + B3 + C4 원본:  ~600-700개 (추정)
CorrelationRemover: 297개 (축소율 ~50%)
```

#### 성능 (LightGBM)
```
Recall:    0.5443
F1 Score:  0.6420
F1 Macro:  0.8066
Threshold: 0.7785
```

#### 비교
```
Member C v4 단독:   F1 0.6660 ✅
test_member_abc:    F1 0.6420 ⬇️ (-2.4%p)
Member B v3:        F1 0.6224
```

#### 장점
- ✅ 세 전략의 Feature를 모두 활용
- ✅ CorrelationRemover로 중복 제거

#### 약점
- ❌ **개별 전략보다 성능 낮음**
- ❌ Feature가 많아 학습 무거움 (297개)
- ❌ 해석 어려움 (ma4_, mb3_, mc4_ 혼재)
- ❌ A4의 결측치 문제 상속

#### 결론
**병합이 오히려 성능 저하** → 개별 전략 사용 권장

---

## 💡 핵심 인사이트

### 1. Feature 설계의 중요성
```
Member C v4 (직관적 집계):      F1 0.6660 ✅
Member B v3 (복잡한 파생):      F1 0.6224
test_member_abc (병합):         F1 0.6420
```
→ **단순하고 직관적인 Feature가 더 효과적**

### 2. 최적화의 효과
```
Member B v3 (최적화 완료):
  - Feature Selection: 326개 → 100개
  - Hyperparameter Tuning: Recall 최적화
  - Threshold 최적화: F1 maximization
  
Member C v4 (최적화 전):
  - 기본 설정만으로 F1 0.6660 달성
  - 최적화 시 F1 0.72+ 예상
```
→ **최적화는 중요하지만, 기본 전략이 더 중요**

### 3. 병합의 효과
```
개별 전략 최고 (Member C v4):  F1 0.6660
병합 전략 (test_member_abc):   F1 0.6420 (-3.6%)
```
→ **무작정 병합은 오히려 역효과**

### 4. Feature Selection
```
Member B v3: 326개 → 100개 (69% 감소)
  → 과적합 방지, 학습 속도 향상
  
Member C v4: 190개 (선택 안 함)
  → 선택 없이도 우수 = Feature 설계가 좋음
```
→ **좋은 Feature 설계 > Feature Selection**

---

## 🎯 최종 권장사항

### 🥇 최우선: Member C v4 최적화

#### 현재 성능
```
Recall:    0.6568
F1 Score:  0.6660 (최고)
F1 Macro:  0.8172
최적화:    ❌ (기본값만)
```

#### 작업 순서
1. **Threshold 최적화** (30분, +3~4%p 예상)
   ```bash
   python -m src.optimization.shared.threshold_optimizer --strategy member_c_strategy_4
   ```

2. **Hyperparameter Tuning** (2~3시간, +1~2%p 예상)
   ```bash
   # Member C 커스텀 튜닝 (F1 최적화)
   python -m src.optimization.member_c.custom_tuning --trials 100
   
   # 또는 공통 튜닝 (Recall 최적화)
   python -m src.optimization.shared.hyperparameter_tuner --strategy member_c_strategy_4 --trials 100
   ```

3. **config.py 업데이트**
   ```python
   STRATEGY_THRESHOLDS = {
       "member_b_strategy_3": 0.3819,
       "member_c_strategy_4": 0.XXXX,  # threshold_analysis.json에서 확인
   }
   ```

#### 예상 최종 성능
```
현재:  Recall 0.6568, F1 0.6660
예상:  Recall 0.70+,   F1 0.72+  ⭐⭐⭐
```

---

### 🥈 차선책: Member B v3 (CatBoost)

#### 현재 성능
```
Model:     CatBoost
Recall:    0.6919 (최고)
F1 Score:  0.6351
F1 Macro:  0.7983
최적화:    ✅ 완료
```

#### 장점
- ✅ Recall이 가장 높음 (FN 최소화)
- ✅ 최적화 완료
- ✅ 자동 적용 시스템

#### 단점
- F1은 Member C v4보다 낮음

#### 사용 시나리오
- **Recall 우선** (사기 놓치면 안 됨)
- FP(오탐)보다 FN(미탐) 최소화가 중요

---

### ❌ 비권장

#### test_member_abc (A+B+C 병합)
```
이유:
- 개별 전략보다 성능 낮음 (F1 -3.6%)
- Feature 많아 학습 무거움 (297개)
- 해석 어려움
- 실용성 낮음
```

#### Member A v4
```
이유:
- 결측치 오류로 실행 불가
- 수정 필요
```

---

## 📈 예상 최종 성능 (최적화 후)

| 전략 | 현재 F1 | 최적화 후 F1 (예상) | 개선폭 | 권장도 |
|------|---------|---------------------|--------|--------|
| **Member C v4** | **0.6660** | **0.72+** | **+6%p** | ⭐⭐⭐ 최고 |
| Member B v3 (catboost) | 0.6351 | 0.64~0.65 | +1~2%p | ⭐⭐ 높음 |
| test_member_abc | 0.6420 | - | - | ❌ 비권장 |
| Member A v4 | ❌ | - | - | ⚠️ 수정 필요 |

---

## 🔧 다음 단계

### 즉시 실행 (오늘)
1. **Member C v4 Threshold 최적화** (30분)
2. Member C v4 성능 확인

### 단기 (1~2일)
3. **Member C v4 Hyperparameter Tuning** (2~3시간)
4. config.py 업데이트
5. 최종 성능 검증

### 중기 (1주일)
6. 최적 모델 선택 (LightGBM vs CatBoost vs XGBoost)
7. 앙상블 기법 검토 (선택)
8. 최종 보고서 작성

---

## 📊 성능 비교 차트

### F1 Score
```
Member C v4:        ████████████████ 0.6660 🏆
test_member_abc:    ███████████████  0.6420
Member B v3:        ██████████████   0.6224
```

### Recall
```
Member B v3 (cat):  █████████████████ 0.6919 🏆
Member C v4:        ███████████████   0.6568
test_member_abc:    ████████████      0.5443
```

### Feature 수
```
test_member_abc:    ███████████████████████████████ 297
Member C v4:        ███████████████████ 190
Member B v3:        ██████████ 100
```

---

## 🎓 결론

### 최고 성능 조합
```
전략:   Member C v4
모델:   LightGBM (또는 CatBoost)
최적화: Threshold + Hyperparameter Tuning (예정)
예상:   F1 0.72+
```

### 핵심 교훈
1. **단순하고 직관적인 Feature 설계**가 가장 중요
2. **최적화는 필수**이지만, 기본 전략이 먼저
3. **무작정 병합은 역효과** (test_member_abc)
4. **Feature Selection**은 좋은 설계를 대체할 수 없음

---

**작성**: 전략 비교 분석  
**최종 수정**: 2026-05-13  
**다음 작업**: Member C v4 최적화 시작
