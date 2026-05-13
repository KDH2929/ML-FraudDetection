# 팀원별 Feature Engineering 컴포넌트

## 폴더 구조

```
features/
├── __init__.py              # import 통합
├── README.md               # 이 파일
├── member_a_features.py    # Member A 전용
├── member_b_features.py    # Member B 전용
└── member_c_features.py    # Member C 전용
```

## 목적

**Git Merge Conflict 방지**: 각 팀원이 독립적인 파일에서 작업

## 사용 방법

### Member A
```python
# member_a_strategy.py
from src.preprocessing.components.features.member_a_features import (
    CustomFeatureA,  # Member A가 만든 Feature
)
```

### Member B
```python
# member_b_strategy.py
from src.preprocessing.components.features.member_b_features import (
    DeviationFeature,   # v1
    BurstFeature,
    GraphFeature,
    AmountFeature,      # v2
    TimeFeature,
    MedicalFeature,
)
```

### Member C
```python
# member_c_strategy.py
from src.preprocessing.components.features.member_c_features import (
    CustomFeatureC,  # Member C가 만든 Feature
)
```

## 공통 Feature

공통 Feature는 `feature_engineer.py`에 있습니다:
- ClaimFeature: 고객 단위 청구 요약
- RatioFeature: 청구 비율
- HospFeature: 병원 이용 패턴

```python
# 공통 Feature 사용
from src.preprocessing.components.feature_engineer import (
    ClaimFeature,
    RatioFeature,
    HospFeature,
)
```

## 새 Feature 추가 시

1. **자신의 파일만 수정**
   - Member A → `member_a_features.py`
   - Member B → `member_b_features.py`
   - Member C → `member_c_features.py`

2. **Feature 클래스 작성**
```python
class MyCustomFeature:
    """Feature 설명"""
    
    def __init__(self):
        self.claim_features_ = None
    
    def fit(self, X, y=None, claim_df=None):
        if claim_df is None:
            raise ValueError("...")
        self.claim_features_ = self._build_features(claim_df)
        return self
    
    def _build_features(self, claim_df):
        # Feature 생성 로직
        return result_df
    
    def transform(self, X):
        X = X.copy()
        X = X.merge(self.claim_features_, on="CUST_ID", how="left")
        new_cols = [c for c in self.claim_features_.columns if c != "CUST_ID"]
        X[new_cols] = X[new_cols].fillna(0)
        return X
```

3. **전략에서 사용**
```python
# member_x_strategy.py
from src.preprocessing.components.features.member_x_features import MyCustomFeature

class MemberXStrategy(BaseStrategy):
    def preprocess(self, X, y, claim_df=None):
        feat = MyCustomFeature()
        X = feat.fit(X, y, claim_df=claim_df).transform(X)
        return X
```

## 주의사항

❌ **하지 말 것**:
- 다른 팀원의 파일 수정
- `feature_engineer.py`에 개인 Feature 추가 (공통 Feature만)

✅ **할 것**:
- 자신의 파일만 수정
- Git commit 전 import 테스트
- Feature 설명 주석 작성

---

**최종 업데이트**: 2026-05-12  
**작성자**: Member B (with Claude Sonnet 4.5)
