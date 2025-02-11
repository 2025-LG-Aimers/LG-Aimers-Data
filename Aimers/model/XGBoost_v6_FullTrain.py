import pandas as pd
import numpy as np
import xgboost as xgb  # ✅ XGBoost 전체 모듈 사용
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

# -------------- 📌 데이터 로딩 --------------
train = pd.read_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/train.csv').drop(columns=['ID'])
test = pd.read_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/test.csv').drop(columns=['ID'])

# -------------- 📌 타겟 변수 분리 --------------
X = train.drop(columns=['임신 성공 여부'])  # 입력 데이터 (Feature)
y = train['임신 성공 여부']  # 타겟 변수 (Label)

print(f"🔹 전체 학습 데이터 크기: {X.shape}")

# ✅ 1. 명목형(Nominal) vs. 순서형(Ordinal) 컬럼 구분
ordinal_columns = [
    '시술 당시 나이', '총 시술 횟수', '클리닉 내 총 시술 횟수', 'IVF 시술 횟수',
    'DI 시술 횟수', '총 임신 횟수', 'IVF 임신 횟수', 'DI 임신 횟수',
    '총 출산 횟수', 'IVF 출산 횟수'
]

nominal_columns = [
    '시술 시기 코드', '시술 유형', '특정 시술 유형', '배란 유도 유형',
    '배아 생성 주요 이유', '난자 출처', '정자 출처',
    '난자 기증자 나이', '정자 기증자 나이'
]

# ✅ 2. 편향된 컬럼 탐색 & 제거
threshold = 0.99
biased_columns = [col for col in X.columns if X[col].value_counts(normalize=True).max() >= threshold]
X.drop(columns=biased_columns, inplace=True, errors='ignore')
test.drop(columns=biased_columns, inplace=True, errors='ignore')

print(f"✅ 제거된 컬럼: {biased_columns}")

# ✅ 3. 결측치 처리 (수치형 데이터 & 범주형 데이터 분리)
numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# ✅ 3-1. 수치형 데이터 결측치 평균값 대체
num_imputer = SimpleImputer(strategy='mean')
X[numeric_columns] = num_imputer.fit_transform(X[numeric_columns])
test[numeric_columns] = num_imputer.transform(test[numeric_columns])

# ✅ 3-2. 명목형 데이터 결측치는 "missing"으로 채우기
X[nominal_columns] = X[nominal_columns].fillna("missing")
test[nominal_columns] = test[nominal_columns].fillna("missing")

# ✅ 4. 순서형 데이터(Ordinal) → Ordinal Encoding 적용
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[ordinal_columns] = ordinal_encoder.fit_transform(X[ordinal_columns])
test[ordinal_columns] = ordinal_encoder.transform(test[ordinal_columns])

# ✅ 5. 명목형 데이터(Nominal) → Target Encoding 적용
for col in nominal_columns:
    target_mean = train.groupby(col)['임신 성공 여부'].mean()  # ✅ 원본 데이터에서 그룹화
    X[col] = X[col].map(target_mean)  # 학습 데이터 변환
    test[col] = test[col].map(target_mean).fillna(X[col].mean())  # 없는 카테고리는 평균값으로 대체

# -------------- 📌 XGBoost 모델 학습 --------------
params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 6,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}

# XGBoost 전용 DMatrix 생성
dtrain = xgb.DMatrix(X, label=y)  # ✅ 전체 데이터를 학습에 사용

# 🔥 모델 학습 (조기 종료 없음)
xgb_model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=500,  # ✅ Early Stopping 제거했으므로 500번 학습
    verbose_eval=True
)

# -------------- 📌 테스트 데이터 예측 및 결과 저장 --------------
dtest = xgb.DMatrix(test)
test_pred_proba = xgb_model.predict(dtest)

sample_submission = pd.read_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/sample_submission.csv')
sample_submission['probability'] = test_pred_proba
sample_submission.to_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/XGBoost_TgtEnc_FullTrain.csv', index=False)

print("✅ XGBoost 모델 학습 & 예측 완료, 결과 저장됨.")
