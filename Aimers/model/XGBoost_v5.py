# 전체 데이터(train.csv)를 100% 학습에 사용하는 XGBoost 코드

import pandas as pd
import numpy as np
import xgboost as xgb  # ✅ XGBoost 전체 모듈 사용
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer  # ✅ 결측치 평균 대체 추가
from sklearn.decomposition import PCA

# -------------- 📌 데이터 로딩 --------------
train = pd.read_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/train.csv').drop(columns=['ID'])
test = pd.read_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/test.csv').drop(columns=['ID'])

# -------------- 📌 타겟 변수 분리 --------------
X = train.drop(columns=['임신 성공 여부'])  # 입력 데이터 (Feature)
y = train['임신 성공 여부']  # 타겟 변수 (Label)

# 제거할 컬럼 리스트
columns_to_remove = ["신선 배아 사용 여부", "미세주입된 난자 수", "혼합된 난자 수"]

# # ✅ PCA 적용할 변수 선택
# pca_features = ["미세주입된 난자 수", "미세주입에서 생성된 배아 수"]

# # ✅ NaN을 평균값으로 대체
# X[pca_features] = X[pca_features].fillna(X[pca_features].mean())

# # ✅ PCA 변환 수행
# pca = PCA(n_components=1)
# X_pca = pca.fit_transform(X[pca_features])

# # ✅ PCA 변수를 추가한 후 원본 변수 제거
# X["PCA_배아_난자"] = X_pca
# test[pca_features] = test[pca_features].fillna(test[pca_features].mean())
# test_pca = pca.transform(test[pca_features])  # 테스트 데이터 변환
# test["PCA_배아_난자"] = test_pca

# X.drop(columns=pca_features, inplace=True, errors='ignore')
# test.drop(columns=pca_features, inplace=True, errors='ignore')

# print(f"✅ PCA 적용 완료! 변환된 변수: PCA_배아_난자")

# 편향된 컬럼 탐색 (95% 이상 한 값으로 채워진 컬럼)
threshold = 0.95
biased_columns = [col for col in X.columns if X[col].value_counts(normalize=True).max() >= threshold]

# 전체 제거할 컬럼 리스트
columns_to_remove.extend(biased_columns)

# 컬럼 삭제 적용
X.drop(columns=columns_to_remove, inplace=True, errors='ignore')
test.drop(columns=columns_to_remove, inplace=True, errors='ignore')

print(f"✅ 제거된 컬럼: {columns_to_remove}")

# -------------- 📌 결측치 처리 --------------
missing_percentage = (X.isnull().sum() / len(X)) * 100

# ✅ 1. 결측치 비율이 80% 이상인 컬럼 삭제
high_missing_columns = missing_percentage[missing_percentage >= 80].index
X.drop(columns=high_missing_columns, inplace=True, errors='ignore')
test.drop(columns=high_missing_columns, inplace=True, errors='ignore')

# ✅ 2. 결측치 비율이 15% ~ 30%인 컬럼 평균값으로 대체
mid_missing_columns = missing_percentage[(missing_percentage >= 15) & (missing_percentage < 30)].index
imputer = SimpleImputer(strategy='mean')  # 평균값 대체
X[mid_missing_columns] = imputer.fit_transform(X[mid_missing_columns])
test[mid_missing_columns] = imputer.transform(test[mid_missing_columns])

# ✅ 3. 범주형 컬럼 자동 감지 & 인코딩 (Ordinal Encoding)
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
print(f"📌 감지된 범주형 컬럼: {categorical_columns}")

# 🔥 테스트 데이터에서도 동일한 컬럼 유지
test = test[X.columns]

# 범주형 데이터를 문자열로 변환
for col in categorical_columns:
    X[col] = X[col].astype(str)
    test[col] = test[col].astype(str)

# ✅ OrdinalEncoder 설정 및 변환
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[categorical_columns] = ordinal_encoder.fit_transform(X[categorical_columns])
test[categorical_columns] = ordinal_encoder.transform(test[categorical_columns])

# -------------- 📌 수치형 컬럼 자동 감지 & 결측치 처리 --------------
numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(f"📌 감지된 수치형 컬럼: {numeric_columns}")

# NaN을 0으로 채우기 (추가적인 결측치 처리)
X[numeric_columns] = X[numeric_columns].fillna(0)
test[numeric_columns] = test[numeric_columns].fillna(0)

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
sample_submission.to_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/XGBoost_full_train(threshold_0.95).csv', index=False)

print("✅ XGBoost 모델 학습 & 예측 완료, 결과 저장됨.")