import pandas as pd
import numpy as np
import xgboost as xgb  # ✅ XGBoost 전체 모듈 사용
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer  # ✅ 결측치 평균 대체 추가
from sklearn.metrics import roc_auc_score, accuracy_score  # ✅ Accuracy Score 추가

# -------------- 📌 데이터 로딩 --------------
train = pd.read_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/train.csv').drop(columns=['ID'])
test = pd.read_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/test.csv').drop(columns=['ID'])

# -------------- 📌 타겟 변수 분리 --------------
X = train.drop(columns=['임신 성공 여부'])  # 입력 데이터 (Feature)
y = train['임신 성공 여부']  # 타겟 변수 (Label)

# ✅ 1. "총 생성 배아 수" 이상치 제거 (19~51 범위 제거)
outlier_feature = "총 생성 배아 수"
outlier_lower, outlier_upper = 19, 51
X = X[(X[outlier_feature] < outlier_lower) | (X[outlier_feature] > outlier_upper)]
y = y.loc[X.index]  # X에서 데이터가 삭제되었으므로, y도 동일하게 인덱스를 맞춤

print(f"✅ 이상치 제거 완료: {outlier_feature} {outlier_lower}~{outlier_upper} 사이 값 제거")
print(f"🔹 이상치 제거 후 데이터 크기: {X.shape}")

# ✅ 2. 학습(80%) - 검증(20%) 데이터 분할
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"🔹 학습 데이터 크기: {X_train.shape}, 검증 데이터 크기: {X_valid.shape}")

# ✅ 3. 편향된 컬럼 탐색 & 제거 (99% 이상 한 값)
threshold = 0.99
biased_columns = [col for col in X_train.columns if X_train[col].value_counts(normalize=True).max() >= threshold]

X_train.drop(columns=biased_columns, inplace=True, errors='ignore')
X_valid.drop(columns=biased_columns, inplace=True, errors='ignore')
test.drop(columns=biased_columns, inplace=True, errors='ignore')

print(f"✅ 제거된 컬럼: {biased_columns}")

# ✅ 4. 결측치 처리 (수치형 & 범주형)
numeric_columns = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

# ✅ 4-1. 수치형 데이터 결측치 평균값 대체
num_imputer = SimpleImputer(strategy='mean')
X_train[numeric_columns] = num_imputer.fit_transform(X_train[numeric_columns])
X_valid[numeric_columns] = num_imputer.transform(X_valid[numeric_columns])
test[numeric_columns] = num_imputer.transform(test[numeric_columns])

# ✅ 4-2. 범주형 데이터 결측치는 "missing"으로 채우기
categorical_columns = X_train.select_dtypes(include=['object']).columns.tolist()
X_train[categorical_columns] = X_train[categorical_columns].fillna("missing")
X_valid[categorical_columns] = X_valid[categorical_columns].fillna("missing")
test[categorical_columns] = test[categorical_columns].fillna("missing")

# ✅ 5. 범주형 데이터 Ordinal Encoding 적용
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_train[categorical_columns] = ordinal_encoder.fit_transform(X_train[categorical_columns])
X_valid[categorical_columns] = ordinal_encoder.transform(X_valid[categorical_columns])
test[categorical_columns] = ordinal_encoder.transform(test[categorical_columns])

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

# ✅ XGBoost DMatrix 생성
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)

# 🔥 모델 학습 (조기 종료 적용)
watchlist = [(dtrain, "train"), (dvalid, "valid")]
xgb_model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=1000,
    evals=watchlist,
    early_stopping_rounds=50,
    verbose_eval=True
)

# -------------- 📌 검증 데이터에서 성능 평가 --------------
valid_pred_proba = xgb_model.predict(dvalid)
valid_pred_class = (valid_pred_proba > 0.5).astype(int)

auc_score = roc_auc_score(y_valid, valid_pred_proba)
accuracy = accuracy_score(y_valid, valid_pred_class)

print(f"🔥 검증 데이터 ROC-AUC Score: {auc_score:.10f}")
print(f"✅ 검증 데이터 Accuracy Score: {accuracy:.10f}")