import pandas as pd
import numpy as np
import xgboost as xgb  # ✅ XGBoost 전체 모듈 사용
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer  # ✅ 결측치 평균 대체 추가
from sklearn.metrics import roc_auc_score, accuracy_score  # ✅ Accuracy Score 추가

# -------------- 📌 데이터 로딩 --------------
train = pd.read_csv('./train.csv').drop(columns=['ID'])
test = pd.read_csv('./test.csv').drop(columns=['ID'])

# -------------- 📌 타겟 변수 분리 --------------
X = train.drop(columns=['임신 성공 여부'])  # 입력 데이터 (Feature)
y = train['임신 성공 여부']  # 타겟 변수 (Label)

# 제거할 컬럼 리스트
columns_to_remove = []

# 편향된 컬럼 탐색 (95% 이상 한 값으로 채워진 컬럼)
threshold = 0.99
biased_columns = [col for col in X.columns if X[col].value_counts(normalize=True).max() >= threshold]

# 전체 제거할 컬럼 리스트
columns_to_remove.extend(biased_columns)

# 컬럼 삭제 적용
X.drop(columns=columns_to_remove, inplace=True, errors='ignore')
test.drop(columns=columns_to_remove, inplace=True, errors='ignore')

print(f"✅ 제거된 컬럼: {columns_to_remove}")


# -------------- 📌 Train-Test Split --------------
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"🔹 학습 데이터 크기: {X_train.shape}, 검증 데이터 크기: {X_valid.shape}")

# -------------- 📌 결측치 처리 --------------
missing_percentage = (X_train.isnull().sum() / len(X_train)) * 100

# ✅ 1. 결측치 비율이 80% 이상인 컬럼 삭제
high_missing_columns = missing_percentage[missing_percentage >= 80].index
X_train.drop(columns=high_missing_columns, inplace=True, errors='ignore')
X_valid.drop(columns=high_missing_columns, inplace=True, errors='ignore')
test.drop(columns=high_missing_columns, inplace=True, errors='ignore')

# ✅ 2. 결측치 비율이 15% ~ 30%인 컬럼 평균값으로 대체
mid_missing_columns = missing_percentage[(missing_percentage >= 15) & (missing_percentage < 30)].index
imputer = SimpleImputer(strategy='mean')  # 평균값 대체
X_train[mid_missing_columns] = imputer.fit_transform(X_train[mid_missing_columns])
X_valid[mid_missing_columns] = imputer.transform(X_valid[mid_missing_columns])
test[mid_missing_columns] = imputer.transform(test[mid_missing_columns])

# ✅ 3. 범주형 컬럼 자동 감지 & 인코딩 (Ordinal Encoding)
categorical_columns = X_train.select_dtypes(include=['object']).columns.tolist()
print(f"📌 감지된 범주형 컬럼: {categorical_columns}")

# 🔥 테스트 데이터에서도 동일한 컬럼 유지
test = test[X_train.columns]

# 범주형 데이터를 문자열로 변환
for col in categorical_columns:
    X_train[col] = X_train[col].astype(str)
    X_valid[col] = X_valid[col].astype(str)
    test[col] = test[col].astype(str)

# ✅ OrdinalEncoder 설정 및 변환
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_train[categorical_columns] = ordinal_encoder.fit_transform(X_train[categorical_columns])
X_valid[categorical_columns] = ordinal_encoder.transform(X_valid[categorical_columns])
test[categorical_columns] = ordinal_encoder.transform(test[categorical_columns])

# -------------- 📌 수치형 컬럼 자동 감지 & 결측치 처리 --------------
numeric_columns = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(f"📌 감지된 수치형 컬럼: {numeric_columns}")

# NaN을 0으로 채우기 (추가적인 결측치 처리)
X_train[numeric_columns] = X_train[numeric_columns].fillna(0)
X_valid[numeric_columns] = X_valid[numeric_columns].fillna(0)
test[numeric_columns] = test[numeric_columns].fillna(0)

# -------------- 📌 XGBoost 모델 학습 (Early Stopping 적용) --------------
params = {
    "objective": "binary:logistic",
    "eval_metric": ["logloss"],
    "max_depth": 5,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.83,
    "random_state": 42
}

# XGBoost 전용 DMatrix 생성
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)

# 🔥 Early Stopping 적용 (조기 종료: 50 라운드 연속 개선 없으면 중단)
watchlist = [(dtrain, "train"), (dvalid, "valid")]
xgb_model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=1000,
    evals=watchlist,
    early_stopping_rounds=50,
    verbose_eval=True
)

# ✅ Feature Importance 가져오기 (Gain 기준)
importances_gain = xgb_model.get_score(importance_type='gain')

# DataFrame 변환
importance_df = pd.DataFrame({
    'Feature': list(importances_gain.keys()),
    'Gain': list(importances_gain.values())  # 정보량 기여도
})

# 🔥 중요도 낮은 순으로 정렬
importance_df = importance_df.sort_values(by='Gain', ascending=True)

# Gain 기준으로 제거할 피처 찾기
low_gain_threshold = 5.0
low_gain_features = importance_df[importance_df['Gain'] < low_gain_threshold]['Feature'].tolist()

# 피처 제거 적용
X_train.drop(columns=low_gain_features, inplace=True, errors='ignore')
X_valid.drop(columns=low_gain_features, inplace=True, errors='ignore')
test.drop(columns=low_gain_features, inplace=True, errors='ignore')

# DMatrix 다시 생성
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)

# 다시 학습
xgb_model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=1000,
    evals=[(dtrain, "train"), (dvalid, "valid")],
    early_stopping_rounds=50,
    verbose_eval=True
)

# -------------- 📌 검증 데이터에서 ROC-AUC 및 Accuracy 평가 --------------
valid_pred_proba = xgb_model.predict(dvalid)
valid_pred_class = (valid_pred_proba > 0.5).astype(int)

auc_score = roc_auc_score(y_valid, valid_pred_proba)
accuracy = accuracy_score(y_valid, valid_pred_class)
print(f"📌 제거된 피처: {low_gain_features}")
print(f"🔥 검증 데이터 ROC-AUC Score: {auc_score:.4f}")
print(f"✅ 검증 데이터 Accuracy Score: {accuracy:.4f}")

#-------------- 📌 테스트 데이터 예측 및 결과 저장 --------------
dtest = xgb.DMatrix(test)
test_pred_proba = xgb_model.predict(dtest)

sample_submission = pd.read_csv('./sample_submission.csv')
sample_submission['probability'] = test_pred_proba
sample_submission.to_csv('./XGBoost_gain(under 5.0).csv', index=False)
