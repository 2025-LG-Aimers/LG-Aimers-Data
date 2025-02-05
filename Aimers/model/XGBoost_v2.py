import pandas as pd
import numpy as np
import xgboost as xgb  # ✅ XGBoost 전체 모듈 사용
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score, accuracy_score  # ✅ Accuracy Score 추가

# -------------- 📌 데이터 로딩 -------------- 
train = pd.read_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/train_rebalancing_v3.csv').drop(columns=['ID'])
test = pd.read_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/test.csv').drop(columns=['ID'])

# -------------- 📌 타겟 변수 분리 -------------- 
X = train.drop(columns=['임신 성공 여부'])  # 입력 데이터 (Feature)
y = train['임신 성공 여부']  # 타겟 변수 (Label)

# -------------- 📌 Train-Test Split -------------- 
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"🔹 학습 데이터 크기: {X_train.shape}, 검증 데이터 크기: {X_valid.shape}")

# -------------- 📌 결측치 처리 -------------- 
missing_percentage = (X_train.isnull().sum() / len(X_train)) * 100

# 결측치 비율이 50% 이상인 컬럼 삭제
high_missing_columns = missing_percentage[missing_percentage >= 50].index
X_train = X_train.drop(columns=high_missing_columns, errors='ignore')
X_valid = X_valid.drop(columns=high_missing_columns, errors='ignore')
test = test.drop(columns=high_missing_columns, errors='ignore')

# 결측치 비율이 0~10%인 컬럼이 포함된 행 삭제
low_missing_columns = missing_percentage[(missing_percentage > 0) & (missing_percentage < 10)].index
if len(low_missing_columns) > 0:
    missing_rows = X_train[low_missing_columns].isnull().any(axis=1)
    X_train = X_train[~missing_rows]
    y_train = y_train[~missing_rows]  # 🔥 y_train도 동일한 행 삭제

# 🔥 인덱스 리셋 (Train-Test Split 후 데이터 정렬 문제 해결)
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

# -------------- 📌 범주형 컬럼 자동 감지 & 인코딩 -------------- 
categorical_columns = X_train.select_dtypes(include=['object']).columns.tolist()
print(f"📌 감지된 범주형 컬럼: {categorical_columns}")

# 🔥 테스트 데이터에서도 동일한 컬럼 유지
test = test[X_train.columns]

# 범주형 데이터를 문자열로 변환
for col in categorical_columns:
    X_train[col] = X_train[col].astype(str)
    X_valid[col] = X_valid[col].astype(str)
    test[col] = test[col].astype(str)

# OrdinalEncoder 설정
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# 학습 데이터 인코딩
X_train_encoded = X_train.copy()
X_train_encoded[categorical_columns] = ordinal_encoder.fit_transform(X_train[categorical_columns])

# 검증 데이터 & 테스트 데이터 인코딩
X_valid_encoded = X_valid.copy()
X_valid_encoded[categorical_columns] = ordinal_encoder.transform(X_valid[categorical_columns])

X_test_encoded = test.copy()
X_test_encoded[categorical_columns] = ordinal_encoder.transform(test[categorical_columns])

# 🔥 컬럼 이름을 유지하도록 변환
X_test_encoded = pd.DataFrame(X_test_encoded, columns=X_train_encoded.columns)

# -------------- 📌 수치형 컬럼 자동 감지 & 결측치 처리 -------------- 
numeric_columns = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(f"📌 감지된 수치형 컬럼: {numeric_columns}")

# NaN을 0으로 채우기
X_train_encoded[numeric_columns] = X_train_encoded[numeric_columns].fillna(0)
X_valid_encoded[numeric_columns] = X_valid_encoded[numeric_columns].fillna(0)
X_test_encoded[numeric_columns] = X_test_encoded[numeric_columns].fillna(0)

# -------------- 📌 XGBoost 모델 학습 (Early Stopping 적용) -------------- 
params = {
    "objective": "binary:logistic",
    "eval_metric": ["logloss"],  # 경고 메시지 방지
    "max_depth": 6,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}

# XGBoost 전용 DMatrix 생성
dtrain = xgb.DMatrix(X_train_encoded, label=y_train)
dvalid = xgb.DMatrix(X_valid_encoded, label=y_valid)

# 🔥 Early Stopping 적용 (조기 종료: 50 라운드 연속 개선 없으면 중단)
watchlist = [(dtrain, "train"), (dvalid, "valid")]
xgb_model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=1000,  # 최대 트리 개수
    evals=watchlist,
    early_stopping_rounds=50,
    verbose_eval=True
)

# -------------- 📌 검증 데이터에서 ROC-AUC 및 Accuracy 평가 -------------- 
valid_pred_proba = xgb_model.predict(dvalid)
valid_pred_class = (valid_pred_proba > 0.5).astype(int)  # 확률 → 클래스 변환

auc_score = roc_auc_score(y_valid, valid_pred_proba)
accuracy = accuracy_score(y_valid, valid_pred_class)  # ✅ Accuracy Score 추가

print(f"🔥 검증 데이터 ROC-AUC Score: {auc_score:.4f}")
print(f"✅ 검증 데이터 Accuracy Score: {accuracy:.4f}")

# -------------- 📌 테스트 데이터 예측 및 결과 저장 -------------- 
dtest = xgb.DMatrix(X_test_encoded)
test_pred_proba = xgb_model.predict(dtest)
test_pred_class = (test_pred_proba > 0.5).astype(int)  # 확률 → 클래스 변환

sample_submission = pd.read_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/sample_submission.csv')
sample_submission['probability'] = test_pred_class  # 🔥 0 또는 1로 저장
sample_submission.to_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/baseline_submit_xgboost.csv', index=False)

print("✅ XGBoost 모델 학습 & 예측 완료, 결과 저장됨.")
