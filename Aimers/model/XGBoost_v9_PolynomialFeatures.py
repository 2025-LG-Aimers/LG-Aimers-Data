import pandas as pd
import numpy as np
import xgboost as xgb  # ✅ XGBoost 전체 모듈 사용
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score

# -------------- 📌 데이터 로딩 --------------
train = pd.read_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/train.csv').drop(columns=['ID'])
test = pd.read_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/test.csv').drop(columns=['ID'])

# -------------- 📌 타겟 변수 분리 --------------
X = train.drop(columns=['임신 성공 여부'])
y = train['임신 성공 여부']

# ✅ 제거할 컬럼 리스트
columns_to_remove = [
    "신선 배아 사용 여부", "미세주입된 난자 수", "IVF 시술 횟수",
    "IVF 임신 횟수", "총 출산 횟수", "정자 출처", "배란 자극 여부"
]

# ✅ 편향된 컬럼 탐색 (95% 이상 한 값 제거)
threshold = 0.95
biased_columns = [col for col in X.columns if X[col].value_counts(normalize=True).max() >= threshold]
columns_to_remove.extend(biased_columns)

# ✅ 컬럼 삭제 적용
X.drop(columns=columns_to_remove, inplace=True, errors='ignore')
test.drop(columns=columns_to_remove, inplace=True, errors='ignore')

print(f"✅ 제거된 컬럼: {columns_to_remove}")

# -------------- 📌 결측치 처리 --------------
# ✅ 1. 결측치 비율이 80% 이상인 컬럼 삭제
missing_percentage = (X.isnull().sum() / len(X)) * 100
high_missing_columns = missing_percentage[missing_percentage >= 80].index.tolist()

X.drop(columns=high_missing_columns, inplace=True, errors='ignore')
test.drop(columns=high_missing_columns, inplace=True, errors='ignore')

# ✅ 2. 범주형 컬럼 인코딩 (Ordinal Encoding)
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
print(f"📌 감지된 범주형 컬럼: {categorical_columns}")

test = test[X.columns]  # 🔥 동일한 컬럼 유지

ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[categorical_columns] = ordinal_encoder.fit_transform(X[categorical_columns])
test[categorical_columns] = ordinal_encoder.transform(test[categorical_columns])

# ✅ 3. 수치형 데이터만 결측치 평균 대체
numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
imputer = SimpleImputer(strategy='mean')
X[numeric_columns] = imputer.fit_transform(X[numeric_columns])
test[numeric_columns] = imputer.transform(test[numeric_columns])

# ✅ NaN 개수 확인 (확인용)
print(f"🔍 NaN 개수 확인: {X.isnull().sum().sum()}")

# -------------- 📌 다항 Feature 생성 --------------
print(f"📌 감지된 수치형 컬럼: {numeric_columns}")

# ✅ Polynomial Features 적용 (2차 상호작용 항만 추가)
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X[numeric_columns])
test_poly = poly.transform(test[numeric_columns])

# ✅ 다항 Feature DataFrame 변환 및 기존 데이터와 병합
poly_feature_names = poly.get_feature_names_out(numeric_columns)
X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names, index=X.index)
test_poly_df = pd.DataFrame(test_poly, columns=poly_feature_names, index=test.index)

X = pd.concat([X, X_poly_df], axis=1)
test = pd.concat([test, test_poly_df], axis=1)

print(f"✅ 추가된 다항 Features: {len(poly_feature_names)} 개")
print(f"🔹 다항 Feature 예시: {poly_feature_names[:5]}")

# ✅ 데이터 타입 변환 (float32로 변환하여 오류 방지)
X = X.astype(np.float32)
test = test.astype(np.float32)

# ✅ Feature 개수 확인
print(f"🔍 Feature 개수 (X): {X.shape}, (test): {test.shape}")

# ✅ Train-Test Split (8:2 비율)
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"🔹 학습 데이터 크기: {X_train.shape}, 검증 데이터 크기: {X_valid.shape}")

# ✅ X_train, X_valid가 DataFrame인지 확인
print(f"🔍 X_train 데이터 타입 확인: {type(X_train)}")
print(f"🔍 X_valid 데이터 타입 확인: {type(X_valid)}")

# ✅ XGBoost DMatrix 변환 전 NumPy 배열로 변환
X_train = X_train.to_numpy()
X_valid = X_valid.to_numpy()

# ✅ XGBoost 전용 DMatrix 생성
try:
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
except Exception as e:
    print(f"🚨 DMatrix 변환 오류 발생: {e}")
    exit()

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

# ✅ 학습 데이터 예측 및 평가
train_pred_proba = xgb_model.predict(dtrain)
train_pred_class = (train_pred_proba > 0.5).astype(int)

train_auc_score = roc_auc_score(y_train, train_pred_proba)
train_accuracy = accuracy_score(y_train, train_pred_class)

print(f"🔥 학습 데이터 ROC-AUC Score: {train_auc_score:.10f}")
print(f"✅ 학습 데이터 Accuracy Score: {train_accuracy:.10f}")

# ✅ 검증 데이터에서 ROC-AUC 및 Accuracy 평가
valid_pred_proba = xgb_model.predict(dvalid)
valid_pred_class = (valid_pred_proba > 0.5).astype(int)

valid_auc_score = roc_auc_score(y_valid, valid_pred_proba)
valid_accuracy = accuracy_score(y_valid, valid_pred_class)

print(f"🔥 검증 데이터 ROC-AUC Score: {valid_auc_score:.10f}")
print(f"✅ 검증 데이터 Accuracy Score: {valid_accuracy:.10f}")

# ✅ 테스트 데이터 예측 및 결과 저장
test = test.to_numpy()  # ✅ XGBoost DMatrix 변환 전 NumPy 배열로 변환
dtest = xgb.DMatrix(test)
test_pred_proba = xgb_model.predict(dtest)

sample_submission = pd.read_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/sample_submission.csv')
sample_submission['probability'] = test_pred_proba
sample_submission.to_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/XGBoost_PolyFeatures_v1.csv', index=False)

print("✅ 다항식 Feature 적용 후 XGBoost 모델 학습 & 예측 완료, 결과 저장됨.")
