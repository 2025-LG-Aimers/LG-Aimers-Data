import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split  # ✅ 데이터 분할 추가
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score  # ✅ 성능 평가 추가

# -------------- 📌 데이터 로딩 --------------
train = pd.read_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/train.csv').drop(columns=['ID'])
test = pd.read_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/test.csv').drop(columns=['ID'])

# -------------- 📌 타겟 변수 분리 --------------
X = train.drop(columns=['임신 성공 여부'])  # 입력 데이터 (Feature)
y = train['임신 성공 여부']  # 타겟 변수 (Label)

print(f"🔹 전체 학습 데이터 크기: {X.shape}")

# ✅ 1. 훈련 데이터 66% / 검증 데이터 34%로 분할
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.34, random_state=42, stratify=y
)

print(f"🔹 학습 데이터 크기: {X_train.shape}, 검증 데이터 크기: {X_valid.shape}")

# ✅ 2. 명목형(Nominal) vs. 순서형(Ordinal) 컬럼 구분
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

# ✅ 3. 편향된 컬럼 탐색 & 제거
threshold = 0.99
biased_columns = [col for col in X_train.columns if X_train[col].value_counts(normalize=True).max() >= threshold]
X_train.drop(columns=biased_columns, inplace=True, errors='ignore')
X_valid.drop(columns=biased_columns, inplace=True, errors='ignore')
test.drop(columns=biased_columns, inplace=True, errors='ignore')

print(f"✅ 제거된 컬럼: {biased_columns}")

# ✅ 4. 결측치 처리 (수치형 데이터 & 범주형 데이터 분리)
numeric_columns = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

# ✅ 4-1. 수치형 데이터 결측치 평균값 대체
num_imputer = SimpleImputer(strategy='mean')
X_train[numeric_columns] = num_imputer.fit_transform(X_train[numeric_columns])
X_valid[numeric_columns] = num_imputer.transform(X_valid[numeric_columns])
test[numeric_columns] = num_imputer.transform(test[numeric_columns])

# ✅ 4-2. 명목형 데이터 결측치는 "missing"으로 채우기
X_train[nominal_columns] = X_train[nominal_columns].fillna("missing")
X_valid[nominal_columns] = X_valid[nominal_columns].fillna("missing")
test[nominal_columns] = test[nominal_columns].fillna("missing")

# ✅ 5. 순서형 데이터(Ordinal) → Ordinal Encoding 적용
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_train[ordinal_columns] = ordinal_encoder.fit_transform(X_train[ordinal_columns])
X_valid[ordinal_columns] = ordinal_encoder.transform(X_valid[ordinal_columns])
test[ordinal_columns] = ordinal_encoder.transform(test[ordinal_columns])

# ✅ 6. 명목형 데이터(Nominal) → Target Encoding 적용
for col in nominal_columns:
    target_mean = train.groupby(col)['임신 성공 여부'].mean()  # ✅ 원본 데이터에서 그룹화
    X_train[col] = X_train[col].map(target_mean)
    X_valid[col] = X_valid[col].map(target_mean).fillna(X_train[col].mean())
    test[col] = test[col].map(target_mean).fillna(X_train[col].mean())

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

# ✅ XGBoost 전용 DMatrix 생성
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)

# 🔥 Early Stopping 적용 (조기 종료: 50 라운드 연속 개선 없으면 중단)
watchlist = [(dtrain, "train"), (dvalid, "valid")]
xgb_model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=1000,  # ✅ 최대 1000 라운드
    evals=watchlist,
    early_stopping_rounds=50,  # ✅ 검증 데이터 개선 없으면 조기 종료
    verbose_eval=True
)

# -------------- 📌 검증 데이터에서 성능 평가 --------------
valid_pred_proba = xgb_model.predict(dvalid)
valid_pred_class = (valid_pred_proba > 0.5).astype(int)

auc_score = roc_auc_score(y_valid, valid_pred_proba)
accuracy = accuracy_score(y_valid, valid_pred_class)

print(f"🔥 검증 데이터 ROC-AUC Score: {auc_score:.4f}")
print(f"✅ 검증 데이터 Accuracy Score: {accuracy:.4f}")

# -------------- 📌 테스트 데이터 예측 및 결과 저장 --------------
dtest = xgb.DMatrix(test)
test_pred_proba = xgb_model.predict(dtest)

sample_submission = pd.read_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/sample_submission.csv')
sample_submission['probability'] = test_pred_proba
sample_submission.to_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/XGBoost_TgtEnc_66-34.csv', index=False)

print("✅ XGBoost 모델 학습 & 예측 완료, 결과 저장됨.")
