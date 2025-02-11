import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMClassifier, early_stopping  # ✅ 조기 종료 추가
from sklearn.metrics import roc_auc_score, accuracy_score

# -------------- 📌 데이터 로딩 -------------- 
train = pd.read_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/train.csv').drop(columns=['ID'])
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

# ✅ (1) 결측치가 80% 이상인 컬럼 제거
high_missing_columns = missing_percentage[missing_percentage >= 80].index
X_train = X_train.drop(columns=high_missing_columns, errors='ignore')
X_valid = X_valid.drop(columns=high_missing_columns, errors='ignore')
test = test.drop(columns=high_missing_columns, errors='ignore')

# ✅ (2) 결측치가 15% ~ 30% 사이인 컬럼을 평균값으로 대체
medium_missing_columns = missing_percentage[(missing_percentage >= 15) & (missing_percentage < 30)].index
X_train[medium_missing_columns] = X_train[medium_missing_columns].fillna(X_train[medium_missing_columns].mean())
X_valid[medium_missing_columns] = X_valid[medium_missing_columns].fillna(X_train[medium_missing_columns].mean())  # 훈련 데이터 평균 사용
test[medium_missing_columns] = test[medium_missing_columns].fillna(X_train[medium_missing_columns].mean())  # 동일하게 처리

# ✅ (3) 결측치 비율이 0~10%인 컬럼이 포함된 행 삭제
low_missing_columns = missing_percentage[(missing_percentage > 0) & (missing_percentage < 10)].index
if len(low_missing_columns) > 0:
    missing_rows = X_train[low_missing_columns].isnull().any(axis=1)
    X_train = X_train[~missing_rows]
    y_train = y_train[~missing_rows]  # 🔥 y_train도 동일한 행 삭제

# 🔥 인덱스 리셋
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

# ✅ OrdinalEncoder 설정
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# ✅ 학습 데이터 인코딩
X_train_encoded = X_train.copy()
X_train_encoded[categorical_columns] = ordinal_encoder.fit_transform(X_train[categorical_columns])

# ✅ 검증 데이터 & 테스트 데이터 인코딩
X_valid_encoded = X_valid.copy()
X_valid_encoded[categorical_columns] = ordinal_encoder.transform(X_valid[categorical_columns])

X_test_encoded = test.copy()
X_test_encoded[categorical_columns] = ordinal_encoder.transform(test[categorical_columns])

# 🔥 컬럼 이름 유지
X_test_encoded = pd.DataFrame(X_test_encoded, columns=X_train_encoded.columns)

# -------------- 📌 모델 학습 (LightGBM + 조기 종료) -------------- 
model = LGBMClassifier(
    n_estimators=1000,   # ✅ 최대 트리 개수 (Early Stopping이 적절한 값 찾음)
    max_depth=6,       
    learning_rate=0.05,  
    subsample=0.8,      
    colsample_bytree=0.8,  
    random_state=42
)

# ✅ 조기 종료(Early Stopping) 적용 (검증 데이터 사용)
model.fit(
    X_train_encoded, y_train,
    eval_set=[(X_valid_encoded, y_valid)],
    eval_metric="logloss",
    callbacks=[early_stopping(stopping_rounds=50, verbose=True)],  # ✅ 조기 종료 적용
)

# -------------- 📌 검증 데이터에서 성능 평가 -------------- 
valid_pred_proba = model.predict_proba(X_valid_encoded)[:, 1]  
auc_score = roc_auc_score(y_valid, valid_pred_proba)
valid_pred_class = model.predict(X_valid_encoded)
accuracy = accuracy_score(y_valid, valid_pred_class)

print(f"🔥 검증 데이터 ROC-AUC Score: {auc_score:.4f}")
print(f"✅ 검증 데이터 Accuracy: {accuracy:.4f}")

# -------------- 📌 테스트 데이터 예측 및 저장 -------------- 
test_pred_proba = model.predict_proba(X_test_encoded)[:, 1]  # 확률값으로 저장

sample_submission = pd.read_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/sample_submission.csv')
sample_submission['probability'] = test_pred_proba  # ✅ 확률값 저장 (0 또는 1 X)
sample_submission.to_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/baseline_submit_lightgbm.csv', index=False)

print("✅ LightGBM 모델 학습 & 예측 완료, 결과 저장됨.")
