import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score

# -------------- 📌 데이터 로딩 -------------- 
train = pd.read_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/train.csv').drop(columns=['ID'])
test = pd.read_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/test.csv').drop(columns=['ID'])

# -------------- 📌 타겟 변수 분리 -------------- 
X = train.drop(columns=['임신 성공 여부'])  
y = train['임신 성공 여부']  

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
imputer = SimpleImputer(strategy='mean')  
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

# ✅ 랜덤 포레스트로 변경
# -------------- 📌 랜덤 포레스트 모델 학습 -------------- 
rf_model = RandomForestClassifier(
    n_estimators=500,  # ✅ 트리 개수
    max_depth=10,  # ✅ 최대 깊이 제한
    min_samples_split=5,  # ✅ 최소 분할 샘플 수
    min_samples_leaf=2,  # ✅ 최소 리프 노드 샘플 수
    random_state=42,
    n_jobs=-1  # ✅ 병렬 처리 활성화
)

rf_model.fit(X_train, y_train)  # ✅ 랜덤 포레스트 학습

# ✅ [변경됨] 학습 데이터(X_train) 성능 평가 추가
# -------------- 📌 학습 데이터(X_train) 평가 -------------- 
train_pred_proba = rf_model.predict_proba(X_train)[:, 1]  # 확률 예측
train_pred_class = rf_model.predict(X_train)  # 클래스 예측

train_auc_score = roc_auc_score(y_train, train_pred_proba)
train_accuracy = accuracy_score(y_train, train_pred_class)

print(f"🟢 학습 데이터 ROC-AUC Score: {train_auc_score:.4f}")
print(f"🟢 학습 데이터 Accuracy Score: {train_accuracy:.4f}")

# -------------- 📌 검증 데이터(X_valid) 평가 -------------- 
valid_pred_proba = rf_model.predict_proba(X_valid)[:, 1]  
valid_pred_class = rf_model.predict(X_valid)  

valid_auc_score = roc_auc_score(y_valid, valid_pred_proba)
valid_accuracy = accuracy_score(y_valid, valid_pred_class)

print(f"🔥 검증 데이터 ROC-AUC Score: {valid_auc_score:.4f}")
print(f"🔥 검증 데이터 Accuracy Score: {valid_accuracy:.4f}")

# -------------- 📌 테스트 데이터 예측 및 결과 저장 -------------- 
test_pred_proba = rf_model.predict_proba(test)[:, 1]  

sample_submission = pd.read_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/sample_submission.csv')
sample_submission['probability'] = test_pred_proba
sample_submission.to_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/RandomForest_with_TrainEval.csv', index=False)

print("✅ 랜덤 포레스트 모델 학습 & 예측 완료, 결과 저장됨.")
