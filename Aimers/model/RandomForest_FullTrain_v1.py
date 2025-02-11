import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score

# -------------- 📌 데이터 로딩 -------------- 
train = pd.read_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/train.csv').drop(columns=['ID'])
test = pd.read_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/test.csv').drop(columns=['ID'])

# -------------- 📌 타겟 변수 분리 -------------- 
X = train.drop(columns=['임신 성공 여부'])  # ✅ 전체 데이터를 학습에 사용
y = train['임신 성공 여부']  

# ✅ 1️⃣ 수치형 컬럼과 범주형 컬럼을 분리
numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()

# ✅ 2️⃣ 수치형 데이터 결측치 평균값 대체
num_imputer = SimpleImputer(strategy='mean')
X[numeric_columns] = num_imputer.fit_transform(X[numeric_columns])
test[numeric_columns] = num_imputer.transform(test[numeric_columns])

# ✅ 3️⃣ 범주형 데이터 결측치는 "missing"으로 채우기
X[categorical_columns] = X[categorical_columns].fillna("missing")
test[categorical_columns] = test[categorical_columns].fillna("missing")

# -------------- 📌 범주형 컬럼 인코딩 -------------- 
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[categorical_columns] = ordinal_encoder.fit_transform(X[categorical_columns])
test[categorical_columns] = ordinal_encoder.transform(test[categorical_columns])

# -------------- 📌 랜덤 포레스트 모델 학습 -------------- 
rf_model = RandomForestClassifier(
    n_estimators=500,  
    max_depth=10,  
    min_samples_split=5,  
    min_samples_leaf=2,  
    random_state=42,
    n_jobs=-1  
)

rf_model.fit(X, y)  # ✅ 전체 데이터를 학습에 사용

# -------------- 📌 학습 데이터(X) 평가 -------------- 
train_pred_proba = rf_model.predict_proba(X)[:, 1]  
train_pred_class = rf_model.predict(X)  

train_auc_score = roc_auc_score(y, train_pred_proba)
train_accuracy = accuracy_score(y, train_pred_class)

print(f"🟢 학습 데이터 ROC-AUC Score: {train_auc_score:.4f}")
print(f"🟢 학습 데이터 Accuracy Score: {train_accuracy:.4f}")

# -------------- 📌 테스트 데이터 예측 및 결과 저장 -------------- 
test_pred_proba = rf_model.predict_proba(test)[:, 1]  

sample_submission = pd.read_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/sample_submission.csv')
sample_submission['probability'] = test_pred_proba
sample_submission.to_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/RandomForest_FullTrain.csv', index=False)

print("✅ 랜덤 포레스트(전체 데이터 학습) 완료, 결과 저장됨.")
