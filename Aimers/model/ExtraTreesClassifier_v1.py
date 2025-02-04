import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

# @데이터 로딩@
train = pd.read_csv('C:/Users/ANTL/Desktop/LG Aimers Data/train_rebalancing_v1.csv').drop(columns=['ID'])
test = pd.read_csv('C:/Users/ANTL/Desktop/LG Aimers Data/test.csv').drop(columns=['ID'])

X = train.drop(columns=['임신 성공 여부'])
y = train['임신 성공 여부']

# 🔥 Train-Test Split (80:20으로 분할) 🔥
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# -------------- 결측치 처리 --------------
missing_percentage = (X_train.isnull().sum() / len(X_train)) * 100
high_missing_columns = missing_percentage[missing_percentage >= 50].index
X_train = X_train.drop(columns=high_missing_columns, errors='ignore')
X_valid = X_valid.drop(columns=high_missing_columns, errors='ignore')

# -------------- 범주형 데이터 인코딩 --------------
categorical_columns = X_train.select_dtypes(include=['object']).columns.tolist()
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

X_train[categorical_columns] = ordinal_encoder.fit_transform(X_train[categorical_columns])
X_valid[categorical_columns] = ordinal_encoder.transform(X_valid[categorical_columns])
test[categorical_columns] = ordinal_encoder.transform(test[categorical_columns])

# 🔥 NaN을 0으로 채우기 🔥
numeric_columns = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
X_train[numeric_columns] = X_train[numeric_columns].fillna(0)
X_valid[numeric_columns] = X_valid[numeric_columns].fillna(0)
test[numeric_columns] = test[numeric_columns].fillna(0)

# -------------- 모델 학습 (ExtraTreesClassifier) --------------
model = ExtraTreesClassifier(random_state=42)
model.fit(X_train, y_train)

# -------------- 학습 데이터 예측 및 평가 --------------
train_pred_proba = model.predict_proba(X_train)[:, 1]
train_pred_class = model.predict(X_train)

train_auc = roc_auc_score(y_train, train_pred_proba)
train_acc = accuracy_score(y_train, train_pred_class)

print(f"🔥 학습 데이터 ROC-AUC Score: {train_auc:.4f}")
print(f"✅ 학습 데이터 Accuracy Score: {train_acc:.4f}")

# -------------- 검증 데이터 예측 및 평가 --------------
valid_pred_proba = model.predict_proba(X_valid)[:, 1]
valid_pred_class = model.predict(X_valid)

valid_auc = roc_auc_score(y_valid, valid_pred_proba)
valid_acc = accuracy_score(y_valid, valid_pred_class)

print(f"🔥 검증 데이터 ROC-AUC Score: {valid_auc:.4f}")
print(f"✅ 검증 데이터 Accuracy Score: {valid_acc:.4f}")

# -------------- 테스트 데이터 예측 및 저장 --------------
pred_proba = model.predict_proba(test)[:, 1]

sample_submission = pd.read_csv('C:/Users/ANTL/Desktop/LG Aimers Data/sample_submission.csv')
sample_submission['probability'] = pred_proba
sample_submission.to_csv('C:/Users/ANTL/Desktop/LG Aimers Data/baseline_submit.csv', index=False)

print("✅ ExtraTrees 모델 학습 & 예측 완료, 결과 저장됨.")
