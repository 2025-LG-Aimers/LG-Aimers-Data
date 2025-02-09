import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
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

# -------------- 📌 수치형 컬럼 자동 감지 & 정규화 --------------
numeric_columns = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(f"📌 감지된 수치형 컬럼: {numeric_columns}")

# ✅ MinMaxScaler 적용 (신경망 학습 안정성 향상)
scaler = MinMaxScaler()
X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
X_valid[numeric_columns] = scaler.transform(X_valid[numeric_columns])
test[numeric_columns] = scaler.transform(test[numeric_columns])

# -------------- 📌 MLP 신경망 모델 구축 --------------
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),  # 입력층
    Dropout(0.3),  # 과적합 방지
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # 출력층 (이진 분류 → sigmoid)
])

# -------------- 📌 모델 컴파일 --------------
model.compile(
    optimizer=Adam(learning_rate=0.001),  # Adam Optimizer
    loss='binary_crossentropy',  # 이진 분류이므로 Binary Crossentropy 사용
    metrics=['accuracy']
)

# -------------- 📌 Early Stopping 설정 --------------
early_stopping = EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True
)

# -------------- 📌 모델 학습 --------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    epochs=100,  # 최대 100 Epoch까지 학습
    batch_size=512,  # 배치 크기 (너무 작으면 느리고, 너무 크면 메모리 부족)
    callbacks=[early_stopping],  # Early Stopping 적용
    verbose=1
)

# -------------- 📌 검증 데이터에서 ROC-AUC 및 Accuracy 평가 --------------
valid_pred_proba = model.predict(X_valid).flatten()
valid_pred_class = (valid_pred_proba > 0.5).astype(int)

auc_score = roc_auc_score(y_valid, valid_pred_proba)
accuracy = accuracy_score(y_valid, valid_pred_class)

print(f"🔥 검증 데이터 ROC-AUC Score: {auc_score:.4f}")
print(f"✅ 검증 데이터 Accuracy Score: {accuracy:.4f}")

# -------------- 📌 테스트 데이터 예측 및 결과 저장 --------------
test_pred_proba = model.predict(test).flatten()

sample_submission = pd.read_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/sample_submission.csv')
sample_submission['probability'] = test_pred_proba
sample_submission.to_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/baseline_submit_mlp.csv', index=False)

print("✅ MLP 모델 학습 & 예측 완료, 결과 저장됨.")
