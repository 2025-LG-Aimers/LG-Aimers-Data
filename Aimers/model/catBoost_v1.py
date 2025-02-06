import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool  # ✅ CatBoost 추가
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
X_valid[medium_missing_columns] = X_valid[medium_missing_columns].fillna(X_train[medium_missing_columns].mean())
test[medium_missing_columns] = test[medium_missing_columns].fillna(X_train[medium_missing_columns].mean())

# ✅ (3) 결측치 비율이 0~10%인 컬럼이 포함된 행 삭제
low_missing_columns = missing_percentage[(missing_percentage > 0) & (missing_percentage < 10)].index
if len(low_missing_columns) > 0:
    missing_rows = X_train[low_missing_columns].isnull().any(axis=1)
    X_train = X_train[~missing_rows]
    y_train = y_train[~missing_rows]  # 🔥 y_train도 동일한 행 삭제

# 🔥 인덱스 리셋
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

# -------------- 📌 범주형 컬럼 자동 감지 -------------- 
categorical_columns = X_train.select_dtypes(include=['object']).columns.tolist()
print(f"📌 감지된 범주형 컬럼: {categorical_columns}")

# 🔥 테스트 데이터에서도 동일한 컬럼 유지
test = test[X_train.columns]

# ✅ CatBoost는 범주형 변수 처리를 자동으로 하지만, 명시적으로 지정하면 더 좋음
for col in categorical_columns:
    X_train[col] = X_train[col].astype(str)
    X_valid[col] = X_valid[col].astype(str)
    test[col] = test[col].astype(str)

# -------------- 📌 CatBoost 데이터 변환 (Pool 사용) -------------- 
train_pool = Pool(X_train, label=y_train, cat_features=categorical_columns)
valid_pool = Pool(X_valid, label=y_valid, cat_features=categorical_columns)
test_pool = Pool(test, cat_features=categorical_columns)

# -------------- 📌 CatBoost 모델 학습 -------------- 
model = CatBoostClassifier(
    iterations=1000,  # ✅ 최대 트리 개수 (조기 종료가 적절한 값 찾음)
    depth=6,
    learning_rate=0.05,
    loss_function='Logloss',
    eval_metric='AUC',  # ✅ ROC-AUC를 평가 지표로 사용
    random_seed=42,
    verbose=100,  # 100 라운드마다 출력
    early_stopping_rounds=50  # ✅ 조기 종료 설정
)

# ✅ CatBoost 모델 학습 (조기 종료 포함)
model.fit(
    train_pool,
    eval_set=valid_pool,
    early_stopping_rounds=50,
    use_best_model=True  # ✅ 조기 종료 후 최적 모델 사용
)

# -------------- 📌 검증 데이터에서 성능 평가 -------------- 
valid_pred_proba = model.predict_proba(valid_pool)[:, 1]  
auc_score = roc_auc_score(y_valid, valid_pred_proba)
valid_pred_class = model.predict(valid_pool)
accuracy = accuracy_score(y_valid, valid_pred_class)

print(f"🔥 검증 데이터 ROC-AUC Score: {auc_score:.4f}")
print(f"✅ 검증 데이터 Accuracy: {accuracy:.4f}")

# -------------- 📌 테스트 데이터 예측 및 저장 -------------- 
test_pred_proba = model.predict_proba(test_pool)[:, 1]  # ✅ 확률값 저장

sample_submission = pd.read_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/sample_submission.csv')
sample_submission['probability'] = test_pred_proba  # ✅ 확률값 저장 (0 또는 1 X)
sample_submission.to_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/baseline_submit_catboost.csv', index=False)

print("✅ CatBoost 모델 학습 & 예측 완료, 결과 저장됨.")
