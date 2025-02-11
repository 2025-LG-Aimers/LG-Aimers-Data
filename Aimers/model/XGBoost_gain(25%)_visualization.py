import pandas as pd
import numpy as np
import xgboost as xgb  # ✅ XGBoost 사용
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score


# ✅ 한글 폰트 경로 설정
plt.rc('font', family='Malgun Gothic')  # Windows: 'Malgun Gothic', Mac: 'AppleGothic'

# ✅ 음수 기호(마이너스) 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False  

# -------------- 📌 데이터 로딩 --------------
train = pd.read_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/train.csv').drop(columns=['ID'])
test = pd.read_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/test.csv').drop(columns=['ID'])

# -------------- 📌 타겟 변수 분리 --------------
X = train.drop(columns=['임신 성공 여부'])  # Feature 데이터
y = train['임신 성공 여부']  # Target 데이터

# ✅ 편향된 컬럼 탐색 & 제거 (95% 이상 같은 값)
threshold = 0.95
biased_columns = [col for col in X.columns if X[col].value_counts(normalize=True).max() >= threshold]

X.drop(columns=biased_columns, inplace=True, errors='ignore')
test.drop(columns=biased_columns, inplace=True, errors='ignore')

print(f"✅ 제거된 컬럼: {biased_columns}")

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

test = test[X_train.columns]  # 🔥 테스트 데이터 컬럼 유지

# ✅ Ordinal Encoding 적용
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_train[categorical_columns] = ordinal_encoder.fit_transform(X_train[categorical_columns])
X_valid[categorical_columns] = ordinal_encoder.transform(X_valid[categorical_columns])
test[categorical_columns] = ordinal_encoder.transform(test[categorical_columns])

# -------------- 📌 XGBoost 모델 학습 --------------
params = {
    "objective": "binary:logistic",
    "eval_metric": ["logloss"],
    "max_depth": 5,  # 6 -> 5 (성능 개선)
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.83,
    "random_state": 42
}

# ✅ DMatrix 생성
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)

# 🔥 Early Stopping 적용
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

# 🔥 Feature Importance (Gain) 높은 순으로 정렬
importance_df = importance_df.sort_values(by='Gain', ascending=False)

# ✅ 상위 20개 Feature 시각화
plt.figure(figsize=(12, 6))
plt.barh(importance_df['Feature'][:20], importance_df['Gain'][:20], color='skyblue')
plt.xlabel("Feature Importance (Gain)")
plt.ylabel("Features")
plt.title("Top 20 Most Important Features (by Gain)")
plt.gca().invert_yaxis()  # 가장 중요한 Feature가 위로 오도록 반전
plt.show()

# ✅ 하위 25% Feature 제거
threshold = importance_df['Gain'].quantile(0.25)
low_gain_features = importance_df[importance_df['Gain'] < threshold]['Feature'].tolist()

# 🔥 Feature 제거 후 다시 학습
X_train.drop(columns=low_gain_features, inplace=True)
X_valid.drop(columns=low_gain_features, inplace=True)
test.drop(columns=low_gain_features, inplace=True)

print(f"📌 제거된 피처 (하위 25%): {low_gain_features}")

# ✅ DMatrix 다시 생성 후 재학습
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)

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
print(f"🔥 검증 데이터 ROC-AUC Score: {auc_score:.4f}")
print(f"✅ 검증 데이터 Accuracy Score: {accuracy:.4f}")

# -------------- 📌 테스트 데이터 예측 및 결과 저장 --------------
dtest = xgb.DMatrix(test)
test_pred_proba = xgb_model.predict(dtest)

sample_submission = pd.read_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/sample_submission.csv')
sample_submission['probability'] = test_pred_proba
sample_submission.to_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/XGBoost_gain(25%).csv', index=False)

print("✅ XGBoost 모델 학습 & 예측 완료, 결과 저장됨.")
