import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score

# -------------- 📌 데이터 로딩 --------------
train = pd.read_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/train.csv').drop(columns=['ID'])
test = pd.read_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/test.csv').drop(columns=['ID'])

# -------------- 📌 타겟 변수 분리 --------------
X = train.drop(columns=['임신 성공 여부'])  # 입력 데이터 (Feature)
y = train['임신 성공 여부']  # 타겟 변수 (Label)

# ✅ 편향된 컬럼 제거
columns_to_remove = ["신선 배아 사용 여부", "미세주입된 난자 수", "IVF 시술 횟수", 
                     "IVF 임신 횟수", "총 출산 횟수", "정자 출처"]

# ✅ 95% 이상 편향된 컬럼 제거
threshold = 0.95
biased_columns = [col for col in X.columns if X[col].value_counts(normalize=True).max() >= threshold]
columns_to_remove.extend(biased_columns)

X.drop(columns=columns_to_remove, inplace=True, errors='ignore')
test.drop(columns=columns_to_remove, inplace=True, errors='ignore')

print(f"✅ 제거된 편향된 컬럼: {biased_columns}")

# -------------- 📌 결측치 처리 --------------
missing_percentage = (X.isnull().sum() / len(X)) * 100

# ✅ 80% 이상 결측치 컬럼 삭제
high_missing_columns = missing_percentage[missing_percentage >= 80].index.tolist()
X.drop(columns=high_missing_columns, inplace=True, errors='ignore')
test.drop(columns=high_missing_columns, inplace=True, errors='ignore')

print(f"✅ 제거된 결측치 높은 컬럼(80% 이상): {high_missing_columns}")

# ✅ 15% ~ 30% 결측치 평균값으로 대체
mid_missing_columns = missing_percentage[(missing_percentage >= 15) & (missing_percentage < 30)].index.tolist()
imputer = SimpleImputer(strategy='mean')
X[mid_missing_columns] = imputer.fit_transform(X[mid_missing_columns])
test[mid_missing_columns] = imputer.transform(test[mid_missing_columns])

# ✅ 범주형 컬럼 인코딩 (Ordinal Encoding)
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
print(f"📌 감지된 범주형 컬럼: {categorical_columns}")

# 🔥 테스트 데이터에서도 동일한 컬럼 유지
test = test[X.columns]

# 범주형 데이터를 문자열로 변환
for col in categorical_columns:
    X[col] = X[col].astype(str)
    test[col] = test[col].astype(str)

# ✅ OrdinalEncoder 설정 및 변환
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[categorical_columns] = ordinal_encoder.fit_transform(X[categorical_columns])
test[categorical_columns] = ordinal_encoder.transform(test[categorical_columns])

# ✅ 수치형 컬럼 NaN 채우기
numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
X[numeric_columns] = X[numeric_columns].fillna(0)
test[numeric_columns] = test[numeric_columns].fillna(0)

# -------------- 📌 XGBoost 교차 검증 & 성능 측정 --------------
params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 6,
    "learning_rate": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}

# ✅ Stratified K-Fold 설정 (5-Fold)
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

auc_scores = []
accuracy_scores = []

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
    print(f"🔹 Fold {fold} 시작...")

    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    # ✅ XGBoost 학습
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=1000,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    # ✅ 검증 데이터 예측
    valid_pred_proba = model.predict(dvalid)
    valid_pred_class = (valid_pred_proba > 0.5).astype(int)

    # ✅ 성능 측정
    auc = roc_auc_score(y_valid, valid_pred_proba)
    accuracy = accuracy_score(y_valid, valid_pred_class)

    auc_scores.append(auc)
    accuracy_scores.append(accuracy)

    print(f"✅ Fold {fold} - ROC-AUC: {auc:.6f}, Accuracy: {accuracy:.6f}")

# ✅ Cross Validation 평균 성능 출력
mean_auc = np.mean(auc_scores)
std_auc = np.std(auc_scores)
mean_accuracy = np.mean(accuracy_scores)
std_accuracy = np.std(accuracy_scores)

print("\n🔥 Cross Validation Results 🔥")
print(f"✅ Mean ROC-AUC Score: {mean_auc:.6f} ± {std_auc:.6f}")
print(f"✅ Mean Accuracy Score: {mean_accuracy:.6f} ± {std_accuracy:.6f}")

# -------------- 📌 최적 트리 개수 찾기 --------------
dtrain = xgb.DMatrix(X, label=y)
cv_results = xgb.cv(
    params=params,
    dtrain=dtrain,
    num_boost_round=1000,
    folds=skf,
    early_stopping_rounds=50,
    verbose_eval=True
)

best_num_boost_round = len(cv_results)
print(f"🔥 최적의 트리 개수: {best_num_boost_round}")

# -------------- 📌 최종 모델 학습 --------------
final_model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=best_num_boost_round,
    verbose_eval=True
)
