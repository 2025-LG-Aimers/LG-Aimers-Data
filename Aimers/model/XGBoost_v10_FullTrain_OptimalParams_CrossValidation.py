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

threshold = 0.95
biased_columns = [col for col in X.columns if X[col].value_counts(normalize=True).max() >= threshold]
columns_to_remove.extend(biased_columns)

X.drop(columns=columns_to_remove, inplace=True, errors='ignore')
test.drop(columns=columns_to_remove, inplace=True, errors='ignore')

# ✅ 결측치 처리
missing_percentage = (X.isnull().sum() / len(X)) * 100
high_missing_columns = missing_percentage[missing_percentage >= 80].index.tolist()
X.drop(columns=high_missing_columns, inplace=True, errors='ignore')
test.drop(columns=high_missing_columns, inplace=True, errors='ignore')

mid_missing_columns = missing_percentage[(missing_percentage >= 15) & (missing_percentage < 30)].index.tolist()
imputer = SimpleImputer(strategy='mean')
X[mid_missing_columns] = imputer.fit_transform(X[mid_missing_columns])
test[mid_missing_columns] = imputer.transform(test[mid_missing_columns])

# ✅ 범주형 컬럼 인코딩
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
test = test[X.columns]

for col in categorical_columns:
    X[col] = X[col].astype(str)
    test[col] = test[col].astype(str)

ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[categorical_columns] = ordinal_encoder.fit_transform(X[categorical_columns])
test[categorical_columns] = ordinal_encoder.transform(test[categorical_columns])

# ✅ 수치형 컬럼 NaN 채우기
numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
X[numeric_columns] = X[numeric_columns].fillna(0)
test[numeric_columns] = test[numeric_columns].fillna(0)

# -------------- 📌 Focal Loss 적용 --------------
def focal_loss(predt, dtrain, gamma=0.705350567650623, alpha=0.14010485785305138):
    y = dtrain.get_label()
    p = 1 / (1 + np.exp(-predt))
    grad = alpha * (1 - p) ** gamma * (p - y)
    hess = alpha * (1 - p) ** gamma * p * (1 - p) * (1 + gamma * (1 - p) * (y - p))
    return grad, hess

# -------------- 📌 XGBoost 교차 검증 --------------
params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 4,
    "learning_rate":  0.04405038339177713,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
auc_scores = []
accuracy_scores = []
cv_results = []

dtrain_full = xgb.DMatrix(X, label=y)

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=1000,
        obj=lambda predt, dtrain: focal_loss(predt, dtrain, gamma=2.0, alpha=0.25),
        evals=[(dvalid, "valid")],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    valid_pred_proba = model.predict(dvalid)
    auc = roc_auc_score(y_valid, valid_pred_proba)
    accuracy = accuracy_score(y_valid, (valid_pred_proba > 0.5).astype(int))

    auc_scores.append(auc)
    accuracy_scores.append(accuracy)
    cv_results.append(model.best_iteration)

# ✅ 최적의 num_boost_round 찾기
best_num_boost_round = int(np.mean(cv_results))
print(f"🔥 최적의 트리 개수: {best_num_boost_round}")

# ✅ 최종 모델 학습 (Full Training)
final_model = xgb.train(
    params=params,
    dtrain=dtrain_full,
    num_boost_round=best_num_boost_round,
    verbose_eval=True
)

# ✅ 테스트 데이터 예측 및 저장
sample_submission = pd.read_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/sample_submission.csv')
dtest = xgb.DMatrix(test)
sample_submission['probability'] = final_model.predict(dtest)
sample_submission.to_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/XGBoost_CV_5Fold_FullTrain.csv', index=False)

print("✅ XGBoost 교차 검증 완료 & 테스트 예측 결과 저장됨.")
