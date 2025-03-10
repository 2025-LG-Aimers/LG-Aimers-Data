import numpy as np
import pandas as pd
import optuna
import pickle
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

# ✅ 데이터 로드
file_path_train = "C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/train.csv"
file_path_test = "C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/test.csv"
sample_submission_path = "C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/sample_submission.csv"

df_train = pd.read_csv(file_path_train)
df_test = pd.read_csv(file_path_test)
df_sample_submission = pd.read_csv(sample_submission_path)

# ✅ ID 컬럼 저장
test_ids = df_sample_submission["ID"]

# ✅ 타겟 변수 분리
target_col = "임신 성공 여부"
X = df_train.drop(columns=["ID", target_col], errors="ignore")
y = df_train[target_col]

# ✅ 편향된 컬럼 제거
columns_to_remove = [
    "신선 배아 사용 여부", "미세주입된 난자 수", 
    "IVF 시술 횟수", "IVF 임신 횟수", "총 출산 횟수", "정자 출처", " 배란 자극 여부"
]
threshold = 0.95
biased_columns = [col for col in X.columns if X[col].value_counts(normalize=True).max() >= threshold]
columns_to_remove.extend(biased_columns)

X.drop(columns=columns_to_remove, inplace=True, errors="ignore")
df_test.drop(columns=columns_to_remove, inplace=True, errors="ignore")

# ✅ 결측치 처리
missing_percentage = (X.isnull().sum() / len(X)) * 100

# ✅ 1. 결측치 비율이 80% 이상인 컬럼 삭제
high_missing_columns = missing_percentage[missing_percentage >= 80].index
X.drop(columns=high_missing_columns, inplace=True, errors="ignore")
df_test.drop(columns=high_missing_columns, inplace=True, errors="ignore")

# ✅ 2. 결측치 비율이 15% ~ 30%인 컬럼 평균값으로 대체
mid_missing_columns = missing_percentage[(missing_percentage >= 15) & (missing_percentage < 30)].index
imputer = SimpleImputer(strategy="mean")  # 평균값 대체
X[mid_missing_columns] = imputer.fit_transform(X[mid_missing_columns])
df_test[mid_missing_columns] = imputer.transform(df_test[mid_missing_columns])

# ✅ 3. 범주형 컬럼 자동 감지 & 인코딩 (Ordinal Encoding)
categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X[categorical_columns] = ordinal_encoder.fit_transform(X[categorical_columns])
df_test[categorical_columns] = ordinal_encoder.transform(df_test[categorical_columns])

# ✅ XGBoost Optuna 최적화 (K-Fold 교차 검증 적용)
def objective(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        "random_state": 42,
        "use_label_encoder": False
    }
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []
    
    for train_idx, valid_idx in kf.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)

        model = xgb.train(params, dtrain, num_boost_round=500, evals=[(dvalid, "valid")], early_stopping_rounds=50, verbose_eval=False)

        valid_preds = model.predict(dvalid)
        auc_scores.append(roc_auc_score(y_valid, valid_preds))

    return np.mean(auc_scores)

# ✅ Optuna 실행 (최적화)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

# ✅ 최적 파라미터 저장 (`pkl` 파일)
best_params = study.best_params
best_params["random_state"] = 42
best_params["eval_metric"] = "auc"
best_params["objective"] = "binary:logistic"
best_params["use_label_encoder"] = False

# ✅ 최적화된 하이퍼파라미터 저장
params_save_path = "best_xgboost_params.pkl"
with open(params_save_path, "wb") as f:
    pickle.dump(best_params, f)

print(f"📁 최적화된 하이퍼파라미터 저장 완료: {params_save_path}")
print(f"🎯 최적의 하이퍼파라미터: {best_params}")

# ✅ 최적 하이퍼파라미터로 전체 데이터 학습
dtrain = xgb.DMatrix(X, label=y)
final_model = xgb.train(best_params, dtrain, num_boost_round=500, verbose_eval=True)

# ✅ 테스트 데이터 예측
dtest = xgb.DMatrix(df_test)
test_pred_proba = final_model.predict(dtest)

# ✅ sample_submission 형식으로 변환
df_submission = pd.DataFrame({"ID": test_ids, "probability": test_pred_proba})

# ✅ 최종 CSV 저장
submission_file_path = "C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/XGBoost_Optuna_KFold.csv"
df_submission.to_csv(submission_file_path, index=False)

print(f"✅ 최적화된 XGBoost 모델의 예측 결과가 '{submission_file_path}' 로 저장되었습니다.")
