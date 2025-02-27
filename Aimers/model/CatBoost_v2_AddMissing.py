import numpy as np
import pandas as pd
import optuna
import pickle
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# 🚀 **1. 데이터 로드**
train_path = "C:/Users/mch2d/Desktop/LG-Aimers-Data-main/train.csv"
test_path = "C:/Users/mch2d/Desktop/LG-Aimers-Data-main/test.csv"
submission_path = "C:/Users/mch2d/Desktop/LG-Aimers-Data-main/sample_submission.csv"

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df_sample_submission = pd.read_csv(submission_path)

# 📌 **2. ID 컬럼 저장**
test_ids = df_sample_submission["ID"]

# 🛠️ **5. 특정 시술 유형('DI')에서 결측치 여부를 새로운 컬럼으로 추가**
target_columns = [
    "단일 배아 이식 여부", "총 생성 배아 수", "미세주입에서 생성된 배아 수", "이식된 배아 수",
    "미세주입 배아 이식 수", "저장된 배아 수", "미세주입 후 저장된 배아 수", "해동된 배아 수",
    "수집된 신선 난자 수", "파트너 정자와 혼합된 난자 수", "기증자 정자와 혼합된 난자 수", "동결 배아 사용 여부"
]

# 'DI'인 경우 조건 생성
condition_train = df_train["시술 유형"] == "DI"
condition_test = df_test["시술 유형"] == "DI"

for col in target_columns:
    df_train.loc[condition_train, f"{col}_IS_MISSING"] = df_train.loc[condition_train, col].isnull().astype(int)
    df_test.loc[condition_test, f"{col}_IS_MISSING"] = df_test.loc[condition_test, col].isnull().astype(int)

# 🎯 **3. 타겟 변수 분리**
target = "임신 성공 여부"
X = df_train.drop(columns=["ID", target], errors="ignore")
y = df_train[target]

# 🔥 **4. 편향된 컬럼 제거**
biased_cols = [
   "신선 배아 사용 여부", "미세주입된 난자 수", "혼합된 난자 수",
    "총 시술 횟수", "총 임신 횟수", "총 출산 횟수", "정자 출처"
]

threshold = 0.95
biased_cols += [col for col in X.columns if X[col].value_counts(normalize=True).max() >= threshold]

X.drop(columns=biased_cols, inplace=True, errors="ignore")
df_test.drop(columns=biased_cols, inplace=True, errors="ignore")

# 🔥 X, df_test 업데이트
X = df_train.drop(columns=["ID", target], errors="ignore")
df_test = df_test.reindex(columns=X.columns, fill_value=0)

# 🛠️ **6. 결측치 처리**
missing_ratio = (X.isnull().sum() / len(X)) * 100

# 80% 이상 결측 -> 컬럼 삭제
to_drop = missing_ratio[missing_ratio >= 80].index.tolist()
X.drop(columns=to_drop, inplace=True, errors="ignore")
df_test.drop(columns=to_drop, inplace=True, errors="ignore")

# 15%~30% 결측 -> 평균값 대체
to_fill_mean = missing_ratio[(missing_ratio >= 15) & (missing_ratio < 30)].index.tolist()
X[to_fill_mean] = X[to_fill_mean].fillna(X[to_fill_mean].mean())
df_test[to_fill_mean] = df_test[to_fill_mean].fillna(X[to_fill_mean].mean())

# 🔹 **7. 범주형 데이터 확인 & 정리**
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

# 🔥 테스트 데이터 컬럼 맞추기
df_test = df_test.reindex(columns=X.columns, fill_value=0)

# 🏷️ **범주형 변수 결측치는 최빈값(Mode)으로 대체**
for col in categorical_features:
    most_frequent = X[col].mode()[0]  # 최빈값 찾기
    X[col] = X[col].fillna(most_frequent).astype(str)
    df_test[col] = df_test[col].fillna(most_frequent).astype(str)

# ⚖️ **8. 클래스 가중치 설정**
weights = {0: 0.2583, 1: 0.7417}  # 실패(0): 0.2583, 성공(1): 0.7417

# 🎯 **9. Optuna를 활용한 하이퍼파라미터 최적화**
def objective(trial):
    config = {
        "iterations": trial.suggest_int("iterations", 500, 3000),  
        "depth": trial.suggest_int("depth", 4, 10),  
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),  
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 50.0, log=True),  
        "border_count": trial.suggest_int("border_count", 16, 64),  
        "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Lossguide", "Depthwise"]),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),  
        "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),  
        "class_weights": [weights[0], weights[1]],
        "random_seed": 42,
        "eval_metric": "AUC",
        "loss_function": "Logloss",
        "task_type": "GPU",
        "devices": "0",
        "verbose": 0  
    }
    
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []
    
    for train_idx, valid_idx in kfold.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = CatBoostClassifier(**config)
        model.fit(
            X_train, y_train,
            eval_set=(X_valid, y_valid),
            cat_features=categorical_features,
            early_stopping_rounds=100,
            verbose=0  
        )
        
        valid_preds = model.predict_proba(X_valid)[:, 1]
        auc_scores.append(roc_auc_score(y_valid, valid_preds))
    
    return np.mean(auc_scores)

# 🚀 **10. Optuna 실행 (50번 반복)**
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)  # ✅ 시행 횟수 50번으로 변경

# 💾 **11. 최적 파라미터 저장**
best_config = study.best_params
best_config.update({
    "random_seed": 42,
    "eval_metric": "AUC",
    "loss_function": "Logloss",
    "task_type": "GPU",
    "devices": "0",
    "verbose": 0  
})

# 📂 **12. 최적화된 파라미터 저장**
param_path = "C:/Users/mch2d/Desktop/LG-Aimers-Data-main/Params/best_catboost_params_v3.pkl"
with open(param_path, "wb") as f:
    pickle.dump(best_config, f)

# 🚀 **13. 모델 학습 및 예측**
best_model = CatBoostClassifier(**best_config)
best_model.fit(X, y, cat_features=categorical_features, verbose=0)

predictions = best_model.predict_proba(df_test)[:, 1]
submission = pd.DataFrame({"ID": test_ids, "probability": predictions})

submission.to_csv("C:/Users/mch2d/Desktop/LG-Aimers-Data-main/catboost_final_submission_v3.csv", index=False)

print("✅ 최종 예측 완료 및 CSV 저장 완료! 🚀")
