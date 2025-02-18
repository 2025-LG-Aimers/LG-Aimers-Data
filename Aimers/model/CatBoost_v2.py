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

# 🎯 **3. 타겟 변수 분리**
target = "임신 성공 여부"
X = df_train.drop(columns=["ID", target], errors="ignore")
y = df_train[target]

# 🔥 **4. 편향된 컬럼 제거**
biased_cols = [
    "신선 배아 사용 여부", "미세주입된 난자 수", "IVF 시술 횟수",
    "IVF 임신 횟수", "총 출산 횟수", "정자 출처"
]

threshold = 0.95
biased_cols += [col for col in X.columns if X[col].value_counts(normalize=True).max() >= threshold]

X.drop(columns=biased_cols, inplace=True, errors="ignore")
df_test.drop(columns=biased_cols, inplace=True, errors="ignore")

# 🛠️ **5. 결측치 처리**
missing_ratio = (X.isnull().sum() / len(X)) * 100

# 🎭 80% 이상 결측 -> 컬럼 삭제
to_drop = missing_ratio[missing_ratio >= 80].index.tolist()
X.drop(columns=to_drop, inplace=True, errors="ignore")
df_test.drop(columns=to_drop, inplace=True, errors="ignore")

# 🎯 15%~30% 결측 -> 평균값 대체
to_fill_mean = missing_ratio[(missing_ratio >= 15) & (missing_ratio < 30)].index.tolist()
X[to_fill_mean] = X[to_fill_mean].fillna(X[to_fill_mean].mean())
df_test[to_fill_mean] = df_test[to_fill_mean].fillna(X[to_fill_mean].mean())

# 🔹 **6. 범주형 데이터 확인 & 정리**
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

# 🔥 테스트 데이터 컬럼 맞추기
df_test = df_test.reindex(columns=X.columns, fill_value=0)

# 🏷️ **범주형 변수 결측치는 최빈값(Mode)으로 대체**
for col in categorical_features:
    most_frequent = X[col].mode()[0]  # 최빈값 찾기
    X[col] = X[col].fillna(most_frequent).astype(str)
    df_test[col] = df_test[col].fillna(most_frequent).astype(str)

# ⚖️ **7. 클래스 가중치 설정**
weights = {0: 0.25, 1: 0.75}  # 실패(0): 0.25, 성공(1): 0.75

# 🎯 **8. Optuna를 활용한 하이퍼파라미터 최적화**
def objective(trial):
    config = {
        "iterations": trial.suggest_int("iterations", 500, 3000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 50.0, log=True),
        "border_count": trial.suggest_int("border_count", 16, 64),
        "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Lossguide", "Depthwise"]),
        "class_weights": [weights[0], weights[1]],
        "random_seed": 42,
        "eval_metric": "AUC",
        "loss_function": "Logloss",
        "task_type": "GPU",
        "devices": "0",
        "verbose": 0  # 🔥 학습 과정 출력 제거
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
            verbose=0  # 🔥 출력 제거
        )
        
        valid_preds = model.predict_proba(X_valid)[:, 1]
        auc_scores.append(roc_auc_score(y_valid, valid_preds))
    
    return np.mean(auc_scores)

# 🚀 **9. Optuna 실행 (하이퍼파라미터 최적화)**
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# 💾 **10. 최적 파라미터 저장**
best_config = study.best_params
best_config.update({
    "random_seed": 42,
    "eval_metric": "AUC",
    "loss_function": "Logloss",
    "task_type": "GPU",
    "devices": "0",
    "verbose": 0  # 🔥 출력 제거
})

# 🔥 GPU 불필요 파라미터 제거
if "colsample_bylevel" in best_config:
    del best_config["colsample_bylevel"]

# 📂 **최적화된 파라미터 저장**
param_path = "best_catboost_params.pkl"
with open(param_path, "wb") as f:
    pickle.dump(best_config, f)

print(f"📁 하이퍼파라미터 저장 완료: {param_path}")
print(f"🎯 최적의 파라미터: {best_config}")

# 🏆 **11. 최적 모델 학습**
best_config["class_weights"] = [weights[0], weights[1]]

try:
    print("🚀 모델 학습 시작...")
    best_model = CatBoostClassifier(**best_config)
    best_model.fit(
        X, y,
        cat_features=categorical_features,
        verbose=0  # 🔥 최종 학습 과정 출력 제거
    )

    # 🔎 **12. 테스트 데이터 예측**
    X_test = df_test
    predictions = best_model.predict_proba(X_test)[:, 1]

    # 📝 **13. sample_submission 생성**
    submission = pd.DataFrame({"ID": test_ids, "probability": predictions})

    # 💾 **14. 최종 CSV 저장**
    final_csv_path = "C:/Users/mch2d/Desktop/LG-Aimers-Data-main/catboost_final_submission.csv"
    submission.to_csv(final_csv_path, index=False)

    print(f"✅ 예측 결과 저장 완료: {final_csv_path}")

except Exception as e:
    print(f"❌ 모델 학습 중 오류 발생: {e}")
