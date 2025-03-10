import numpy as np
import pandas as pd
import optuna
import pickle
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# 데이터 로드
file_path_train = "C:/Users/IT/Desktop/LG-Aimers-Data-main/LG-Aimers-Data-main/train.csv"
file_path_test = "C:/Users/IT/Desktop/LG-Aimers-Data-main/LG-Aimers-Data-main/test.csv"
sample_submission_path = "C:/Users/IT/Desktop/LG-Aimers-Data-main/LG-Aimers-Data-main/sample_submission.csv"

df_train = pd.read_csv(file_path_train)
df_test = pd.read_csv(file_path_test)
df_sample_submission = pd.read_csv(sample_submission_path)

# 'ID' 컬럼 유지 (sample_submission을 위해 필요)
test_ids = df_sample_submission["ID"]

# 타겟 컬럼 분리
target_col = "임신 성공 여부"
X = df_train.drop(columns=["ID", target_col], errors="ignore")
y = df_train[target_col]

# 편향된 컬럼 제거
columns_to_remove = [
    "신선 배아 사용 여부", "미세주입된 난자 수", "IVF 시술 횟수",
    "IVF 임신 횟수", "총 출산 횟수", "정자 출처"
]

threshold = 0.95
biased_columns = [col for col in X.columns if X[col].value_counts(normalize=True).max() >= threshold]
columns_to_remove.extend(biased_columns)

X.drop(columns=columns_to_remove, inplace=True, errors="ignore")
df_test.drop(columns=columns_to_remove, inplace=True, errors="ignore")

# 결측치 처리
missing_percentage = (X.isnull().sum() / len(X)) * 100

# 80% 이상 결측치 컬럼 삭제
high_missing_columns = missing_percentage[missing_percentage >= 80].index.tolist()
X.drop(columns=high_missing_columns, inplace=True, errors="ignore")
df_test.drop(columns=high_missing_columns, inplace=True, errors="ignore")

# 15%~30% 결측치 컬럼 평균값으로 대체
mid_missing_columns = missing_percentage[(missing_percentage >= 15) & (missing_percentage < 30)].index.tolist()
X[mid_missing_columns] = X[mid_missing_columns].fillna(X[mid_missing_columns].mean())
df_test[mid_missing_columns] = df_test[mid_missing_columns].fillna(X[mid_missing_columns].mean())

# 범주형 컬럼 확인 (CatBoost에서 직접 처리 가능)
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

# 테스트 데이터 컬럼 맞추기
df_test = df_test.reindex(columns=X.columns, fill_value=0)

# 범주형 변수 NaN 값 처리 → "missing" 문자열로 변환
for col in cat_features:
    X[col] = X[col].fillna("missing").astype(str)
    df_test[col] = df_test[col].fillna("missing").astype(str)

# 클래스 가중치 설정 (불균형 데이터 보정)
class_weights = {0: 0.25, 1: 0.75}  # 실패(0) -> 0.25, 성공(1) -> 0.75

# Optuna를 활용한 하이퍼파라미터 최적화 (K-Fold 적용)
def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 500, 3000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),  # ⚠️ `suggest_loguniform` → `suggest_float(log=True)`
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 50.0, log=True),
        "border_count": trial.suggest_int("border_count", 16, 64),
        "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Lossguide", "Depthwise"]),
        "class_weights": [class_weights[0], class_weights[1]],  # 가중치 적용
        "random_seed": 42,
        "eval_metric": "AUC",
        "loss_function": "Logloss",
        "task_type": "GPU",   # ✅ GPU 사용 설정
        "devices": "0",       # ✅ 특정 GPU (GPU 0번) 사용
        "verbose": 0
    }
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # 5-Fold 교차 검증
    auc_scores = []
    
    for train_idx, valid_idx in kf.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = CatBoostClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=(X_valid, y_valid),
            cat_features=cat_features,
            early_stopping_rounds=100,
            verbose=0
        )
        
        valid_preds = model.predict_proba(X_valid)[:, 1]
        auc_scores.append(roc_auc_score(y_valid, valid_preds))
    
    return np.mean(auc_scores)  # K-Fold 평균 AUC 반환

# Optuna 실행 (최적화)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

# 최적 파라미터 저장 (`pkl` 파일)
best_params = study.best_params
best_params["random_seed"] = 42
best_params["eval_metric"] = "AUC"
best_params["loss_function"] = "Logloss"
best_params["task_type"] = "GPU"   # ✅ GPU 사용
best_params["devices"] = "0"       # ✅ 특정 GPU 사용
best_params["verbose"] = 100

# GPU 사용 시 불필요한 파라미터 제거
if "colsample_bylevel" in best_params:
    del best_params["colsample_bylevel"]

# 최적화된 하이퍼파라미터를 pkl 파일로 저장
params_save_path = "best_catboost_params_kfold.pkl"
with open(params_save_path, "wb") as f:
    pickle.dump(best_params, f)

print(f"📁 최적화된 하이퍼파라미터가 저장되었습니다: {params_save_path}")
print(f"🎯 최적의 하이퍼파라미터: {best_params}")

# 최적 하이퍼파라미터 적용하여 전체 데이터 학습 (K-Fold X, 최종 모델 생성)
best_params["class_weights"] = [class_weights[0], class_weights[1]]

final_model = CatBoostClassifier(**best_params)
final_model.fit(
    X, y,  # 전체 데이터 사용
    cat_features=cat_features,
    verbose=100
)

# 테스트 데이터 예측 (확률값 저장)
X_test = df_test
test_preds = final_model.predict_proba(X_test)[:, 1]  # 확률값 저장

# sample_submission 형식으로 변환
df_submission = pd.DataFrame({"ID": test_ids, "probability": test_preds})

# 최종 CSV 파일 저장
submission_file_path = "C:/Users/IT/Desktop/LG-Aimers-Data-main/LG-Aimers-Data-main/catboost_kfold_weight.csv"
df_submission.to_csv(submission_file_path, index=False)

print(f"✅ 최적화된 CatBoost 모델의 예측 결과가 '{submission_file_path}' 로 저장되었습니다.")