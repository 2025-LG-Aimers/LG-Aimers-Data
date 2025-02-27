"""
 * Project : LG Aimers - 20개의 피처만 두고 CatBoost의 최적의 파라미터를 찾는 코드
 * Program Purpose and Features :
 * - .pkl 파일로 최적의 파라미터 정보 저장, .cbm파일로 catboost model을 파일로 저장
 * Author : SP Hong
 * First Write Date : 2025.02.25
 * ============================================================
 * Program history
 * ============================================================
 * Author    		Date		    Version		            History
   SP Hong          2025.02.25      Model_with_Yolo.v1      20개 피처에서 최적의 파라미터 생성
"""

import numpy as np
import pandas as pd
import optuna
import pickle
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# ✅ **1. 사용하려는 20개 Feature 선택**
selected_features = [
    '이식된 배아 수', '시술 당시 나이', '배아 이식 경과일', '저장된 배아 수', '총 생성 배아 수',
    '특정 시술 유형', '동결 배아 사용 여부', '미세주입에서 생성된 배아 수',
    'IVF 시술 횟수', 'DI 시술 횟수', 'IVF 임신 횟수', '시술 시기 코드',
    '배아 생성 주요 이유', '클리닉 내 총 시술 횟수', 'IVF 출산 횟수',
    '해동된 배아 수', '파트너 정자와 혼합된 난자 수', '난자 출처',
    '난자 기증자 나이', '정자 기증자 나이'
]

# ✅ **2. 데이터 로드**
train_path = "C:/Users/mch2d/Desktop/LG-Aimers-Data-main/train.csv"
test_path = "C:/Users/mch2d/Desktop/LG-Aimers-Data-main/test.csv"
submission_path = "C:/Users/mch2d/Desktop/LG-Aimers-Data-main/sample_submission.csv"

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df_sample_submission = pd.read_csv(submission_path)

# ✅ **3. ID 컬럼 저장**
test_ids = df_sample_submission["ID"]

# ✅ **4. 타겟 변수 분리 (20개 Feature만 선택)**
X = df_train[selected_features].copy()
y = df_train["임신 성공 여부"].copy()

# ✅ **5. 결측치 처리**
missing_ratio = (X.isnull().sum() / len(X)) * 100

# 🎯 (1) 80% 이상 결측 → 컬럼 삭제
to_drop = missing_ratio[missing_ratio >= 80].index.tolist()
X.drop(columns=to_drop, inplace=True, errors="ignore")
df_test.drop(columns=to_drop, inplace=True, errors="ignore")

# 🎯 (2) 수치형 변수 → 중앙값(median)으로 대체
num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
X[num_features] = X[num_features].apply(lambda col: col.fillna(col.median()))
df_test[num_features] = df_test[num_features].apply(lambda col: col.fillna(col.median()))

# 🎯 (3) 범주형 변수 → 최빈값(Mode)으로 대체
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
for col in categorical_features:
    most_frequent = X[col].mode()[0]  # 최빈값 찾기
    X[col] = X[col].fillna(most_frequent).astype(str)
    df_test[col] = df_test[col].fillna(most_frequent).astype(str)

# ✅ **6. 테스트 데이터 컬럼 맞추기**
df_test = df_test.reindex(columns=X.columns, fill_value=0)

# ✅ **7. 클래스 가중치 설정**
weights = {0: 0.2583, 1: 0.7417}  # 실패(0): 0.2583, 성공(1): 0.7417

# 🎯 **8. Optuna를 활용한 하이퍼파라미터 최적화**
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

# ✅ **9. Optuna 실행 (40번 반복)**
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=40)

# ✅ **10. 최적 파라미터 저장**
best_config = study.best_params
best_config.update({
    "random_seed": 42,
    "eval_metric": "AUC",
    "loss_function": "Logloss",
    "task_type": "GPU",
    "devices": "0",
    "verbose": 0  
})

# ✅ **11. 최적화된 파라미터를 CSV 및 PKL로 저장**
param_path_pkl = "C:/Users/mch2d/Desktop/LG-Aimers-Data-main/Params/best_catboost_params_v4.pkl"
param_path_csv = "C:/Users/mch2d/Desktop/LG-Aimers-Data-main/Params/best_catboost_params_v4.csv"

with open(param_path_pkl, "wb") as f:
    pickle.dump(best_config, f)

pd.DataFrame([best_config]).to_csv(param_path_csv, index=False)

print(f"📁 최적의 하이퍼파라미터 저장 완료: {param_path_pkl}, {param_path_csv}")

# ✅ **12. 최적 모델 학습 및 저장**
print("🚀 최적 파라미터로 CatBoost 모델 학습 시작...")
best_model = CatBoostClassifier(**best_config)
best_model.fit(X, y, cat_features=categorical_features, verbose=0)

# ✅ **13. 학습된 모델 저장**
model_path = "C:/Users/mch2d/Desktop/LG-Aimers-Data-main/Models/best_catboost_model_v4.cbm"
best_model.save_model(model_path)

print(f"✅ 학습된 모델 저장 완료: {model_path}")

# ✅ **14. 테스트 데이터 예측 및 저장**
predictions = best_model.predict_proba(df_test)[:, 1]
submission = pd.DataFrame({"ID": test_ids, "probability": predictions})
final_csv_path = "C:/Users/mch2d/Desktop/LG-Aimers-Data-main/catboost_final_submission_v4.csv"
submission.to_csv(final_csv_path, index=False)

print(f"✅ 최종 예측 결과 저장 완료: {final_csv_path}")
