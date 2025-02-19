import numpy as np
import pandas as pd
import pickle
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# ✅ 최적의 하이퍼파라미터 로드
param_path = "C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/Params/best_catboost_params.pkl"

with open(param_path, "rb") as f:
    best_params = pickle.load(f)

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
biased_cols = [
    "신선 배아 사용 여부", "미세주입된 난자 수", "IVF 시술 횟수",
    "IVF 임신 횟수", "총 출산 횟수", "정자 출처"
]
threshold = 0.95
biased_cols += [col for col in X.columns if X[col].value_counts(normalize=True).max() >= threshold]

X.drop(columns=biased_cols, inplace=True, errors="ignore")
df_test.drop(columns=biased_cols, inplace=True, errors="ignore")

# ✅ 결측치 처리
missing_ratio = (X.isnull().sum() / len(X)) * 100

# 80% 이상 결측치 컬럼 삭제
to_drop = missing_ratio[missing_ratio >= 80].index.tolist()
X.drop(columns=to_drop, inplace=True, errors="ignore")
df_test.drop(columns=to_drop, inplace=True, errors="ignore")

# 15%~30% 결측치 컬럼 평균값으로 대체
to_fill_mean = missing_ratio[(missing_ratio >= 15) & (missing_ratio < 30)].index.tolist()
X[to_fill_mean] = X[to_fill_mean].fillna(X[to_fill_mean].mean())
df_test[to_fill_mean] = df_test[to_fill_mean].fillna(X[to_fill_mean].mean())

# ✅ 범주형 데이터 확인 & 정리
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

# 테스트 데이터 컬럼 맞추기
df_test = df_test.reindex(columns=X.columns, fill_value=0)

# ✅ 범주형 변수 결측치는 최빈값(Mode)으로 대체
for col in categorical_features:
    most_frequent = X[col].mode()[0]  # 최빈값 찾기
    X[col] = X[col].fillna(most_frequent).astype(str)
    df_test[col] = df_test[col].fillna(most_frequent).astype(str)

# ✅ `random_state`를 0~100까지 변경하면서 K-Fold 교차 검증
random_states = range(0, 101)
auc_scores = []

for rs in random_states:
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)  # ✅ random_state 변경

    fold_auc_scores = []
    
    for train_idx, valid_idx in kf.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = CatBoostClassifier(**best_params)  # ✅ 최적의 하이퍼파라미터 적용

        model.fit(
            X_train, y_train,
            eval_set=(X_valid, y_valid),
            cat_features=categorical_features,
            early_stopping_rounds=100,
            verbose=0
        )

        valid_preds = model.predict_proba(X_valid)[:, 1]
        fold_auc_scores.append(roc_auc_score(y_valid, valid_preds))

    mean_auc = np.mean(fold_auc_scores)
    auc_scores.append([rs, mean_auc])

    print(f"🎲 random_state={rs} -> 평균 AUC: {mean_auc:.4f}")

# ✅ 최적의 random_state 찾기
df_results = pd.DataFrame(auc_scores, columns=["random_state", "AUC"])
best_random_state = df_results.loc[df_results["AUC"].idxmax(), "random_state"]
best_auc = df_results["AUC"].max()

print(f"\n🏆 최적의 random_state: {best_random_state} (AUC: {best_auc:.4f})")

# ✅ 결과 CSV로 저장
df_results.to_csv("C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/random_state_experiment_results.csv", index=False)
print("📁 random_state 실험 결과 저장 완료: random_state_experiment_results.csv")
