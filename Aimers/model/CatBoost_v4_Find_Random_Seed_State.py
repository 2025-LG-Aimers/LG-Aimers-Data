import numpy as np
import pandas as pd
import optuna
import pickle
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# 데이터 로드
train_path = "C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/train.csv"
test_path = "C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/test.csv"
submission_path = "C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/sample_submission.csv"
param_path = "C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/Params/best_catboost_params.pkl"  # ✅ 저장된 최적 파라미터

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df_sample_submission = pd.read_csv(submission_path)

# 'ID' 컬럼 저장
test_ids = df_sample_submission["ID"]

# 타겟 변수 분리
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

# 80% 이상 결측 -> 컬럼 삭제
to_drop = missing_percentage[missing_percentage >= 80].index.tolist()
X.drop(columns=to_drop, inplace=True, errors="ignore")
df_test.drop(columns=to_drop, inplace=True, errors="ignore")

# 15%~30% 결측 -> 평균값 대체
to_fill_mean = missing_percentage[(missing_percentage >= 15) & (missing_percentage < 30)].index.tolist()
X[to_fill_mean] = X[to_fill_mean].fillna(X[to_fill_mean].mean())
df_test[to_fill_mean] = df_test[to_fill_mean].fillna(X[to_fill_mean].mean())

# 범주형 컬럼 확인 (CatBoost에서 직접 처리 가능)
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

# 테스트 데이터 컬럼 맞추기
df_test = df_test.reindex(columns=X.columns, fill_value=0)

# 범주형 변수 결측치는 최빈값(Mode)으로 대체
for col in cat_features:
    most_frequent = X[col].mode()[0]  # 최빈값 찾기
    X[col] = X[col].fillna(most_frequent).astype(str)
    df_test[col] = df_test[col].fillna(most_frequent).astype(str)

# 저장된 최적의 파라미터 로드
try:
    with open(param_path, "rb") as f:
        best_params = pickle.load(f)
    print(f"✅ 최적의 하이퍼파라미터 로드 완료: {best_params}")
except Exception as e:
    print(f"❌ 하이퍼파라미터 파일 로드 실패: {e}")
    exit()

# 여러 random_seed 값으로 테스트
random_seeds = [42, 100, 2024, 777]  # ✅ 테스트할 random_seed 목록
results = []  # 결과 저장 리스트

# 5-Fold Cross Validation으로 random_seed 변경하면서 성능 테스트
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for seed in random_seeds:
    print(f"\n🚀 테스트 중: random_seed = {seed}")
    
    best_params["random_seed"] = seed  # ✅ 변경된 random_seed 적용
    auc_scores = []
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y), 1):
        print(f"  🔹 Fold {fold} 시작...")

        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = CatBoostClassifier(**best_params)
        model.fit(
            X_train, y_train,
            eval_set=(X_valid, y_valid),
            cat_features=cat_features,
            early_stopping_rounds=100,
            verbose=0  # ✅ 출력 제거
        )

        valid_preds = model.predict_proba(X_valid)[:, 1]
        auc = roc_auc_score(y_valid, valid_preds)
        auc_scores.append(auc)

    # ✅ 평균 AUC 계산
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    results.append({"random_seed": seed, "Mean AUC": mean_auc, "Std AUC": std_auc})

# 결과 정리 & 출력
df_results = pd.DataFrame(results)
print("\n🔥 Random Seed별 AUC 성능 비교 🔥")
print(df_results)

# 최적 random_seed 선택 후 모델 최종 학습
best_random_seed = df_results.sort_values("Mean AUC", ascending=False).iloc[0]["random_seed"]
print(f"\n🎯 최적 random_seed 선택: {best_random_seed}")

best_params["random_seed"] = int(best_random_seed)  # ✅ 최적 random_seed 적용

final_model = CatBoostClassifier(**best_params)
final_model.fit(
    X, y,  # 전체 데이터 사용
    cat_features=cat_features,
    verbose=100
)

# 테스트 데이터 예측 (확률값 저장)
X_test = df_test
test_preds = final_model.predict_proba(X_test)[:, 1]  # 확률값 저장

# sample_submission 생성
df_submission = pd.DataFrame({"ID": test_ids, "probability": test_preds})

# 최종 CSV 파일 저장
submission_file_path = "C:/Users/IT/Desktop/LG-Aimers-Data-main/LG-Aimers-Data-main/catboost_kfold_weight_best_seed.csv"
df_submission.to_csv(submission_file_path, index=False)

print(f"✅ 최적 random_seed 모델의 예측 결과가 '{submission_file_path}' 로 저장되었습니다.")
