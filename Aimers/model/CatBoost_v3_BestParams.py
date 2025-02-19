import numpy as np
import pandas as pd
import pickle
from catboost import CatBoostClassifier

#  데이터 로드
train_path = "C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/train.csv"
test_path = "C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/test.csv"
submission_path = "C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/sample_submission.csv"
param_path = "C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/Params/best_catboost_params.pkl" # 저장된 최적 파라미터 파일

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df_sample_submission = pd.read_csv(submission_path)

# ID 컬럼 저장
test_ids = df_sample_submission["ID"]

# 타겟 변수 분리
target = "임신 성공 여부"
X = df_train.drop(columns=["ID", target], errors="ignore")
y = df_train[target]

# 편향된 컬럼 제거
biased_cols = [
    "신선 배아 사용 여부", "미세주입된 난자 수", "IVF 시술 횟수",
    "IVF 임신 횟수", "총 출산 횟수", "정자 출처"
]

threshold = 0.95
biased_cols += [col for col in X.columns if X[col].value_counts(normalize=True).max() >= threshold]

X.drop(columns=biased_cols, inplace=True, errors="ignore")
df_test.drop(columns=biased_cols, inplace=True, errors="ignore")

# 결측치 처리
missing_ratio = (X.isnull().sum() / len(X)) * 100

# 80% 이상 결측 -> 컬럼 삭제
to_drop = missing_ratio[missing_ratio >= 80].index.tolist()
X.drop(columns=to_drop, inplace=True, errors="ignore")
df_test.drop(columns=to_drop, inplace=True, errors="ignore")

# 15%~30% 결측 -> 평균값 대체
to_fill_mean = missing_ratio[(missing_ratio >= 15) & (missing_ratio < 30)].index.tolist()
X[to_fill_mean] = X[to_fill_mean].fillna(X[to_fill_mean].mean())
df_test[to_fill_mean] = df_test[to_fill_mean].fillna(X[to_fill_mean].mean())

# 6. 범주형 데이터 확인 & 정리
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

# 테스트 데이터 컬럼 맞추기
df_test = df_test.reindex(columns=X.columns, fill_value=0)

# 범주형 변수 결측치는 최빈값(Mode)으로 대체
for col in categorical_features:
    most_frequent = X[col].mode()[0]  # 최빈값 찾기
    X[col] = X[col].fillna(most_frequent).astype(str)
    df_test[col] = df_test[col].fillna(most_frequent).astype(str)

# 저장된 최적의 파라미터 로드
try:
    with open(param_path, "rb") as f:
        best_config = pickle.load(f)
    print(f"✅ 최적의 하이퍼파라미터 로드 완료: {best_config}")
except Exception as e:
    print(f"❌ 하이퍼파라미터 파일 로드 실패: {e}")
    exit()

# 모델 학습 (불필요한 출력 제거)
best_config["verbose"] = 0  # 🔥 학습 과정 출력 제거
best_model = CatBoostClassifier(**best_config)

print("🚀 최적의 하이퍼파라미터로 모델 학습 시작...")
best_model.fit(
    X, y,
    cat_features=categorical_features,
    verbose=0  # 🔥 최종 학습 과정 출력 제거
)

# 테스트 데이터 예측
X_test = df_test
predictions = best_model.predict_proba(X_test)[:, 1]

# sample_submission 생성
submission = pd.DataFrame({"ID": test_ids, "probability": predictions})

# 최종 CSV 저장
final_csv_path = "C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/catboost_final_submission_best_params.csv"
submission.to_csv(final_csv_path, index=False)

print(f"✅ 최적화된 모델 예측 결과 저장 완료: {final_csv_path}")
