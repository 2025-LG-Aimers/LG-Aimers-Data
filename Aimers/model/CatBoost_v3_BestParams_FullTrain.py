import numpy as np
import pandas as pd
import pickle
from catboost import CatBoostClassifier

# ✅ 데이터 로드
train_path = "C:/Users/mch2d/Desktop/LG-Aimers-Data-main/train.csv"
test_path = "C:/Users/mch2d/Desktop/LG-Aimers-Data-main/test.csv"
submission_path = "C:/Users/mch2d/Desktop/LG-Aimers-Data-main/sample_submission.csv"
param_path = "C:/Users/mch2d/Desktop/LG-Aimers-Data-main/Params/best_catboost_params_v3.pkl"

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df_sample_submission = pd.read_csv(submission_path)

# ✅ ID 컬럼 저장
test_ids = df_sample_submission["ID"]

# ✅ 타겟 변수 분리
target = "임신 성공 여부"
X = df_train.drop(columns=["ID", target], errors="ignore")
y = df_train[target]

# # 🛠️ **특정 시술 유형('DI')에서 결측치 여부를 새로운 컬럼으로 추가**
# target_columns = [
#     "단일 배아 이식 여부", "총 생성 배아 수", "미세주입에서 생성된 배아 수", "이식된 배아 수",
#     "미세주입 배아 이식 수", "저장된 배아 수", "미세주입 후 저장된 배아 수", "해동된 배아 수",
#     "수집된 신선 난자 수", "파트너 정자와 혼합된 난자 수", "기증자 정자와 혼합된 난자 수", "동결 배아 사용 여부"
# ]

# # 🔥 '시술 유형' 컬럼이 존재하는 경우에만 실행
# if "시술 유형" in df_train.columns:
#     for col in target_columns:
#         df_train[f"{col}_IS_MISSING"] = df_train[col].isnull().astype(int)

# if "시술 유형" in df_test.columns:
#     for col in target_columns:
#         df_test[f"{col}_IS_MISSING"] = df_test[col].isnull().astype(int)

# ✅ 편향된 컬럼 제거
biased_cols = [
    "신선 배아 사용 여부", "미세주입된 난자 수", "혼합된 난자 수",
    "총 시술 횟수", "총 임신 횟수", "총 출산 횟수", "정자 출처"
]
threshold = 0.95
biased_cols += [col for col in X.columns if X[col].value_counts(normalize=True).max() >= threshold]

X.drop(columns=biased_cols, inplace=True, errors="ignore")
df_test.drop(columns=biased_cols, inplace=True, errors="ignore")

# ✅ 결측치 처리
missing_ratio = (X.isnull().sum() / len(X)) * 100

# 80% 이상 결측 -> 컬럼 삭제
to_drop = missing_ratio[missing_ratio >= 80].index.tolist()
X.drop(columns=to_drop, inplace=True, errors="ignore")
df_test.drop(columns=to_drop, inplace=True, errors="ignore")

# 15%~30% 결측 -> 평균값 대체
to_fill_mean = missing_ratio[(missing_ratio >= 15) & (missing_ratio < 30)].index.tolist()
X[to_fill_mean] = X[to_fill_mean].fillna(X[to_fill_mean].mean())
df_test[to_fill_mean] = df_test[to_fill_mean].fillna(X[to_fill_mean].mean())

# ✅ 범주형 데이터 확인 & 정리
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

# ✅ 모든 범주형 변수를 문자열(str)로 변환 (CatBoost가 인식하도록)
for col in categorical_features:
    X[col] = X[col].astype(str)
    df_test[col] = df_test[col].astype(str)

# ✅ 최종 남아 있는 컬럼 확인
remaining_columns = X.columns.tolist()

# ✅ 출력
print("✅ 최종 남아 있는 컬럼 개수:", len(remaining_columns))
print("✅ 최종 남아 있는 컬럼 목록:")
print(remaining_columns)

# ✅ 테스트 데이터 컬럼 맞추기
df_test = df_test.reindex(columns=X.columns, fill_value=0)

# ✅ 저장된 최적의 파라미터 로드
try:
    with open(param_path, "rb") as f:
        best_config = pickle.load(f)
    print(f"✅ 최적의 하이퍼파라미터 로드 완료: {best_config}")
except Exception as e:
    print(f"❌ 하이퍼파라미터 파일 로드 실패: {e}")
    exit()

# ✅ GPU 사용 시 AUC 지원 문제 해결 (Logloss로 변경)
if best_config.get("task_type") == "GPU":
    best_config["eval_metric"] = "Logloss"

# ✅ CatBoost 모델 전체 데이터(Full Train) 학습
print("🚀 Full Train 모델 학습 시작...")
best_config.pop("verbose", None)
best_model = CatBoostClassifier(**best_config, verbose=0)
best_model.fit(X, y, cat_features=categorical_features)

# ✅ 테스트 데이터 예측 (원본 모델)
print("\n🚀 테스트 데이터 예측 중...")
test_preds = best_model.predict_proba(df_test)[:, 1]

# ✅ sample_submission 생성
submission = pd.DataFrame({"ID": test_ids, "probability": test_preds})

# ✅ 최종 CSV 저장
final_csv_path = "C:/Users/mch2d/Desktop/LG-Aimers-Data-main/catboost_Best_Params_v3_FullTrain.csv"
submission.to_csv(final_csv_path, index=False)

print(f"✅ 원본 모델 예측 결과 저장 완료: {final_csv_path}")
