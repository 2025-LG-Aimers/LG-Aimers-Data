import numpy as np
import pandas as pd
import pickle
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score

# ✅ 데이터 로드
train_path = "C:/Users/mch2d/Desktop/LG-Aimers-Data-main/train.csv"
test_path = "C:/Users/mch2d/Desktop/LG-Aimers-Data-main/test.csv"
submission_path = "C:/Users/mch2d/Desktop/LG-Aimers-Data-main/sample_submission.csv"
param_path = "C:/Users/mch2d/Desktop/LG-Aimers-Data-main/Params/best_catboost_params.pkl"

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df_sample_submission = pd.read_csv(submission_path)

# ✅ ID 컬럼 저장
test_ids = df_sample_submission["ID"]

# ✅ 타겟 변수 분리
target = "임신 성공 여부"
X = df_train.drop(columns=["ID", target], errors="ignore")
y = df_train[target]

# ✅ 편향된 컬럼 제거
biased_cols = [
    "신선 배아 사용 여부", "미세주입된 난자 수", "혼합된 난자 수", "IVF 시술 횟수",
    "IVF 임신 횟수", "총 출산 횟수", "정자 출처"
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

# 테스트 데이터 컬럼 맞추기
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

# ✅ 8:2로 데이터 분할 (Stratify 사용하여 클래스 비율 유지)
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"📊 훈련 데이터 크기: {X_train.shape}, 검증 데이터 크기: {X_valid.shape}")

# ✅ CatBoost 모델 학습
best_config.pop("verbose", None)  # verbose 제거
best_model = CatBoostClassifier(**best_config, verbose=0)

print("🚀 모델 학습 시작...")
best_model.fit(X_train, y_train, cat_features=categorical_features)

# ✅ 검증 데이터 예측 (확률 예측)
valid_probs = best_model.predict_proba(X_valid)[:, 1]
valid_auc = roc_auc_score(y_valid, valid_probs)

# ✅ 검증 데이터 예측 (클래스 예측)
valid_preds = best_model.predict(X_valid)
valid_acc = accuracy_score(y_valid, valid_preds)

print(f"✅ 모델 평가 완료! AUC: {valid_auc:.10f} | Accuracy: {valid_acc:.10f}")

# ✅ 후처리 캘리브레이션 (Platt Scaling) 적용
cal_model = CalibratedClassifierCV(best_model, method="sigmoid", cv="prefit")
cal_model.fit(X_valid.values, y_valid)

# ✅ 캘리브레이션 후 검증 데이터 예측 (확률 예측)
valid_calibrated_probs = cal_model.predict_proba(X_valid.values)[:, 1]
valid_calibrated_auc = roc_auc_score(y_valid, valid_calibrated_probs)

# ✅ 캘리브레이션 후 검증 데이터 예측 (클래스 예측)
valid_calibrated_preds = cal_model.predict(X_valid.values)
valid_calibrated_acc = accuracy_score(y_valid, valid_calibrated_preds)

print(f"🎯 캘리브레이션 후 평가! AUC: {valid_calibrated_auc:.10f} | Accuracy: {valid_calibrated_acc:.10f}")

# ✅ 테스트 데이터 예측
test_preds = best_model.predict_proba(df_test)[:, 1]
calibrated_test_preds = cal_model.predict_proba(df_test.values)[:, 1]

# ✅ sample_submission 생성
submission_raw = pd.DataFrame({"ID": test_ids, "probability": test_preds})
submission_calibrated = pd.DataFrame({"ID": test_ids, "probability": calibrated_test_preds})

# ✅ 최종 CSV 저장
raw_csv_path = "C:/Users/mch2d/Desktop/LG-Aimers-Data-main/catboost_final_submission_raw.csv"
calibrated_csv_path = "C:/Users/mch2d/Desktop/LG-Aimers-Data-main/catboost_final_submission_calibrated.csv"

submission_raw.to_csv(raw_csv_path, index=False)
submission_calibrated.to_csv(calibrated_csv_path, index=False)

print(f"✅ 원본 모델 예측 결과 저장 완료: {raw_csv_path}")
print(f"✅ 캘리브레이션 적용 모델 예측 결과 저장 완료: {calibrated_csv_path}")
