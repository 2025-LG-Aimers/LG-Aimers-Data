import numpy as np
import pandas as pd
import pickle
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

# ✅ 데이터 로드
train_path = "C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/train.csv"
test_path = "C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/test.csv"
submission_path = "C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/sample_submission.csv"
param_path = "C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/Params/best_catboost_params.pkl"

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
    "신선 배아 사용 여부", "미세주입된 난자 수", "IVF 시술 횟수",
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

# 테스트 데이터 컬럼 맞추기
df_test = df_test.reindex(columns=X.columns, fill_value=0)

# ✅ 범주형 변수 결측치는 최빈값(Mode)으로 대체
for col in categorical_features:
    most_frequent = X[col].mode()[0]  # 최빈값 찾기
    X[col] = X[col].fillna(most_frequent).astype(str)
    df_test[col] = df_test[col].fillna(most_frequent).astype(str)

# ✅ 저장된 최적의 파라미터 로드
try:
    with open(param_path, "rb") as f:
        best_config = pickle.load(f)
    print(f"✅ 최적의 하이퍼파라미터 로드 완료: {best_config}")
except Exception as e:
    print(f"❌ 하이퍼파라미터 파일 로드 실패: {e}")
    exit()

# ✅ K-Fold 교차 검증 적용 (StratifiedKFold)
n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# ✅ K-Fold 학습 및 예측 저장
test_preds = np.zeros(len(df_test))  # 테스트 데이터 예측값 저장
calibrated_preds = np.zeros(len(df_test))  # 캘리브레이션된 예측값 저장
auc_scores = []  # AUC 점수 저장
calibrated_auc_scores = []  # 캘리브레이션된 AUC 점수 저장

for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"🚀 Fold {fold + 1} 학습 시작...")
    
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # 모델 학습
    # 'verbose' 키 제거 후 전달
    best_config.pop("verbose", None)  # verbose가 존재하면 제거
    best_model = CatBoostClassifier(**best_config, verbose=0)
    best_model.fit(X_train, y_train, cat_features=categorical_features)

    # ✅ 검증 데이터 예측
    valid_preds = best_model.predict_proba(X_valid)[:, 1]
    fold_auc = roc_auc_score(y_valid, valid_preds)
    auc_scores.append(fold_auc)  # AUC 저장

    # ✅ 후처리 캘리브레이션 (Platt Scaling) 적용
    cal_model = CalibratedClassifierCV(best_model, method="sigmoid", cv=5)
    cal_model.fit(X_valid, y_valid)

    # ✅ 검증 데이터 캘리브레이션 후 예측
    valid_calibrated_preds = cal_model.predict_proba(X_valid)[:, 1]
    fold_calibrated_auc = roc_auc_score(y_valid, valid_calibrated_preds)
    calibrated_auc_scores.append(fold_calibrated_auc)  # 캘리브레이션된 AUC 저장

    # ✅ 원본 모델 예측
    fold_pred = best_model.predict_proba(df_test)[:, 1]
    test_preds += fold_pred / n_splits  # K-Fold 평균 앙상블

    # ✅ 캘리브레이션된 모델 예측
    fold_calibrated_pred = cal_model.predict_proba(df_test)[:, 1]
    calibrated_preds += fold_calibrated_pred / n_splits  # K-Fold 평균 앙상블

    print(f"✅ Fold {fold + 1} 완료! AUC: {fold_auc:.10f} | 캘리브레이션 후 AUC: {fold_calibrated_auc:.10f}")

# ✅ 평균 AUC 출력
mean_auc = np.mean(auc_scores)
mean_calibrated_auc = np.mean(calibrated_auc_scores)

print(f"\n🏆 K-Fold 평균 AUC: {mean_auc:.10f}")
print(f"🎯 캘리브레이션 적용 후 평균 AUC: {mean_calibrated_auc:.10f}")

# ✅ sample_submission 생성 (캘리브레이션 적용 O / X 비교)
submission_raw = pd.DataFrame({"ID": test_ids, "probability": test_preds})  # 원본 예측값
submission_calibrated = pd.DataFrame({"ID": test_ids, "probability": calibrated_preds})  # 캘리브레이션된 예측값

# ✅ 최종 CSV 저장
raw_csv_path = "C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/catboost_final_submission_kfold_raw.csv"
calibrated_csv_path = "C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/catboost_final_submission_kfold_calibrated.csv"

submission_raw.to_csv(raw_csv_path, index=False)
submission_calibrated.to_csv(calibrated_csv_path, index=False)

print(f"✅ 원본 모델 예측 결과 저장 완료: {raw_csv_path}")
print(f"✅ 캘리브레이션 적용 모델 예측 결과 저장 완료: {calibrated_csv_path}")
