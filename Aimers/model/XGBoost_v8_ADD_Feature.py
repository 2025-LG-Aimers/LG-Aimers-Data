import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score

# -------------- 📌 검증 데이터 예측 및 평가


# -------------- 📌 데이터 로딩 --------------
train = pd.read_csv('C:/Users/mch2d/Desktop/LG-Aimers-Data-main/train.csv').drop(columns=['ID'])
test = pd.read_csv('C:/Users/mch2d/Desktop/LG-Aimers-Data-main/test.csv').drop(columns=['ID'])

# -------------- 📌 타겟 변수 분리 --------------
y = train['임신 성공 여부']  # 타겟 변수 (Label)
X = train.drop(columns=['임신 성공 여부'])  # 입력 데이터 (Feature)

# -------------- 📌 Feature Engineering --------------
# ✅ "배아 사용률" Feature 추가
if "이식된 배아 수" in X.columns and "저장된 배아 수" in X.columns:
    X["배아 사용률"] = X["이식된 배아 수"] / (X["이식된 배아 수"] + X["저장된 배아 수"] + 1e-5)
    test["배아 사용률"] = test["이식된 배아 수"] / (test["이식된 배아 수"] + test["저장된 배아 수"] + 1e-5)

# ✅ "부부 주 불임 원인" 및 "부부 부 불임 원인" Feature 추가
if "남성 주 불임 원인" in X.columns and "여성 주 불임 원인" in X.columns:
    X["부부 주 불임 원인2"] = X["남성 주 불임 원인"] | X["여성 주 불임 원인"]
    test["부부 주 불임 원인2"] = test["남성 주 불임 원인"] | test["여성 주 불임 원인"]

if "남성 부 불임 원인" in X.columns and "여성 부 불임 원인" in X.columns:
    X["부부 부 불임 원인2"] = X["남성 부 불임 원인"] | X["여성 부 불임 원인"]
    test["부부 부 불임 원인2"] = test["남성 부 불임 원인"] | test["여성 부 불임 원인"]

# ✅ "미세주입 성공률" & "미세주입 이식율" Feature 추가
X["미세주입 성공률"] = X["미세주입에서 생성된 배아 수"] / (X["미세주입된 난자 수"] + 1e-5)
test["미세주입 성공률"] = test["미세주입에서 생성된 배아 수"] / (test["미세주입된 난자 수"] + 1e-5)

X["미세주입 이식율"] = X["미세주입 배아 이식 수"] / (X["미세주입에서 생성된 배아 수"] + 1e-5)
test["미세주입 이식율"] = test["미세주입 배아 이식 수"] / (test["미세주입에서 생성된 배아 수"] + 1e-5)

# ✅ 숫자 변환 함수 정의 (횟수 변환)
def convert_to_number(value):
    if isinstance(value, str):
        value = value.replace("회", "").strip()
        if "이상" in value:
            return 6  
        return int(value)
    return value

# ✅ "IVF 임신 횟수", "DI 임신 횟수" 등 숫자로 변환
for col in ["IVF 임신 횟수", "DI 임신 횟수", "총 임신 횟수", "IVF 출산 횟수", "DI 출산 횟수", "총 출산 횟수"]:
    X[col] = X[col].apply(convert_to_number)
    test[col] = test[col].apply(convert_to_number)

# ✅ IVF & DI 비율 Feature 추가
X["IVF_임신_비율"] = X["IVF 임신 횟수"] / (X["총 임신 횟수"] + 1e-5)
X["DI_임신_비율"] = X["DI 임신 횟수"] / (X["총 임신 횟수"] + 1e-5)
X["IVF_출산_비율"] = X["IVF 출산 횟수"] / (X["총 출산 횟수"] + 1e-5)
X["DI_출산_비율"] = X["DI 출산 횟수"] / (X["총 출산 횟수"] + 1e-5)

test["IVF_임신_비율"] = test["IVF 임신 횟수"] / (test["총 임신 횟수"] + 1e-5)
test["DI_임신_비율"] = test["DI 임신 횟수"] / (test["총 임신 횟수"] + 1e-5)
test["IVF_출산_비율"] = test["IVF 출산 횟수"] / (test["총 출산 횟수"] + 1e-5)
test["DI_출산_비율"] = test["DI 출산 횟수"] / (test["총 출산 횟수"] + 1e-5)

# -------------- 📌 결측치 처리 --------------
missing_percentage = (X.isnull().sum() / len(X)) * 100

# ✅ 1. 결측치 비율이 50% 이상인 컬럼 삭제
high_missing_columns = missing_percentage[missing_percentage >= 50].index.tolist()
X.drop(columns=high_missing_columns, inplace=True, errors='ignore')
test.drop(columns=high_missing_columns, inplace=True, errors='ignore')

# ✅ 2. 결측치 비율이 15% ~ 30%인 컬럼 처리
mid_missing_columns = missing_percentage[(missing_percentage >= 15) & (missing_percentage < 30)].index.tolist()
imputer_numeric = SimpleImputer(strategy='mean')  # 평균값 대체
X[mid_missing_columns] = imputer_numeric.fit_transform(X[mid_missing_columns])
test[mid_missing_columns] = imputer_numeric.transform(test[mid_missing_columns])

# ✅ 3. 범주형 컬럼 최빈값 대체 + 인코딩
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
imputer_categorical = SimpleImputer(strategy='most_frequent')
X[categorical_columns] = imputer_categorical.fit_transform(X[categorical_columns])
test[categorical_columns] = imputer_categorical.transform(test[categorical_columns])

ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[categorical_columns] = ordinal_encoder.fit_transform(X[categorical_columns])
test[categorical_columns] = ordinal_encoder.transform(test[categorical_columns])

# ✅ 최종 남아있는 컬럼 확인
remaining_columns = X.columns.tolist()
print(f"\n✅ 최종 남아있는 컬럼 개수: {len(remaining_columns)}")
print(f"✅ 최종 남아있는 컬럼 목록: {remaining_columns}")


# -------------- 📌 XGBoost 모델 학습 --------------
params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 6,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}

# -------------- 📌 Gain 하위 40% Feature 제거 --------------
dtrain_temp = xgb.DMatrix(X, label=y)

xgb_model_temp = xgb.train(
    params=params,  # ✅ params가 이제 정상적으로 정의됨
    dtrain=dtrain_temp,
    num_boost_round=500,
    verbose_eval=False
)

feature_importances = xgb_model_temp.get_score(importance_type="gain")
importance_df = pd.DataFrame({
    "Feature": list(feature_importances.keys()),
    "Gain": list(feature_importances.values())
}).sort_values(by="Gain", ascending=False)

# ✅ 하위 40% Feature 탐색
gain_threshold = np.percentile(importance_df["Gain"], 40)
low_gain_features = importance_df[importance_df["Gain"] <= gain_threshold]["Feature"].tolist()

# ✅ 하위 40% Feature 제거
X.drop(columns=low_gain_features, inplace=True, errors='ignore')
test.drop(columns=low_gain_features, inplace=True, errors='ignore')

print(f"\n🚀 Gain 하위 40% (≤ {gain_threshold:.2f}) Feature 개수: {len(low_gain_features)}")
print(f"📌 제거된 Feature 목록: {low_gain_features}")

# ✅ 컬럼 삭제 적용 (train & test)
columns_to_remove = [
    "신선 배아 사용 여부", "미세주입된 난자 수", "혼합된 난자 수",
    "총 시술 횟수", "총 임신 횟수", "총 출산 횟수", "정자 출처",
    "IVF 임신 횟수", "IVF 출산 횟수", "DI 출산 횟수",
    "부부 부 불임 원인", "남성 부 불임 원인",
    "불임 원인 - 난관 질환", "불임 원인 - 남성 요인", "불임 원인 - 여성 요인", "불임 원인 - 자궁경부 문제", "불임 원인 - 정자 면역학적 요인",
    "시술 유형"
]
X.drop(columns=columns_to_remove, inplace=True, errors='ignore')
test.drop(columns=columns_to_remove, inplace=True, errors='ignore')


# ✅ 최종 남아있는 컬럼 확인
remaining_columns = X.columns.tolist()
print(f"\n✅ Gain 하위 40% 제거 후 남아있는 컬럼 개수: {len(remaining_columns)}")
print(f"✅ Gain 하위 40% 제거 후 남아있는 컬럼 목록: {remaining_columns}")

# -------------- 📌 🔥 Train-Test Split (8:2) --------------
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ✅ XGBoost 전용 DMatrix 생성
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)
dtest = xgb.DMatrix(test)

# -------------- 📌 XGBoost 모델 학습 --------------
params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 6,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}

# ✅ XGBoost 전용 DMatrix 생성
dtrain = xgb.DMatrix(X, label=y)  # ✅ 전체 데이터를 학습에 사용

# ✅ 모델 학습 (조기 종료 없음)
xgb_model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=500,  # ✅ Early Stopping 제거했으므로 500번 학습
    verbose_eval=True
)

# # ✅ 조기 종료(Early Stopping) 추가
# xgb_model = xgb.train(
#     params=params,
#     dtrain=dtrain,
#     num_boost_round=500,
#     evals=[(dtrain, "train"), (dvalid, "valid")],
#     early_stopping_rounds=50,  # 50라운드 동안 개선 없으면 멈춤
#     verbose_eval=50
# )

# # -------------- 📌 검증 데이터 예측 및 평가 --------------
# valid_pred_proba = xgb_model.predict(dvalid)  # 확률값 예측
# valid_pred_labels = (valid_pred_proba >= 0.5).astype(int)  # 0.5 기준으로 이진 분류

# # ✅ LogLoss Score
# logloss_score = log_loss(y_valid, valid_pred_proba)

# # ✅ ROC-AUC Score
# roc_auc = roc_auc_score(y_valid, valid_pred_proba)

# # ✅ Accuracy Score
# accuracy = accuracy_score(y_valid, valid_pred_labels)

# # -------------- 📌 평가 결과 출력 --------------
# print(f"\n✅ 검증 데이터 LogLoss Score: {logloss_score:.5f}")
# print(f"✅ 검증 데이터 ROC-AUC Score: {roc_auc:.5f}")
# print(f"✅ 검증 데이터 Accuracy Score: {accuracy:.5f}")


# -------------- 📌 테스트 데이터 예측 및 결과 저장 --------------
dtest = xgb.DMatrix(test)
test_pred_proba = xgb_model.predict(dtest)

sample_submission = pd.read_csv('C:/Users/mch2d/Desktop/LG-Aimers-Data-main/sample_submission.csv')
sample_submission['probability'] = test_pred_proba
sample_submission.to_csv('C:/Users/mch2d/Desktop/LG-Aimers-Data-main/XGBoost_full_Original_HighScore_v12.csv', index=False)

print("\n✅ XGBoost 모델 학습 & 예측 완료, 결과 저장됨.")
