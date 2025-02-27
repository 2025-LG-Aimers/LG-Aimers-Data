import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

# ✅ 한글 폰트 깨짐 방지 (Windows / Mac 대응)
import platform
if platform.system() == "Windows":
    plt.rc("font", family="Malgun Gothic")
else:
    plt.rc("font", family="AppleGothic")

plt.rcParams["axes.unicode_minus"] = False  # 음수 기호 깨짐 방지

# -------------- 📌 데이터 로딩 --------------
train = pd.read_csv('C:/Users/mch2d/Desktop/LG-Aimers-Data-main/train.csv').drop(columns=['ID'])

# -------------- 📌 타겟 변수 분리 --------------
X = train.drop(columns=['임신 성공 여부'])
y = train['임신 성공 여부']

# ✅ "배아 사용률" Feature 추가
if "이식된 배아 수" in X.columns and "저장된 배아 수" in X.columns:
    X["배아 사용률"] = X["이식된 배아 수"] / (X["이식된 배아 수"] + X["저장된 배아 수"] + 1e-5)
    print("✅ '배아 사용률' Feature 추가 완료!")

# ✅ "부부 주 불임 원인" 및 "부부 부 불임 원인" Feature 추가
if "남성 주 불임 원인" in X.columns and "여성 주 불임 원인" in X.columns:
    X["부부 주 불임 원인2"] = X["남성 주 불임 원인"] | X["여성 주 불임 원인"]

if "남성 부 불임 원인" in X.columns and "여성 부 불임 원인" in X.columns:
    X["부부 부 불임 원인2"] = X["남성 부 불임 원인"] | X["여성 부 불임 원인"]

# ✅ "미세주입 성공률" & "미세주입 이식율" Feature 추가
X["미세주입 성공률"] = X["미세주입에서 생성된 배아 수"] / (X["미세주입된 난자 수"] + 1e-5)
X["미세주입 이식율"] = X["미세주입 배아 이식 수"] / (X["미세주입에서 생성된 배아 수"] + 1e-5)

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

# ✅ IVF & DI 비율 Feature 추가
X["IVF_임신_비율"] = X["IVF 임신 횟수"] / (X["총 임신 횟수"] + 1e-5)
X["DI_임신_비율"] = X["DI 임신 횟수"] / (X["총 임신 횟수"] + 1e-5)
X["IVF_출산_비율"] = X["IVF 출산 횟수"] / (X["총 출산 횟수"] + 1e-5)
X["DI_출산_비율"] = X["DI 출산 횟수"] / (X["총 출산 횟수"] + 1e-5)

print("✅ IVF & DI 관련 파생 변수 추가 완료!")

# ✅ 불필요한 컬럼 제거
column_to_remove = [
    "남성 주 불임 원인", "여성 주 불임 원인", "부부 주 불임 원인",
    "남성 부 불임 원인", "여성 부 불임 원인", "부부 부 불임 원인",
    "동결 배아 사용 여부", "미세주입된 난자 수", "혼합된 난자 수"
]
X.drop(columns=column_to_remove, inplace=True, errors='ignore')

# -------------- 📌 결측치 처리 --------------
# ✅ 범주형 변수 Ordinal Encoding
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[categorical_columns] = ordinal_encoder.fit_transform(X[categorical_columns])

# ✅ 수치형 변수 결측치 평균 대체
numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
imputer_numeric = SimpleImputer(strategy='mean')
X[numeric_columns] = imputer_numeric.fit_transform(X[numeric_columns])

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
dtrain = xgb.DMatrix(X, label=y)
xgb_model = xgb.train(params=params, dtrain=dtrain, num_boost_round=500, verbose_eval=False)

# -------------- 📌 Feature Importance 계산 --------------
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Weight": [xgb_model.get_score(importance_type="weight").get(f, 0) for f in X.columns],
    "Gain": [xgb_model.get_score(importance_type="gain").get(f, 0) for f in X.columns],
    "Cover": [xgb_model.get_score(importance_type="cover").get(f, 0) for f in X.columns]
}).sort_values(by="Gain", ascending=False)

# ✅ Feature Importance 전체 출력
print("\n🔥 Feature Importance (전체 출력):")
print(importance_df.to_string(index=False))

# ✅ IVF & DI 관련 Feature 간 상관관계 분석
ivf_di_features = [
    "IVF_임신_비율", "DI_임신_비율", "IVF_출산_비율", "DI_출산_비율",
    "IVF 임신 횟수", "DI 임신 횟수", "총 임신 횟수",
    "IVF 출산 횟수", "DI 출산 횟수", "총 출산 횟수",
    "배아 사용률", "미세주입 성공률"
]
ivf_di_corr = X[ivf_di_features].corr()

# ✅ IVF & DI 관련 Feature 상관관계 출력
print("\n🔍 IVF & DI 관련 Feature 간 상관관계:")
print(ivf_di_corr.to_string())

# ✅ IVF & DI 관련 Feature Heatmap 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(ivf_di_corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("IVF & DI 관련 Feature 간 상관관계 Heatmap")
plt.show()
