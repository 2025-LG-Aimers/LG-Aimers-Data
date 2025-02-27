import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

# ✅ 한글 폰트 깨짐 해결 (Windows/Mac 대응)
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

# ✅ "부부 주 불임 원인" 및 "부부 부 불임 원인" Feature 추가
if "남성 주 불임 원인" in X.columns and "여성 주 불임 원인" in X.columns:
    X["부부 주 불임 원인2"] = X["남성 주 불임 원인"] | X["여성 주 불임 원인"]
if "남성 부 불임 원인" in X.columns and "여성 부 불임 원인" in X.columns:
    X["부부 부 불임 원인2"] = X["남성 부 불임 원인"] | X["여성 부 불임 원인"]

# ✅ "미세주입 성공률" Feature 추가
X["미세주입 성공률"] = X["미세주입에서 생성된 배아 수"] / (X["미세주입된 난자 수"] + 1e-5)

# ✅ "미세주입 이식율" Feature 추가
X["미세주입 이식율"] = X["미세주입 배아 이식 수"] / (X["미세주입에서 생성된 배아 수"] + 1e-5)

# -------------- 📌 결측치 처리 --------------
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[categorical_columns] = ordinal_encoder.fit_transform(X[categorical_columns])

numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
imputer = SimpleImputer(strategy='mean')
X[numeric_columns] = imputer.fit_transform(X[numeric_columns])

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

xgb_model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=500,
    verbose_eval=False
)

# -------------- 📌 Feature Importance 계산 --------------
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Gain": [xgb_model.get_score(importance_type="gain").get(f, 0) for f in X.columns],
    "Weight": [xgb_model.get_score(importance_type="weight").get(f, 0) for f in X.columns],
    "Cover": [xgb_model.get_score(importance_type="cover").get(f, 0) for f in X.columns]
})

# ✅ Feature Importance 정렬
importance_df = importance_df.sort_values(by="Gain", ascending=False)

# ✅ Gain 값 하위 30% 기준으로 Feature 선택
gain_threshold = np.percentile(importance_df["Gain"], 30)  # 하위 30% 기준값
low_gain_features = importance_df[importance_df["Gain"] <= gain_threshold]
high_gain_features = importance_df[importance_df["Gain"] > gain_threshold]

# ✅ Gain 값 분포 시각화 (막대 그래프)
plt.figure(figsize=(14, 6))
barplot = sns.barplot(x="Gain", y="Feature", data=importance_df, palette="coolwarm", hue=None, legend=False)
plt.xlabel("Gain 값")
plt.ylabel("Feature 이름")
plt.title("Feature Gain 값 분포")

# 🔴 하위 30% Feature에 빨간색 강조
for index, patch in enumerate(barplot.patches):
    if importance_df["Feature"].iloc[index] in low_gain_features["Feature"].values:
        patch.set_color("red")

# 하위 30% 기준선 추가
plt.axvline(gain_threshold, color="black", linestyle="--", label=f"하위 30% 기준 ({gain_threshold:.2f})")
plt.legend()
plt.grid(axis="x")
plt.show()

# ✅ Feature 간 상관관계 분석 (Gain 낮은 Feature들만 선택)
low_gain_feature_names = low_gain_features["Feature"].tolist()

# ✅ 상관관계 히트맵 시각화 (하위 30% Feature만)
if len(low_gain_feature_names) > 1:  # Feature가 2개 이상일 경우만 실행
    plt.figure(figsize=(12, 10))
    sns.heatmap(X[low_gain_feature_names].corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Gain 낮은 Feature 간 상관관계 히트맵")
    plt.show()

# ✅ Feature Importance 상세 출력
print("\n🔥 전체 Feature Importance (Gain, Weight, Cover 값 포함):")
print(importance_df.to_string(index=False))  # 전체 Feature Importance 테이블 출력

# ✅ 결과 출력
print(f"\n🔍 Gain 하위 30% (≤ {gain_threshold:.2f}) Feature 개수: {len(low_gain_features)}")
print(f"📌 해당 Feature 목록: {low_gain_features['Feature'].tolist()}")

print(f"\n🔥 Gain 상위 70% (> {gain_threshold:.2f}) Feature 개수: {len(high_gain_features)}")
print(f"📌 해당 Feature 목록: {high_gain_features['Feature'].tolist()}")
