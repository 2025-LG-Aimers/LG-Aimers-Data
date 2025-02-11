import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb  # ✅ XGBoost 사용
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

# ✅ Matplotlib 한글 폰트 설정 (Windows: 'Malgun Gothic', Mac: 'AppleGothic')
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# ✅ 데이터 로딩
train = pd.read_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/train.csv', encoding='utf-8')

# ✅ 타겟 변수 분리
X = train.drop(columns=['임신 성공 여부'])
y = train['임신 성공 여부']

# ✅ 1. 수치형 & 범주형 데이터 분리
numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()

# ✅ 2. 수치형 데이터 결측치 처리 (평균 대체)
num_imputer = SimpleImputer(strategy='mean')
X[numeric_columns] = num_imputer.fit_transform(X[numeric_columns])

# ✅ 3. 범주형 데이터 결측치 처리 ("missing"으로 대체)
X[categorical_columns] = X[categorical_columns].fillna("missing")

# ✅ 4. 범주형 데이터 Ordinal Encoding
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[categorical_columns] = ordinal_encoder.fit_transform(X[categorical_columns])

# ✅ Train-Test Split (Feature Importance 계산용)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ✅ 5. XGBoost 모델 학습 (Feature Importance 구하기 위함)
params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 5,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.83,
    "random_state": 42
}

dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)

xgb_model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=500,
    evals=[(dtrain, "train"), (dvalid, "valid")],
    verbose_eval=False
)

# ✅ 6. Feature Importance (Gain 기준) 저장
importances_gain = xgb_model.get_score(importance_type='gain')
importance_df = pd.DataFrame({'Feature': list(importances_gain.keys()), 'Gain': list(importances_gain.values())})
importance_df = importance_df.sort_values(by='Gain', ascending=False)

# ✅ Feature Importance CSV 저장
importance_csv_path = 'C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/feature_importance.csv'
importance_df.to_csv(importance_csv_path, index=False)
print(f"✅ Feature Importance 저장 완료: {importance_csv_path}")

# ✅ 7. Gain이 높은 상위 10개 Feature 선택
top_features = importance_df.head(20)['Feature'].tolist()

# ✅ 선택된 Feature들의 상관행렬 계산
X_selected = X_train[top_features]
corr_matrix = X_selected.corr()

# ✅ 8. 상삼각 행렬(Upper Triangle) 제거 (반만 표시)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# 🔥 Heatmap 시각화
plt.figure(figsize=(12, 10))  # ✅ 그래프 크기 조정 (너비=12, 높이=10)
sns.heatmap(
    corr_matrix, mask=mask, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5,
    annot_kws={"size": 8}  # ✅ 숫자 크기 조절 (기본: 10~12, 줄이려면 8~9 추천)
)
plt.xticks(rotation=45, ha='right', fontsize=10)  # ✅ X축 글자 45도 회전 + 오른쪽 정렬
plt.yticks(fontsize=10)  # ✅ Y축 폰트 크기 조절
plt.title("Gain 높은 Feature 간 상관관계 Heatmap", fontsize=14)  # ✅ 제목 크기 조절
plt.subplots_adjust(bottom=0.3)  # ✅ X축 글자가 너무 아래로 내려가면 여백 추가
plt.show()

