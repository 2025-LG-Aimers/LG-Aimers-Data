import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score

# -------------- 📌 데이터 로딩 --------------
train = pd.read_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/train.csv').drop(columns=['ID'])
test = pd.read_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/test.csv').drop(columns=['ID'])

# -------------- 📌 타겟 변수 분리 --------------
X = train.drop(columns=['임신 성공 여부'])  # 입력 데이터 (Feature)
y = train['임신 성공 여부']  # 타겟 변수 (Label)

# ✅ 편향된 컬럼 제거
columns_to_remove = ["신선 배아 사용 여부", "미세주입된 난자 수", "IVF 시술 횟수", 
                     "IVF 임신 횟수", "총 출산 횟수", "정자 출처"]

# ✅ 95% 이상 편향된 컬럼 제거
threshold = 0.95
biased_columns = [col for col in X.columns if X[col].value_counts(normalize=True).max() >= threshold]
columns_to_remove.extend(biased_columns)

X.drop(columns=columns_to_remove, inplace=True, errors='ignore')
test.drop(columns=columns_to_remove, inplace=True, errors='ignore')

print(f"✅ 제거된 편향된 컬럼: {biased_columns}")

# -------------- 📌 결측치 처리 --------------
missing_percentage = (X.isnull().sum() / len(X)) * 100

# ✅ 80% 이상 결측치 컬럼 삭제
high_missing_columns = missing_percentage[missing_percentage >= 80].index.tolist()
X.drop(columns=high_missing_columns, inplace=True, errors='ignore')
test.drop(columns=high_missing_columns, inplace=True, errors='ignore')

print(f"✅ 제거된 결측치 높은 컬럼(80% 이상): {high_missing_columns}")

# ✅ 15% ~ 30% 결측치 평균값으로 대체
mid_missing_columns = missing_percentage[(missing_percentage >= 15) & (missing_percentage < 30)].index.tolist()
imputer = SimpleImputer(strategy='mean')
X[mid_missing_columns] = imputer.fit_transform(X[mid_missing_columns])
test[mid_missing_columns] = imputer.transform(test[mid_missing_columns])

# ✅ 범주형 컬럼 인코딩 (Ordinal Encoding)
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
print(f"📌 감지된 범주형 컬럼: {categorical_columns}")

# 🔥 테스트 데이터에서도 동일한 컬럼 유지
test = test[X.columns]

# 범주형 데이터를 문자열로 변환
for col in categorical_columns:
    X[col] = X[col].astype(str)
    test[col] = test[col].astype(str)

# ✅ OrdinalEncoder 설정 및 변환
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[categorical_columns] = ordinal_encoder.fit_transform(X[categorical_columns])
test[categorical_columns] = ordinal_encoder.transform(test[categorical_columns])

# ✅ 수치형 컬럼 NaN 채우기
numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
X[numeric_columns] = X[numeric_columns].fillna(0)
test[numeric_columns] = test[numeric_columns].fillna(0)

# -------------- 📌 Focal Loss 적용 --------------
def focal_loss(predt, dtrain, gamma=0.705350567650623, alpha=0.14010485785305138):
    """ Focal Loss 커스텀 함수 (Gradient & Hessian 계산) """
    y = dtrain.get_label()
    p = 1 / (1 + np.exp(-predt))  # 시그모이드 변환
    grad = alpha * (1 - p) ** gamma * (p - y)
    hess = alpha * (1 - p) ** gamma * p * (1 - p) * (1 + gamma * (1 - p) * (y - p))
    return grad, hess

# -------------- 📌 XGBoost 교차 검증 --------------
params = {
    "objective": "binary:logistic",  # 기본적으로 사용하지만 무시됨 (커스텀 로스 적용할 것이므로)
    "eval_metric": "logloss",  # 평가 지표는 logloss 유지
    "max_depth": 4,
    "learning_rate":  0.04405038339177713,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}

# ✅ Stratified K-Fold 설정 (5-Fold)
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# ✅ XGBoost 교차 검증 진행 및 성능 평가
auc_scores = []
accuracy_scores = []

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
    print(f"🔹 Fold {fold} 시작...")

    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    # ✅ XGBoost 학습 (Focal Loss 적용)
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=1000,
        obj=lambda predt, dtrain: focal_loss(predt, dtrain, gamma=2.0, alpha=0.25),  # ✅ Focal Loss 적용
        evals=[(dvalid, "valid")],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    # ✅ 검증 데이터 예측 및 성능 측정
    valid_pred_proba = model.predict(dvalid)
    valid_pred_class = (valid_pred_proba > 0.5).astype(int)

    auc = roc_auc_score(y_valid, valid_pred_proba)
    accuracy = accuracy_score(y_valid, valid_pred_class)

    auc_scores.append(auc)
    accuracy_scores.append(accuracy)

    print(f"✅ Fold {fold} - ROC-AUC: {auc:.6f}, Accuracy: {accuracy:.6f}")

# ✅ Cross Validation 평균 성능 출력
print("\n🔥 Cross Validation Results 🔥")
print(f"✅ Mean ROC-AUC Score: {np.mean(auc_scores):.6f} ± {np.std(auc_scores):.6f}")
print(f"✅ Mean Accuracy Score: {np.mean(accuracy_scores):.6f} ± {np.std(accuracy_scores):.6f}")
