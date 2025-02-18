import optuna
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

# -------------- 📌 데이터 로딩 --------------
train = pd.read_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/train.csv').drop(columns=['ID'])
test = pd.read_csv('C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/test.csv').drop(columns=['ID'])

# -------------- 📌 타겟 변수 분리 --------------
X = train.drop(columns=['임신 성공 여부'])
y = train['임신 성공 여부']

# ✅ 편향된 컬럼 제거
columns_to_remove = ["신선 배아 사용 여부", "미세주입된 난자 수", "IVF 시술 횟수", 
                     "IVF 임신 횟수", "총 출산 횟수", "정자 출처"]

# ✅ 95% 이상 편향된 컬럼 제거
threshold = 0.95
biased_columns = [col for col in X.columns if X[col].value_counts(normalize=True).max() >= threshold]
columns_to_remove.extend(biased_columns)

X.drop(columns=columns_to_remove, inplace=True, errors='ignore')
test.drop(columns=columns_to_remove, inplace=True, errors='ignore')

# -------------- 📌 결측치 처리 --------------
missing_percentage = (X.isnull().sum() / len(X)) * 100

# ✅ 80% 이상 결측치 컬럼 삭제
high_missing_columns = missing_percentage[missing_percentage >= 80].index.tolist()
X.drop(columns=high_missing_columns, inplace=True, errors='ignore')
test.drop(columns=high_missing_columns, inplace=True, errors='ignore')

# ✅ 15% ~ 30% 결측치 평균값으로 대체
mid_missing_columns = missing_percentage[(missing_percentage >= 15) & (missing_percentage < 30)].index.tolist()
imputer = SimpleImputer(strategy='mean')
X[mid_missing_columns] = imputer.fit_transform(X[mid_missing_columns])
test[mid_missing_columns] = imputer.transform(test[mid_missing_columns])

# ✅ 범주형 컬럼 인코딩 (Ordinal Encoding)
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()

# 🔥 테스트 데이터에서도 동일한 컬럼 유지
test = test[X.columns]

# 범주형 데이터를 문자열로 변환 후 인코딩
for col in categorical_columns:
    X[col] = X[col].astype(str)
    test[col] = test[col].astype(str)

ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[categorical_columns] = ordinal_encoder.fit_transform(X[categorical_columns])
test[categorical_columns] = ordinal_encoder.transform(test[categorical_columns])

# ✅ 수치형 컬럼 NaN 채우기
numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
X[numeric_columns] = X[numeric_columns].fillna(0)
test[numeric_columns] = test[numeric_columns].fillna(0)

# -------------- 📌 Focal Loss 정의 --------------
def focal_loss(predt, dtrain, gamma, alpha):
    """ Focal Loss 커스텀 함수 (Gradient & Hessian 계산) """
    y = dtrain.get_label()
    p = 1 / (1 + np.exp(-predt))  # 시그모이드 변환
    grad = alpha * (1 - p) ** gamma * (p - y)
    hess = alpha * (1 - p) ** gamma * p * (1 - p) * (1 + gamma * (1 - p) * (y - p))
    return grad, hess

# -------------- 📌 Optuna 최적화 함수 --------------
def objective(trial):
    """ Optuna 하이퍼파라미터 최적화 함수 """
    # ✅ 최적화할 하이퍼파라미터 범위 설정
    max_depth = trial.suggest_int("max_depth", 3, 10)
    learning_rate = trial.suggest_float("learning_rate", 0.005, 0.05, log=True)
    gamma = trial.suggest_float("gamma", 0.5, 3.0)
    alpha = trial.suggest_float("alpha", 0.1, 0.5)

    # ✅ XGBoost 파라미터 설정
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    }

    # ✅ 5-Fold Stratified K-Fold 설정
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []

    # ✅ Cross Validation 수행
    for train_idx, valid_idx in skf.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)

        # ✅ XGBoost 학습 (Focal Loss 적용)
        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=500,
            obj=lambda predt, dtrain: focal_loss(predt, dtrain, gamma, alpha),
            evals=[(dvalid, "valid")],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        # ✅ 검증 데이터 예측 및 성능 측정
        valid_pred_proba = model.predict(dvalid)
        auc = roc_auc_score(y_valid, valid_pred_proba)
        auc_scores.append(auc)

    return np.mean(auc_scores)  # ✅ 평균 AUC 반환 (최대화 목표)

# -------------- 📌 Optuna 실행 --------------
study = optuna.create_study(direction="maximize")  # AUC 최대화
study.optimize(objective, n_trials=50)  # 50번 최적화 시도

# ✅ 최적의 하이퍼파라미터 출력
print("\n🔥 최적의 하이퍼파라미터 🔥")
print(study.best_params)
print(f"✅ 최고 AUC Score: {study.best_value:.6f}")
