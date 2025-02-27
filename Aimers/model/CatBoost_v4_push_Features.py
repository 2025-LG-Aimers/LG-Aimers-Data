import numpy as np
import pandas as pd
import pickle
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

# ✅ 최적 Feature 조합 (이미 선정된 Feature만 사용)
selected_features = [
    '시술 시기 코드', '시술 당시 나이', '특정 시술 유형', '배란 자극 여부', 
    '배란 유도 유형', '단일 배아 이식 여부', '불명확 불임 원인', 
    '불임 원인 - 난관 질환', '불임 원인 - 남성 요인', '불임 원인 - 배란 장애',
    '불임 원인 - 자궁내막증', '배아 생성 주요 이유', '클리닉 내 총 시술 횟수', 'IVF 시술 횟수', 
    'DI 시술 횟수', 'IVF 임신 횟수', 'IVF 출산 횟수', '총 생성 배아 수', '미세주입에서 생성된 배아 수', 
    '이식된 배아 수', '미세주입 배아 이식 수', '저장된 배아 수', '미세주입 후 저장된 배아 수', 
    '해동된 배아 수', '수집된 신선 난자 수', '파트너 정자와 혼합된 난자 수', '기증자 정자와 혼합된 난자 수', 
    '난자 출처', '난자 기증자 나이', 
    '정자 기증자 나이', '동결 배아 사용 여부', '배아 이식 경과일'
]

# ✅ 데이터 로드
train_path = "C:/Users/mch2d/Desktop/LG-Aimers-Data-main/train.csv"
test_path = "C:/Users/mch2d/Desktop/LG-Aimers-Data-main/test.csv"
param_path = "C:/Users/mch2d/Desktop/LG-Aimers-Data-main/Params/best_catboost_params_v3.pkl"

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

X = df_train[selected_features].copy()
y = df_train["임신 성공 여부"].copy()

# ✅ 1️⃣ 편향된 컬럼 제거 (특정 값이 95% 이상인 컬럼 삭제)
threshold = 0.95
biased_cols = [col for col in selected_features if X[col].value_counts(normalize=True).max() >= threshold]
X.drop(columns=biased_cols, inplace=True, errors="ignore")
df_test.drop(columns=biased_cols, inplace=True, errors="ignore")

# ✅ 2️⃣ 결측치 처리
missing_ratio = (X.isnull().sum() / len(X)) * 100

# ✅ (1) 80% 이상 결측 → 컬럼 삭제
to_drop = missing_ratio[missing_ratio >= 80].index.tolist()
X.drop(columns=to_drop, inplace=True, errors="ignore")
df_test.drop(columns=to_drop, inplace=True, errors="ignore")

# ✅ (2) 수치형 변수 → 중앙값(median)으로 대체
num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
X[num_features] = X[num_features].apply(lambda col: col.fillna(col.median()))
df_test[num_features] = df_test[num_features].apply(lambda col: col.fillna(col.median()))

# ✅ (3) 범주형 변수 → "missing" 값으로 대체
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
for col in categorical_features:
    X[col] = X[col].fillna("missing").astype(str)
    df_test[col] = df_test[col].fillna("missing").astype(str)

# ✅ 테스트 데이터 컬럼 맞추기
df_test = df_test.reindex(columns=X.columns, fill_value=0)

# ✅ 저장된 최적의 하이퍼파라미터 로드
with open(param_path, "rb") as f:
    best_config = pickle.load(f)

# ✅ 'verbose' 키 제거 (중복 방지)
best_config.pop("verbose", None)

# ✅ GPU 사용 시 평가 지표 변경
if best_config.get("task_type") == "GPU":
    best_config["eval_metric"] = "Logloss"

# ✅ K-Fold 설정 (5-Fold)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ✅ 각 Fold의 점수 저장
auc_scores = []
accuracy_scores = []

print("\n🚀 K-Fold 교차 검증 시작...\n")

# ✅ K-Fold 학습 & 평가
for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # ✅ 모델 학습
    model = CatBoostClassifier(**best_config, verbose=0)
    model.fit(X_train, y_train, cat_features=categorical_features)

    # ✅ 예측
    valid_preds_proba = model.predict_proba(X_valid)[:, 1]  # ROC-AUC용 확률 예측값
    valid_preds = model.predict(X_valid)  # Accuracy용 예측값

    # ✅ 성능 평가
    auc = roc_auc_score(y_valid, valid_preds_proba)
    acc = accuracy_score(y_valid, valid_preds)

    auc_scores.append(auc)
    accuracy_scores.append(acc)

    print(f"🔹 Fold {fold + 1}: AUC = {auc:.5f}, Accuracy = {acc:.5f}")

# ✅ 평균 점수 출력
mean_auc = np.mean(auc_scores)
mean_acc = np.mean(accuracy_scores)

print("\n🎯 최종 K-Fold 평균 점수:")
print(f"✅ 평균 ROC-AUC Score: {mean_auc:.10f}")
print(f"✅ 평균 Accuracy Score: {mean_acc:.10f}")
