"""
 * Project : LG Aimers - 20개 Feature 기반 CatBoost 학습 (기본 파라미터)
 * Program Purpose and Features :
 * - 원본 Feature vs Polynomial Feature 비교
 * - 기본 CatBoost 설정으로 학습 진행
 * - 5-Fold 교차 검증을 통해 성능 비교 (ROC-AUC, Accuracy)
 * Author : SP Hong
 * First Write Date : 2025.02.25
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

# ✅ **1. 사용하려는 20개 Feature 선택**
selected_features = [
    '이식된 배아 수', '시술 당시 나이', '배아 이식 경과일', '저장된 배아 수', '총 생성 배아 수',
    '특정 시술 유형', '동결 배아 사용 여부', '미세주입에서 생성된 배아 수',
    'IVF 시술 횟수', 'DI 시술 횟수', 'IVF 임신 횟수', '시술 시기 코드',
    '배아 생성 주요 이유', '클리닉 내 총 시술 횟수', 'IVF 출산 횟수',
    '해동된 배아 수', '파트너 정자와 혼합된 난자 수', '난자 출처',
    '난자 기증자 나이', '정자 기증자 나이'
]

# ✅ **2. 데이터 로드**
train_path = "C:/Users/mch2d/Desktop/LG-Aimers-Data-main/train.csv"
test_path = "C:/Users/mch2d/Desktop/LG-Aimers-Data-main/test.csv"

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# ✅ **3. 타겟 변수 분리**
X = df_train[selected_features].copy()
y = df_train["임신 성공 여부"].copy()

# ✅ **4. 결측치 처리**
missing_ratio = (X.isnull().sum() / len(X)) * 100

# 🎯 (1) 80% 이상 결측 → 컬럼 삭제
to_drop = missing_ratio[missing_ratio >= 80].index.tolist()
X.drop(columns=to_drop, inplace=True, errors="ignore")
df_test.drop(columns=to_drop, inplace=True, errors="ignore")

# 🎯 (2) 수치형 변수 → 중앙값(median)으로 대체
num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
X[num_features] = X[num_features].apply(lambda col: col.fillna(col.median()))
df_test[num_features] = df_test[num_features].apply(lambda col: col.fillna(col.median()))

# 🎯 (3) 범주형 변수 → 최빈값(Mode)으로 대체
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
for col in categorical_features:
    most_frequent = X[col].mode()[0]
    X[col] = X[col].fillna(most_frequent).astype(str)
    df_test[col] = df_test[col].fillna(most_frequent).astype(str)

# ✅ **5. Polynomial Features (다항 특징) 적용**
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)

# ✅ 수치형 변수만 변환
X_poly = poly.fit_transform(X[num_features])
df_test_poly = poly.transform(df_test[num_features])

# ✅ Polynomial Feature 이름 설정
poly_feature_names = poly.get_feature_names_out(num_features)

# ✅ 다항 피처를 DataFrame으로 변환
X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names, index=X.index)
df_test_poly_df = pd.DataFrame(df_test_poly, columns=poly_feature_names, index=df_test.index)

# ✅ 기존 데이터 + Polynomial Features 결합
X_poly_final = pd.concat([X, X_poly_df], axis=1)
df_test_poly_final = pd.concat([df_test, df_test_poly_df], axis=1)

# ✅ 중복된 컬럼 제거
X_poly_final = X_poly_final.loc[:, ~X_poly_final.columns.duplicated()]
df_test_poly_final = df_test_poly_final.loc[:, ~df_test_poly_final.columns.duplicated()]

# ✅ 테스트 데이터 컬럼 맞추기
df_test_poly_final = df_test_poly_final.reindex(columns=X_poly_final.columns, fill_value=0)

# ✅ **6. CatBoost 기본 파라미터 설정**
default_params = {
    "iterations": 1000,
    "depth": 6,
    "learning_rate": 0.03,
    "l2_leaf_reg": 3,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "task_type": "GPU",
    "devices": "0",
    "verbose": 100
}

# ✅ **7. K-Fold 설정 (5-Fold)**
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ✅ 각 Fold의 점수 저장 (원본 Feature & Polynomial Feature 비교)
auc_scores_original = []
accuracy_scores_original = []

auc_scores_poly = []
accuracy_scores_poly = []

print("\n🚀 K-Fold 교차 검증 시작...\n")

# ✅ **8. K-Fold 학습 & 평가**
for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    X_train_poly, X_valid_poly = X_poly_final.iloc[train_idx], X_poly_final.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # ✅ 원본 Feature로 학습
    model_original = CatBoostClassifier(**default_params)
    model_original.fit(X_train, y_train, cat_features=categorical_features, verbose=0)

    # ✅ Polynomial Features로 학습
    model_poly = CatBoostClassifier(**default_params)
    model_poly.fit(X_train_poly, y_train, cat_features=categorical_features, verbose=0)

    # ✅ 예측 (원본 Feature)
    valid_preds_proba_original = model_original.predict_proba(X_valid)[:, 1]
    valid_preds_original = model_original.predict(X_valid)

    auc_original = roc_auc_score(y_valid, valid_preds_proba_original)
    acc_original = accuracy_score(y_valid, valid_preds_original)

    auc_scores_original.append(auc_original)
    accuracy_scores_original.append(acc_original)

    # ✅ 예측 (Polynomial Features)
    valid_preds_proba_poly = model_poly.predict_proba(X_valid_poly)[:, 1]
    valid_preds_poly = model_poly.predict(X_valid_poly)

    auc_poly = roc_auc_score(y_valid, valid_preds_proba_poly)
    acc_poly = accuracy_score(y_valid, valid_preds_poly)

    auc_scores_poly.append(auc_poly)
    accuracy_scores_poly.append(acc_poly)

    print(f"🔹 Fold {fold + 1}:")
    print(f"   ➜ 원본 Feature: AUC = {auc_original:.10f}, Accuracy = {acc_original:.10f}")
    print(f"   ➜ Polynomial Feature: AUC = {auc_poly:.10f}, Accuracy = {acc_poly:.10f}")

# ✅ **9. 최종 결과 출력**
print("\n🎯 최종 K-Fold 평균 점수 비교:")
print(f"✅ 원본 Feature - 평균 ROC-AUC Score: {np.mean(auc_scores_original):.10f}")
print(f"✅ 원본 Feature - 평균 Accuracy Score: {np.mean(accuracy_scores_original):.10f}")
print(f"✅ Polynomial Feature - 평균 ROC-AUC Score: {np.mean(auc_scores_poly):.10f}")
print(f"✅ Polynomial Feature - 평균 Accuracy Score: {np.mean(accuracy_scores_poly):.10f}")
