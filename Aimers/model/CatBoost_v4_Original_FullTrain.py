"""
 * Project : LG Aimers - 20개 Feature 기반 CatBoost 학습 (기본 파라미터)
 * Program Purpose and Features :
 * - 기본 CatBoost 설정으로 Full Train 진행
 * - 학습된 모델을 .cbm 파일로 저장
 * - sample_submission.csv를 활용해 최종 예측 결과 저장
 * Author : SP Hong
 * First Write Date : 2025.02.25
"""

import numpy as np
import pandas as pd
import os
from catboost import CatBoostClassifier

# ✅ **1. 사용하려는 20개 Feature 선택**
selected_features = [
    '시술 시기 코드', 
    '시술 당시 나이', 
    '특정 시술 유형', 
    '배란 자극 여부', 
    '배란 유도 유형', 
    '단일 배아 이식 여부', 
    '불명확 불임 원인', 
    '불임 원인 - 난관 질환', 
    '불임 원인 - 남성 요인', 
    '불임 원인 - 배란 장애',
    '불임 원인 - 자궁내막증', 
    '배아 생성 주요 이유', 
    '클리닉 내 총 시술 횟수', 
    'IVF 시술 횟수', 
    'DI 시술 횟수', 
    'IVF 임신 횟수', 
    'IVF 출산 횟수', 
    '총 생성 배아 수', 
    '미세주입에서 생성된 배아 수', 
    '이식된 배아 수', 
    '미세주입 배아 이식 수', 
    '저장된 배아 수', 
    '미세주입 후 저장된 배아 수', 
    '해동된 배아 수', 
    '수집된 신선 난자 수', 
    '파트너 정자와 혼합된 난자 수', 
    '기증자 정자와 혼합된 난자 수', 
    '난자 출처', 
    '난자 기증자 나이', 
    '정자 기증자 나이', 
    '동결 배아 사용 여부', 
    '배아 이식 경과일',
    '남성 주 불임 원인',
    '남성 부 불임 원인',
    '여성 주 불임 원인',
    '여성 부 불임 원인',
    '부부 주 불임 원인',
    '부부 부 불임 원인'

]

# ✅ **2. 데이터 로드**
train_path = "C:/Users/mch2d/Desktop/LG-Aimers-Data-main/train.csv"
test_path = "C:/Users/mch2d/Desktop/LG-Aimers-Data-main/test.csv"
submission_path = "C:/Users/mch2d/Desktop/LG-Aimers-Data-main/sample_submission.csv"

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df_sample_submission = pd.read_csv(submission_path)

# ✅ **3. ID 컬럼 저장**
test_ids = df_sample_submission["ID"]

# ✅ **4. 타겟 변수 분리**
X = df_train[selected_features].copy()
y = df_train["임신 성공 여부"].copy()

# ✅ **5. 결측치 처리**
missing_ratio = (X.isnull().sum() / len(X)) * 100

# 🎯 (1) 80% 이상 결측 → 컬럼 삭제
to_drop = missing_ratio[missing_ratio >= 80].index.tolist()
X.drop(columns=to_drop, inplace=True, errors="ignore")
df_test.drop(columns=to_drop, inplace=True, errors="ignore")

# 🎯 (2) 수치형 변수 → 평균값(mean)으로 대체
num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
X[num_features] = X[num_features].apply(lambda col: col.fillna(col.mean()))
df_test[num_features] = df_test[num_features].apply(lambda col: col.fillna(col.mean()))

# 🎯 (3) 범주형 변수 → 최빈값(Mode)으로 대체
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
for col in categorical_features:
    most_frequent = X[col].mode()[0]
    X[col] = X[col].fillna(most_frequent).astype(str)
    df_test[col] = df_test[col].fillna(most_frequent).astype(str)

# ✅ **6. 테스트 데이터 컬럼 맞추기**
df_test = df_test.reindex(columns=X.columns, fill_value=0)

# ✅ **7. CatBoost 기본 파라미터 설정**
default_params = {
    "iterations": 1000,
    "depth": 6,
    "learning_rate": 0.03,
    "l2_leaf_reg": 3,
    "loss_function": "Logloss",  # ✅ AUC 대신 Logloss 사용 (GPU에서 AUC 지원 X)
    "eval_metric": "Logloss",
    "task_type": "GPU",
    "devices": "0",
    "verbose": 100
}

# ✅ **8. CatBoost Full Train 모델 학습**
print("\n🚀 CatBoost 기본 파라미터로 Full Train 모델 학습 시작...")
best_model = CatBoostClassifier(**default_params)
best_model.fit(X, y, cat_features=categorical_features, verbose=100)

# ✅ **9. 학습된 모델 저장**
model_dir = "C:/Users/mch2d/Desktop/LG-Aimers-Data-main/Models"
os.makedirs(model_dir, exist_ok=True)  # 📁 저장 폴더 없으면 생성

model_path = os.path.join(model_dir, "catboost_fulltrain_default.bin")
best_model.save_model(model_path, format="cbm")

print(f"✅ 학습된 모델 저장 완료: {model_path}")

# ✅ **10. 테스트 데이터 예측 및 저장**
predictions = best_model.predict_proba(df_test)[:, 1]
submission = pd.DataFrame({"ID": test_ids, "probability": predictions})

final_csv_path = "C:/Users/mch2d/Desktop/LG-Aimers-Data-main/CatBoost_Orginal_FullTrain_v1.csv"
submission.to_csv(final_csv_path, index=False)

print(f"✅ 최종 예측 결과 저장 완료: {final_csv_path}")
