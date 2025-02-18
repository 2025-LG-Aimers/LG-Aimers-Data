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

# ✅ 1. 편향된 컬럼 제거
columns_to_remove = ["신선 배아 사용 여부", "미세주입된 난자 수", "IVF 시술 횟수", 
                     "IVF 임신 횟수", "총 출산 횟수", "정자 출처"]

# ✅ 95% 이상 편향된 컬럼 제거
threshold = 0.95
biased_columns = [col for col in X.columns if X[col].value_counts(normalize=True).max() >= threshold]
columns_to_remove.extend(biased_columns)

X.drop(columns=columns_to_remove, inplace=True, errors='ignore')
test.drop(columns=columns_to_remove, inplace=True, errors='ignore')

# ✅ 2. 결측치 처리 (컬럼별 결측치 비율 계산)
missing_percentage = (X.isnull().sum() / len(X)) * 100

# 🔥 80% 이상 결측치 컬럼 삭제
high_missing_columns = missing_percentage[missing_percentage >= 80].index.tolist()
X.drop(columns=high_missing_columns, inplace=True, errors='ignore')
test.drop(columns=high_missing_columns, inplace=True, errors='ignore')

# 🔥 15% ~ 30% 결측치 평균값으로 대체
mid_missing_columns = missing_percentage[(missing_percentage >= 10) & (missing_percentage < 50)].index.tolist()
imputer = SimpleImputer(strategy='mean')
X[mid_missing_columns] = imputer.fit_transform(X[mid_missing_columns])
test[mid_missing_columns] = imputer.transform(test[mid_missing_columns])

# ✅ 3. 제거되지도 않고 결측치도 대체되지 않은 컬럼 찾기
remaining_columns = list(set(X.columns) - set(mid_missing_columns))

# ✅ 4. 최종 출력
print("\n🔥 [제거된 컬럼 목록]")
print(f"✅ 편향된 컬럼: {biased_columns}")
print(f"✅ 80% 이상 결측치 컬럼: {high_missing_columns}")
print(f"🔥 최종 제거된 컬럼: {columns_to_remove + high_missing_columns}")

print("\n🔥 [결측치를 평균값으로 대체한 컬럼]")
print(mid_missing_columns)

print("\n🔥 [제거되지도 않고 결측치를 대체하지도 않은 컬럼]")
print(remaining_columns)
