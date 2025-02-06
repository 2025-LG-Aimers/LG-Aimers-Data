import pandas as pd

df = pd.read_csv("C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/train.csv")  
print(df.info())  # 데이터 타입, 결측치 확인
print(df.describe())  # 수치형 변수 기본 통계량
print(df.head())  # 데이터 일부 샘플 확인


num_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = df.select_dtypes(include=['object']).columns.tolist()

print(f"수치형 변수 개수: {len(num_features)}, 범주형 변수 개수: {len(cat_features)}")


missing_ratio = df.isnull().sum() / len(df) * 100
print(missing_ratio[missing_ratio > 0].sort_values(ascending=False))  # 결측치 많은 변수 확인

df.fillna(df.median(), inplace=True)  # 수치형 변수는 중앙값으로 채우기
df.fillna("Unknown", inplace=True)  # 범주형 변수는 'Unknown'으로 채우기
