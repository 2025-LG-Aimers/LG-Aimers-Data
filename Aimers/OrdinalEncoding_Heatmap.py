import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder

# -------------- 📌 데이터 로딩 --------------
df = pd.read_csv("C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/train.csv")

# 🔹 한글 깨짐 방지 (Windows 환경)
plt.rc("font", family="Malgun Gothic")
plt.rc("axes", unicode_minus=False)

# 🔹 현재 남아있는 범주형 컬럼
categorical_columns = [
    "시술 시기 코드", "시술 당시 나이", "특정 시술 유형", "배란 유도 유형", "배아 생성 주요 이유",
    "총 시술 횟수", "클리닉 내 총 시술 횟수", "IVF 시술 횟수", "DI 시술 횟수",
    "총 임신 횟수", "IVF 임신 횟수", "총 출산 횟수", "IVF 출산 횟수",
    "난자 출처", "정자 출처", "난자 기증자 나이", "정자 기증자 나이"
]

# 🔹 범주형 데이터를 Label Encoding (Ordinal Encoding)
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
df_encoded = df[categorical_columns].astype(str)  # 문자형 변환
df_encoded = encoder.fit_transform(df_encoded)

# 🔹 데이터프레임 변환 (컬럼 이름 유지)
df_encoded = pd.DataFrame(df_encoded, columns=categorical_columns)

# 🔹 상관계수 계산
corr_matrix = df_encoded.corr()

# 🔹 Heatmap 그리기
plt.figure(figsize=(12, 8))
sns.heatmap(
    corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5
)
plt.title("범주형 변수 간 상관계수 Heatmap")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.show()
