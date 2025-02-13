import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ 한글 폰트 설정 (Windows: "Malgun Gothic", Mac: "AppleGothic")
import matplotlib
matplotlib.rc("font", family="Malgun Gothic")  # 한글 폰트 설정
matplotlib.rc("axes", unicode_minus=False)  # 마이너스(-) 기호 깨짐 방지

# 데이터 로드
df = pd.read_csv("C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/train.csv")

# 이상치를 체크할 수치형 컬럼 리스트
numeric_cols = [
    "총 생성 배아 수", "미세주입된 난자 수", "미세주입에서 생성된 배아 수", "이식된 배아 수", "미세주입 배아 이식 수",
    "저장된 배아 수", "미세주입 후 저장된 배아 수", "해동된 배아 수", "수집된 신선 난자 수",
    "혼합된 난자 수", "파트너 정자와 혼합된 난자 수"
]

# ✅ IQR 기반 이상치 탐지 함수
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)  # 1사분위수 (25%)
    Q3 = df[column].quantile(0.75)  # 3사분위수 (75%)
    IQR = Q3 - Q1  # IQR 계산
    lower_bound = Q1 - 1.5 * IQR  # 이상치 하한선
    upper_bound = Q3 + 1.5 * IQR  # 이상치 상한선

    # 이상치 데이터만 필터링
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
    
    return outliers, lower_bound, upper_bound

# ✅ 박스플롯(Box Plot) 시각화
plt.figure(figsize=(20, 12))  # 전체 그래프 크기 조정
for i, col in enumerate(numeric_cols):
    if col in df.columns:  # 데이터에 해당 컬럼이 있는지 확인
        outliers, lower, upper = detect_outliers_iqr(df, col)
        
        # 박스플롯 그리기
        plt.subplot(4, 4, i + 1)  # 4행 4열 서브플롯
        sns.boxplot(y=df[col], color="skyblue", fliersize=5)  # 이상치는 빨간 점으로 표시됨
        plt.axhline(y=lower, color="red", linestyle="--", label="하한선")
        plt.axhline(y=upper, color="red", linestyle="--", label="상한선")
        plt.title(f"{col} (이상치 개수: {len(outliers)})", fontsize=10)
        plt.xticks([])  # x축 눈금 제거

plt.tight_layout()  # 간격 조정
plt.show()
