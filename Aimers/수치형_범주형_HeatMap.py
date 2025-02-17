import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols

# 🔹 한글 깨짐 방지 (Windows 환경)
plt.rc("font", family="Malgun Gothic")
plt.rc("axes", unicode_minus=False)

# 🔹 데이터 로드
df = pd.read_csv("C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/train.csv")

# 🔹 수치형 & 범주형 컬럼 리스트
numeric_columns = [
    "배란 자극 여부", "단일 배아 이식 여부", "불명확 불임 원인", "불임 원인 - 난관 질환", "불임 원인 - 남성 요인",
    "불임 원인 - 배란 장애", "불임 원인 - 자궁내막증", "총 생성 배아 수", "미세주입에서 생성된 배아 수", 
    "이식된 배아 수", "미세주입 배아 이식 수", "저장된 배아 수", "미세주입 후 저장된 배아 수", "해동된 배아 수",
    "수집된 신선 난자 수", "파트너 정자와 혼합된 난자 수", "기증자 정자와 혼합된 난자 수", "동결 배아 사용 여부",
    "배아 이식 경과일", "배아 해동 경과일"
]

categorical_columns = [
    "시술 시기 코드", "시술 당시 나이", "특정 시술 유형", "배란 유도 유형", "배아 생성 주요 이유",
    "총 시술 횟수", "클리닉 내 총 시술 횟수", "DI 시술 횟수", "총 임신 횟수", "IVF 출산 횟수",
    "난자 출처", "난자 기증자 나이", "정자 기증자 나이"
]

# ✅ 1. 수치형 데이터 간 상관계수 (피어슨 상관계수)
df_numeric = df[numeric_columns]
corr_numeric = df_numeric.corr()

# ✅ 2. 범주형 ↔ 수치형 상관관계 (ANOVA η² 값)
def eta_squared(y, x):
    """ANOVA를 사용하여 η² 값 계산"""
    try:
        # 범주형 데이터를 문자열(str)로 변환하여 ANOVA가 올바르게 수행되도록 처리
        df[x] = df[x].astype(str)
        model = ols(f"{y} ~ C({x})", data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        # ANOVA에서 Sum of Squares 계산
        ss_between = anova_table["sum_sq"][0]  # 범주형에 의한 변동량
        ss_total = anova_table["sum_sq"].sum()  # 전체 변동량
        eta2 = ss_between / ss_total if ss_total > 0 else 0  # η² 값 계산

        # 너무 작은 값(0.01 미만)은 0으로 처리하여 시각적으로 의미 있도록 함
        return eta2 if eta2 >= 0.01 else 0
    except:
        return np.nan  # 계산 불가능한 경우 NaN 반환

# 🔹 범주형 ↔ 수치형 η² 값 행렬 생성
eta_matrix = np.zeros((len(numeric_columns), len(categorical_columns)))

for i, num_col in enumerate(numeric_columns):
    for j, cat_col in enumerate(categorical_columns):
        eta_matrix[i, j] = eta_squared(num_col, cat_col)

# 🔹 DataFrame 변환
eta_df = pd.DataFrame(eta_matrix, index=numeric_columns, columns=categorical_columns)

# ✅ 3. 수치형 ↔ 범주형 결합된 상관행렬 만들기
# 🔹 피어슨 상관계수 + ANOVA η² 값 결합
combined_corr = pd.concat([corr_numeric, eta_df], axis=1)

# 🔹 NaN 값은 0으로 채우되, 최소값을 보존하여 시각적으로 표시 가능하게 함
combined_corr.fillna(0, inplace=True)

# ✅ 4. 최종 Heatmap 그리기
plt.figure(figsize=(18, 12))
sns.heatmap(
    combined_corr, annot=True, fmt=".3f", cmap="coolwarm", linewidths=0.5, cbar=True
)
plt.title("수치형 ↔ 수치형 (피어슨) & 범주형 ↔ 수치형 (ANOVA η²) Heatmap")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.show()
