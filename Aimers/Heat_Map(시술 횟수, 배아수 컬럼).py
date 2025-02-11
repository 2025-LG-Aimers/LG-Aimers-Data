import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

# 🔥 한글 폰트 설정 (Windows 기준)
matplotlib.rc('font', family='Malgun Gothic')
matplotlib.rc('axes', unicode_minus=False)

# ✅ 사용할 피처 목록
feature_cols = [
    "총 시술 횟수", "클리닉 내 총 시술 횟수", "IVF 시술 횟수", "DI 시술 횟수", 
    "총 임신 횟수", "IVF 임신 횟수", "총 출산 횟수", "IVF 출산 횟수", "DI 출산 횟수",
    "총 생성 배아 수", "미세주입된 난자 수", "미세주입에서 생성된 배아 수",
    "이식된 배아 수", "미세주입 배아 이식 수", "저장된 배아 수",
    "미세주입 후 저장된 배아 수", "해동된 배아 수", "해동 난자 수",
    "수집된 신선 난자 수", "저장된 신선 난자 수", "혼합된 난자 수",
    "파트너 정자와 혼합된 난자 수", "기증자 정자와 혼합된 난자 수",
    "임신 성공 여부"
]

# ✅ 데이터 불러오기
df = pd.read_csv("C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/train.csv", encoding="utf-8")

# ✅ 실제 존재하는 컬럼만 선택
cols_in_df = [col for col in feature_cols if col in df.columns]
df_subset = df[cols_in_df].copy()

# 🔥 'n회' → 숫자로 변환하는 함수
def convert_round_to_int(x):
    if isinstance(x, str):
        x = x.strip()
        if x in ["0회", "1회", "2회", "3회", "4회", "5회"]:
            return int(x.replace("회", ""))
        elif x in ["6회", "6회이상", "6회 이상"]:
            return 6
    return None  

# ✅ 'n회' 관련 컬럼 변환 적용
round_cols = [
    "총 시술 횟수", "클리닉 내 총 시술 횟수", "IVF 시술 횟수", "DI 시술 횟수",
    "총 임신 횟수", "IVF 임신 횟수", "총 출산 횟수", "IVF 출산 횟수", "DI 출산 횟수"
]
round_cols_in_df = [c for c in round_cols if c in df_subset.columns]

for c in round_cols_in_df:
    df_subset[c] = df_subset[c].astype(str).apply(convert_round_to_int)

# ✅ 상관계수 계산
corr = df_subset.corr()

# 🔥 상삼각 행렬 마스크 만들기 (반만 보이게 설정)
mask = np.triu(np.ones_like(corr, dtype=bool))  # 상삼각 행렬(True) → 가려질 부분

# ✅ Heatmap 그리기
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr, mask=mask,  # 🔥 상삼각 행렬 가리기
    cmap='coolwarm',
    annot=True, fmt='.2f',
    annot_kws={"size": 7},
    cbar_kws={"shrink": 0.8}
)

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.title('시술 횟수(문자→숫자 변환) & 난자·배아 관련 + 임신 성공 여부 Heatmap (반만 표시)')
plt.tight_layout()
plt.show()
