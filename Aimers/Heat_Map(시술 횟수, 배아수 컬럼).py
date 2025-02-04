import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import LabelEncoder

matplotlib.rc('font', family='Malgun Gothic')   # 한글 폰트(Windows)
matplotlib.rc('axes', unicode_minus=False)

# 히트맵에 포함될 수치형 컬럼 (난자·배아 관련) + 'n회' 컬럼 + 라벨
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

# CSV 파일 로드
df = pd.read_csv("C:/Users/ANTL/Desktop/LG Aimers Data/train.csv", encoding="utf-8")

# 실제 DataFrame에 존재하는 컬럼만 선별
cols_in_df = [col for col in feature_cols if col in df.columns]
df_subset = df[cols_in_df].copy()

# --------------------------------------------------------------------------------
# 1) 'n회' 형식 문자열을 숫자로 변환하기 위한 매핑 함수/사전
# --------------------------------------------------------------------------------
def convert_round_to_int(x):
    """
    '0회', '1회', '2회', '3회', '4회', '5회', '6회', '6회 이상' 등 
    문자열을 정수로 변환 (0,1,2,3,4,5,6).
    만약 다른 값이 들어오면 None(결측치)로 처리.
    """
    if isinstance(x, str):
        x = x.strip()  # 공백 제거
        if x in ["0회", "1회", "2회", "3회", "4회", "5회"]:
            return int(x.replace("회", ""))  # 'n회' → n
        elif x in ["6회", "6회이상", "6회 이상"]:
            return 6
    return None  # 매핑 안 되는 값은 결측치로 처리하거나 필요에 맞게 조정

# --------------------------------------------------------------------------------
# 2) 실제 'n회'가 들어있는 컬럼들만 골라서 변환
# --------------------------------------------------------------------------------
round_cols = [
    "총 시술 횟수", "클리닉 내 총 시술 횟수", "IVF 시술 횟수", "DI 시술 횟수",
    "총 임신 횟수", "IVF 임신 횟수", "총 출산 횟수", "IVF 출산 횟수", "DI 출산 횟수"
]
round_cols_in_df = [c for c in round_cols if c in df_subset.columns]

for c in round_cols_in_df:
    df_subset[c] = df_subset[c].astype(str).apply(convert_round_to_int)

# --------------------------------------------------------------------------------
# 3) 상관계수 계산을 위해 임신 성공 여부도 수치형(0/1)인지 확인
#    (만약 문자열이라면 astype(int) 등으로 변환)
# --------------------------------------------------------------------------------
if "임신 성공 여부" in df_subset.columns:
    # 예: 만약 문자열 '0'/'1'이라면 int로 변환
    # df_subset["임신 성공 여부"] = df_subset["임신 성공 여부"].astype(int)
    pass

# --------------------------------------------------------------------------------
# 4) 상관계수 및 히트맵 시각화
# --------------------------------------------------------------------------------
corr = df_subset.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr,
    cmap='coolwarm',
    annot=True,     
    fmt='.2f',
    annot_kws={"size": 7},
    cbar_kws={"shrink": 0.8}
)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.title('시술 횟수(문자→숫자 변환) & 난자·배아 관련 + 임신 성공 여부 Heatmap')
plt.tight_layout()
plt.show()
