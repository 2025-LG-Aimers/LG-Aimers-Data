import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import matplotlib

matplotlib.rc('font', family='Malgun Gothic')   # Windows는 맑은 고딕
matplotlib.rc('axes', unicode_minus=False)

# 사용할 컬럼 목록 (불임 원인 관련)
subset_cols = [
    "남성 주 불임 원인", "남성 부 불임 원인", "여성 주 불임 원인", "여성 부 불임 원인",
    "부부 주 불임 원인", "부부 부 불임 원인", "불명확 불임 원인",
    "불임 원인 - 난관 질환", "불임 원인 - 남성 요인", "불임 원인 - 배란 장애",
    "불임 원인 - 여성 요인", "불임 원인 - 자궁경부 문제", "불임 원인 - 자궁내막증",
    "불임 원인 - 정자 농도", "불임 원인 - 정자 면역학적 요인",
    "불임 원인 - 정자 운동성", "불임 원인 - 정자 형태", "임신 성공 여부"
]

# CSV 파일 로드 (인코딩 상황에 맞춰 조정)
df = pd.read_csv('C:/Users/ANTL/Desktop/LG Aimers Data/train.csv', encoding='utf-8')

# 실제 DataFrame에 존재하는 컬럼만 골라서 추출
cols_in_df = [col for col in subset_cols if col in df.columns]
df_subset = df[cols_in_df].copy()

# 라벨 인코딩: 텍스트 → 숫자
for col in df_subset.columns:
    df_subset[col] = df_subset[col].astype(str)
    le = LabelEncoder()
    df_subset[col] = le.fit_transform(df_subset[col])

# 상관계수 계산
corr_subset = df_subset.corr()

# 히트맵 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_subset,
    cmap='coolwarm',
    annot=True,   # 각 셀에 상관계수 값 표시
    fmt='.2f',
    annot_kws={"size": 7},
    cbar_kws={"shrink": 0.8}
)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.title('불임 원인 관련 컬럼 간 상관관계 Heatmap')
plt.tight_layout()
plt.show()
