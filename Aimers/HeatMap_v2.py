import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import LabelEncoder

# 한글 폰트 설정 (Windows 기준)
matplotlib.rc('font', family='Malgun Gothic')   
matplotlib.rc('axes', unicode_minus=False)

# ------------------------------ Heatmap 함수 정의 ------------------------------ #
def plot_heatmap(csv_path, feature_cols):
    """
    주어진 컬럼 목록에 해당하는 데이터를 이용해 Heatmap을 생성하는 함수.
    
    Parameters:
        csv_path (str) : 분석할 CSV 파일 경로
        feature_cols (list) : Heatmap에 포함할 컬럼 리스트 (수치형, n회, 범주형 가능)
    
    Returns:
        None
    """

    # 1) CSV 데이터 불러오기
    df = pd.read_csv(csv_path, encoding="utf-8")

    # 2) 실제 DataFrame에 존재하는 컬럼만 선별
    cols_in_df = [col for col in feature_cols if col in df.columns]
    df_subset = df[cols_in_df].copy()

    # 3) "n회" 형식 문자열을 숫자로 변환하기 위한 매핑 함수
    def convert_round_to_int(x):
        """
        '0회', '1회', '2회', ..., '6회 이상'을 숫자로 변환 (0~6)
        """
        if isinstance(x, str):
            x = x.strip()
            if x in ["0회", "1회", "2회", "3회", "4회", "5회"]:
                return int(x.replace("회", ""))
            elif x in ["6회", "6회이상", "6회 이상"]:
                return 6
        return None  # 변환 불가한 값은 결측치 처리

    # 4) "n회" 데이터 변환 (문자 → 숫자)
    round_cols = ["총 시술 횟수", "클리닉 내 총 시술 횟수", "IVF 시술 횟수", "DI 시술 횟수",
                  "총 임신 횟수", "IVF 임신 횟수", "총 출산 횟수", "IVF 출산 횟수", "DI 출산 횟수"]
    
    round_cols_in_df = [c for c in round_cols if c in df_subset.columns]
    
    for c in round_cols_in_df:
        df_subset[c] = df_subset[c].astype(str).apply(convert_round_to_int)

    # 5) 범주형 데이터 변환 (Label Encoding)
    categorical_columns = df_subset.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_columns:
        le = LabelEncoder()
        df_subset[col] = le.fit_transform(df_subset[col].astype(str))

    # 6) 임신 성공 여부가 존재하면 정수형으로 변환
    if "임신 성공 여부" in df_subset.columns:
        df_subset["임신 성공 여부"] = df_subset["임신 성공 여부"].astype(int)

    # 7) 상관계수 계산 및 Heatmap 시각화
    corr = df_subset.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr, cmap='coolwarm', annot=True, fmt='.2f', annot_kws={"size": 8},
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title('선택된 컬럼들의 Heatmap')
    plt.tight_layout()
    plt.show()

# ------------------------------  함수 실행 ------------------------------ #
csv_path = "C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/train.csv"

# 사용자가 원하는 컬럼 입력
selected_features = [
    # 수치형 데이터
    "이식된 배아 수", "미세주입 배아 이식 수", "저장된 배아 수",
    
    # "n회" 데이터
    "총 시술 횟수",
    
    # 범주형 데이터
    "시술 당시 나이", "시술 유형", "특정 시술 유형"
    
    # 타겟 변수
    "임신 성공 여부"
]

plot_heatmap(csv_path, selected_features)
