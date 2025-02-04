import pandas as pd

# CSV 파일 로드(인코딩 상황 맞춰 조정)
df = pd.read_csv('C:/Users/ANTL/Desktop/LG Aimers Data/train.csv', encoding='utf-8')

# "총 시술 횟수"와 "임신 성공 여부"별 개수 집계
df_counts = df.groupby(['총 시술 횟수', '임신 성공 여부']).size().unstack(fill_value=0)

# 결과 출력
print(df_counts)
