import pandas as pd
from sklearn.utils import resample

# CSV 파일 불러오기
df = pd.read_csv('C:/Users/ANTL/Desktop/LG Aimers Data/train.csv', encoding='utf-8')

# 라벨(임신 성공 여부)에 따라 데이터 분리
df_false = df[df['임신 성공 여부'] == 0]  # 다수 클래스
df_true  = df[df['임신 성공 여부'] == 1]  # 소수 클래스

# 데이터 개수 확인
count_false = len(df_false)  # 190123
count_true  = len(df_true)   # 66228
print(f"원본 데이터: 0 = {count_false}, 1 = {count_true}")

# ──────────── 언더샘플링 ────────────
# 다수 클래스(0)를 소수 클래스(1)의 개수만큼 줄이기
df_false_undersampled = resample(
    df_false,
    replace=False,          # 중복추출 없이
    n_samples=count_true,   # 66,228로 맞춤
    random_state=42
)

# 언더샘플링된 결과 합치기
df_rebalanced = pd.concat([df_false_undersampled, df_true], axis=0)

# 리밸런싱 후 각 클래스 개수 확인
rebalanced_count_false = len(df_rebalanced[df_rebalanced['임신 성공 여부'] == 0])
rebalanced_count_true  = len(df_rebalanced[df_rebalanced['임신 성공 여부'] == 1])
print(f"리밸런싱된 데이터(언더샘플링): 0 = {rebalanced_count_false}, 1 = {rebalanced_count_true}")

# 언더샘플링된 데이터 저장
df_rebalanced.to_csv('C:/Users/ANTL/Desktop/LG Aimers Data/train_rebalancing_v1.csv', 
                     index=False, encoding='utf-8-sig')
