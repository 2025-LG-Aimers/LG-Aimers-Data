import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 한글 폰트 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

train = pd.read_csv('C:/Users/ANTL/Desktop/LG Aimers Data/train.csv', encoding='utf-8')

# 원하는 나이 순서 지정
age_order = ["만18-34세", "만35-37세", "만38-39세", "만40-42세", "만43-44세", "만45-50세", "알 수 없음"]

# 서브플롯 생성 (1행 2열)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# (1) 왼쪽 그래프: 나이별 임신 성공 여부 '개수'
sns.countplot(
    data=train,
    x='시술 당시 나이',
    hue='임신 성공 여부',
    order=age_order,                     # 지정한 순서 적용
    palette=['#1f77b4', '#ff7f0e'],      # 실패=파랑, 성공=주황
    ax=axes[0]
)
axes[0].set_title('시술 당시 나이별 임신 성공 여부 분포')
axes[0].tick_params(axis='x', rotation=45)

# 범례 수정 (0→실패, 1→성공)
handles_left, labels_left = axes[0].get_legend_handles_labels()
axes[0].legend(
    handles_left, 
    ['실패', '성공'],
    title='임신 성공 여부'
)

# (2) 오른쪽 그래프: 나이별 성공/실패 '비율'
df_count = train.groupby(['시술 당시 나이', '임신 성공 여부']).size().reset_index(name='count')
df_count['proportion'] = df_count.groupby('시술 당시 나이')['count'].transform(lambda x: x / x.sum())

sns.barplot(
    data=df_count,
    x='시술 당시 나이',
    y='proportion',
    hue='임신 성공 여부',
    order=age_order,                     # 지정한 순서 적용
    palette=['#1f77b4', '#ff7f0e'],      # 실패=파랑, 성공=주황
    ax=axes[1]
)
axes[1].set_title('시술 당시 나이별 성공/실패율')
axes[1].set_ylim(0, 1)
axes[1].set_ylabel('비율')
axes[1].tick_params(axis='x', rotation=45)

# 범례 수정 (0→실패, 1→성공)
handles_right, labels_right = axes[1].get_legend_handles_labels()
axes[1].legend(
    handles_right, 
    ['실패', '성공'],
    title='임신 성공 여부'
)

plt.tight_layout()
plt.show()
