import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 🔹 한글 깨짐 방지 (Windows 환경)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# 🔹 데이터 로딩
df = pd.read_csv("C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/train.csv")

# 🔹 그래프 스타일 설정
sns.set(style="whitegrid")

# -------------------- 📌 1. 이식된 배아 수별 성공/실패 개수 --------------------
plt.figure(figsize=(10, 5))
sns.countplot(
    x="이식된 배아 수",
    hue="임신 성공 여부",
    data=df,
    palette="coolwarm",
    dodge=True  # 막대 옆으로 정렬
)
plt.xlabel("이식된 배아 수")
plt.ylabel("개수")
plt.title("이식된 배아 수별 임신 성공/실패 개수")
plt.legend(title="임신 성공 여부", labels=["실패(0)", "성공(1)"])
plt.show()

# -------------------- 📌 2. 이식된 배아 수별 성공 확률 --------------------
plt.figure(figsize=(8, 5))
sns.barplot(
    x="이식된 배아 수",
    y="임신 성공 여부",
    data=df,
    estimator=lambda x: sum(x) / len(x),  # 성공률(비율) 계산
    ci=None,  # 신뢰구간 제거
    palette="coolwarm"
)
plt.xlabel("이식된 배아 수")
plt.ylabel("임신 성공 확률")
plt.title("이식된 배아 수에 따른 임신 성공 확률")
plt.ylim(0, 1)  # 확률 범위 (0~1)
plt.show()

# -------------------- 📌 3. 히트맵 (Heatmap) --------------------
# 🔹 교차표 생성 (각 배아 수에서 성공/실패 개수)
pivot_table = df.pivot_table(
    index="이식된 배아 수",
    columns="임신 성공 여부",
    aggfunc="size",
    fill_value=0
)

# 🔹 히트맵 시각화
plt.figure(figsize=(6, 4))
sns.heatmap(pivot_table, annot=True, fmt="d", cmap="coolwarm", linewidths=1, cbar=False)
plt.xlabel("임신 성공 여부")
plt.ylabel("이식된 배아 수")
plt.title("이식된 배아 수 vs 임신 성공 여부 히트맵")
plt.show()
