import pandas as pd

# 파일 로드 (인코딩 확인 후 적절한 방식 선택)
file_path = "C:/Users/ANTL/Desktop/LG Aimers Data/train.csv"
df = pd.read_csv(file_path, encoding="utf-8")

# 삭제할 컬럼 리스트 (기존 삭제 대상 + 추가 삭제 대상 포함)
columns_to_drop = [
    # 기존 삭제 컬럼
    "난자 해동 경과일",
    "PGS 시술 여부",
    "PGD 시술 여부",
    "착상 전 유전 검사 사용 여부",
    "임신 시도 또는 마지막 임신 경과 연수",
    "배아 해동 경과일",
    
    # 추가 삭제 (수치형 데이터)
    "난자 채취 경과일",
    "난자 혼합 경과일",
    "기증자 정자와 혼합된 난자 수",
    "해동된 배아 수",
    
    # 추가 삭제 (범주형 데이터)
    "시술 시기 코드",
    "배란 유도 유형",
    "정자 기증자 나이",
    "정자 출처",
    "난자 출처",
    
    # 불임 원인 관련 컬럼 (중복 정보가 포함된 것으로 판단)
    "불임 원인 - 난관 질환",
    "불임 원인 - 남성 요인",
    "불임 원인 - 배란 장애",
    "불임 원인 - 여성 요인",
    "불임 원인 - 자궁경부 문제",
    "불임 원인 - 자궁내막증",
    "불임 원인 - 정자 농도",
    "불임 원인 - 정자 면역학적 요인",
    "불임 원인 - 정자 운동성",
    "불임 원인 - 정자 형태"
]

# 해당 컬럼 삭제
df.drop(columns=columns_to_drop, inplace=True)

# '배아 생성 주요 이유' 컬럼에서 '난자 저장용'인 행 제거
df = df[df["배아 생성 주요 이유"] != "난자 저장용"]
df = df[df["배아 생성 주요 이유"] != "배아 저장용"]

# 저장 경로
output_path = "C:/Users/ANTL/Desktop/LG Aimers Data/train_modified_v2.csv"

# 변경된 데이터프레임 저장 (utf-8-sig 사용)
df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"파일 저장 완료: {output_path}")
