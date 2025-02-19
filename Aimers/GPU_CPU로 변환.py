import pickle

# 📂 원본 pkl 파일 경로
original_param_path = "C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/Params/best_catboost_params.pkl"

# 📂 변경된 pkl 파일 저장 경로
new_param_path = "C:/Users/ANTL/Documents/GitHub/LG-Aimers-Data/Params/best_catboost_params_cpu.pkl"

# 🔄 pkl 파일 로드
with open(original_param_path, "rb") as f:
    best_config = pickle.load(f)

# 🔄 GPU → CPU 변환 (새로운 딕셔너리 생성)
new_config = best_config.copy()  # 원본을 유지하기 위해 복사
new_config["task_type"] = "CPU"  # CPU로 변경
new_config.pop("devices", None)  # GPU 관련 설정 제거

# 💾 변경된 내용을 새 파일로 저장
with open(new_param_path, "wb") as f:
    pickle.dump(new_config, f)

print(f"✅ 새로운 pkl 파일이 CPU 모드로 저장되었습니다: {new_param_path}")
