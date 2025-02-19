import os
import pandas as pd

# data_path: voice-command 디렉토리의 로컬 경로
data_path = "C:/Users/004/Desktop/data/voice-command"

train_path = f"{data_path}/Training"
val_path = f"{data_path}/Validation"

output_dir = "./dataset"  # CSV 저장할 디렉토리
output_csv_path = os.path.join(output_dir, "audio_path_val.csv")

# 결과 저장을 위한 리스트 초기화
audio_data = []

# train_path 내 [원천]으로 시작하는 디렉토리 순회
for root, dirs, files in os.walk(val_path):
    for file in files:
        if file.endswith(".wav"):
            file_id = os.path.splitext(file)[0]  # 확장자 제거한 ID
            file_path = os.path.join(root, file).replace("\\", "/")  # 경로 저장
            
            audio_data.append([file_id, file_path])

# 데이터프레임 생성
df = pd.DataFrame(audio_data, columns=["id", "audio_path"])

# CSV 파일 저장
df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

print(f"처리한 audio file 개수: {len(df)}")
print(f"CSV 파일이 저장되었습니다: {output_csv_path}")
