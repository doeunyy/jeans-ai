import pandas as pd

# CSV 파일 읽기
csv_file = "dataset_val.csv"  # 여기에 CSV 파일명을 입력
df = pd.read_csv(csv_file)

# 필요 없는 컬럼 삭제
columns_to_drop = ["gender", "age", "region", "dialect"]
df = df.drop(columns=columns_to_drop, errors="ignore")  # 해당 컬럼이 없으면 무시

# 기본 오디오 파일 경로 설정
base_path = "C:/Users/004/Desktop/data/voice-command/Validation/"

# 새로운 audio_path 컬럼 생성
df["audio_path"] = df["id"].apply(lambda x: f"{base_path}{x}.wav")

# 변경된 데이터 확인
print(df.head())

# 새로운 CSV 파일로 저장 (선택 사항)
df.to_csv("dataset_val_updated.csv", index=False)
