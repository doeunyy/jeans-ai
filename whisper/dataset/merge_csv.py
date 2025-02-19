import pandas as pd

dataset_type = "val"

# CSV 파일 경로 설정
file1_path = f"./dataset/audio_path_{dataset_type}.csv"
file2_path = f"./dataset/whisper_{dataset_type}.csv"
file3_path = f"./dataset/annotation_{dataset_type}.csv"

# CSV 파일 로드
df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)
df3 = pd.read_csv(file3_path)

# 2번 파일을 id 기준으로 병합
df_merged = df1.merge(df2, on="id", how="left")

# 3번 파일에서 labelText 컬럼만 선택하여 병합
df3 = df3[["id", "labelText"]]
df_merged = df_merged.merge(df3, on="id", how="left")

# 병합된 데이터 저장
output_csv_path = f"./dataset/dataset_{dataset_type}_final.csv"
df_merged.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

print(f"병합된 CSV 파일이 저장되었습니다: {output_csv_path}")
