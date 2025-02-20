import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 파일 로드
train_df = pd.read_csv("dataset_train.csv")
valid_df = pd.read_csv("dataset_val.csv")

# 📌 [Step 1] Train 데이터셋 5개로 나누기
train_splits = np.array_split(train_df, 5)

# 📌 [Step 2] Valid 데이터셋 5개로 나누고 일부를 Test 데이터셋으로 사용
valid_train, test_df = train_test_split(valid_df, test_size=0.2, random_state=42)  # 20%를 test set으로 분할
valid_splits = np.array_split(valid_train, 5)  # 나머지 80%를 5개로 나누기

# 📌 [Step 3] 분할된 데이터 저장
for i, df in enumerate(train_splits):
    df.to_csv(f"./final/dataset_train_part{i+1}.csv", index=False)

for i, df in enumerate(valid_splits):
    df.to_csv(f"./final/dataset_valid_part{i+1}.csv", index=False)

test_df.to_csv("./final/dataset_test.csv", index=False)  # Test 데이터셋 저장

print(f"Train 데이터셋: 총 {len(train_df)}, Valid 데이터셋: 총 {len(valid_train)}, Test 데이터셋: 총 {len(test_df)} -> 성공적으로 분할 및 저장되었습니다!")
