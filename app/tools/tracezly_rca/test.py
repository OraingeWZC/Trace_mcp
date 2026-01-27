import pandas as pd

path = "processed/tianchi_processed_data/test.csv"
df = pd.read_csv(path)

# 去掉 NaN 后取唯一值
unique_rc = df['RootCause'].dropna().unique().tolist()
print(f"共 {len(unique_rc)} 个 RootCause 唯一值：")
for v in unique_rc:
    print(v)