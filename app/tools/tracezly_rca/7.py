import pandas as pd

# 合并 normal 文件
df_normal = pd.concat([
    pd.read_csv('normal/2025-06-08_spans.csv'),
    pd.read_csv('normal/2025-06-09_spans.csv'),
    pd.read_csv('normal/2025-06-10_spans.csv'),
    pd.read_csv('normal/2025-06-11_spans.csv')
], ignore_index=True)
df_normal.to_csv('normal/2025-06-08_09_10_11_spans.csv', index=False)

# 合并 service 文件 
df_abnormal = pd.concat([
    pd.read_csv('service/2025-06-08_spans.csv'),
    pd.read_csv('service/2025-06-09_spans.csv'),
    pd.read_csv('service/2025-06-10_spans.csv'),
    pd.read_csv('service/2025-06-11_spans.csv')
], ignore_index=True)
df_abnormal.to_csv('service/2025-06-08_09_10_11_spans.csv', index=False)