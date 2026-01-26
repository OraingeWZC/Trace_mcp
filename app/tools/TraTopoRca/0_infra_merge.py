import pandas as pd
import os
import numpy as np

# ================= 配置路径 =================
# 原始的两个天池指标文件
FILE_NORMAL = 'dataset/tianchi/normal_metrics_2e5_1622.csv'
FILE_FAULT  = 'dataset/tianchi/all_metrics_30s.csv'

# 目标输出路径 (模拟 AIOps 的标准命名)
OUTPUT_DIR  = 'dataset/tianchi'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'merged_all_infra.csv')
# ===========================================

def merge_csv():
    print(f"🚀 开始合并指标文件...")
    
    # 1. 读取
    if not os.path.exists(FILE_NORMAL) or not os.path.exists(FILE_FAULT):
        print("❌ 错误：找不到原始 CSV 文件，请检查路径。")
        return
    
    # 读取时指定类型，防止大数被截断
    df1 = pd.read_csv(FILE_NORMAL)
    df2 = pd.read_csv(FILE_FAULT)
    print(f"   - Normal 集: {len(df1)} 行")
    print(f"   - Fault 集:  {len(df2)} 行")

    # 2. 合并
    df_merged = pd.concat([df1, df2], ignore_index=True)
    
    # 3. 标准化关键列 (这步非常重要！)
    print("🛠 正在标准化关键列...")
    
    # 3.1 统一时间戳列名 -> timeMs (毫秒)
    if 'timeMs' not in df_merged.columns:
        if 'timestamp' in df_merged.columns:
            # 天池数据通常是纳秒，需要转毫秒
            df_merged['timeMs'] = df_merged['timestamp'].astype(np.int64) // 1000000
        elif 'time' in df_merged.columns:
            df_merged['timeMs'] = pd.to_datetime(df_merged['time']).astype('int64') // 10**6
    
    # 3.2 统一主机名列名 -> kubernetes_node
    if 'kubernetes_node' not in df_merged.columns:
        if 'instance_id' in df_merged.columns:
            df_merged['kubernetes_node'] = df_merged['instance_id'].astype(str)

    # 4. 去重 (防止两个文件有时间重叠)
    print("🧹 正在去重...")
    if 'timeMs' in df_merged.columns and 'kubernetes_node' in df_merged.columns:
        df_merged.drop_duplicates(subset=['timeMs', 'kubernetes_node'], keep='last', inplace=True)
    else:
        print("⚠️ 警告：关键列缺失，跳过去重")

    # 5. 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"💾 正在保存到: {OUTPUT_FILE}")
    # index=False 很重要，避免生成多余的 Unnamed: 0 列
    df_merged.to_csv(OUTPUT_FILE, index=False)
    
    print(f"✅ 合并完成！总行数: {len(df_merged)}")

if __name__ == '__main__':
    merge_csv()