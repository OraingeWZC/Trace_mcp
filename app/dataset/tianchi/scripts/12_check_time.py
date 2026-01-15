import pandas as pd
from datetime import datetime, timedelta
import os
import numpy as np

def ms_timestamp_to_beijing(ms_timestamp):
    """
    将毫秒级Unix时间戳转换为北京时间字符串 (UTC+8)
    :param ms_timestamp: 毫秒级时间戳（数值型）
    :return: 格式化的北京时间字符串，格式：YYYY-MM-DD HH:MM:SS.ms
    """
    try:
        # 毫秒转秒（保留毫秒精度）
        sec_timestamp = ms_timestamp / 1000
        
        # 1. 获取 UTC 时间
        utc_datetime = datetime.utcfromtimestamp(sec_timestamp)
        
        # 2. 转换为北京时间 (UTC + 8小时)
        beijing_datetime = utc_datetime + timedelta(hours=8)
        
        # 3. 格式化（保留3位毫秒）
        ms = int((sec_timestamp - int(sec_timestamp)) * 1000)
        return beijing_datetime.strftime(f"%Y-%m-%d %H:%M:%S.{ms:03d}")
    except Exception as e:
        return f"转换失败: {str(e)}"

def process_csv_time_columns(csv_path):
    """
    处理CSV文件，提取指定列的极值并转换为北京时间（内存优化版）
    :param csv_path: CSV文件路径
    """
    # 1. 检查文件是否存在
    if not os.path.exists(csv_path):
        print(f"错误：文件 {csv_path} 不存在！")
        return

    print(f"正在分析文件: {csv_path} ...")

    # 初始化全局极值
    global_start_min = float('inf')
    global_start_max = float('-inf')
    global_end_min = float('inf')
    global_end_max = float('-inf')
    
    found_data = False
    
    # 2. 分块读取 CSV (防止内存溢出)
    # chunksize=100000 表示每次只读 10万行到内存
    try:
        reader = pd.read_csv(csv_path, usecols=['StartTimeMs', 'EndTimeMs'], chunksize=100000)
        
        for i, chunk in enumerate(reader):
            # 3. 数据清洗 (转为数字，非数字转NaN)
            chunk['StartTimeMs'] = pd.to_numeric(chunk['StartTimeMs'], errors='coerce')
            chunk['EndTimeMs'] = pd.to_numeric(chunk['EndTimeMs'], errors='coerce')
            
            # 删除空值
            chunk.dropna(subset=['StartTimeMs', 'EndTimeMs'], inplace=True)
            
            if chunk.empty:
                continue
                
            found_data = True
            
            # 4. 更新当前块的极值
            c_start_min = chunk['StartTimeMs'].min()
            c_start_max = chunk['StartTimeMs'].max()
            c_end_min = chunk['EndTimeMs'].min()
            c_end_max = chunk['EndTimeMs'].max()
            
            # 5. 更新全局极值
            if c_start_min < global_start_min: global_start_min = c_start_min
            if c_start_max > global_start_max: global_start_max = c_start_max
            if c_end_min < global_end_min: global_end_min = c_end_min
            if c_end_max > global_end_max: global_end_max = c_end_max
            
            # print(f"  已处理批次 {i+1}...", end='\r')

    except Exception as e:
        print(f"\n读取文件出错: {e}")
        return

    if not found_data:
        print("\n错误：文件中没有有效的 StartTimeMs/EndTimeMs 数据。")
        return

    # 6. 转换为北京时间
    result = {
        "StartTimeMs 最小值": {
            "时间戳(ms)": global_start_min,
            "北京时间": ms_timestamp_to_beijing(global_start_min)
        },
        "StartTimeMs 最大值": {
            "时间戳(ms)": global_start_max,
            "北京时间": ms_timestamp_to_beijing(global_start_max)
        },
        "EndTimeMs 最小值": {
            "时间戳(ms)": global_end_min,
            "北京时间": ms_timestamp_to_beijing(global_end_min)
        },
        "EndTimeMs 最大值": {
            "时间戳(ms)": global_end_max,
            "北京时间": ms_timestamp_to_beijing(global_end_max)
        }
    }

    # 7. 输出结果
    print("\n" + "=" * 60)
    print("CSV时间列极值及北京时间转换结果：")
    print("=" * 60)
    for key, value in result.items():
        print(f"{key}:")
        print(f"  原始时间戳(ms)：{value['时间戳(ms)']}")
        print(f"  转换结果     ：{value['北京时间']}")
    print("=" * 60)

# ------------------- 主执行区 -------------------
if __name__ == "__main__":
    # 请修改为你的CSV文件路径（绝对路径/相对路径均可）
    CSV_FILE_PATH = "/root/wzc/Trace_mcp/app/dataset/tianchi/data/NormalData/normal_traces2e5_30s_6h.csv"  # 示例："D:/data/time_records.csv"
    
    # 执行处理
    process_csv_time_columns(CSV_FILE_PATH)