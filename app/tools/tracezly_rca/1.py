import pandas as pd
import gc

try:
    # 预处理normal数据
    normal_path = 'normal/2025-06-06_spans.csv'
    abnormal_path = 'service/2025-06-06_spans.csv'

    print("正在读取CSV文件...")
    # 读取CSV文件，指定低内存模式
    normal_df = pd.read_csv(normal_path, low_memory=False)
    abnormal_df = pd.read_csv(abnormal_path, low_memory=False)
    
    print(f"Normal数据形状: {normal_df.shape}")
    print(f"Abnormal数据形状: {abnormal_df.shape}")
    
    print("正在处理normal数据...")
    # 检查必要的列是否存在
    required_cols = ['EndTimeMs', 'StartTimeMs', 'URL', 'SpanId']
    for col in required_cols:
        if col not in normal_df.columns:
            print(f"警告: normal数据中缺少列 '{col}'")
        if col not in abnormal_df.columns:
            print(f"警告: abnormal数据中缺少列 '{col}'")
    
    # 数据处理步骤 - 分步处理以避免内存问题
    if 'EndTimeMs' in normal_df.columns and 'StartTimeMs' in normal_df.columns:
        normal_df['Duration'] = normal_df['EndTimeMs'] - normal_df['StartTimeMs']
    
    if 'EndTimeMs' in abnormal_df.columns and 'StartTimeMs' in abnormal_df.columns:
        abnormal_df['Duration'] = abnormal_df['EndTimeMs'] - abnormal_df['StartTimeMs']

    if 'URL' in normal_df.columns:
        normal_df['OperationName'] = normal_df['URL']
    
    if 'URL' in abnormal_df.columns:
        abnormal_df['OperationName'] = abnormal_df['URL']

    # 重命名列 - 只重命名存在的列
    normal_rename_dict = {}
    abnormal_rename_dict = {}
    
    if 'EndTimeMs' in normal_df.columns:
        normal_rename_dict['EndTimeMs'] = 'EndTime'
    if 'StartTimeMs' in normal_df.columns:
        normal_rename_dict['StartTimeMs'] = 'StartTime'
    if 'SpanId' in normal_df.columns:
        normal_rename_dict['SpanId'] = 'SpanID'
    
    if 'EndTimeMs' in abnormal_df.columns:
        abnormal_rename_dict['EndTimeMs'] = 'EndTime'
    if 'StartTimeMs' in abnormal_df.columns:
        abnormal_rename_dict['StartTimeMs'] = 'StartTime'
    if 'SpanId' in abnormal_df.columns:
        abnormal_rename_dict['SpanId'] = 'SpanID'
    
    if normal_rename_dict:
        normal_df = normal_df.rename(columns=normal_rename_dict)
    if abnormal_rename_dict:
        abnormal_df = abnormal_df.rename(columns=abnormal_rename_dict)

    # 添加异常标记
    normal_df['Anomaly'] = False
    abnormal_df['Anomaly'] = True

    # 重命名fault相关列（如果存在）
    fault_rename_dict = {}
    if 'fault_type' in normal_df.columns:
        fault_rename_dict['fault_type'] = 'FaultCategory'
    if 'fault_instance' in normal_df.columns:
        fault_rename_dict['fault_instance'] = 'RootCause'
    
    if fault_rename_dict:
        normal_df = normal_df.rename(columns=fault_rename_dict)

    fault_rename_dict = {}
    if 'fault_type' in abnormal_df.columns:
        fault_rename_dict['fault_type'] = 'FaultCategory'
    if 'fault_instance' in abnormal_df.columns:
        fault_rename_dict['fault_instance'] = 'RootCause'
    
    if fault_rename_dict:
        abnormal_df = abnormal_df.rename(columns=fault_rename_dict)

    print("正在保存文件...")
    # 分批保存以避免内存问题
    normal_df.to_csv(normal_path, index=False, chunksize=10000)
    print("Normal数据保存完成")
    
    abnormal_df.to_csv(abnormal_path, index=False, chunksize=10000)
    print("Abnormal数据保存完成")
    
    # 清理内存
    del normal_df, abnormal_df
    gc.collect()
    
    print("数据预处理完成！")

except MemoryError:
    print("内存不足，请尝试处理更小的数据块")
except FileNotFoundError as e:
    print(f"文件未找到: {e}")
except KeyError as e:
    print(f"列不存在: {e}")
except Exception as e:
    print(f"发生错误: {e}")
    import traceback
    traceback.print_exc()