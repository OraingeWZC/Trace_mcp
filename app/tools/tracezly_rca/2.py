import pandas as pd
import json
from datetime import datetime, timedelta
import pytz
import numpy as np
import gc

def get_normal_traces_far_from_faults_optimized(data_path, target_count):
    """优化版本：使用分块处理和内存管理"""
    
    try:
        print("开始读取数据...")
        # 分块读取大文件
        chunk_size = 100000
        chunks = []
        
        for chunk in pd.read_csv(data_path, chunksize=chunk_size, low_memory=False):
            normal_chunk = chunk[chunk['Anomaly'] == False].copy()
            if len(normal_chunk) > 0:
                chunks.append(normal_chunk)
            del chunk  # 释放内存
            gc.collect()
        
        if not chunks:
            print("没有找到正常数据")
            return None, None
            
        print(f"读取了 {len(chunks)} 个数据块")
        
        # 合并数据块
        print("合并数据块...")
        normal_df = pd.concat(chunks, ignore_index=True)
        del chunks  # 释放内存
        gc.collect()
        
        print(f"总正常数据: {len(normal_df['TraceID'].unique())} 个Trace")
        
        # 检查必要的列
        required_cols = ['TraceID', 'StartTime', 'EndTime', 'Anomaly']
        missing_cols = [col for col in required_cols if col not in normal_df.columns]
        if missing_cols:
            print(f"缺少必要的列: {missing_cols}")
            return None, None
        
        # 数据类型转换和异常值处理
        print("处理数据类型...")
        normal_df['StartTime'] = pd.to_numeric(normal_df['StartTime'], errors='coerce')
        normal_df['EndTime'] = pd.to_numeric(normal_df['EndTime'], errors='coerce')
        
        # 移除有异常值的行
        normal_df = normal_df.dropna(subset=['StartTime', 'EndTime'])
        
        if len(normal_df) == 0:
            print("所有数据都包含异常值")
            return None, None
        
        # 获取正常数据的时间范围（转换为秒级时间戳）
        normal_start_time = normal_df['StartTime'].min() / 1000
        normal_end_time = normal_df['EndTime'].max() / 1000
        
        print(f"正常数据时间范围: {datetime.fromtimestamp(normal_start_time)} 到 {datetime.fromtimestamp(normal_end_time)}")
        
        # 读取故障时间段
        fault_periods = []
        try:
            with open('groundtruths.jsonl', 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():  # 跳过空行
                        try:
                            fault_data = json.loads(line.strip())
                            start_time = datetime.fromisoformat(fault_data['start_time'].replace('Z', '+00:00'))
                            end_time = datetime.fromisoformat(fault_data['end_time'].replace('Z', '+00:00'))
                            
                            fault_start_timestamp = start_time.timestamp()
                            fault_end_timestamp = end_time.timestamp()
                            
                            if fault_start_timestamp <= normal_end_time and fault_end_timestamp >= normal_start_time:
                                fault_periods.append((fault_start_timestamp, fault_end_timestamp))
                                print(f"包含故障时间段: {start_time} 到 {end_time}")
                        except (json.JSONDecodeError, KeyError, ValueError) as e:
                            print(f"跳过无效的故障数据行: {e}")
                            continue
        except FileNotFoundError:
            print("未找到 groundtruths.jsonl 文件")
            fault_periods = []
        
        print(f"发现 {len(fault_periods)} 个位于正常数据时间范围内的故障时间段")
        
        # 计算trace统计信息
        print("计算trace统计信息...")
        try:
            trace_stats = normal_df.groupby('TraceID').agg({
                'StartTime': ['min', 'count'],
                'EndTime': 'max'
            }).reset_index()
            
            trace_stats.columns = ['TraceID', 'start_time', 'span_count', 'end_time']
            trace_stats['start_time'] = trace_stats['start_time'] / 1000
            trace_stats['end_time'] = trace_stats['end_time'] / 1000
            
        except Exception as e:
            print(f"计算trace统计时出错: {e}")
            return None, None
        
        # 如果没有故障时间段，直接返回
        if len(fault_periods) == 0:
            print("没有故障时间段，返回所有正常数据")
            trace_stats['distance_to_fault'] = float('inf')
            selected_count = min(target_count, len(trace_stats))
            selected_traces = trace_stats.head(selected_count)
            selected_trace_ids = selected_traces['TraceID'].tolist()
            final_data = normal_df[normal_df['TraceID'].isin(selected_trace_ids)]
            
            output_file = 'normal_traces_far_from_faults.csv'
            final_data.to_csv(output_file, index=False)
            print(f"数据已保存到: {output_file}")
            return final_data, selected_traces
        
        # 分批计算距离以避免内存问题
        print("分批计算距离...")
        fault_array = np.array(fault_periods)
        batch_size = 10000
        n_traces = len(trace_stats)
        all_distances = []
        
        for i in range(0, n_traces, batch_size):
            end_idx = min(i + batch_size, n_traces)
            batch_traces = trace_stats.iloc[i:end_idx]
            
            print(f"处理批次 {i//batch_size + 1}/{(n_traces-1)//batch_size + 1}")
            
            try:
                batch_distances = calculate_batch_distances(
                    batch_traces['start_time'].values,
                    batch_traces['end_time'].values,
                    fault_array
                )
                all_distances.extend(batch_distances)
            except Exception as e:
                print(f"计算距离时出错: {e}")
                # 使用默认距离
                all_distances.extend([0.0] * len(batch_traces))
            
            gc.collect()  # 释放内存
        
        trace_stats['distance_to_fault'] = all_distances
        
        # 排序和选择
        trace_stats = trace_stats.sort_values('distance_to_fault', ascending=False)
        
        print(f"可用的normal traces: {len(trace_stats)}")
        if len(trace_stats) > 0:
            print(f"距离故障最远的trace距离: {trace_stats.iloc[0]['distance_to_fault'] / 3600:.2f} 小时")
            print(f"距离故障最近的trace距离: {trace_stats.iloc[-1]['distance_to_fault'] / 3600:.2f} 小时")
        
        # 选择traces
        selected_count = min(target_count, len(trace_stats))
        selected_traces = trace_stats.head(selected_count)
        
        print(f"\n选择了 {selected_count} 个traces")

        print(f"在选择的trace中的距离故障的距离")
        if len(selected_traces) > 0:
            print(f"距离故障最远的trace距离: {selected_traces.iloc[0]['distance_to_fault'] / 3600:.2f} 小时")
            print(f"距离故障最近的trace距离: {selected_traces.iloc[-1]['distance_to_fault'] / 3600:.2f} 小时")

        
        # 获取最终数据
        selected_trace_ids = selected_traces['TraceID'].tolist()
        final_data = normal_df[normal_df['TraceID'].isin(selected_trace_ids)]
        
        print(f"最终数据量: {len(final_data)} 条记录")
        
        # 保存结果
        output_file = data_path.split('_spans.csv')[0] + '_normal_traces.csv'
        final_data.to_csv(output_file, index=False)
        print(f"\n数据已保存到: {output_file}")
        
        return final_data, selected_traces
        
    except Exception as e:
        print(f"发生未预期的错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def calculate_batch_distances(trace_starts, trace_ends, fault_array):
    """分批计算距离，避免内存问题"""
    n_traces = len(trace_starts)
    n_faults = len(fault_array)
    
    if n_faults == 0:
        return [float('inf')] * n_traces
    
    min_distances = []
    
    for i in range(n_traces):
        trace_start = trace_starts[i]
        trace_end = trace_ends[i]
        
        min_dist = float('inf')
        
        for j in range(n_faults):
            fault_start = fault_array[j, 0]
            fault_end = fault_array[j, 1]
            
            # 检查重叠
            if trace_start <= fault_end and trace_end >= fault_start:
                min_dist = 0.0
                break
            
            # 计算距离
            if trace_end < fault_start:
                dist = fault_start - trace_end
            elif trace_start > fault_end:
                dist = trace_start - fault_end
            else:
                dist = 0.0
            
            min_dist = min(min_dist, dist)
        
        min_distances.append(min_dist)
    
    return min_distances

# 执行
if __name__ == "__main__":
    try:
        data_path = 'normal/2025-06-07_spans.csv'
        result_data, trace_info = get_normal_traces_far_from_faults_optimized(
            data_path=data_path, 
            target_count=60000
        )
        
        if result_data is not None:
            print("处理完成!")
        else:
            print("处理失败!")
            
    except Exception as e:
        print(f"主程序错误: {e}")
        import traceback
        traceback.print_exc()