import pandas as pd
import random

# 合并正常和异常数据

def merge_csvs(normal_file, anomaly_file, output_file):
    """合并正常和异常数据"""
    
    # 设置随机种子
    random.seed(42)
    
    print("加载正常数据...")
    normal_df = pd.read_csv(normal_file)
    normal_df = normal_df.drop(columns=['Normalized_StartTime'])
    normal_df = normal_df.drop(columns=['Normalized_EndTime'])
    print(normal_df.columns)
    print(f"正常数据: {len(normal_df)} 行, {normal_df['TraceID'].nunique()} 个Trace")
    
    print("加载异常数据...")
    anomaly_df = pd.read_csv(anomaly_file)
    anomaly_df = anomaly_df[anomaly_df['Anomaly'] == 1]
    anomaly_df = anomaly_df.drop(columns=['Normalized_StartTime'])
    anomaly_df = anomaly_df.drop(columns=['Normalized_EndTime'])
    print(anomaly_df.columns)
    print(f"异常数据: {len(anomaly_df)} 行, {anomaly_df['TraceID'].nunique()} 个Trace")
    
    # 处理正常数据
    print("处理正常数据...")
    normal_df['Anomaly'] = 0

    # 处理异常数据
    print("处理异常数据...")
    # 确保列名一致
    if 'SpanId' in anomaly_df.columns:
        anomaly_df = anomaly_df.rename(columns={'SpanId': 'SpanID'})
    
    
    # 合并数据
    print("合并数据...")
    merged_df = pd.concat([normal_df, anomaly_df], ignore_index=True)
    print(f"合并后: {len(merged_df)} 行, {merged_df['TraceID'].nunique()} 个Trace")
    
    # 删除节点数小于2的图
    print("删除节点数小于2的图...")
    trace_counts = merged_df['TraceID'].value_counts()
    valid_traces = trace_counts[trace_counts >= 2].index
    
    # 统计被删除的数据
    filtered_out = merged_df[~merged_df['TraceID'].isin(valid_traces)]
    print(f"删除了 {len(filtered_out)} 行数据，{filtered_out['TraceID'].nunique()} 个Trace（节点数<2）")
    
    # 过滤数据
    merged_df = merged_df[merged_df['TraceID'].isin(valid_traces)]
    print(f"过滤后: {len(merged_df)} 行, {merged_df['TraceID'].nunique()} 个Trace")
    
    # 保存合并后的数据
    merged_df.to_csv(output_file, index=False)
    print(f"已保存到 {output_file}")
    
    # 统计异常分布
    anomaly_count = merged_df['Anomaly'].sum()
    total_count = len(merged_df)
    print(f"异常比例: {anomaly_count/total_count:.3f}")
    
    # 按TraceID统计异常分布
    trace_anomaly_stats = merged_df.groupby('TraceID')['Anomaly'].first()
    normal_traces = (trace_anomaly_stats == False).sum()
    anomaly_traces = (trace_anomaly_stats == True).sum()
    print(f"Trace级别统计: 正常 {normal_traces} 个, 异常 {anomaly_traces} 个")
    
    # 分别统计正常和异常数据的节点数分布
    print("\n" + "="*50)
    print("节点数分布统计")
    print("="*50)
    
    # 整体节点数分布
    final_trace_counts = merged_df['TraceID'].value_counts()
    print(f"整体节点数分布:")
    print(f"最小节点数: {final_trace_counts.min()}")
    print(f"最大节点数: {final_trace_counts.max()}")
    print(f"平均节点数: {final_trace_counts.mean():.2f}")
    
    # 获取每个trace的异常标签和节点数
    trace_info = merged_df.groupby('TraceID').agg({
        'Anomaly': 'first',  # 每个trace的异常标签
        'SpanID': 'count'    # 每个trace的节点数
    }).rename(columns={'SpanID': 'NodeCount'})
    
    # 正常数据节点数分布
    normal_node_counts = trace_info[trace_info['Anomaly'] == 0]['NodeCount']
    print(f"\n正常数据节点数分布:")
    print(f"数量: {len(normal_node_counts)} 个Trace")
    print(f"最小节点数: {normal_node_counts.min()}")
    print(f"最大节点数: {normal_node_counts.max()}")
    print(f"平均节点数: {normal_node_counts.mean():.2f}")
    
    # 异常数据节点数分布
    anomaly_node_counts = trace_info[trace_info['Anomaly'] == 1]['NodeCount']
    print(f"\n异常数据节点数分布:")
    print(f"数量: {len(anomaly_node_counts)} 个Trace")
    print(f"最小节点数: {anomaly_node_counts.min()}")
    print(f"最大节点数: {anomaly_node_counts.max()}")
    print(f"平均节点数: {anomaly_node_counts.mean():.2f}")

if __name__ == "__main__":
    normal_file = 'normal/2025-06-07_normal_traces.csv' 
    anomaly_file = 'service/2025-06-07_spans.csv' 
    output_file= 'output/'+ anomaly_file.split('/')[-1].split('_spans.csv')[0] + '_merged_traces.csv'
    merge_csvs(normal_file, anomaly_file, output_file)