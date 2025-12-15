import os
import pandas as pd
import sys
from typing import Set, Tuple


def get_trace_ids(csv_path: str) -> Set[str]:
    """
    读取CSV文件，提取去重后的TraceID集合（排除空值）
    :param csv_path: CSV文件路径
    :return: TraceID的集合（元素为字符串）
    """
    # 检查文件是否存在
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"文件不存在: {csv_path}")
    
    try:
        # 只读取TraceID列，提高效率（忽略低内存警告）
        df = pd.read_csv(csv_path, usecols=['TraceID'], low_memory=False)
    except Exception as e:
        raise RuntimeError(f"读取文件失败: {str(e)}")
    
    # 检查是否包含TraceID列
    if 'TraceID' not in df.columns:
        raise ValueError(f"文件 {csv_path} 中未找到'TraceID'列")
    
    # 提取非空的TraceID，转换为字符串（避免类型差异导致的比较问题），去重后转为集合
    trace_ids = df['TraceID'].dropna().astype(str).unique()
    return set(trace_ids)


def calculate_trace_stats(trace_set1: Set[str], trace_set2: Set[str]) -> Tuple[int, int]:
    """
    计算两个TraceID集合的不同总数和重复数量
    :param trace_set1: 第一个文件的TraceID集合
    :param trace_set2: 第二个文件的TraceID集合
    :return: (不同的Trace总数, 重复的Trace数量)
    """
    # 不同的Trace总数 = 两个集合的并集大小
    total_unique = len(trace_set1.union(trace_set2))
    # 重复的Trace数量 = 两个集合的交集大小
    duplicate_count = len(trace_set1.intersection(trace_set2))
    return total_unique, duplicate_count


def main():
    # 检查命令行参数（需要传入两个CSV文件路径）
    if len(sys.argv) != 3:
        print("用法: python trace_duplicate_check.py <第一个CSV文件路径> <第二个CSV文件路径>")
        sys.exit(1)
    
    csv1_path = sys.argv[1].strip()
    csv2_path = sys.argv[2].strip()
    
    try:
        # 1. 提取两个文件的TraceID集合
        print(f"正在读取第一个文件: {csv1_path}")
        trace_set1 = get_trace_ids(csv1_path)
        print(f"第一个文件中共有 {len(trace_set1)} 个不重复的TraceID")
        
        print(f"\n正在读取第二个文件: {csv2_path}")
        trace_set2 = get_trace_ids(csv2_path)
        print(f"第二个文件中共有 {len(trace_set2)} 个不重复的TraceID")
        
        # 2. 计算统计结果
        total_unique, duplicate_count = calculate_trace_stats(trace_set1, trace_set2)
        
        # 3. 输出结果
        print("\n" + "="*60)
        print(f"两个文件中不同的Trace总数: {total_unique}")
        print(f"两个文件中重复的Trace数量: {duplicate_count}")
        print("="*60)
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()