import csv
import argparse
from collections import defaultdict
'''
加载数据：按 TraceID 对所有 Span 进行分组。
原始统计：统计过滤前，不同 Span 数量区间的 Trace 分布。
识别并过滤：识别出“多根节点”的 Trace（根节点定义：ParentID == -1 或 ParentID 在该 Trace 中找不到对应的 SpanID）。
过滤后统计：统计剔除多根节点后，剩余“单根纯净” Trace 的区间分布。
'''

def get_interval_name(count):
    """根据节点数返回所属区间名称"""
    if count == 2: return "2个节点"
    if count == 3: return "3个节点"
    if count == 4: return "4个节点"
    if count == 5: return "5个节点"
    if 6 <= count <= 10: return "6~10个节点"
    if 11 <= count <= 15: return "11~15个节点"
    if count > 15: return "15个节点以上"
    return "其他 (1个节点)"

def analyze_traces(csv_file_path, trace_col="TraceID", span_col="SpanID", parent_col="ParentID"):
    # 1. 存储结构：trace_id -> {'spans': set(SpanIDs), 'parents': list(ParentIDs)}
    # 使用 set 存储 SpanID 方便后续判断 ParentID 是否存在
    trace_map = defaultdict(lambda: {'span_ids': set(), 'parent_ids': []})
    
    print(f"📖 正在读取文件: {csv_file_path} ...")
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                tid = row[trace_col].strip()
                sid = row[span_col].strip()
                pid = row[parent_col].strip()
                
                if tid and sid:
                    trace_map[tid]['span_ids'].add(sid)
                    trace_map[tid]['parent_ids'].append(pid)
    except Exception as e:
        print(f"❌ 读取错误: {e}")
        return

    # 2. 核心逻辑：计算每个 Trace 的节点数和根节点数
    # results 存储: { trace_id: (node_count, root_count) }
    results = {}
    for tid, data in trace_map.items():
        span_ids = data['span_ids']
        node_count = len(span_ids)
        
        # 根节点判定：ParentID == -1 或 ParentID 不在当前 Trace 的 SpanID 集合中
        root_count = 0
        for pid in data['parent_ids']:
            if pid == "-1" or pid not in span_ids:
                root_count += 1
        
        results[tid] = {
            'node_count': node_count,
            'is_multi_root': root_count >= 2
        }

    # 3. 定义区间顺序用于打印
    interval_order = [
        "2个节点", "3个节点", "4个节点", "5个节点", 
        "6~10个节点", "11~15个节点", "15个节点以上"
    ]

    # 4. 统计：过滤前
    before_stats = defaultdict(int)
    for res in results.values():
        interval = get_interval_name(res['node_count'])
        before_stats[interval] += 1

    # 5. 统计：过滤后 (只统计非多根的)
    after_stats = defaultdict(int)
    multi_root_count = 0
    for res in results.values():
        if res['is_multi_root']:
            multi_root_count += 1
            continue
        interval = get_interval_name(res['node_count'])
        after_stats[interval] += 1

    # 6. 输出结果
    total_traces = len(results)
    
    print("\n" + "="*70)
    print(f"📊 Trace 综合分析报告")
    print("="*70)
    print(f"📈 原始总 Trace 数量: {total_traces}")
    print(f"🔴 多根节点 Trace 数量: {multi_root_count} ({multi_root_count/total_traces*100:.2f}%)")
    print(f"✅ 纯净单根 Trace 数量: {total_traces - multi_root_count}")
    print("-" * 70)

    print(f"{'区间范围':<20} | {'过滤前 (原始)':<15} | {'过滤后 (去除多根)':<15}")
    print("-" * 70)
    
    for interval in interval_order:
        b_val = before_stats.get(interval, 0)
        a_val = after_stats.get(interval, 0)
        print(f"{interval:<24} | {b_val:<15} | {a_val:<15}")
    
    print("="*70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="整合统计：节点区间分布 + 多根节点过滤")
    parser.add_argument("-f", "--file", default="normal_traces.csv", help="CSV文件路径")
    parser.add_argument("--trace-col", default="TraceID", help="TraceID列名")
    parser.add_argument("--span-col", default="SpanId", help="SpanID列名")
    parser.add_argument("--parent-col", default="ParentID", help="ParentID列名")
    
    args = parser.parse_args()
    
    analyze_traces(
        csv_file_path=args.csv_file,
        trace_col=args.trace_col,
        span_col=args.span_col,
        parent_col=args.parent_col
    )