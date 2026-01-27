import csv
import random
import os
from collections import defaultdict

# 划分数据集

def split_data(merged_file):
    """按约束分割数据"""
    
    random.seed(42)
    
    print("加载合并后的数据...")
    
    # 使用字典统计Trace特征，避免加载整个DataFrame到内存
    trace_stats = {}
    trace_anomaly_map = {}
    total_rows = 0
    
    print("统计Trace特征...")
    normal_count = 0
    anomaly_count = 0
    
    with open(merged_file, 'r') as f:
        reader = csv.DictReader(f)
        print(f"CSV列名: {reader.fieldnames}")
        
        for row in reader:
            total_rows += 1
            trace_id = row['TraceID']
            span_id = row['SpanID']
            
            # 更严格的异常判断
            anomaly_val = str(row.get('Anomaly', '0')).strip().lower()
            is_anomaly = anomaly_val in ['1', 'true', '1.0']
            
            if is_anomaly:
                anomaly_count += 1
            else:
                normal_count += 1
            
            # 统计每个Trace的SpanID数量和异常状态
            if trace_id not in trace_stats:
                trace_stats[trace_id] = set()
                trace_anomaly_map[trace_id] = False
            
            # 关键修正：只要trace中有任何一个span是异常的，整个trace就标记为异常
            if is_anomaly:
                trace_anomaly_map[trace_id] = True
            
            trace_stats[trace_id].add(span_id)
    
    print(f"行级别统计: 正常 {normal_count} 行, 异常 {anomaly_count} 行")
    print(f"总数据: {total_rows} 行, {len(trace_stats)} 个Trace")
    
    # 调试：检查SpanID的唯一性
    print("\n=== 调试信息：检查前10个Trace的SpanID ===")
    debug_count = 0
    for trace_id, span_ids in trace_stats.items():
        if debug_count >= 10:
            break
        print(f"Trace {trace_id}: {len(span_ids)} 个唯一SpanID")
        if len(span_ids) <= 3:  # 只显示节点数较少的trace的详细信息
            print(f"  SpanIDs: {list(span_ids)}")
        debug_count += 1
    
    # 统计Trace级别的正常/异常分布
    trace_normal_count = sum(1 for is_anom in trace_anomaly_map.values() if not is_anom)
    trace_anomaly_count = sum(1 for is_anom in trace_anomaly_map.values() if is_anom)
    print(f"Trace级别统计: 正常 {trace_normal_count} 个, 异常 {trace_anomaly_count} 个")
    
    # 过滤节点数小于2的Trace
    valid_traces = []
    node_count_distribution = {}
    single_node_traces = []  # 调试：记录单节点traces
    
    for trace_id, span_ids in trace_stats.items():
        node_count = len(span_ids)
        is_anomaly = trace_anomaly_map[trace_id]
        
        # 统计节点数分布
        if node_count not in node_count_distribution:
            node_count_distribution[node_count] = {'normal': 0, 'anomaly': 0}
        
        if is_anomaly:
            node_count_distribution[node_count]['anomaly'] += 1
        else:
            node_count_distribution[node_count]['normal'] += 1
        
        # 记录单节点traces用于调试
        if node_count == 1:
            single_node_traces.append((trace_id, is_anomaly))
        
        if node_count >= 2:  # 删除节点数小于2的图
            valid_traces.append((trace_id, node_count, is_anomaly))
    
    print(f"有效Trace: {len(valid_traces)} 个")
    print(f"单节点Trace: {len(single_node_traces)} 个")
    
    # 显示节点数分布
    print("节点数分布:")
    for node_count in sorted(node_count_distribution.keys())[:20]:
        dist = node_count_distribution[node_count]
        total_for_count = dist['normal'] + dist['anomaly']
        print(f"  {node_count}节点: 正常{dist['normal']}个, 异常{dist['anomaly']}个, 总计{total_for_count}个")
    
    # 分离正常和异常
    normal_traces = [t for t in valid_traces if not t[2]]
    anomaly_traces = [t for t in valid_traces if t[2]]
    
    print(f"正常Trace: {len(normal_traces)} 个")
    print(f"异常Trace: {len(anomaly_traces)} 个")
    
    if len(normal_traces) == 0:
        print("错误: 没有找到正常的Trace!")
        return
    
    # 目标数量
    target_train = 45000
    target_val = 4500
    target_test = 4000
    test_anomaly_rate = 0.50
    
    # 检查数据是否充足
    total_needed = target_train + target_val + target_test
    if len(valid_traces) < total_needed:
        print(f"警告: 总Trace数({len(valid_traces)})小于需求({total_needed})")
        print("将按比例缩放目标数量...")
        scale_factor = len(valid_traces) / total_needed
        target_train = int(target_train * scale_factor)
        target_val = int(target_val * scale_factor)
        target_test = len(valid_traces) - target_train - target_val
        print(f"调整后: 训练{target_train}, 验证{target_val}, 测试{target_test}")
    
    # 选择测试集异常样本
    if len(anomaly_traces) > 0:
        desired_anom = min(int(round(target_test * test_anomaly_rate)), len(anomaly_traces))
        random.shuffle(anomaly_traces)
        test_anom_ids = [t[0] for t in anomaly_traces[:desired_anom]]
    else:
        test_anom_ids = []
        print("警告: 测试集中没有异常样本")
    
    # 修正后的测试集正常样本选择逻辑
    print("开始选择测试集正常样本...")
    normal_remaining = [t for t in normal_traces]
    needed_norm = target_test - len(test_anom_ids)
    needed_norm = min(needed_norm, len(normal_remaining))
    
    print(f"需要选择 {needed_norm} 个正常样本用于测试集")
    print(f"可用正常样本: {len(normal_remaining)} 个")
    
    # 按节点数分组正常样本
    normal_by_nodes = defaultdict(list)
    for trace in normal_remaining:
        node_count = trace[1]
        normal_by_nodes[node_count].append(trace)
    
    print("正常样本按节点数分组:")
    for node_count in sorted(normal_by_nodes.keys())[:10]:
        count = len(normal_by_nodes[node_count])
        print(f"  {node_count}节点: {count} 个")
    
    # 新的测试集选择策略：确保多样性
    test_norm_selected = []
    
    # 第一步：从每种节点数中选择一定数量，确保多样性
    target_per_node_count = max(1, needed_norm // len(normal_by_nodes))
    print(f"目标每种节点数选择: {target_per_node_count} 个")
    
    # 从每种节点数中选择样本
    remaining_need = needed_norm
    for node_count in sorted(normal_by_nodes.keys()):
        available = normal_by_nodes[node_count]
        if remaining_need <= 0:
            break
            
        # 计算这种节点数应该选择多少个
        select_count = min(target_per_node_count, len(available), remaining_need)
        
        # 随机选择
        random.shuffle(available)
        selected = available[:select_count]
        test_norm_selected.extend(selected)
        
        remaining_need -= select_count
        print(f"从{node_count}节点中选择了 {select_count} 个样本")
        
        # 从可用列表中移除已选择的
        normal_by_nodes[node_count] = available[select_count:]
    
    # 第二步：如果还没达到目标数量，随机补充
    if remaining_need > 0:
        print(f"还需要补充 {remaining_need} 个样本")
        
        # 收集所有剩余的样本
        remaining_traces = []
        for traces in normal_by_nodes.values():
            remaining_traces.extend(traces)
        
        # 随机选择补充
        random.shuffle(remaining_traces)
        additional = remaining_traces[:remaining_need]
        test_norm_selected.extend(additional)
        
        print(f"随机补充了 {len(additional)} 个样本")
    
    test_norm_ids = [t[0] for t in test_norm_selected]
    test_ids = set(test_anom_ids + test_norm_ids)
    
    print(f"测试集最终: {len(test_ids)} 个 (异常: {len(test_anom_ids)}, 正常: {len(test_norm_ids)})")
    
    # 分析测试集的节点数分布
    test_normal_node_dist = defaultdict(int)
    for trace in test_norm_selected:
        node_count = trace[1]
        test_normal_node_dist[node_count] += 1
    
    print("测试集正常样本节点数分布:")
    for node_count in sorted(test_normal_node_dist.keys()):
        count = test_normal_node_dist[node_count]
        print(f"  {node_count}节点: {count} 个")
    
    # 训练集选择
    print("开始训练集选图...")
    
    # ✅ 关键修复：排除测试集已选择的traces，确保训练集和测试集不重叠
    train_normal_traces = [t for t in normal_traces if t[0] not in test_ids]
    print(f"排除测试集后，可用于训练集的正常traces: {len(train_normal_traces)} 个 (原始: {len(normal_traces)} 个)")
    
    # 按节点数分组
    traces_by_nodes = defaultdict(list)
    for trace in train_normal_traces:  # 使用排除测试集后的traces
        node_count = trace[1]
        traces_by_nodes[node_count].append(trace)
    
    # 统计2-100节点数范围内的图
    print("2-100节点数范围内的图分布:")
    target_range_traces = []
    range_coverage = {}
    
    for node_count in range(2, 100):  # 2到100节点
        if node_count in traces_by_nodes:
            available = len(traces_by_nodes[node_count])
            range_coverage[node_count] = available
            target_range_traces.extend(traces_by_nodes[node_count])
            print(f"  {node_count}节点: {available} 个")
        else:
            range_coverage[node_count] = 0
            print(f"  {node_count}节点: 0 个 (缺失)")
    
    print(f"2-100节点范围内总计: {len(target_range_traces)} 个图")
    missing_nodes = [n for n, count in range_coverage.items() if count == 0]
    if missing_nodes:
        print(f"缺失的节点数: {missing_nodes}")
    
    # 训练集选择策略
    train_selected = []
    
    # 第一步：确保2-100节点数的每种类型都有代表
    # 改进：尽可能多地选择图，尽量在第一步就满足大部分需求
    print("第一步: 确保2-100节点数覆盖，尽可能多选择...")
    
    # 计算第一步应该选择的总数（目标占总目标的50-70%，尽量多选）
    # 设置一个较高的目标，让第一步尽可能多地选择
    first_step_target = max(50000, int(target_train * 0.7))  # 至少50000个，或占总目标的70%
    
    # 统计有数据的节点数
    available_node_counts = [nc for nc in range(2, 100) if nc in traces_by_nodes]
    num_available_nodes = len(available_node_counts)
    
    if num_available_nodes > 0:
        # 基础配额：每种节点数至少选择的基础数量（提高基础配额）
        base_quota_per_node = max(200, first_step_target // num_available_nodes)
        
        # 计算每种节点数应该选择的数量（考虑可用数量）
        total_first_step = 0
        selection_plan = {}
        
        for node_count in available_node_counts:
            available_count = len(traces_by_nodes[node_count])
            # 每种节点数选择：尽可能多选，最多不超过可用数量的60-70%
            # 这样可以确保第一步选择足够多的图，同时还有样本留给第二步补充
            max_select_ratio = 0.65  # 提高到65%
            max_select = min(available_count, int(available_count * max_select_ratio))
            select_count = min(max(base_quota_per_node, max_select), available_count)
            selection_plan[node_count] = select_count
            total_first_step += select_count
        
        # 如果总数还不够目标，尝试提高比例上限，尽可能多选
        if total_first_step < first_step_target:
            # 计算还差多少
            shortage = first_step_target - total_first_step
            # 尝试逐步提高比例上限，最多到80%
            for ratio in [0.70, 0.75, 0.80]:
                total_first_step = sum(selection_plan.values())
                if total_first_step >= first_step_target:
                    break
                
                shortage = first_step_target - total_first_step
                if shortage <= 0:
                    break
                    
                for node_count in available_node_counts:
                    if total_first_step >= first_step_target:
                        break
                        
                    available_count = len(traces_by_nodes[node_count])
                    current_plan = selection_plan.get(node_count, 0)
                    # 计算在更高比例下可以多选多少
                    max_at_ratio = int(available_count * ratio)
                    if max_at_ratio > current_plan:
                        additional = min(max_at_ratio - current_plan, shortage)
                        if additional > 0:
                            selection_plan[node_count] = current_plan + additional
                            shortage -= additional
                
                # 重新计算总数
                total_first_step = sum(selection_plan.values())
        
        # 如果总数超过目标，按比例缩减（但尽量保持较多数量）
        elif total_first_step > first_step_target:
            scale_factor = first_step_target / total_first_step
            for node_count in selection_plan:
                selection_plan[node_count] = max(1, int(selection_plan[node_count] * scale_factor))
        
        # 执行选择
        for node_count in available_node_counts:
            if node_count in selection_plan:
                available_traces = traces_by_nodes[node_count]
                random.shuffle(available_traces)
                select_count = selection_plan[node_count]
                selected = available_traces[:select_count]
                train_selected.extend(selected)
                
                # 从可用列表中移除已选择的
                traces_by_nodes[node_count] = available_traces[select_count:]
                
                if select_count > 0:
                    print(f"  {node_count}节点: 选择了 {select_count} 个 (可用: {len(available_traces)})")
    
    print(f"第一步完成，已选择: {len(train_selected)} 个图 (目标: {first_step_target})")
    
    # 第二步：如果还没达到目标数量，继续选择
    # 改进：按比例选择，保持节点数平衡
    remaining_needed = target_train - len(train_selected)
    print(f"第二步: 需要补充 {remaining_needed} 个图...")
    
    if remaining_needed > 0:
        # 统计剩余traces按节点数的分布
        remaining_by_nodes = defaultdict(list)
        selected_trace_ids = {t[0] for t in train_selected}
        
        for node_count, traces in traces_by_nodes.items():
            remaining_by_nodes[node_count] = [t for t in traces if t[0] not in selected_trace_ids]
        
        # 计算每种节点数剩余的数量
        remaining_counts = {nc: len(traces) for nc, traces in remaining_by_nodes.items()}
        total_remaining = sum(remaining_counts.values())
        
        print(f"剩余可选择的图: {total_remaining} 个")
        
        if total_remaining >= remaining_needed:
            # 按比例分配：每种节点数按照其剩余数量占总剩余数量的比例来选择
            second_step_selected = []
            
            for node_count in sorted(remaining_by_nodes.keys()):
                if remaining_needed <= 0:
                    break
                    
                available = remaining_by_nodes[node_count]
                if len(available) == 0:
                    continue
                
                # 计算这种节点数应该选择的比例
                proportion = remaining_counts[node_count] / total_remaining
                target_count = int(remaining_needed * proportion)
                
                # 确保不超过可用数量，且至少选择一些
                select_count = min(target_count, len(available), remaining_needed)
                
                if select_count > 0:
                    random.shuffle(available)
                    selected = available[:select_count]
                    second_step_selected.extend(selected)
                    remaining_needed -= select_count
                    
                    # 从剩余列表中移除
                    remaining_by_nodes[node_count] = available[select_count:]
            
            # 如果还有剩余需求，从所有剩余traces中随机补充
            if remaining_needed > 0:
                all_remaining = []
                for traces in remaining_by_nodes.values():
                    all_remaining.extend(traces)
                
                if len(all_remaining) >= remaining_needed:
                    random.shuffle(all_remaining)
                    second_step_selected.extend(all_remaining[:remaining_needed])
                else:
                    second_step_selected.extend(all_remaining)
                    print(f"警告: 无法完全满足需求，实际补充了 {len(second_step_selected)} 个")
            
            train_selected.extend(second_step_selected)
            print(f"第二步完成，补充了 {len(second_step_selected)} 个图")
        else:
            # 如果剩余的图不够，全部加入
            print(f"警告: 剩余图数量不足，只能选择 {total_remaining} 个")
            for traces in remaining_by_nodes.values():
                train_selected.extend(traces)
    
    train_ids = set([t[0] for t in train_selected])
    print(f"训练集最终选择: {len(train_ids)} 个图")
    
    # 统计训练集的节点数分布
    train_node_distribution = defaultdict(int)
    for trace in train_selected:
        node_count = trace[1]
        train_node_distribution[node_count] += 1
    
    print("训练集节点数分布:")
    for node_count in sorted(train_node_distribution.keys()):
        count = train_node_distribution[node_count]
        print(f"  {node_count}节点: {count} 个")
    
    # 验证集从剩余的正常图中随机选择
    remaining_for_val = [t for t in normal_traces if t[0] not in train_ids and t[0] not in test_ids]
    random.shuffle(remaining_for_val)
    actual_val_count = min(target_val, len(remaining_for_val))
    val_ids = set([t[0] for t in remaining_for_val[:actual_val_count]])
    
    print(f"验证集: {len(val_ids)} 个")
    
    # 统计节点数分布
    def get_node_distribution(trace_ids, valid_traces):
        node_counts = [t[1] for t in valid_traces if t[0] in trace_ids]
        if node_counts:
            return {
                'min': min(node_counts),
                'max': max(node_counts),
                'avg': sum(node_counts) / len(node_counts),
                'large_graphs': len([n for n in node_counts if n >= 80]),
                'medium_graphs': len([n for n in node_counts if 20 <= n <= 79]),
                'small_graphs': len([n for n in node_counts if 8 <= n <= 19]),
                'tiny_graphs': len([n for n in node_counts if 2 <= n <= 7])
            }
        return {'min': 0, 'max': 0, 'avg': 0, 'large_graphs': 0, 'medium_graphs': 0, 'small_graphs': 0, 'tiny_graphs': 0}
    
    train_stats = get_node_distribution(train_ids, valid_traces)
    val_stats = get_node_distribution(val_ids, valid_traces)
    test_stats = get_node_distribution(test_ids, valid_traces)
    
    print(f"\n数据集划分统计:")
    print(f"  训练集: {len(train_ids)} 个Trace")
    print(f"    节点数: {train_stats['min']}-{train_stats['max']} (平均: {train_stats['avg']:.1f})")
    print(f"    微图(2-7节点): {train_stats['tiny_graphs']} 个")
    print(f"    小图(8-19节点): {train_stats['small_graphs']} 个")
    print(f"    中图(20-79节点): {train_stats['medium_graphs']} 个")
    print(f"    大图(≥80节点): {train_stats['large_graphs']} 个")
    print(f"  验证集: {len(val_ids)} 个Trace")
    print(f"    节点数: {val_stats['min']}-{val_stats['max']} (平均: {val_stats['avg']:.1f})")
    print(f"  测试集: {len(test_ids)} 个Trace")
    print(f"    节点数: {test_stats['min']}-{test_stats['max']} (平均: {test_stats['avg']:.1f})")
    print(f"    微图(2-7节点): {test_stats['tiny_graphs']} 个")
    print(f"    小图(8-19节点): {test_stats['small_graphs']} 个")
    print(f"    中图(20-79节点): {test_stats['medium_graphs']} 个")
    print(f"    大图(≥80节点): {test_stats['large_graphs']} 个")
    print(f"    异常样本: {len(test_anom_ids)} 个")
    
    print("\n保存分割后的数据...")
    
    # 提取日期部分并创建输出目录
    date_part = merged_file.split('/')[-1].split('_merged_traces.csv')[0]
    output_dir = f'4.1/tracezly_rca/processed/{date_part}'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"输出目录: {output_dir}")

    with open(merged_file, 'r', encoding='utf-8') as src_file, \
         open(f'{output_dir}/train.csv', 'w', newline='', encoding='utf-8') as train_file, \
         open(f'{output_dir}/val.csv', 'w', newline='', encoding='utf-8') as val_file, \
         open(f'{output_dir}/test.csv', 'w', newline='', encoding='utf-8') as test_file:

        reader = csv.DictReader(src_file)

        # 初始化写入器
        fieldnames = reader.fieldnames
        train_writer = csv.DictWriter(train_file, fieldnames=fieldnames)
        val_writer = csv.DictWriter(val_file, fieldnames=fieldnames)
        test_writer = csv.DictWriter(test_file, fieldnames=fieldnames)

        train_writer.writeheader()
        val_writer.writeheader()
        test_writer.writeheader()

        # 开始写入
        train_rows = 0
        val_rows = 0
        test_rows = 0
        for row in reader:
            trace_id = row['TraceID']
            if trace_id in train_ids:
                train_writer.writerow(row)
                train_rows += 1
            elif trace_id in val_ids:
                val_writer.writerow(row)
                val_rows += 1
            elif trace_id in test_ids:
                test_writer.writerow(row)
                test_rows += 1

        # 打印结果
        print(f"已保存分割后的数据到 {output_dir}/")
        print(f"  训练集: {train_rows} 行")
        print(f"  验证集: {val_rows} 行")
        print(f"  测试集: {test_rows} 行")

if __name__ == "__main__":
    merged_file = '/mnt/sdb/zly/4.1/tianchi/tianchi_processed_data2.csv'
    split_data(merged_file)