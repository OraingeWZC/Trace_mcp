"""
分析数据集中 root_cause 标注和 service_id 映射的一致性
"""
import os
import yaml
from collections import defaultdict, Counter
from loguru import logger
from tracegnn.data.bytes_db import BytesSqliteDB
from tracegnn.data.trace_graph_db import TraceGraphDB
from tracegnn.data.trace_graph import TraceGraph, TraceGraphIDManager

def load_service_mapping(yaml_path):
    """加载 service_id.yml 映射"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        name_to_id = yaml.safe_load(f)
    id_to_name = {int(v): k for k, v in name_to_id.items()}
    return name_to_id, id_to_name

def get_service_prefix(service_name):
    """提取服务名前缀"""
    if isinstance(service_name, str) and '.' in service_name:
        return service_name.split('.')[0]
    return service_name

def analyze_dataset(dataset_path, service_id_yaml_path):
    """分析数据集中的root_cause映射问题"""

    logger.info(f"正在分析数据集: {dataset_path}")
    logger.info(f"Service映射文件: {service_id_yaml_path}")

    # 加载service映射
    name_to_id, id_to_name = load_service_mapping(service_id_yaml_path)

    logger.info(f"\n加载了 {len(id_to_name)} 个service映射:")
    for sid in sorted(id_to_name.keys())[:10]:
        logger.info(f"  {sid}: {id_to_name[sid]}")
    if len(id_to_name) > 10:
        logger.info(f"  ... 还有 {len(id_to_name) - 10} 个")

    # 统计数据
    total_traces = 0
    anomalous_traces = 0
    root_cause_counter = Counter()
    root_cause_to_services_in_graph = defaultdict(lambda: defaultdict(int))
    root_cause_in_graph_count = 0
    root_cause_not_in_graph = []

    # 遍历数据集
    with TraceGraphDB(BytesSqliteDB(dataset_path)) as db:
        logger.info(f"\n开始分析 {len(db)} 个trace...")

        for i in range(len(db)):
            graph: TraceGraph = db.get(i)
            total_traces += 1

            if graph.anomaly > 0:
                anomalous_traces += 1

                if graph.root_cause is not None:
                    rc = graph.root_cause
                    root_cause_counter[rc] += 1

                    # 获取root_cause的名称
                    rc_name = id_to_name.get(rc, f"unknown-{rc}")
                    rc_prefix = get_service_prefix(rc_name)

                    # 收集trace中的所有service_ids
                    services_in_trace = set()
                    for _, node in graph.iter_bfs():
                        if node.service_id is not None:
                            services_in_trace.add(node.service_id)

                    # 检查root_cause是否在trace中
                    if rc in services_in_trace:
                        root_cause_in_graph_count += 1
                    else:
                        # 检查是否有相同前缀的服务
                        prefix_matches = []
                        for sid in services_in_trace:
                            service_name = id_to_name.get(sid, f"unknown-{sid}")
                            service_prefix = get_service_prefix(service_name)
                            if service_prefix == rc_prefix:
                                prefix_matches.append((sid, service_name))

                        root_cause_not_in_graph.append({
                            'trace_idx': i,
                            'root_cause_id': rc,
                            'root_cause_name': rc_name,
                            'root_cause_prefix': rc_prefix,
                            'services_in_trace': services_in_trace,
                            'prefix_matches': prefix_matches
                        })

                    # 统计trace中出现的所有服务
                    for sid in services_in_trace:
                        service_name = id_to_name.get(sid, f"unknown-{sid}")
                        root_cause_to_services_in_graph[rc][sid] += 1

    # 输出分析结果
    logger.info(f"\n{'='*80}")
    logger.info(f"数据集统计:")
    logger.info(f"  总trace数: {total_traces}")
    logger.info(f"  异常trace数: {anomalous_traces}")
    logger.info(f"  有root_cause标注: {sum(root_cause_counter.values())}")
    logger.info(f"  root_cause在图中: {root_cause_in_graph_count}")
    logger.info(f"  root_cause不在图中: {len(root_cause_not_in_graph)}")

    logger.info(f"\n{'='*80}")
    logger.info(f"Root Cause 分布:")
    for rc, count in root_cause_counter.most_common():
        rc_name = id_to_name.get(rc, f"unknown-{rc}")
        logger.info(f"  service_id={rc} ('{rc_name}'): {count} 次")

    if root_cause_not_in_graph:
        logger.info(f"\n{'='*80}")
        logger.info(f"Root Cause 不在 Trace 图中的样本 ({len(root_cause_not_in_graph)} 个):")

        # 按root_cause分组
        by_rc = defaultdict(list)
        for item in root_cause_not_in_graph:
            by_rc[item['root_cause_id']].append(item)

        for rc, items in sorted(by_rc.items()):
            rc_name = id_to_name.get(rc, f"unknown-{rc}")
            rc_prefix = get_service_prefix(rc_name)
            logger.info(f"\n  Root Cause: service_id={rc} ('{rc_name}', 前缀='{rc_prefix}') - {len(items)} 个样本")

            # 统计这个root_cause对应的trace中最常出现的服务
            service_counter = Counter()
            prefix_match_counter = Counter()

            for item in items:
                for sid in item['services_in_trace']:
                    service_counter[sid] += 1
                for sid, _ in item['prefix_matches']:
                    prefix_match_counter[sid] += 1

            logger.info(f"    该root_cause的trace中最常见的服务:")
            for sid, count in service_counter.most_common(5):
                service_name = id_to_name.get(sid, f"unknown-{sid}")
                service_prefix = get_service_prefix(service_name)
                is_prefix_match = "★" if service_prefix == rc_prefix else " "
                logger.info(f"      {is_prefix_match} service_id={sid} ('{service_name}'): 出现{count}次")

            if prefix_match_counter:
                logger.info(f"    前缀匹配的服务:")
                for sid, count in prefix_match_counter.most_common():
                    service_name = id_to_name.get(sid, f"unknown-{sid}")
                    logger.info(f"      ✓ service_id={sid} ('{service_name}'): 出现{count}次")
                    logger.info(f"        建议: 将 root_cause={rc} 映射到 service_id={sid}")

    # 生成修正建议
    logger.info(f"\n{'='*80}")
    logger.info(f"修正建议:")

    if len(root_cause_not_in_graph) > 0:
        logger.info(f"\n发现 {len(root_cause_not_in_graph)} 个样本的root_cause不在trace图中。")
        logger.info(f"\n可能的解决方案:")
        logger.info(f"  方案1: 修改root_cause标注")
        logger.info(f"    将不在图中的root_cause替换为前缀匹配的service_id")

        logger.info(f"\n  方案2: 使用前缀匹配进行评估")
        logger.info(f"    在evaluate_with_root_cause函数中使用前缀匹配而不是精确匹配")

        logger.info(f"\n  方案3: 检查数据采集问题")
        logger.info(f"    异常可能导致某些服务的span没有被采集，需要改进数据采集")

        # 生成映射建议
        logger.info(f"\n建议的 root_cause 映射修正:")
        for rc, items in sorted(by_rc.items()):
            rc_name = id_to_name.get(rc, f"unknown-{rc}")

            # 找最常见的前缀匹配服务
            prefix_match_counter = Counter()
            for item in items:
                for sid, _ in item['prefix_matches']:
                    prefix_match_counter[sid] += 1

            if prefix_match_counter:
                most_common_sid, count = prefix_match_counter.most_common(1)[0]
                most_common_name = id_to_name.get(most_common_sid, f"unknown-{most_common_sid}")
                logger.info(f"  {rc} ('{rc_name}') → {most_common_sid} ('{most_common_name}')  [{count}/{len(items)} 个trace中出现]")
    else:
        logger.info(f"  ✓ 所有root_cause都在对应的trace图中，无需修正！")

    logger.info(f"\n{'='*80}")

    return {
        'total_traces': total_traces,
        'anomalous_traces': anomalous_traces,
        'root_cause_in_graph': root_cause_in_graph_count,
        'root_cause_not_in_graph': root_cause_not_in_graph,
        'mapping_suggestions': by_rc if root_cause_not_in_graph else {}
    }

def main():
    # 配置
    dataset_root = "dataset/tianchi2"
    processed_dir = os.path.join(dataset_root, "processed")
    test_dataset = os.path.join(processed_dir, "test")
    service_id_yaml = os.path.join(processed_dir, "service_id.yml")

    logger.info("="*80)
    logger.info("Root Cause 映射一致性分析工具")
    logger.info("="*80)

    # 分析数据集
    result = analyze_dataset(test_dataset, service_id_yaml)

    logger.info(f"\n分析完成！")
    logger.info(f"如需生成修正脚本，请查看上面的建议。")

if __name__ == "__main__":
    main()
