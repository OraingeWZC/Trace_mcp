# debug_k8s_query.py
# -*- coding: utf-8 -*-
import os
import sys
import json
import time
from datetime import datetime

# === 1. 环境准备 (保持不变) ===
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.getcwd())

try:
    import config
    from tools.common import create_cms_client, execute_cms_query
    from tools.paas_entity_tools import umodel_get_entities
    from tools.paas_data_tools import umodel_get_golden_metrics
    from tools.constants import REGION_ID, WORKSPACE_NAME
except ImportError as e:
    print(f"❌ 环境依赖错误: {e}")
    sys.exit(1)

# === 2. 配置区域 ===

# 🕒 查询时间段 (过去 10 分钟)
START_TIME_STR = "2025-09-16 00:00:00"
END_TIME_STR = "2025-09-24 23:59:59"

START_TIME = datetime.strptime(START_TIME_STR, "%Y-%m-%d %H:%M:%S")
START_TIME = int(START_TIME.timestamp())
END_TIME = datetime.strptime(END_TIME_STR, "%Y-%m-%d %H:%M:%S")
END_TIME = int(END_TIME.timestamp())

# ============================================

def timestamp_to_str(ts):
    try:
        ts = float(ts)
        if ts > 1e14: ts /= 1000000 
        if ts > 1e11: ts /= 1000    
        return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    except:
        return str(ts)

def explore_and_query():
    print(f"🔌 连接 CMS 客户端 (Region: {REGION_ID})...")
    client = create_cms_client(REGION_ID)

    print("\n🔍 步骤 1: 正在查找活跃的 k8s.node 实体...")
    
    # 尝试查询 k8s 域
    entity_query = {
        "domain": "k8s",               # 切换为 k8s
        "entity_set_name": "k8s.node", # 切换为 k8s.node
        "from_time": START_TIME,
        "to_time": END_TIME,
        "limit": 50  # 先查前5个看看
    }
    entity_query = {
    "domain": "acs",
    "entity_set_name": "acs.ecs.instance",
    "from_time": START_TIME,
    "to_time": END_TIME,
    "limit": 500  # 假设节点数不超过 500，如果超过需分页
    }
    
    res = umodel_get_entities.invoke(entity_query)
    nodes = res.data
    print(f"✅ 找到 {len(nodes)} 个 K8s 节点。列表如下：")
    
    target_node_id = None
    target_node_name = None

    for i, node in enumerate(nodes):
        output_filename = "k8s_nodes_list.json"
        open(output_filename, 'w', encoding='utf-8').write(json.dumps(node, ensure_ascii=False, indent=2))
        nodename = node.get('provider_id') 
        name = node.get('name') or node.get('nodeName') or node.get('instance_id')
        
        print(f"   [{i}] Name: {name:<25} | ID: {nodename}")
        
        # 默认选中第一个作为测试目标
        if i == 0:
            target_node_id = nodename
            target_node_name = name

    res = umodel_get_entities.invoke(entity_query)
    nodes = res.data
    print(f"✅ 找到 {len(nodes)} 个 K8s 节点。列表如下：")
    
    target_node_id = None
    target_node_name = None

    for i, node in enumerate(nodes):
        output_filename = "k8s_nodes_list.json"
        open(output_filename, 'w', encoding='utf-8').write(json.dumps(node, ensure_ascii=False, indent=2))
        nodename = node.get('instance_id') 
        k8sname = node.get('instance_ip')
        
        print(f"   [{i}] k8sname: {k8sname:<25} | NodeName: {nodename}")
        
        # 默认选中第一个作为测试目标
        if i == 0:
            target_node_id = nodename
            target_node_name = k8sname

    print(f"\n🎯 选定测试节点: {target_node_name} (ID: {target_node_id})")


    # # ---------------------------------------------------------
    # # 步骤 2: 拉取该节点的所有可用指标 (Golden Metrics)
    # # ---------------------------------------------------------
    # print(f"\n🔍 步骤 2: 拉取该节点的 Golden Metrics (自动发现可用指标)...")
    
    # gm_query = {
    #     "domain": "k8s", # 保持和上面一致 (k8s 或 cc)
    #     "entity_set_name": "k8s.node",
    #     "entity_ids": [target_node_id],
    #     "from_time": START_TIME,
    #     "to_time": END_TIME
    # }
    
    # gm_res = umodel_get_golden_metrics.invoke(gm_query)
    
    # available_metrics = set()
    # if gm_res and gm_res.data:
    #     print(f"✅ 成功获取 Golden Metrics 数据! (以下是包含数据的指标名)")
    #     for item in gm_res.data:
    #         metric_name = item.get('metric')
    #         available_metrics.add(metric_name)
            
    #         # === 这里打印一条完整的原始数据给你看 ===
    #         if len(available_metrics) == 1: # 只打印第一条详情，避免刷屏
    #             print("\n[👀 数据样例 - 原始 JSON 内容]")
    #             print(json.dumps(item, indent=2, ensure_ascii=False))
    #             print("-" * 50)
        
    #     print("\n📊 该节点当前可用的指标列表 (Target Metrics 候选):")
    #     for m in sorted(list(available_metrics)):
    #         print(f"   - {m}")
    # else:
    #     print("⚠️ 该节点没有 Golden Metrics 数据，尝试直接查询常用指标...")
    #     available_metrics = {"cpu_total", "memory_usage_bytes", "network_receive_bytes"}

    # # ---------------------------------------------------------
    # # 步骤 3: 使用底层 CMS 接口查询特定指标
    # # ---------------------------------------------------------
    # # 从发现的列表中选一个，或者使用默认的
    # test_metric = list(available_metrics)[0] if available_metrics else "node_cpu_utilization"
    
    # print(f"\n🔍 步骤 3: 使用 CMS 底层接口查询单项指标 [{test_metric}]...")
    
    # # 构建查询语句 (注意这里 domain 和 name 的变化)
    # query_stmt = (
    #     f".entity_set "
    #     f"with(domain='{entity_query['domain']}', name='k8s.node', ids=['{target_node_id}']) "
    #     f"| entity-call get_metric('{test_metric}')"
    # )
    
    # print(f"   Query: {query_stmt}")
    
    # try:
    #     res = execute_cms_query(client, WORKSPACE_NAME, query_stmt, START_TIME, END_TIME)
        
    #     if res and res.data:
    #         print(f"✅ CMS 查询成功! 获取到 {len(res.data)} 个数据点。")
            
    #         # 打印完整数据供参考
    #         output_file = "k8s_query_result.json"
    #         with open(output_file, "w", encoding='utf-8') as f:
    #             json.dump(res.data, f, indent=2, ensure_ascii=False)
            
    #         # 打印前 3 条看看格式
    #         for i, p in enumerate(res.data[:3]):
    #             val = p.get('value') or p.get(test_metric)
    #             ts = p.get('timestamp') or p.get('ts')
    #             print(f"   [{i}] Time: {timestamp_to_str(ts)} | Value: {val}")
                
    #         print(f"\n💾 完整 JSON 结果已保存至: {output_file}")
    #     else:
    #         print("⚠️ CMS 查询返回空数据。")
            
    # except Exception as e:
    #     print(f"❌ CMS 查询报错: {e}")

if __name__ == "__main__":
    explore_and_query()