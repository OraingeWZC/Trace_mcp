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
START_TIME_STR = "2026-01-20 18:20:33"
END_TIME_STR = "2026-01-30 22:20:33"

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

    print("\n🔍 步骤 1: 正在查找活跃的 acs 实体...")
    
    # # 尝试查询 k8s 域
    # entity_query = {
    #     "domain": "k8s",               # 切换为 k8s
    #     "entity_set_name": "k8s.node", # 切换为 k8s.node
    #     "from_time": START_TIME,
    #     "to_time": END_TIME,
    #     "limit": 50  # 先查前5个看看
    # }
    # res = umodel_get_entities.invoke(entity_query)
    # nodes = res.data
    # print(f"✅ 找到 {len(nodes)} 个节点。列表如下：")

    # for i, node in enumerate(nodes):
    #     output_filename = "k8s_nodes_list.json"
    #     open(output_filename, 'w', encoding='utf-8').write(json.dumps(node, ensure_ascii=False, indent=2))
    #     nodename = node.get('provider_id') 
    #     name = node.get('name') or node.get('nodeName') or node.get('instance_id')
        
    #     print(f"   [{i}] Name: {name:<25} | ID: {nodename}")

    entity_query = {
        "domain": "acs",
        "entity_set_name": "acs.ecs.instance",
        "from_time": START_TIME,
        "to_time": END_TIME,
        "limit": 500  # 假设节点数不超过 500，如果超过需分页
    }
    
    res = umodel_get_entities.invoke(entity_query)
    nodes = res.data
    print(f"✅ 找到 {len(nodes)} 个节点。列表如下：")

    for i, node in enumerate(nodes):
        output_filename = "acs_nodes_list.json"
        open(output_filename, 'w', encoding='utf-8').write(json.dumps(node, ensure_ascii=False, indent=2))
        nodename = node.get('provider_id') 
        name = node.get('name') or node.get('nodeName') or node.get('instance_id')
        
        print(f"   [{i}] Name: {name:<25} | ID: {nodename}")
    
    # 获取节点指标
    target_node = nodes[0]
    entity_id = target_node.get('__entity_id__')
    instance_id = node.get('instance_id')
    print(f"✅ 选定目标节点: {instance_id} (ID: {entity_id})")

    # 2. 调用黄金指标接口获取该节点的所有时序数据
    print(f"📊 正在请求全量指标详情...")
    metrics_res = umodel_get_golden_metrics.invoke({
        "domain": "acs",
        "entity_set_name": "acs.ecs.instance",
        "entity_ids": [entity_id],
        "from_time": START_TIME,
        "to_time": END_TIME
    })

    # 3. 封装并保存为 JSON
    output_data = {
        "meta": {
            "instance_id": instance_id,
            "entity_id": entity_id,
            "start_time": START_TIME,
            "end_time": END_TIME,
            "workspace": WORKSPACE_NAME
        },
        "raw_response": metrics_res.data if metrics_res else []
    }

    output_file = "raw_all_metrics.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    
    print(f"\n🎉 成功！原始指标数据已保存至: {output_file}")
    print(f"💡 该文件中包含了 {len(output_data['raw_response'])} 组不同的指标数据。")

if __name__ == "__main__":
    explore_and_query()