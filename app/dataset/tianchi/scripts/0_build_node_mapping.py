# 0_build_node_mapping.py
# -*- coding: utf-8 -*-
"""
全局节点映射生成工具
功能：扫描指定时间段内活跃的所有 ECS 节点，构建 IP/HostName -> InstanceID 的完整映射表。
输出：dataset/ecs_mapping_index.json
"""
import os
import sys
import json
import time
from datetime import datetime

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 鉴权配置
import config 
os.environ["ALIBABA_CLOUD_ROLE_SESSION_NAME"] = "mapping-builder"

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from tools.paas_entity_tools import umodel_get_entities
except ImportError as e:
    print(f"❌ 依赖导入失败: {e}")
    sys.exit(1)

def build_mapping(start_time_str, end_time_str, output_path):
    print(f"🔍 正在扫描全量 ECS 节点 ({start_time_str} ~ {end_time_str})...")
    
    # 转换时间戳
    s_ts = int(datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S").timestamp())
    e_ts = int(datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S").timestamp())

    # 查询所有 ECS (limit 设大一点以覆盖全量，或者分也拉取)
    query = {
        "domain": "acs",
        "entity_set_name": "acs.ecs.instance",
        "from_time": s_ts,
        "to_time": e_ts,
        "limit": 500  # 假设节点数不超过 500，如果超过需分页
    }

    res = umodel_get_entities.invoke(query)
    if not res or not res.data:
        print("❌ 未查询到任何节点信息！")
        return

    mapping_data = {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_unique_ids": 0
        },
        "ip_to_id": {},    # IP -> i-xxx
        "name_to_id": {},  # K8sNodeName -> i-xxx
        "all_instance_ids": []
    }

    ids = set()
    
    for node in res.data:
        instance_id = node.get('instance_id')
        if not instance_id: continue
        
        ids.add(instance_id)

        # 1. 映射 IP
        raw_ip = node.get('instance_ip') or node.get('privateIpAddress')
        ip_list = []
        if isinstance(raw_ip, list):
            ip_list = raw_ip
        elif isinstance(raw_ip, str):
            ip_list = raw_ip.split(',')
        
        for ip in ip_list:
            ip = ip.strip()
            if ip:
                mapping_data["ip_to_id"][ip] = instance_id
                # 同时也映射带前缀的版本 (适配 Trace 中的常见格式)
                mapping_data["name_to_id"][f"cn-qingdao.{ip}"] = instance_id

        # 2. 映射 Hostname / K8s Node Name
        # 如果 Entity 里有 hostname 字段，也加进去
        hostname = node.get('hostname') or node.get('instance_name')
        if hostname:
            mapping_data["name_to_id"][hostname] = instance_id

    mapping_data["meta"]["total_unique_ids"] = len(ids)
    mapping_data["all_instance_ids"] = list(ids)

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mapping_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 映射文件已生成: {output_path}")
    print(f"   - 覆盖节点数: {len(ids)}")
    print(f"   - IP 映射条目: {len(mapping_data['ip_to_id'])}")
    print(f"   - Name 映射条目: {len(mapping_data['name_to_id'])}")

if __name__ == "__main__":
    # 建议时间范围覆盖整个比赛/数据采集周期
    START_TIME = "2025-09-16 00:00:00"
    END_TIME = "2025-09-24 23:59:59" # 根据实际情况调整
    OUTPUT_FILE = "data/ecs_mapping_index.json"
    
    build_mapping(START_TIME, END_TIME, OUTPUT_FILE)