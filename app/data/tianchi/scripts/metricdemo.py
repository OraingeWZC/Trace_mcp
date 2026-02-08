# debug_cms_query.py
# -*- coding: utf-8 -*-
import os
import sys
import json
import time
from datetime import datetime
import pandas as pd

# === 1. 环境准备 ===
# 添加项目路径以导入 tools
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # 如果脚本在 scripts/ 下
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.getcwd()) # 或者是当前目录

print(project_root)

try:
    import config
    from tools.common import create_cms_client, execute_cms_query
    from tools.constants import REGION_ID, WORKSPACE_NAME
except ImportError as e:
    print(f"❌ 环境依赖错误: {e}")
    print("请确保你在正确的项目环境下运行 (例如: python scripts/debug_cms_query.py)")
    sys.exit(1)

# === 2. 配置区域 (请在这里修改你要查的内容) ===

# 🎯 目标节点 ID (列表)
TARGET_NODES = [
    'i-m5ec00yjg8kxv34hyr0j'
]

# 📊 目标指标 (列表)
# 你可以换成 aggregate_node_cpu_usage 等
TARGET_METRICS = [
    "aggregate_node_cpu_usage", 
    "aggregate_node_memory_usage",
    "aggregate_node_tcp_alloc_total_num"
]

# 🕒 查询时间段 (请确保这段时间节点是活跃的)
START_TIME_STR = "2025-09-16 18:20:00"
END_TIME_STR   = "2025-09-16 18:30:00"  # 查10分钟试试

# ============================================

def timestamp_to_str(ts):
    """将毫秒/秒级时间戳转为可读字符串"""
    try:
        # 可能是纳秒或毫秒，统一下
        ts = float(ts)
        if ts > 1e14: ts /= 1000000 # 纳秒转毫秒
        if ts > 1e11: ts /= 1000    # 毫秒转秒
        return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    except:
        return str(ts)

def run_raw_query():
    # 1. 初始化客户端
    print(f"🔌 连接 CMS 客户端 (Region: {REGION_ID})...")
    client = create_cms_client(REGION_ID)

    # 2. 转换时间
    start_ts = int(datetime.strptime(START_TIME_STR, "%Y-%m-%d %H:%M:%S").timestamp())
    end_ts = int(datetime.strptime(END_TIME_STR, "%Y-%m-%d %H:%M:%S").timestamp())
    
    print(f"📅 查询窗口: {START_TIME_STR} ~ {END_TIME_STR} ({end_ts - start_ts}s)")

    # 3. 开始循环查询
    results = {}

    for node_id in TARGET_NODES:
        print(f"\n🔍 正在查询节点: {node_id}")
        results[node_id] = {}
        
        for metric in TARGET_METRICS:
            # =======================================================
            # 🔥 核心：这就是最底层的查询语句构造
            # =======================================================
            query_stmt = (
                f".entity_set "
                f"with(domain='acs', name='acs.ecs.instance', ids=['{node_id}']) "
                f"| entity-call get_metric('{metric}')"
            )
            
            print(f"   Wait... 指标 [{metric}]", end="", flush=True)
            
            try:
                # 调用工具函数执行查询
                # 注意：Workspace 必须要对，否则查不到
                res = execute_cms_query(client, WORKSPACE_NAME, query_stmt, start_ts, end_ts)
                
                if res and res.data:
                    data_points = res.data
                    count = len(data_points)
                    print(f" ✅ 获取到 {count} 个数据点")
                    
                    # 提取数据用于展示
                    parsed_data = []
                    for p in data_points:
                        # 兼容不同的返回格式
                        val = p.get('value') or p.get(metric)
                        ts = p.get('timestamp') or p.get('ts')
                        parsed_data.append({
                            "time_str": timestamp_to_str(ts),
                            "timestamp": ts,
                            "value": val
                        })
                    
                    # 按时间排序
                    parsed_data.sort(key=lambda x: x['timestamp'])
                    results[node_id][metric] = parsed_data
                    
                    # 打印前3条和后3条看看
                    if parsed_data:
                        print(f"      Start: {parsed_data[0]['time_str']} = {parsed_data[0]['value']}")
                        print(f"      End  : {parsed_data[-1]['time_str']} = {parsed_data[-1]['value']}")
                else:
                    print(f" ⚠️ 空数据 (API返回成功但无内容)")
                    results[node_id][metric] = None
                    
            except Exception as e:
                print(f" ❌ 查询报错: {e}")

    # 4. 保存结果到文件以便查看
    output_file = "debug_cms_result.json"
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 完整结果已保存到: {output_file}")

if __name__ == "__main__":
    # 确保 AK/SK 环境变量存在
    if not os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_ID"):
        print("⚠️ 警告: 环境变量 ALIBABA_CLOUD_ACCESS_KEY_ID 未设置，可能导致鉴权失败")
    
    run_raw_query()