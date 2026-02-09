# scripts/6_map_trace_nodes.py
# -*- coding: utf-8 -*-
"""
Trace 节点映射清洗工具
功能：
1. 读取现有的 Trace CSV 文件。
2. 加载 ecs_mapping_index.json 映射表。
3. 将 Trace 中的 NodeName (IP/K8sName) 统一替换为物理 Instance ID。
4. 原有的 NodeName 会被备份到新列 RawNodeName 中。
"""

import os
import csv
import json
import argparse
import sys
from collections import defaultdict

# # 增大 CSV 字段限制，防止 Trace 过长报错
# csv.field_size_limit(sys.maxsize)

def load_mapping(json_path):
    """加载映射文件，构建查找表"""
    if not os.path.exists(json_path):
        print(f"❌ 错误: 映射文件不存在 {json_path}")
        return None
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 合并 lookup table
    # 优先匹配精确的 key (IP 或 Hostname)
    lookup = {}
    if "ip_to_id" in data:
        lookup.update(data["ip_to_id"])
    if "name_to_id" in data:
        lookup.update(data["name_to_id"])
    
    print(f"✅ 已加载映射表，包含 {len(lookup)} 个条目")
    return lookup

def process_file(input_path, output_path, mapping):
    """处理单个 CSV 文件"""
    if not os.path.exists(input_path):
        print(f"⚠️ 跳过: 输入文件不存在 {input_path}")
        return

    print(f"🔄 正在处理: {input_path} ...")
    
    mapped_count = 0
    total_count = 0
    missed_nodes = set()

    with open(input_path, 'r', encoding='utf-8', newline='') as f_in, \
         open(output_path, 'w', encoding='utf-8', newline='') as f_out:
        
        reader = csv.DictReader(f_in)
        # 1. 修改表头：把 NodeName 放到原来的位置，新增 RawNodeName
        fieldnames = list(reader.fieldnames)
        if "RawNodeName" not in fieldnames:
            # 插入到 NodeName 后面，或者最后
            if "NodeName" in fieldnames:
                idx = fieldnames.index("NodeName")
                fieldnames.insert(idx + 1, "RawNodeName")
            else:
                fieldnames.append("RawNodeName")
        
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in reader:
            total_count += 1
            original_name = row.get("NodeName", "").strip()
            
            # 备份原始名字
            row["RawNodeName"] = original_name
            
            # 查找映射
            # 1. 尝试直接匹配
            target_id = mapping.get(original_name)
            
            # 2. 尝试去掉 'cn-qingdao.' 前缀匹配
            if not target_id and original_name.startswith("cn-qingdao."):
                short_name = original_name.replace("cn-qingdao.", "")
                target_id = mapping.get(short_name)
            
            # 3. 尝试作为 IP 匹配 (如果包含在 HostName 里)
            # (这一步视情况而定，如果你的 mapping 足够全，通常不需要模糊匹配)
            
            if target_id:
                row["NodeName"] = target_id
                mapped_count += 1
            else:
                if original_name and original_name.lower() != "none":
                    missed_nodes.add(original_name)
            
            writer.writerow(row)
            
            if total_count % 50000 == 0:
                print(f"   已处理 {total_count} 行...", end='\r')

    print(f"\n   ✅ 完成! 映射成功率: {mapped_count}/{total_count} ({mapped_count/total_count*100:.1f}%)")
    if missed_nodes:
        print(f"   ⚠️ 未命中映射的节点 (Top 5): {list(missed_nodes)[:5]}")
        # 可以把 missed_nodes 写入日志方便排查

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", default="data/ecs_mapping_250916_260131.json", help="映射文件路径")
    parser.add_argument("--inputs", nargs="+", 
                        default=[
                            # "data/NormalData/normal_traces.csv",
                            # "data/ServiceFault/all_fault_traces.csv",
                            # "data/NodeFault/all_fault_traces.csv"
                            "data/NormalData/normal_traces2e5_0120_nostatus.csv"
                        ],
                        help="需要处理的 CSV 文件列表")
    parser.add_argument("--suffix", default="_mapped", help="输出文件后缀 (例如 _mapped)")
    args = parser.parse_args()

    mapping = load_mapping(args.mapping)
    if not mapping:
        return

    for input_file in args.inputs:
        # 构造输出文件名: data/xxx.csv -> data/xxx_mapped.csv
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}{args.suffix}{ext}"
        
        process_file(input_file, output_file, mapping)

if __name__ == "__main__":
    main()