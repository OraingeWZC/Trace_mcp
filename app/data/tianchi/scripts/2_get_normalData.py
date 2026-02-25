#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
正常时段数据获取工具 (Baseline Data Fetcher) - 最终版
- 支持 --window-hours 自定义时间窗
- 支持 --file-name 自定义文件名后缀 (防止覆盖)
- 包含悬浮节点/断链严格检查
"""

import os
import sys
import json
import csv
import time
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import config

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# ================= 🔧 配置区域 =================
# 1. 指标定义
TARGET_METRICS = [
    "aggregate_node_net_receive_packages_errors_per_minute",
    "aggregate_node_tcp_inuse_total_num",
    "aggregate_node_tcp_alloc_total_num",
    "aggregate_node_cpu_usage",
    "aggregate_node_memory_usage",
    "aggregate_node_disk_io_usage"
]

# 2. SLS 配置
PROJECT_NAME = config.SLS_PROJECT_NAME
LOGSTORE_NAME = config.SLS_LOGSTORE_NAME
REGION = config.SLS_REGION

# 3. 鉴权配置
os.environ.setdefault("ALIBABA_CLOUD_ROLE_SESSION_NAME", "normal-data-fetcher")

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 导入工具库
try:
    from tools.paas_entity_tools import umodel_get_entities
    from tools.paas_data_tools import umodel_get_golden_metrics
    from tools.common import create_cms_client, execute_cms_query
    from aliyun.log import LogClient, GetLogsRequest
    from alibabacloud_sts20150401.client import Client as StsClient
    from alibabacloud_sts20150401 import models as sts_models
    from alibabacloud_tea_openapi import models as open_api_models
    from tools.constants import REGION_ID, WORKSPACE_NAME
except ImportError as e:
    print(f"❌ 依赖缺失: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === 悬浮节点检查 ===
def check_orphan_root(spans: list) -> bool:
    """
    检查 Trace 是否存在断链与多根
    找出所有“拓扑根”：即 ParentID 不指向当前 Trace 中任何已知 Span 的节点。(包含三种情况：ParentID为空、ParentID为-1、ParentID指向不存在的ID)
       """
    if not spans: 
        return False
    
    # 1. 建立当前 Trace 所有 SpanID 的集合 (白名单)
    span_ids = set()
    for s in spans:
        # 兼容不同字段名，确保转为字符串并去空
        sid = str(s.get('SpanId', '')).strip()
        if sid: 
            span_ids.add(sid)
    
    # 如果没有有效 Span ID，直接视为无效
    if not span_ids: return False

    root_count = 0
    
    # 2. 遍历所有 Span，统计“拓扑根”的数量
    for s in spans:
        pid = str(s.get('ParentID', '')).strip()
        
        # 核心判定：只要 ParentID 不在 span_ids 里，它就是一个“根”
        # (这自动涵盖了: pid为 -1, pid为 nan, pid为 null, 以及 pid 指向缺失节点的情况)
        if pid not in span_ids:
            root_count += 1
            
    # 3. 严格限制：只能有 1 个根
    return root_count == 1

class NormalDataFetcher:
    def __init__(self, args):
        self.args = args
        self.output_dir = args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.cms_client = create_cms_client(REGION_ID)
        self.sls_client = self._init_sls_client()

    def _init_sls_client(self):
        """初始化 SLS 客户端 (带 STS)"""
        config = open_api_models.Config(
            access_key_id=os.environ["ALIBABA_CLOUD_ACCESS_KEY_ID"],
            access_key_secret=os.environ["ALIBABA_CLOUD_ACCESS_KEY_SECRET"],
            endpoint=f'sts.{REGION}.aliyuncs.com'
        )
        print(config)
        sts_client = StsClient(config)
        resp = sts_client.assume_role(sts_models.AssumeRoleRequest(
            role_arn=os.environ["ALIBABA_CLOUD_ROLE_ARN"],
            role_session_name="normal-fetcher",
            duration_seconds=3600
        ))
        creds = resp.body.credentials
        return LogClient(
            endpoint=f"{REGION}.log.aliyuncs.com",
            accessKeyId=creds.access_key_id,
            accessKey=creds.access_key_secret,
            securityToken=creds.security_token
        )

    def determine_time_window(self):
        """步骤 1: 确定正常时间段"""
        logger.info(f"📅 正在扫描 {self.args.csv} 计算基准时间...")
        min_ts = float('inf')
        
        if os.path.exists(self.args.csv):
            with open(self.args.csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        ts = int(datetime.strptime(row['start_time'], '%Y-%m-%d %H:%M:%S').timestamp())
                        if ts < min_ts: min_ts = ts
                    except: continue
        else:
            logger.warning(f"⚠️ 未找到 {self.args.csv}，使用当前时间作为基准")
            min_ts = int(time.time())
        
        # 定义：最早故障前 window_seconds小时 ~ 前 1小时
        end_time = min_ts - 60 * 60
        window_seconds = int(self.args.window_hours * 3600)
        start_time = end_time - window_seconds
        
        logger.info(f"✅ 选定正常时段: {datetime.fromtimestamp(start_time)} ~ {datetime.fromtimestamp(end_time)}")
        logger.info(f"   (窗口: {self.args.window_hours}h, 基准故障前缓冲: 1h)")
        
        return start_time, end_time

    def fetch_metrics(self, start_ts, end_ts):
        """
        获取节点指标：
        1. 使用分片查询 (Chunking) 防止 API 自动降采样 (解决 60s 粒度问题)
        2. 统一时间戳单位为纳秒 (防止索引报错)
        3. 结合 Golden Metrics 和 CMS 原始接口 (补全缺失指标)
        4. 使用 ffill+fillna 策略 (解决空洞/断层问题)
        """
        logger.info(f"🚀 [Metric] 开始获取正常时段的节点指标 ({start_ts} -> {end_ts})...")
        
        # 1. 获取活跃节点列表
        entity_query = {
            "domain": "acs",
            "entity_set_name": "acs.ecs.instance",
            "from_time": start_ts,
            "to_time": end_ts,
            "limit": 200
        }
        nodes_res = umodel_get_entities.invoke(entity_query)
        if not nodes_res or not nodes_res.data:
            logger.warning("   ⚠️ 未发现活跃节点")
            return

        nodes = nodes_res.data
        logger.info(f"   发现 {len(nodes)} 个活跃节点，正在分片拉取高精度指标...")
        
        # 准备 CSV 输出
        filename = f"normal_metrics_{self.args.file_name}.csv"
        csv_path = os.path.join(self.output_dir, filename)
        headers = ['problem_id', 'fault_type', 'instance_id', 'timestamp'] + sorted(TARGET_METRICS)
        
        rows_to_write = []
        
        # 分片大小设为 30分钟 (1800s)
        # 时间跨度短时，API 会返回原始高精度数据 (如 10s/15s)；跨度长时会自动聚合为 60s
        CHUNK_SIZE = 1800

        for i, node in enumerate(nodes):
            instance_id = node.get('instance_id')
            entity_id = node.get('__entity_id__')
            if not entity_id: continue

            # node_data: { timestamp_ns: { metric_name: value } }
            node_data = {} 

            # === [核心逻辑] 分片循环查询 ===
            current_chunk_start = start_ts
            while current_chunk_start < end_ts:
                current_chunk_end = min(current_chunk_start + CHUNK_SIZE, end_ts)
                
                # 记录本轮分片中找到的指标，用于决定是否需要 CMS 补缺
                chunk_found_metrics = set()

                # --- 策略 A: Golden Metrics (首选) ---
                try:
                    gm_res = umodel_get_golden_metrics.invoke({
                        "domain": "acs",
                        "entity_set_name": "acs.ecs.instance",
                        "entity_ids": [entity_id],
                        "from_time": current_chunk_start,
                        "to_time": current_chunk_end
                    })
                    
                    if gm_res and gm_res.data:
                        for item in gm_res.data:
                            m_name = item.get('metric')
                            if m_name in TARGET_METRICS:
                                chunk_found_metrics.add(m_name)
                                import ast
                                vals = ast.literal_eval(item.get('__value__', '[]'))
                                tss = ast.literal_eval(item.get('__ts__', '[]'))
                                for v, t in zip(vals, tss):
                                    # [修复] 强制转换为纳秒 (19位)，防止与 CMS 毫秒混用导致 Pandas 崩溃
                                    t_int = int(t)
                                    t_ns = t_int * 1000000 if t_int < 1e14 else t_int
                                    
                                    if t_ns not in node_data: node_data[t_ns] = {}
                                    node_data[t_ns][m_name] = v
                except Exception as e:
                    # logger.warning(f"GM Error: {e}")
                    pass

                # --- 策略 B: CMS 原始接口补缺 (备选) ---
                # 如果 Golden Metrics 没拿到某些指标，尝试去查底层接口
                missing = [m for m in TARGET_METRICS if m not in chunk_found_metrics]
                if missing:
                    for m in missing:
                        query = f".entity_set with(domain='acs', name='acs.ecs.instance', ids=['{entity_id}']) | entity-call get_metric('{m}')"
                        try:
                            # 注意：CMS 查询比较慢，这里只查缺失的部分
                            res = execute_cms_query(self.cms_client, WORKSPACE_NAME, query, current_chunk_start, current_chunk_end)
                            if res and res.data:
                                for r in res.data:
                                    v = r.get('value') or r.get(m)
                                    t = r.get('timestamp') or r.get('ts')
                                    if v is not None and t is not None:
                                        t_int = int(t)
                                        t_ns = t_int * 1000000 if t_int < 1e14 else t_int
                                        
                                        if t_ns not in node_data: node_data[t_ns] = {}
                                        node_data[t_ns][m] = v
                        except: 
                            pass
                
                # 推进到下一个分片
                current_chunk_start = current_chunk_end
            
            # === 统一重采样与填充 ===
            if self.args.interval and self.args.interval > 0 and node_data:
                try:
                    df = pd.DataFrame.from_dict(node_data, orient='index')
                    df.index = pd.to_datetime(df.index, unit='ns')
                    
                    # 1. 数据清洗：强制转为数字，处理空字符串
                    for col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # 2. 重采样 + 填充策略
                    df_resampled = df.resample(f'{self.args.interval}s').mean().ffill().fillna(0.0)
                    
                    # 3. 回填
                    new_timestamps = df_resampled.index.astype(np.int64).tolist()
                    for idx, ts_val in enumerate(new_timestamps):
                        row_vals = df_resampled.iloc[idx].to_dict()
                        
                        # 构造 CSV 行
                        row = {
                            'problem_id': 'normal_000',
                            'fault_type': 'normal',
                            'instance_id': instance_id,
                            'timestamp': ts_val
                        }
                        # 填入指标值，缺失的补空字符串(或0)
                        for m in TARGET_METRICS:
                            row[m] = row_vals.get(m, 0.0)
                        
                        rows_to_write.append(row)
                        
                except Exception as e:
                    logger.error(f"   [Node {instance_id}] 重采样/处理失败: {e}")
                    # 出错时降级方案：写入原始数据（防止数据全丢）
                    for ts, metrics in node_data.items():
                        row = {
                            'problem_id': 'normal_000',
                            'fault_type': 'normal',
                            'instance_id': instance_id,
                            'timestamp': ts
                        }
                        for m in TARGET_METRICS:
                            row[m] = metrics.get(m, 0.0)
                        rows_to_write.append(row)

            if (i+1) % 5 == 0: print(f"   已处理 {i+1}/{len(nodes)} 个节点...", end='\r')

        # 统一写入文件
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(rows_to_write)
            logger.info(f"\n✅ [Metric] 已保存 {len(rows_to_write)} 条指标数据至 {csv_path}")
        except Exception as e:
            logger.error(f"❌ 写入 CSV 失败: {e}")

    def fetch_traces(self, start_ts, end_ts):
        """步骤 3: 获取 Trace 数据 (含严格过滤)"""
        logger.info("🚀 [Trace] 开始获取正常时段的 Trace...")
        
        # 1. 初筛: 获取包含至少一个正常Span的候选TraceID
        query = "*"
        limit = self.args.trace_limit
        
        candidate_trace_ids = set()
        offset = 0
        target_candidates = int(limit * 2.0) 
        
        logger.info(f"   目标: 获取 {limit} 条纯净 Trace，预计需扫描 {target_candidates} 个候选 ID...")
        
        while len(candidate_trace_ids) < target_candidates:
            req = GetLogsRequest(PROJECT_NAME, LOGSTORE_NAME, query=query, fromTime=start_ts, toTime=end_ts, line=100, offset=offset)
            try:
                res = self.sls_client.get_logs(req)
                if not res or not res.get_logs(): break
                logs = res.get_logs()
                for log in logs:
                    candidate_trace_ids.add(log.get_contents().get('traceId'))
                offset += len(logs)
                print(f"   已扫描 {offset} 条日志，发现 {len(candidate_trace_ids)} 个候选 TraceID...", end='\r')
                if len(logs) < 100: break
            except Exception as e:
                logger.error(f"SLS Query Error: {e}")
                break
        
        logger.info(f"\n   扫描结束。开始拉取并严格清洗 {len(candidate_trace_ids)} 个 Trace...")
        
        # [修改] 使用后缀构造文件名
        filename = f"normal_traces{self.args.file_name}.csv"
        csv_path = os.path.join(self.output_dir, filename)
        
        csv_headers = [
            'TraceID', 'SpanId', 'ParentID', 'ServiceName', 'NodeName', 'PodName', 
            'URL', 'SpanKind', 'StartTimeMs', 'EndTimeMs', 'DurationMs',
            'StatusCode', 'HttpStatusCode', 'fault_type', 'fault_instance', 'problem_id'
        ]
        
        batch_list = list(candidate_trace_ids)
        valid_trace_count = 0
        total_spans = 0
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writeheader()
            
            # 每次处理 20 个 TraceID
            for i in range(0, len(batch_list), 20):
                if valid_trace_count >= limit: break
                
                batch = batch_list[i:i+20]
                or_query = " OR ".join([f'traceId: "{tid}"' for tid in batch])
                
                # === 内存聚合: 将这20个Trace的所有Span先存起来 ===
                trace_buffer = {tid: [] for tid in batch} 
                
                sub_offset = 0
                while True:
                    req = GetLogsRequest(PROJECT_NAME, LOGSTORE_NAME, query=or_query, fromTime=start_ts, toTime=end_ts, line=100, offset=sub_offset)
                    try:
                        res = self.sls_client.get_logs(req)
                        if not res or not res.get_logs(): break
                        logs = res.get_logs()
                        
                        for log in logs:
                            d = log.get_contents()
                            tid = d.get('traceId')
                            if tid in trace_buffer:
                                # 解析 Span 数据
                                try: res_obj = json.loads(d.get('resources', '{}'))
                                except: res_obj = {}
                                try: att_obj = json.loads(d.get('resources', '{}'))
                                except: att_obj = {}
                                try:
                                    s_ms = int(d.get('startTime', 0)) / 1e6
                                    d_ms = int(d.get('duration', 0)) / 1e6
                                except: s_ms, d_ms = 0, 0
                                
                                # 暂存原始数据对象
                                span_obj = {
                                    'TraceID': tid,
                                    'SpanId': d.get('spanId'),
                                    'ParentID': d.get('parentSpanId'),
                                    'ServiceName': d.get('serviceName'),
                                    'NodeName': res_obj.get('k8s.node.name'),
                                    'PodName': res_obj.get('k8s.pod.name'),
                                    'URL': d.get('spanName'),
                                    'SpanKind': d.get('kind'),
                                    'StartTimeMs': f"{s_ms:.3f}",
                                    'EndTimeMs': f"{s_ms + d_ms:.3f}",
                                    'DurationMs': f"{d_ms:.3f}",
                                    'StatusCode': d.get('statusCode'),
                                    'HttpStatusCode': str(att_obj.get('http.status_code') or att_obj.get('rpc.grpc.status_code', '')),
                                    'fault_type': 'normal',
                                    'fault_instance': "",
                                    'problem_id': 'normal_000'
                                }
                                trace_buffer[tid].append(span_obj)
                        
                        sub_offset += len(logs)
                        if len(logs) < 100: break
                    except: break
                
                # === 严格过滤: 检查每个Trace是否真正“纯净” ===
                rows_to_save = []
                for tid, spans in trace_buffer.items():
                    if not spans: continue
                    if len(spans) < 2: continue

                    is_error = False
                    is_out_of_window = False # [新增] 标记是否超出时间窗

                    for span in spans:
                        # 1. 检查错误状态
                        try:
                            # 兼容处理：有些statusCode可能是空或非数字，视作0
                            sc = int(span['StatusCode']) if span['StatusCode'] and span['StatusCode'].isdigit() else 0
                            if sc > 1: is_error = True; break
                        except: pass
                        
                        # 2. 检查开始时间是否早于窗口起始时间
                        # start_ts 是秒级，StartTimeMs 是字符串毫秒，需转换
                        try:
                            span_start_ms = float(span['StartTimeMs'])
                            if span_start_ms < start_ts * 1000:
                                is_out_of_window = True
                                break # 只要有一个 Span 早于窗口，整条 Trace 丢弃
                        except: pass

                    if is_error: continue
                    if is_out_of_window: continue # [新增] 丢弃跨窗口的 Trace

                    # 3. 严格悬浮检查
                    if not check_orphan_root(spans): continue

                    rows_to_save.extend(spans)
                    valid_trace_count += 1
                
                # 写入文件
                if rows_to_save:
                    writer.writerows(rows_to_save)
                    total_spans += len(rows_to_save)
                
                print(f"   进度: 已获取 {valid_trace_count}/{limit} 条纯净 Trace...", end='\r')

        logger.info(f"\n✅ [Trace] 已保存 {valid_trace_count} 条纯净 Trace ({total_spans} Spans) 至 {csv_path}")

    def run(self):
        # s_ts, e_ts = self.determine_time_window()

        custom_start = "2026-01-20 20:00:00" 
        custom_end = "2026-01-20 23:59:59"
        s_ts = int(datetime.strptime(custom_start, "%Y-%m-%d %H:%M:%S").timestamp())
        e_ts = int(datetime.strptime(custom_end, "%Y-%m-%d %H:%M:%S").timestamp())

        # 获取指标时，额外多往前拉 3 分钟
        self.fetch_metrics(s_ts - 180, e_ts)
        # self.fetch_traces(s_ts, e_ts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="dataset/b_gt.csv", help="故障列表路径")
    parser.add_argument("--output-dir", default="data/NormalData", help="输出目录")
    parser.add_argument("--trace-limit", type=int, default=40, help="获取多少条正常 Trace")
    parser.add_argument("--interval", type=int, default=30, help="指标重采样间隔(秒)")
    parser.add_argument("--window-hours", type=float, default=4.0, help="获取故障前多少小时的数据")
    parser.add_argument("--file-name", type=str, default="4e5_30s_4h_0120", help="输出文件名后缀")
    
    args = parser.parse_args()

    fetcher = NormalDataFetcher(args)
    fetcher.run()