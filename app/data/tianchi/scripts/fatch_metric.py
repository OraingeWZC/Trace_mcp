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

# 添加项目路径（确保能 import 到 tianchi 下的 config/tools）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

# config 在某些测试环境下可能不存在；本脚本仅拉取 metric，不强依赖 config
try:
    import config  # type: ignore
except Exception:
    config = None

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

# 2. SLS 配置（当前脚本仅做 metric 测试，SLS 参数保留但不强制存在）
PROJECT_NAME = getattr(config, "SLS_PROJECT_NAME", "") if config else ""
LOGSTORE_NAME = getattr(config, "SLS_LOGSTORE_NAME", "") if config else ""
REGION = getattr(config, "SLS_REGION", "") if config else ""

# 3. 鉴权配置
os.environ.setdefault("ALIBABA_CLOUD_ROLE_SESSION_NAME", "normal-data-fetcher")

# 导入工具库（metric 拉取只依赖 tools；SLS 相关依赖做成可选）
try:
    from tools.paas_entity_tools import umodel_get_entities
    from tools.paas_data_tools import umodel_get_golden_metrics
except ImportError as e:
    print(f"❌ 依赖缺失(必需 tools.*): {e}")
    sys.exit(1)

# 可选依赖：如果你未来想把 trace/SLS 逻辑也塞回这个脚本，再使用这些包
try:
    from aliyun.log import LogClient, GetLogsRequest  # noqa: F401
    from alibabacloud_sts20150401.client import Client as StsClient  # noqa: F401
    from alibabacloud_sts20150401 import models as sts_models  # noqa: F401
    from alibabacloud_tea_openapi import models as open_api_models  # noqa: F401
except Exception:
    LogClient = None  # type: ignore
    GetLogsRequest = None  # type: ignore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
def fetch_metrics(self, start_ts, end_ts):
    """
    获取节点指标：
    1. 使用分片查询 (Chunking) 防止 API 自动降采样 (解决 60s 粒度问题)
    2. 统一时间戳单位为纳秒 (防止索引报错)
    3. 结合 Golden Metrics 和 CMS 原始接口 (补全缺失指标)
    4. 使用 ffill+fillna 策略 (解决空洞/断层问题)
    """
    logger.info(f"🚀 [Metric] 开始获取正常时段的节点指标 ({start_ts} -> {end_ts})...")

    # 统计策略提取到的数据量（两者都要）：
    # raw_points: 从接口返回并解析到 (timestamp, value) 的数量
    # new_points: 写入 node_data 时新增的 (timestamp, metric) cell 数量
    # overwrite_points: 写入 node_data 时覆盖已有 cell 的数量
    #
    # 说明：当前脚本仅保留 Golden Metrics 策略。
    overall_stat = {
        "golden_metrics": {"raw_points": 0, "new_points": 0, "overwrite_points": 0},
    }

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

    # [关键设置] 分片大小设为 30分钟 (1800s)
    # 时间跨度短时，API 会返回原始高精度数据 (如 10s/15s)；跨度长时会自动聚合为 60s
    CHUNK_SIZE = 1800

    # 运行级统计/追踪：用于解释“发现活跃节点数”和“最终写入节点数”为何不一致
    node_reports = []
    skipped_no_entity_id = 0
    nodes_no_data = 0
    nodes_with_data = 0

    for i, node in enumerate(nodes):
        instance_id = node.get('instance_id')
        entity_id = node.get('__entity_id__')
        if not entity_id:
            skipped_no_entity_id += 1
            node_reports.append({
                "instance_id": instance_id,
                "entity_id": entity_id,
                "status": "skipped",
                "reason": "missing___entity_id__",
                "gm_raw_points": 0,
                "gm_new_points": 0,
                "gm_overwrite_points": 0,
                "rows_written": 0,
            })
            continue

        # node_data: { timestamp_ns: { metric_name: value } }
        node_data = {}

        node_stat = {
            "golden_metrics": {"raw_points": 0, "new_points": 0, "overwrite_points": 0},
        }
        gm_exception = None

        # === [核心逻辑] 分片循环查询 ===
        current_chunk_start = start_ts
        while current_chunk_start < end_ts:
            current_chunk_end = min(current_chunk_start + CHUNK_SIZE, end_ts)

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
                            import ast
                            vals = ast.literal_eval(item.get('__value__', '[]'))
                            tss = ast.literal_eval(item.get('__ts__', '[]'))
                            for v, t in zip(vals, tss):
                                node_stat["golden_metrics"]["raw_points"] += 1
                                overall_stat["golden_metrics"]["raw_points"] += 1

                                # [修复] 强制转换为纳秒 (19位)，防止与 CMS 毫秒混用导致 Pandas 崩溃
                                t_int = int(t)
                                t_ns = t_int * 1000000 if t_int < 1e14 else t_int

                                if t_ns not in node_data:
                                    node_data[t_ns] = {}

                                if m_name in node_data[t_ns]:
                                    node_stat["golden_metrics"]["overwrite_points"] += 1
                                    overall_stat["golden_metrics"]["overwrite_points"] += 1
                                else:
                                    node_stat["golden_metrics"]["new_points"] += 1
                                    overall_stat["golden_metrics"]["new_points"] += 1

                                node_data[t_ns][m_name] = v
            except Exception as e:
                gm_exception = str(e)

            # 推进到下一个分片
            current_chunk_start = current_chunk_end

        # 如果一个节点在该时间段内完全拿不到目标指标，node_data 为空，则不会产生 CSV 行
        if not node_data:
            nodes_no_data += 1
            node_reports.append({
                "instance_id": instance_id,
                "entity_id": entity_id,
                "status": "no_data",
                "reason": "golden_metrics_empty" if not gm_exception else "golden_metrics_exception",
                "gm_exception": gm_exception,
                "gm_raw_points": node_stat["golden_metrics"]["raw_points"],
                "gm_new_points": node_stat["golden_metrics"]["new_points"],
                "gm_overwrite_points": node_stat["golden_metrics"]["overwrite_points"],
                "rows_written": 0,
            })
            continue

        nodes_with_data += 1

        rows_written_for_node = 0

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
                rows_written_for_node = len(new_timestamps)
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
                rows_written_for_node = len(node_data)
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

        # 节点级别策略统计：对比两种策略各自贡献了多少点
        try:
            merged_cells = sum(len(metric_map) for metric_map in node_data.values()) if node_data else 0
        except Exception:
            merged_cells = 0

        logger.info(
            f"   [Node {instance_id}] GM raw/new/ovw="
            f"{node_stat['golden_metrics']['raw_points']}/"
            f"{node_stat['golden_metrics']['new_points']}/"
            f"{node_stat['golden_metrics']['overwrite_points']}; "
            f"merged_cells={merged_cells}"
        )

        node_reports.append({
            "instance_id": instance_id,
            "entity_id": entity_id,
            "status": "ok",
            "reason": None,
            "gm_exception": gm_exception,
            "gm_raw_points": node_stat["golden_metrics"]["raw_points"],
            "gm_new_points": node_stat["golden_metrics"]["new_points"],
            "gm_overwrite_points": node_stat["golden_metrics"]["overwrite_points"],
            "rows_written": rows_written_for_node,
            "merged_cells": merged_cells,
            "min_ts_ns": min(node_data.keys()) if node_data else None,
            "max_ts_ns": max(node_data.keys()) if node_data else None,
        })

        if (i + 1) % 5 == 0:
            print(f"   已处理 {i+1}/{len(nodes)} 个节点...", end='\r')

    # 统一写入文件
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows_to_write)
        logger.info(f"\n✅ [Metric] 已保存 {len(rows_to_write)} 条指标数据至 {csv_path}")

        # 额外保存策略统计（更可追溯，方便你后续做对比）
        try:
            stat_path = os.path.join(
                self.output_dir,
                f"normal_metrics_{self.args.file_name}_strategy_stats.json"
            )
            with open(stat_path, "w", encoding="utf-8") as f:
                json.dump(overall_stat, f, indent=2, ensure_ascii=False)

            logger.info(
                "📊 [Metric] 策略统计(整体)："
                f"GM raw/new/ovw={overall_stat['golden_metrics']['raw_points']}/"
                f"{overall_stat['golden_metrics']['new_points']}/"
                f"{overall_stat['golden_metrics']['overwrite_points']}; "
                f"已写入 {stat_path}"
            )
        except Exception as e:
            logger.warning(f"⚠️ [Metric] 写入策略统计失败: {e}")

        # 保存节点明细，解释“发现活跃节点数”与“最终写入节点数”的差异
        try:
            report_path = os.path.join(
                self.output_dir,
                f"normal_metrics_{self.args.file_name}_node_report.json"
            )
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump({
                    "total_active_nodes": len(nodes),
                    "skipped_no_entity_id": skipped_no_entity_id,
                    "nodes_no_data": nodes_no_data,
                    "nodes_with_data": nodes_with_data,
                    "node_reports": node_reports,
                }, f, indent=2, ensure_ascii=False)
            logger.info(
                "🧾 [Metric] 节点明细报告："
                f"active={len(nodes)}, skipped(no_entity_id)={skipped_no_entity_id}, "
                f"no_data={nodes_no_data}, with_data={nodes_with_data}; "
                f"已写入 {report_path}"
            )
        except Exception as e:
            logger.warning(f"⚠️ [Metric] 写入节点明细报告失败: {e}")

    except Exception as e:
        logger.error(f"❌ 写入 CSV 失败: {e}")


class MetricFetcher:
    """
    方案 A：保留 self 风格（与 2_get_normalData.py 的写法一致），便于你单独拿出来测试。
    - 统计口径：两者都要（raw_points + new_points/overwrite_points）
    """

    def __init__(self, args):
        self.args = args
        self.output_dir = args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # 复用上面的模块级函数作为“方法”实现，避免大段缩进改动
    fetch_metrics = fetch_metrics


def _parse_time_str(time_str):
    """
    解析时间字符串到 datetime。
    支持：
    - 'YYYY-mm-dd HH:MM:SS'
    - 'YYYY-mm-ddTHH:MM:SS'
    """
    if not time_str:
        return None
    s = str(time_str).strip()
    s = s.replace("T", " ")
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def main():
    parser = argparse.ArgumentParser(description="单独测试 metric 拉取（GM + CMS 补缺），并统计两种策略的数据点数")
    parser.add_argument("--output-dir", default="data/demo", help="输出目录")
    parser.add_argument("--interval", type=int, default=30, help="指标重采样间隔(秒)，<=0 表示不重采样")
    parser.add_argument("--file-name", type=str, default="metric_demo", help="输出文件名后缀")

    # 时间窗口：优先用 start/end；否则用 window-hours 回溯
    parser.add_argument("--start-time", type=str, default="2026-01-20 20:00:00", help="开始时间：YYYY-mm-dd HH:MM:SS")
    parser.add_argument("--end-time", type=str, default="2026-01-20 23:59:59", help="结束时间：YYYY-mm-dd HH:MM:SS（默认当前时间）")
    parser.add_argument("--window-hours", type=float, default=4.0, help="未提供 start-time 时，回溯小时数")

    args = parser.parse_args()

    end_dt = _parse_time_str(args.end_time) if args.end_time else datetime.now()
    start_dt = _parse_time_str(args.start_time) if args.start_time else (end_dt - timedelta(hours=args.window_hours))

    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())
    if start_ts >= end_ts:
        raise ValueError(f"时间窗口不合法：start_ts({start_ts}) >= end_ts({end_ts})")

    logger.info(f"📅 测试窗口: {start_dt.strftime('%Y-%m-%d %H:%M:%S')} -> {end_dt.strftime('%Y-%m-%d %H:%M:%S')} ({end_ts - start_ts}s)")

    fetcher = MetricFetcher(args)
    fetcher.fetch_metrics(start_ts, end_ts)


if __name__ == "__main__":
    main()
