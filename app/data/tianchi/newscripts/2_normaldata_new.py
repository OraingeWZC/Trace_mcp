#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2_normaldata_new.py

目标：
在给定时间窗内：
1) 从 SLS tracing logstore 拉取所有 span（Trace/Span 原始日志），并按 2_get_normalData.py 的 trace CSV 格式保存；
2) 根据 trace 中出现的 k8s.node.name（故障注入维度），查询对应的节点指标（Prometheus metricstore，.metric_set 语句），
   并按 2_get_normalData.py 的 metrics CSV 风格保存（problem_id/fault_type/instance_id/timestamp + 指标列）。

说明：
- Trace/Span 数据：走 SLS GetLogs (LogClient + GetLogsRequest)。
- 指标数据：走 CMS SPL（GetEntityStoreDataRequest），使用你从可视化界面拿到的 .metric_set with(...) 模板。
- 为了“尽量和 2_get_normalData.py 输出格式一致”，这里：
  - trace CSV 的列名与 2_get_normalData.py.fetch_traces 生成的 normal_traces*.csv 对齐；
  - metrics CSV 的列名与 2_get_normalData.py.fetch_metrics 的风格对齐（instance_id 字段这里存 k8s.node.name）。

运行示例（在 app/data/tianchi 目录）：
  python .\\scripts\\2_normaldata_new.py --start-time "now()-1h" --end-time "now()"

需要的 .env（至少）：
  ALIBABA_CLOUD_ACCESS_KEY_ID / ALIBABA_CLOUD_ACCESS_KEY_SECRET
  SLS_PROJECT_NAME / SLS_LOGSTORE_NAME / SLS_REGION
  WORKSPACE_NAME（用于 CMS 查询；若未配置会使用默认值）
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 确保从任意工作目录运行都能 import 到 tools / scripts 下的模块
project_root = Path(__file__).resolve().parents[1]  # app/data/tianchi
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# 让脚本与现有目录结构保持一致：scripts/config.py 会 load_dotenv()
import config  # noqa: F401  pylint: disable=unused-import


try:
    from aliyun.log import LogClient
    from aliyun.log.getlogsrequest import GetLogsRequest
except Exception as e:  # pragma: no cover
    raise RuntimeError("缺少 aliyun-log-python-sdk 依赖，请先安装 aliyun-log-python-sdk") from e


try:
    from alibabacloud_sts20150401.client import Client as StsClient
    from alibabacloud_sts20150401 import models as sts_models
    from alibabacloud_tea_openapi import models as open_api_models
except Exception:
    StsClient = None  # type: ignore
    sts_models = None  # type: ignore
    open_api_models = None  # type: ignore


from tools.common import create_cms_client, execute_cms_query
from tools.constants import WORKSPACE_NAME


@dataclass
class TraceRow:
    trace_id: str = ""
    span_id: str = ""
    parent_id: str = ""
    service_name: str = ""
    node_name: str = ""
    pod_name: str = ""
    url: str = ""
    span_kind: str = ""
    start_time_ms: str = ""
    end_time_ms: str = ""
    duration_ms: str = ""
    status_code: str = ""
    http_status_code: str = ""
    fault_type: str = "normal"
    fault_instance: str = ""
    problem_id: str = "normal_000"


TRACE_CSV_HEADERS = [
    "TraceID",
    "SpanId",
    "ParentID",
    "ServiceName",
    "NodeName",
    "PodName",
    "URL",
    "SpanKind",
    "StartTimeMs",
    "EndTimeMs",
    "DurationMs",
    "StatusCode",
    "HttpStatusCode",
    "fault_type",
    "fault_instance",
    "problem_id",
]


def convert_to_unixtime(time_str: str) -> int:
    """
    将时间字符串转换为 Unix 时间戳（秒）。
    支持：
    - now() / now
    - now()-1d / now-1d
    - now()-1h / now-1h
    - now()-10m / now-10m
    - now()-30s / now-30s
    - 2026-02-11 10:00:00 / 2026-02-11
    - 纯数字（秒/毫秒/纳秒会尽量归一到秒）
    """
    if time_str is None:
        raise ValueError("time_str is None")

    s = str(time_str).strip()
    if not s:
        raise ValueError("time_str is empty")

    if s.isdigit():
        ts = int(s)
        # ns / ms / s heuristics
        if ts > 10**15:
            return int(ts / 10**9)
        if ts > 10**12:
            return int(ts / 1000)
        return ts

    if s.startswith("now"):
        now_dt = datetime.now()
        tail = s.replace("now()", "now").replace("now", "").strip()
        if not tail:
            return int(now_dt.timestamp())

        if tail.startswith("-"):
            offset = tail[1:].strip()
            m = re.match(r"^(\d+)([smhdw])$", offset)
            if not m:
                raise ValueError(f"无效的 now 偏移格式: {s}")
            value = int(m.group(1))
            unit = m.group(2)
            if unit == "s":
                dt = now_dt - timedelta(seconds=value)
            elif unit == "m":
                dt = now_dt - timedelta(minutes=value)
            elif unit == "h":
                dt = now_dt - timedelta(hours=value)
            elif unit == "d":
                dt = now_dt - timedelta(days=value)
            elif unit == "w":
                dt = now_dt - timedelta(weeks=value)
            else:
                raise ValueError(f"不支持的单位: {unit}")
            return int(dt.timestamp())

    # absolute datetime
    try:
        if len(s) == 10:
            dt = datetime.strptime(s, "%Y-%m-%d")
        else:
            dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        return int(dt.timestamp())
    except ValueError as e:
        raise ValueError(f"无法解析时间字符串: {s}") from e


def _safe_json_loads(s: Any) -> Dict[str, Any]:
    if not s:
        return {}
    if isinstance(s, dict):
        return s
    if not isinstance(s, str):
        return {}
    try:
        return json.loads(s)
    except Exception:
        return {}


def _ns_to_ms_str(ns_str: Any) -> str:
    """
    tracing logstore 里 startTime/endTime 常是纳秒字符串（19位）。
    这里统一转成毫秒字符串，和 2_get_normalData.py 输出格式一致。
    """
    if ns_str is None:
        return ""
    try:
        v = int(str(ns_str))
    except Exception:
        return ""
    # ns -> ms
    if v > 10**14:
        return str(int(v / 10**6))
    # ms already
    if v > 10**11:
        return str(v)
    # seconds
    return str(int(v * 1000))


def _duration_to_ms_str(duration_any: Any) -> str:
    """
    duration 常见是纳秒字符串（或微秒/毫秒），这里做一个保守转换：
    - >1e14 认为是 ns；>1e11 认为是 us；>1e8 认为是 ms；否则原样。
    """
    if duration_any is None:
        return ""
    try:
        v = float(str(duration_any))
    except Exception:
        return ""

    if v > 1e14:
        return str(int(v / 1e6))
    if v > 1e11:
        return str(int(v / 1e3))
    if v > 1e8:
        return str(int(v))
    return str(int(v))


def _extract_url_from_span(span: Dict[str, Any]) -> str:
    """
    注意：这里的 URL 列按照 2_get_normalData.py 的口径，直接使用 spanName。

    在 tracing 场景里 spanName 更像 “OperationName/接口名/方法名”，
    对 RPC/DB/内部 span 也有意义；而真实 http.url/url.full 往往只在 HTTP span 出现，
    混在同一列会导致你说的“太杂/高基数”问题。
    """
    v = span.get("spanName")
    return v if isinstance(v, str) else ""


def span_to_trace_row(span: Dict[str, Any]) -> Tuple[TraceRow, Optional[str]]:
    """
    将 SLS tracing logstore 的一条 span 日志转换为 trace CSV 行。
    返回 (TraceRow, k8s_node_name)。
    """
    resources = _safe_json_loads(span.get("resources"))
    node_name = resources.get("k8s.node.name") or ""
    pod_name = resources.get("k8s.pod.name") or span.get("hostname") or ""

    row = TraceRow(
        trace_id=str(span.get("traceId") or ""),
        span_id=str(span.get("spanId") or ""),
        parent_id=str(span.get("parentSpanId") or ""),
        service_name=str(span.get("serviceName") or ""),
        node_name=str(node_name),
        pod_name=str(pod_name),
        url=_extract_url_from_span(span),
        span_kind=str(span.get("kind") or ""),
        start_time_ms=_ns_to_ms_str(span.get("startTime")),
        end_time_ms=_ns_to_ms_str(span.get("endTime")),
        duration_ms=_duration_to_ms_str(span.get("duration")),
        status_code=str(span.get("statusCode") or ""),
        http_status_code=str(span.get("httpStatusCode") or ""),
    )

    return row, (str(node_name) if node_name else None)


def _trace_has_single_topology_root(spans: List[Dict[str, Any]]) -> bool:
    """
    参考 2_get_normalData.py 的 check_orphan_root：
    - “拓扑根”定义：ParentID 不指向当前 Trace 的任何 SpanId（包含 ParentID 为空 / -1 / 指向缺失节点等情况）
    - 严格要求：只能有 1 个根；否则视为断链/多根，丢弃该 Trace
    """
    if not spans:
        return False

    span_ids = set()
    for s in spans:
        sid = str(s.get("SpanId", "")).strip()
        if sid:
            span_ids.add(sid)
    if not span_ids:
        return False

    root_count = 0
    for s in spans:
        pid = str(s.get("ParentID", "")).strip()
        if pid not in span_ids:
            root_count += 1
    return root_count == 1


def _strict_trace_filter(spans: List[Dict[str, Any]], start_ts: int) -> bool:
    """
    参考 2_get_normalData.py 的严格清洗逻辑：
    - 至少 2 个 Span
    - 所有 Span 的 StatusCode 不为错误（>1 视为错误）
    - 任一 Span 的 StartTimeMs 早于窗口起点，则整条 Trace 丢弃（避免跨窗口）
    - 只能有一个拓扑根（去除断链/多根）
    """
    if not spans or len(spans) < 2:
        return False

    window_start_ms = float(start_ts) * 1000.0
    for s in spans:
        # 1) 错误状态
        sc_raw = s.get("StatusCode", "")
        try:
            sc = int(sc_raw) if sc_raw and str(sc_raw).isdigit() else 0
        except Exception:
            sc = 0
        if sc > 1:
            return False

        # 2) 跨窗口：只要有一个 span 早于窗口起点，整条 trace 丢弃
        try:
            st_ms = float(s.get("StartTimeMs", "0") or "0")
            if st_ms < window_start_ms:
                return False
        except Exception:
            # 无法解析时间时不强行丢弃（避免因为脏数据把 trace 全砍掉）
            pass

    # 3) 多根/断链
    return _trace_has_single_topology_root(spans)


def init_sls_client(region: str) -> LogClient:
    """
    初始化 SLS LogClient。
    - 若配置了 ALIBABA_CLOUD_ROLE_ARN 且安装了 STS SDK，则走 AssumeRole 获取临时凭证；
    - 否则直接使用 AK/SK 直连。
    """
    access_key_id = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_ID") or ""
    access_key_secret = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_SECRET") or ""
    if not access_key_id or not access_key_secret:
        raise RuntimeError("缺少 ALIBABA_CLOUD_ACCESS_KEY_ID / ALIBABA_CLOUD_ACCESS_KEY_SECRET（请配置 .env）")

    endpoint = f"{region}.log.aliyuncs.com"
    role_arn = os.environ.get("ALIBABA_CLOUD_ROLE_ARN") or ""

    if not role_arn or StsClient is None:
        logger.info("SLS 使用 AK/SK 直连（未配置 RoleArn 或缺少 STS SDK）。")
        return LogClient(endpoint=endpoint, accessKeyId=access_key_id, accessKey=access_key_secret)

    # STS AssumeRole
    logger.info("SLS 使用 STS AssumeRole 获取临时凭证。")
    sts_cfg = open_api_models.Config(  # type: ignore[union-attr]
        access_key_id=access_key_id,
        access_key_secret=access_key_secret,
        endpoint=f"sts.{region}.aliyuncs.com",
    )
    sts_client = StsClient(sts_cfg)  # type: ignore[misc]
    resp = sts_client.assume_role(
        sts_models.AssumeRoleRequest(  # type: ignore[union-attr]
            role_arn=role_arn,
            role_session_name=os.environ.get("ALIBABA_CLOUD_ROLE_SESSION_NAME", "normaldata-new") or "normaldata-new",
            duration_seconds=3600,
        )
    )
    creds = resp.body.credentials
    return LogClient(
        endpoint=endpoint,
        accessKeyId=creds.access_key_id,
        accessKey=creds.access_key_secret,
        securityToken=creds.security_token,
    )


def iter_sls_logs(
    sls_client: LogClient,
    project: str,
    logstore: str,
    start_ts: int,
    end_ts: int,
    query: str,
    page_size: int = 100,
    max_logs: int = 0,
) -> Iterable[Dict[str, Any]]:
    """
    分页遍历 SLS GetLogs 返回的日志 contents(dict)。
    max_logs=0 表示不限制（直到读完窗口内的数据）。
    """
    offset = 0
    total = 0

    while True:
        line = page_size
        if max_logs and (max_logs - total) < line:
            line = max_logs - total
        if line <= 0:
            break

        req = GetLogsRequest(project, logstore, query=query, fromTime=start_ts, toTime=end_ts, line=line, offset=offset)
        res = sls_client.get_logs(req)
        if not res or not res.get_logs():
            break

        logs = res.get_logs()
        if not logs:
            break

        for log in logs:
            d = log.get_contents() or {}
            yield d
            total += 1
            if max_logs and total >= max_logs:
                return

        offset += len(logs)
        if len(logs) < line:
            break


def fetch_k8s_node_entities(
    cms_region: str,
    workspace: str,
    start_ts: int,
    end_ts: int,
    limit: int,
) -> List[Dict[str, Any]]:
    """
    拉取 k8s.node 实体（返回 List[Dict]），用于:
    - 获取 cluster_id
    - name/internal_ip/__entity_id__
    """
    cms_client = create_cms_client(cms_region)
    spl = (
        ".entity_set with(domain='k8s', name='k8s.node') "
        "| entity-call get_entities() "
        f"| limit {int(limit)}"
    )
    res = execute_cms_query(cms_client, workspace, spl, start_ts, end_ts, limit=limit)
    if not res or res.error:
        logger.warning(f"k8s.node 实体查询失败: {getattr(res, 'message', '')}")
        return []
    return res.data or []


def parse_metric_series_rows(rows: List[Dict[str, Any]]) -> Dict[int, float]:
    """
    将 .metric_set 返回的多条序列行，聚合为 timestamp(ns) -> value(float)。
    处理多序列时同一 timestamp 做平均。
    期望每行包含 __ts__ / __value__（通常是字符串形式的列表）。
    """
    acc: Dict[int, Tuple[float, int]] = {}

    for r in rows:
        ts_raw = r.get("__ts__")
        val_raw = r.get("__value__")

        try:
            ts_list = ast.literal_eval(ts_raw) if isinstance(ts_raw, str) else ts_raw
            val_list = ast.literal_eval(val_raw) if isinstance(val_raw, str) else val_raw
        except Exception:
            continue

        if not isinstance(ts_list, list) or not isinstance(val_list, list):
            continue

        for t, v in zip(ts_list, val_list):
            try:
                t_i = int(t)
                v_f = float(v) if v is not None and str(v) != "null" else float("nan")
            except Exception:
                continue
            if t_i not in acc:
                acc[t_i] = (0.0, 0)
            s, c = acc[t_i]
            if not (v_f != v_f):  # NaN check
                acc[t_i] = (s + v_f, c + 1)

    out: Dict[int, float] = {}
    for t, (s, c) in acc.items():
        if c > 0:
            out[t] = s / c
    return out


def build_metric_set_spl(
    region_id: str,
    cluster_id: str,
    metric: str,
    step: str,
    aggregate: str,
    internal_ip: str,
    node_name: str,
    metric_set_name: str = "k8s.metric.high_level_metric_node",
) -> str:
    """
    根据你提供的可视化界面语句模板构造 .metric_set SPL。
    """
    storage_name = f"{region_id}/{cluster_id}"
    filter_query = f"internal_ip = ''{internal_ip}'' and node = ''{node_name}''"

    return (
        ".metric_set with("
        "storage_kind='aliyun_prometheus', "
        "storage_domain='k8s', "
        f"storage_name='{storage_name}', "
        "domain='k8s', "
        f"name='{metric_set_name}', "
        "source='metrics', "
        f"metric='{metric}', "
        f"step='{step}', "
        f"aggregate='{aggregate}', "
        f"query='{filter_query}'"
        ")"
    )


def fetch_k8s_node_metric_series(
    cms_region: str,
    workspace: str,
    start_ts: int,
    end_ts: int,
    region_id: str,
    cluster_id: str,
    node_name: str,
    internal_ip: str,
    metric: str,
    step: str,
    aggregate: str,
    metric_set_name: str,
) -> Dict[int, float]:
    cms_client = create_cms_client(cms_region)
    spl = build_metric_set_spl(
        region_id=region_id,
        cluster_id=cluster_id,
        metric=metric,
        step=step,
        aggregate=aggregate,
        internal_ip=internal_ip,
        node_name=node_name,
        metric_set_name=metric_set_name,
    )
    res = execute_cms_query(cms_client, workspace, spl, start_ts, end_ts, limit=1000)
    if not res or res.error:
        logger.warning(f"k8s.node 指标查询失败: metric={metric}, node={node_name}, msg={getattr(res, 'message', '')}")
        return {}
    rows = res.data or []
    return parse_metric_series_rows(rows)


def write_traces_csv(
    sls_client: LogClient,
    project: str,
    logstore: str,
    start_ts: int,
    end_ts: int,
    query: str,
    out_csv: Path,
    page_size: int,
    max_logs: int,
    trace_log_every: int = 2000,
) -> List[str]:
    """
    写 trace CSV，并返回在 trace 中出现过的 k8s.node.name 列表（去重后保持顺序）。
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    # 为了实现“按 TraceID 分组 + Trace 内按开始时间排序”，需要先把 span 全量缓存在内存里。
    # 对正常数据采集（时间窗通常不大）这是可以接受的；如果你后续要拉超大窗口，再考虑分批/落盘。
    trace_buffer: Dict[str, List[Dict[str, Any]]] = {}

    scanned_rows = 0
    for span in iter_sls_logs(
        sls_client=sls_client,
        project=project,
        logstore=logstore,
        start_ts=start_ts,
        end_ts=end_ts,
        query=query,
        page_size=page_size,
        max_logs=max_logs,
    ):
        row, _ = span_to_trace_row(span)
        if not row.trace_id:
            continue

        row_dict: Dict[str, Any] = {
            "TraceID": row.trace_id,
            "SpanId": row.span_id,
            "ParentID": row.parent_id,
            "ServiceName": row.service_name,
            "NodeName": row.node_name,
            "PodName": row.pod_name,
            "URL": row.url,
            "SpanKind": row.span_kind,
            "StartTimeMs": row.start_time_ms,
            "EndTimeMs": row.end_time_ms,
            "DurationMs": row.duration_ms,
            "StatusCode": row.status_code,
            "HttpStatusCode": row.http_status_code,
            "fault_type": row.fault_type,
            "fault_instance": row.fault_instance,
            "problem_id": row.problem_id,
        }

        trace_buffer.setdefault(row.trace_id, []).append(row_dict)
        scanned_rows += 1

        if trace_log_every > 0 and scanned_rows and scanned_rows % trace_log_every == 0:
            logger.info("Trace 扫描中: 已扫描 span=%d, traces=%d", scanned_rows, len(trace_buffer))

    # 参考 2_get_normalData.py 的严格过滤：去除错误/跨窗口/多根等不“纯净”的 trace
    kept_traces: List[Tuple[float, str, List[Dict[str, Any]]]] = []
    dropped_multi_root = 0
    dropped_other = 0
    for tid, spans in trace_buffer.items():
        if not spans:
            continue
        if not _strict_trace_filter(spans, start_ts=start_ts):
            # 粗略区分一下多根/断链，方便你观察过滤效果
            if not _trace_has_single_topology_root(spans):
                dropped_multi_root += 1
            else:
                dropped_other += 1
            continue

        def _st_key(s: Dict[str, Any]) -> float:
            try:
                return float(s.get("StartTimeMs", "0") or "0")
            except Exception:
                return 0.0

        spans_sorted = sorted(spans, key=_st_key)
        trace_start = _st_key(spans_sorted[0])
        kept_traces.append((trace_start, tid, spans_sorted))

    kept_traces.sort(key=lambda x: x[0])  # Trace 之间按开始时间排序

    # node_list 以最终“保留的 trace spans”为准，避免把已被过滤掉的 trace 节点也带入后续指标对齐逻辑
    seen_nodes = set()
    node_list: List[str] = []
    for _, _, spans_sorted in kept_traces:
        for s in spans_sorted:
            n = str(s.get("NodeName", "") or "").strip()
            if n and n not in seen_nodes:
                seen_nodes.add(n)
                node_list.append(n)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TRACE_CSV_HEADERS)
        writer.writeheader()
        total_rows = 0
        for _, _, spans_sorted in kept_traces:
            writer.writerows(spans_sorted)
            total_rows += len(spans_sorted)

    logger.info(
        "Trace CSV 已保存: %s (spans=%d, traces=%d, unique_nodes=%d, dropped_multi_root=%d, dropped_other=%d)",
        out_csv,
        sum(len(x[2]) for x in kept_traces),
        len(kept_traces),
        len(node_list),
        dropped_multi_root,
        dropped_other,
    )
    return node_list


def write_node_metrics_csv(
    cms_region: str,
    workspace: str,
    start_ts: int,
    end_ts: int,
    sls_region_id: str,
    cluster_id: str,
    nodes: List[str],
    node_entities: List[Dict[str, Any]],
    metrics: List[str],
    step: str,
    aggregate: str,
    metric_set_name: str,
    out_csv: Path,
) -> None:
    """
    将节点指标按 2_get_normalData 的 metrics CSV 风格写出：
      problem_id,fault_type,instance_id,timestamp,<metric columns...>
    其中 instance_id 这里用 k8s.node.name（与故障注入对齐）。
    timestamp 为纳秒（与 metricstore 返回一致）。
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics_sorted = sorted(set(metrics))

    # 建 name -> internal_ip / cluster_id 映射
    name_to_ip: Dict[str, str] = {}
    inferred_cluster_id = cluster_id or ""

    for rec in node_entities:
        n = rec.get("name") or rec.get("nodeName") or rec.get("provider_id")
        ip = rec.get("internal_ip") or rec.get("internalIp") or rec.get("internalIP")
        cid = rec.get("cluster_id") or rec.get("clusterId")
        if cid and not inferred_cluster_id:
            inferred_cluster_id = str(cid)
        if n and ip:
            name_to_ip[str(n)] = str(ip)

    if not inferred_cluster_id:
        raise RuntimeError("无法推断 cluster_id（请在参数中显式传 --cluster-id 或确保 k8s.node 实体返回 cluster_id 字段）")

    headers = ["problem_id", "fault_type", "instance_id", "timestamp"] + metrics_sorted

    total_rows = 0
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        for i, node_name in enumerate(nodes):
            internal_ip = name_to_ip.get(node_name)
            if not internal_ip:
                # 尝试从 node_name 中抽取 IP
                m = re.search(r"(\\d+\\.\\d+\\.\\d+\\.\\d+)", node_name)
                internal_ip = m.group(1) if m else ""

            if not internal_ip:
                logger.warning(f"[metrics] 跳过 node={node_name}（无法得到 internal_ip）")
                continue

            # 每个 metric 拉一条序列，并合并成 ts -> {metric: value}
            ts_map: Dict[int, Dict[str, float]] = {}
            for metric in metrics_sorted:
                series = fetch_k8s_node_metric_series(
                    cms_region=cms_region,
                    workspace=workspace,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    region_id=sls_region_id,
                    cluster_id=inferred_cluster_id,
                    node_name=node_name,
                    internal_ip=internal_ip,
                    metric=metric,
                    step=step,
                    aggregate=aggregate,
                    metric_set_name=metric_set_name,
                )
                for ts_ns, v in series.items():
                    if ts_ns not in ts_map:
                        ts_map[ts_ns] = {}
                    ts_map[ts_ns][metric] = v

            # 写行：每个 timestamp 一行
            for ts_ns in sorted(ts_map.keys()):
                row: Dict[str, Any] = {
                    "problem_id": "normal_000",
                    "fault_type": "normal",
                    "instance_id": node_name,
                    "timestamp": ts_ns,
                }
                for metric in metrics_sorted:
                    row[metric] = ts_map[ts_ns].get(metric, 0.0)
                writer.writerow(row)
                total_rows += 1

            if (i + 1) % 5 == 0:
                logger.info(f"[metrics] 已处理节点: {i+1}/{len(nodes)} (rows={total_rows})")

    logger.info(f"Node metrics CSV 已保存: {out_csv} (rows={total_rows}, nodes={len(nodes)}, metrics={len(metrics_sorted)})")


def main() -> None:
    parser = argparse.ArgumentParser(description="拉取指定时间窗的所有 Trace + 对应节点指标，并按 2_get_normalData 风格导出 CSV")

    parser.add_argument("--start-time", default='2026-02-17 18:00:00', help="开始时间（now()-1h / 'YYYY-MM-DD HH:MM:SS' / 时间戳）")
    parser.add_argument("--end-time", default='2026-02-17 18:30:00', help="结束时间（now() / 'YYYY-MM-DD HH:MM:SS' / 时间戳）")
    # parser.add_argument("--start-time", default='now()-1h', help="开始时间（now()-1h / 'YYYY-MM-DD HH:MM:SS' / 时间戳）")
    # parser.add_argument("--end-time", default='now()', help="结束时间（now() / 'YYYY-MM-DD HH:MM:SS' / 时间戳）")

    parser.add_argument("--workspace", default=os.environ.get("WORKSPACE_NAME", WORKSPACE_NAME), help="CMS workspace（默认读 WORKSPACE_NAME）")
    parser.add_argument("--cms-region", default=os.environ.get("SLS_REGION", "cn-hongkong"), help="CMS region（默认读 SLS_REGION）")
    parser.add_argument("--sls-region", default=os.environ.get("SLS_REGION", "cn-hongkong"), help="SLS region（默认读 SLS_REGION）")
    parser.add_argument("--sls-project", default=getattr(config, "SLS_PROJECT_NAME", "") or os.environ.get("SLS_PROJECT_NAME", ""), help="SLS project")
    parser.add_argument("--sls-logstore", default=getattr(config, "SLS_LOGSTORE_NAME", "") or os.environ.get("SLS_LOGSTORE_NAME", ""), help="SLS logstore（tracing logstore）")

    parser.add_argument("--trace-query", default="*", help="SLS trace 查询语句（默认: *）")
    parser.add_argument("--page-size", type=int, default=100, help="SLS 分页大小（默认: 100）")
    parser.add_argument("--max-trace-logs", type=int, default=400000, help="最多拉取多少条 span 日志（0 表示不限制）")
    parser.add_argument("--trace-log-every",type=int, default=10000, help="Trace 扫描进度日志打印频率（0 表示关闭；默认: 2000）",)
    parser.add_argument("--k8s-node-limit", type=int, default=2000, help="k8s.node 实体查询上限（默认: 2000）")
    parser.add_argument(
        "--node-metrics-scope",
        choices=["all", "trace"],
        default="all",
        help=(
            "节点指标拉取范围："
            "all=按 k8s.node 实体全量拉取；"
            "trace=只对 trace 覆盖到的 k8s.node.name 拉取（默认: all）"
        ),
    )
    parser.add_argument("--cluster-id", default="", help="k8s 集群 ID（可选；为空则尝试从 k8s.node 实体推断）")
    parser.add_argument("--metric-set-name", default="k8s.metric.high_level_metric_node", help="metric_set name（默认: k8s.metric.high_level_metric_node）")
    parser.add_argument("--metrics", default="node_cpu_usage_rate,node_memory_usage_rate,node_disk_usage_rate", help="节点指标列表（逗号分隔）")
    parser.add_argument("--metric-step", default="30s", help="指标 step（默认: 30s）")
    parser.add_argument("--metric-aggregate", default="true", help="指标 aggregate（默认: true）")

    parser.add_argument("--output-dir", default="data/rca", help="输出目录（默认: data/demo）")
    parser.add_argument("--suffix", default="_0217_4e5", help="输出文件名后缀（例如 _0210）")

    args = parser.parse_args()

    start_ts = convert_to_unixtime(args.start_time)
    end_ts = convert_to_unixtime(args.end_time)
    if end_ts <= start_ts:
        raise ValueError("end_time 必须大于 start_time")

    if not args.sls_project or not args.sls_logstore:
        raise RuntimeError("缺少 SLS_PROJECT_NAME / SLS_LOGSTORE_NAME（请配置 .env 或通过参数传入）")

    output_dir = Path(args.output_dir)
    traces_csv = output_dir / f"normal_traces{args.suffix}.csv"
    metrics_csv = output_dir / f"normal_metrics_{args.suffix.lstrip('_') or 'new'}.csv"

    logger.info(f"时间窗: {datetime.fromtimestamp(start_ts)} ~ {datetime.fromtimestamp(end_ts)}")
    logger.info(f"SLS: region={args.sls_region}, project={args.sls_project}, logstore={args.sls_logstore}, query={args.trace_query}")
    logger.info(f"CMS: region={args.cms_region}, workspace={args.workspace}")

    # 1) 拉 trace 并写 CSV，同时收集 node_name
    sls_client = init_sls_client(args.sls_region)
    node_names = write_traces_csv(
        sls_client=sls_client,
        project=args.sls_project,
        logstore=args.sls_logstore,
        start_ts=start_ts,
        end_ts=end_ts,
        query=args.trace_query,
        out_csv=traces_csv,
        page_size=args.page_size,
        max_logs=args.max_trace_logs,
        trace_log_every=args.trace_log_every,
    )
    if not node_names:
        logger.warning("Trace 中未解析到任何 k8s.node.name；后续节点指标将无法对齐（可检查 trace resources 字段）。")

    # 2) 查 k8s.node 实体，用于 internal_ip/cluster_id
    node_entities = fetch_k8s_node_entities(
        cms_region=args.cms_region,
        workspace=args.workspace,
        start_ts=start_ts,
        end_ts=end_ts,
        limit=args.k8s_node_limit,
    )

    # 指标拉取节点范围：trace 覆盖集 vs k8s.node 实体全量集
    entity_nodes: List[str] = []
    if node_entities:
        seen = set()
        for rec in node_entities:
            n = rec.get("name") or rec.get("nodeName") or rec.get("provider_id")
            if not n:
                continue
            n_s = str(n)
            if n_s not in seen:
                seen.add(n_s)
                entity_nodes.append(n_s)

    if args.node_metrics_scope == "all":
        nodes_for_metrics = entity_nodes
        if not nodes_for_metrics:
            # 极端情况：实体查询为空，则退化回 trace 覆盖节点集合，避免直接输出空指标文件
            logger.warning("k8s.node 实体列表为空，无法按全量节点拉指标；将退化为仅对 trace 覆盖节点拉取。")
            nodes_for_metrics = node_names
    else:
        nodes_for_metrics = node_names

    logger.info(
        "[nodes] trace 覆盖节点数=%d, k8s.node 实体节点数=%d, 指标拉取范围=%s, metrics_nodes=%d",
        len(node_names),
        len(entity_nodes),
        args.node_metrics_scope,
        len(nodes_for_metrics),
    )

    metrics_list = [m.strip() for m in str(args.metrics).split(",") if m.strip()]
    if not metrics_list:
        raise ValueError("metrics 参数为空")

    # 3) 拉节点指标并写 CSV（instance_id = k8s.node.name）
    write_node_metrics_csv(
        cms_region=args.cms_region,
        workspace=args.workspace,
        start_ts=start_ts,
        end_ts=end_ts,
        sls_region_id=args.sls_region,
        cluster_id=args.cluster_id,
        nodes=nodes_for_metrics,
        node_entities=node_entities,
        metrics=metrics_list,
        step=args.metric_step,
        aggregate=args.metric_aggregate,
        metric_set_name=args.metric_set_name,
        out_csv=metrics_csv,
    )

    logger.info("完成。")


if __name__ == "__main__":
    main()
