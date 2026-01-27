"""
将 trace_dataset.jsonl 转换为 merged_traces.csv 格式。

输入：每行一个 trace，包含 spans、is_anomaly、root_causes 等字段。
输出：按 span 展开为行的 CSV，字段与 2025-06-06_merged_traces.csv 一致：
TraceID,SpanID,ParentID,NodeName,ServiceName,PodName,URL,HttpStatusCode,StatusCode,
SpanKind,StartTime,EndTime,FaultCategory,RootCause,Duration,OperationName,Anomaly

用法示例：
  python convert_trace_dataset.py \
    --input /mnt/sdb/zly/4.1/tianchi/tianchi-2025/trace_dataset.jsonl \
    --output /mnt/sdb/zly/4.1/tracezly_rca/output/2025-06-06_merged_traces.csv
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _normalize_time_ms(value: Any) -> float:
    """将 startTime 统一为毫秒浮点数。"""
    v = _safe_float(value)
    if v == 0:
        return 0.0
    if v > 1e14:  # 纳秒
        return v / 1e6
    if v > 1e12:  # 微秒
        return v / 1e3
    if v > 1e10:  # 秒（含小数）的毫秒
        return v / 1.0  # 已经是毫秒级时间戳
    if v > 1e6:  # 秒
        return v * 1e3
    return v


def _normalize_duration_ms(value: Any) -> float:
    """将 duration 统一为毫秒浮点数。"""
    v = _safe_float(value)
    if v == 0:
        return 0.0
    if v > 1e9:  # 纳秒
        return v / 1e6
    if v > 1e6:  # 微秒
        return v / 1e3
    return v  # 其余认为已是毫秒


def _parse_dict_like(value: Any) -> Dict[str, Any]:
    """attributes/resources 可能是字典或 JSON 字符串，尝试解析为字典。"""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _extract_field(span: Dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        if key in span and span[key] not in (None, "", "null"):
            return str(span[key])
    return ""


def span_to_row(trace: Dict[str, Any], span: Dict[str, Any], default_root_causes: List[str]) -> Tuple[str, ...]:
    """将单个 span 转为 CSV 行。"""
    attributes = _parse_dict_like(span.get("attributes"))
    resources = _parse_dict_like(span.get("resources"))

    # 关键字段映射
    trace_id = _extract_field(span, ("traceId", "trace_id", "traceid"))
    span_id = _extract_field(span, ("spanId", "span_id", "spanid"))
    parent_id = _extract_field(span, ("parentSpanId", "parent_span_id", "parentspanid", "parentid"))
    service = _extract_field(span, ("serviceName", "service", "servicename"))
    operation = _extract_field(span, ("spanName", "operation_name", "name"))
    span_kind = _extract_field(span, ("kind", "spanKind"))

    # 设备/节点信息（若缺失则留空）
    node = resources.get("host.name") or resources.get("hostname") or resources.get("k8s.node.name") or ""
    pod = resources.get("k8s.pod.name") or resources.get("pod.name") or resources.get("pod") or ""

    # URL / HTTP 状态
    url = attributes.get("http.url") or attributes.get("http.target") or attributes.get("url") or ""
    http_status = attributes.get("http.status_code") or attributes.get("http.status") or attributes.get("status") or ""

    status_code = _extract_field(span, ("statusCode", "status_code", "status"))

    start_ms = _normalize_time_ms(span.get("startTime") or span.get("start_time"))
    duration_ms = _normalize_duration_ms(span.get("duration") or span.get("duration_ms"))
    end_ms = start_ms + duration_ms if start_ms and duration_ms else 0.0

    # 根因/故障分类
    root_causes = trace.get("root_causes") or default_root_causes
    root_cause_str = ";".join(root_causes) if root_causes else ""
    fault_category = root_cause_str  # 若没有专门分类字段，则复用根因

    anomaly_flag = 1 if trace.get("is_anomaly") else 0

    return (
        trace_id,
        span_id,
        parent_id or "-1",
        str(node),
        str(service),
        str(pod),
        str(url),
        str(http_status),
        str(status_code),
        str(span_kind),
        f"{start_ms:.6f}" if start_ms else "",
        f"{end_ms:.6f}" if end_ms else "",
        str(fault_category),
        str(root_cause_str),
        f"{duration_ms:.6f}" if duration_ms else "",
        str(operation),
        str(anomaly_flag),
    )


def convert(input_path: Path, output_path: Path, limit: int = 0):
    header = [
        "TraceID",
        "SpanID",
        "ParentID",
        "NodeName",
        "ServiceName",
        "PodName",
        "URL",
        "HttpStatusCode",
        "StatusCode",
        "SpanKind",
        "StartTime",
        "EndTime",
        "FaultCategory",
        "RootCause",
        "Duration",
        "OperationName",
        "Anomaly",
    ]

    count_traces = 0
    count_spans = 0

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(header)
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                trace = json.loads(line)
            except Exception:
                continue

            spans = trace.get("spans") or []
            default_root_causes = trace.get("root_causes") or []
            for span in spans:
                row = span_to_row(trace, span, default_root_causes)
                writer.writerow(row)
                count_spans += 1

            count_traces += 1
            if limit and count_traces >= limit:
                break

    print(f"✅ 完成转换: traces={count_traces}, spans={count_spans}, 输出={output_path}")


def main():
    parser = argparse.ArgumentParser(description="转换 trace_dataset.jsonl 为 merged_traces.csv")
    parser.add_argument("--input", type=Path, required=True, help="trace_dataset.jsonl 路径")
    parser.add_argument("--output", type=Path, required=True, help="输出 CSV 路径")
    parser.add_argument("--limit", type=int, default=0, help="可选，限制处理的 trace 数（调试用）")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"未找到输入文件: {args.input}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    convert(args.input, args.output, args.limit)


if __name__ == "__main__":
    main()

