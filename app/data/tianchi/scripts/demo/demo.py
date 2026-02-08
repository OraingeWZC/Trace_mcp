# get_trace_detail.py
# -*- coding: utf-8 -*-
import os
import json
import time
from datetime import datetime
from aliyun.log import LogClient, GetLogsRequest
import config

# === 必须引入 STS 相关的库 ===
try:
    from alibabacloud_sts20150401.client import Client as StsClient
    from alibabacloud_sts20150401 import models as sts_models
    from alibabacloud_tea_openapi import models as open_api_models
except ImportError:
    print("❌ 缺少必要的库，请运行: pip install alibabacloud_sts20150401 alibabacloud_tea_openapi")
    exit(1)

# === 配置区域 ===
PROJECT_NAME = config.SLS_PROJECT_NAME
LOGSTORE_NAME = config.SLS_LOGSTORE_NAME
REGION = config.SLS_REGION
ENDPOINT = f"{REGION}.log.aliyuncs.com"

# 环境变量
ACCESS_KEY_ID = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_ID", "")
ACCESS_KEY_SECRET = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_SECRET", "")
ROLE_ARN = os.environ.get("ALIBABA_CLOUD_ROLE_ARN", "")

def get_sts_client():
    """获取带有 STS Token 的 SLS 客户端"""
    print("🔐 1. 正在向阿里云 STS 申请临时凭证...")
    
    # 初始化 STS 客户端
    config = open_api_models.Config(
        access_key_id=ACCESS_KEY_ID,
        access_key_secret=ACCESS_KEY_SECRET,
        endpoint=f'sts.{REGION}.aliyuncs.com'
    )
    sts_client = StsClient(config)
    
    # 发起 AssumeRole 请求
    resp = sts_client.assume_role(sts_models.AssumeRoleRequest(
        role_arn=ROLE_ARN,
        role_session_name="trace_debugger",
        duration_seconds=3600
    ))
    
    creds = resp.body.credentials
    print(f"✅ 凭证获取成功! (Token 有效期至: {creds.expiration})")
    
    # 返回带有 Token 的 LogClient
    return LogClient(
        endpoint=ENDPOINT,
        accessKeyId=creds.access_key_id,
        accessKey=creds.access_key_secret,
        securityToken=creds.security_token  # 关键点
    )

def save_trace_to_file(trace_id, log_items):
    """将 Trace 数据保存为 JSON 文件"""
    output_data = {
        "trace_id": trace_id,
        "span_count": len(log_items),
        "export_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "spans": []
    }

    # 解析每条日志的内容
    for log in log_items:
        content = log.get_contents()
        # 尝试解析 resources 字段 (通常是 JSON 字符串)
        try:
            if 'resources' in content:
                content['resources'] = json.loads(content['resources'])
        except:
            pass
        
        # 尝试解析 attribute 字段
        try:
            if 'attribute' in content:
                content['attribute'] = json.loads(content['attribute'])
        except:
            pass
            
        output_data["spans"].append(content)

    # 写入文件
    filename = f"trace_{trace_id}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    
    print(f"\n💾 数据已写入文件: {os.path.abspath(filename)}")

def run_task():
    # 1. 初始化鉴权
    try:
        client = get_sts_client()
    except Exception as e:
        print(f"❌ 鉴权失败: {e}")
        return

    # 2. 设定时间范围
    now = int(time.time())
    start_time = now - 3600 
    print(f"📅 查询时间范围: {datetime.fromtimestamp(start_time)} ~ {datetime.fromtimestamp(now)}")

    # 3. 寻找一个 TraceID
    print("\n🔍 2. 正在寻找一个正常的 TraceID (StatusCode <= 1)...")
    query_1 = "* | where try_cast(statusCode as bigint) <= 1"
    
    try:
        req1 = GetLogsRequest(PROJECT_NAME, LOGSTORE_NAME, fromTime=start_time, toTime=now, query=query_1, line=10)
        res1 = client.get_logs(req1)
    except Exception as e:
        print(f"❌ 查询失败，请检查 Project/Logstore 配置: {e}")
        return

    if not res1 or not res1.get_logs():
        print("❌ 未找到日志，请确认 Logstore 中是否有数据。")
        return

    target_trace_id = res1.get_logs()[0].get_contents().get('traceId')
    print(f"✅ 找到目标 TraceID: {target_trace_id}")

    # 4. 获取该 Trace 的所有 Span
    print(f"\n🔍 3. 正在拉取该 Trace 的所有 Span 详情...")
    query_2 = f'traceId: "{target_trace_id}"'
    
    # 这里为了演示拉取前 200 条 Span，如果是超大 Trace 需要翻页
    req2 = GetLogsRequest(PROJECT_NAME, LOGSTORE_NAME, fromTime=start_time, toTime=now, query=query_2, line=200)
    res2 = client.get_logs(req2)
    spans = res2.get_logs()
    
    print(f"📦 拉取成功，共 {len(spans)} 个 Span")

    # 5. 打印摘要并保存
    print("\n--- Trace 摘要预览 ---")
    for i, log in enumerate(spans[:3]): # 只打印前3个看看
        c = log.get_contents()
        print(f"[{i+1}] {c.get('serviceName')} -> {c.get('spanName')} (Took: {int(c.get('duration',0))/1000}ms)")
    if len(spans) > 3: print("...")

    # 保存到文件
    save_trace_to_file(target_trace_id, spans)

if __name__ == "__main__":
    if not ACCESS_KEY_ID or not ROLE_ARN:
        print("❌ 错误: 环境变量未设置 (ALIBABA_CLOUD_ACCESS_KEY_ID / SECRET / ROLE_ARN)")
    else:
        run_task()