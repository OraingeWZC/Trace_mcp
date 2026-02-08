# scripts/9_peek_inventory.py
# -*- coding: utf-8 -*-
import os, sys, json, time
from datetime import datetime

# 1. 路径修正 (为了能导入上一级的 config)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

try:
    from aliyun.log import LogClient, GetLogsRequest
    from alibabacloud_sts20150401.client import Client as StsClient
    from alibabacloud_tea_openapi import models as open_api_models
    from alibabacloud_sts20150401 import models as sts_models
except ImportError:
    print("❌ 缺少依赖，请 pip install aliyun-log-python-sdk alibabacloud_sts20150401")
    sys.exit(1)

# 2. 获取客户端 (带STS)
def get_client():
    cfg = open_api_models.Config(
        access_key_id=os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_ID"),
        access_key_secret=os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_SECRET"),
        endpoint=f"sts.{config.SLS_REGION}.aliyuncs.com"
    )
    resp = StsClient(cfg).assume_role(sts_models.AssumeRoleRequest(
        role_arn=os.environ.get("ALIBABA_CLOUD_ROLE_ARN"),
        role_session_name="simple-peek", duration_seconds=900
    ))
    c = resp.body.credentials
    return LogClient(f"{config.SLS_REGION}.log.aliyuncs.com", c.access_key_id, c.access_key_secret, c.security_token)

# 3. 执行查询
if __name__ == "__main__":
    # 任意指定一个包含数据的时间点 (15分钟窗口)
    center_time = "2025-09-16 23:25:00"
    ts = int(datetime.strptime(center_time, "%Y-%m-%d %H:%M:%S").timestamp())
    
    client = get_client()
    
    # 查 inventory 服务，只取 1 条
    req = GetLogsRequest(
        project=config.SLS_PROJECT_NAME,
        logstore=config.SLS_LOGSTORE_NAME,
        fromTime=ts - 600,
        toTime=ts + 600,
        query='serviceName: "frontend"', # 核心：只拿一条
        line=1
    )
    
    res = client.get_logs(req)
    if res and res.get_logs():
        print(json.dumps(res.get_logs()[0].get_contents(), indent=2, ensure_ascii=False))
    else:
        print("⚠️ 未找到数据，请检查时间范围。")