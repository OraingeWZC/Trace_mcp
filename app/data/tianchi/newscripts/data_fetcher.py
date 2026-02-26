#!/usr/bin/env python3
"""
Trace 日志数据拉取脚本（独立版本）

从指定的 logstore 中拉取指定时间区间内的 trace 数据。
logstore 信息通过 CMS API 动态获取。

依赖安装:
    pip install alibabacloud_cms20240330 alibabacloud_sls20201230 alibabacloud_tea_openapi pandas

使用方法:
    # 设置环境变量
    export ALIBABA_CLOUD_ACCESS_KEY_ID=your_access_key_id
    export ALIBABA_CLOUD_ACCESS_KEY_SECRET=your_access_key_secret
    
    # 运行脚本
    python trace_log_fetcher.py --entity-type apm.service
"""

import os
import sys
import json
import re
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from textwrap import dedent

import pandas as pd
from jinja2 import Template

from alibabacloud_cms20240330.client import Client as Cms20240330Client
from alibabacloud_cms20240330 import models as cms_models
from alibabacloud_sls20201230.client import Client as Sls20201230Client
from alibabacloud_sls20201230 import models as sls_models
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_tea_util import models as util_models

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 时间转换工具
# ============================================================================

def convert_to_unixtime(time_str: str) -> int:
    """
    将时间字符串转换为 Unix 时间戳（秒）
    
    支持的格式：
    - now() 或 now：当前时间
    - now()-1d 或 now-1d：当前时间减去1天
    - now()-1h 或 now-1h：当前时间减去1小时
    - now()-1m 或 now-1m：当前时间减去1分钟
    - now()-10s 或 now-10s：当前时间减去10秒
    - now()-1w 或 now-1w：当前时间减去1周
    - 2025-06-11 10:00:00：绝对时间格式
    - 整数时间戳
    """
    if time_str is None:
        return -1
    if isinstance(time_str, int):
        if time_str > 1e12:
            return int(time_str / 1e9)
        elif time_str > 1e9 and len(str(time_str)) >= 13:
            return int(time_str / 1000)
        else:
            return time_str

    time_str = time_str.strip()
    
    if time_str.isdigit():
        timestamp = int(time_str)
        if timestamp > 1e12:
            return int(timestamp / 1e9)
        elif timestamp > 1e9 and len(str(timestamp)) >= 13:
            return int(timestamp / 1000)
        else:
            return timestamp
    
    if time_str.startswith("now"):
        current_time = datetime.now()
        time_str = time_str.replace("now()", "now").replace("now", "").strip()
        
        if not time_str:
            return int(current_time.timestamp())
        
        if time_str.startswith("-"):
            offset_str = time_str[1:].strip()
            pattern = r'^(\d+)([dhmsw])$'
            match = re.match(pattern, offset_str)
            
            if match:
                value = int(match.group(1))
                unit = match.group(2)
                
                if unit == 'w':
                    offset_time = current_time - timedelta(weeks=value)
                elif unit == 'd':
                    offset_time = current_time - timedelta(days=value)
                elif unit == 'h':
                    offset_time = current_time - timedelta(hours=value)
                elif unit == 'm':
                    offset_time = current_time - timedelta(minutes=value)
                elif unit == 's':
                    offset_time = current_time - timedelta(seconds=value)
                else:
                    raise ValueError(f"不支持的时间单位: {unit}")
                
                return int(offset_time.timestamp())
            else:
                raise ValueError(f"无效的时间偏移格式: {offset_str}")
    
    try:
        if len(time_str) == 10:
            dt = datetime.strptime(time_str, '%Y-%m-%d')
        else:
            dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
        return int(dt.timestamp())
    except ValueError:
        raise ValueError(f"无法解析时间字符串: {time_str}")


# ============================================================================
# Region 映射
# ============================================================================

REGION_ENDPOINT_MAP = {
    "cn-hangzhou": "cms.cn-hangzhou.aliyuncs.com",
    "cn-shanghai": "cms.cn-shanghai.aliyuncs.com",
    "cn-beijing": "cms.cn-beijing.aliyuncs.com",
    "cn-shenzhen": "cms.cn-shenzhen.aliyuncs.com",
    "cn-hongkong": "cms.cn-hongkong.aliyuncs.com",
    "cn-qingdao": "cms.cn-qingdao.aliyuncs.com",
    "cn-zhangjiakou": "cms.cn-zhangjiakou.aliyuncs.com",
    "cn-huhehaote": "cms.cn-huhehaote.aliyuncs.com",
    "cn-chengdu": "cms.cn-chengdu.aliyuncs.com",
    "ap-southeast-1": "cms.ap-southeast-1.aliyuncs.com",
    "ap-southeast-2": "cms.ap-southeast-2.aliyuncs.com",
    "ap-southeast-3": "cms.ap-southeast-3.aliyuncs.com",
    "ap-southeast-5": "cms.ap-southeast-5.aliyuncs.com",
    "ap-northeast-1": "cms.ap-northeast-1.aliyuncs.com",
    "ap-south-1": "cms.ap-south-1.aliyuncs.com",
    "us-east-1": "cms.us-east-1.aliyuncs.com",
    "us-west-1": "cms.us-west-1.aliyuncs.com",
    "eu-central-1": "cms.eu-central-1.aliyuncs.com",
    "eu-west-1": "cms.eu-west-1.aliyuncs.com",
    "me-east-1": "cms.me-east-1.aliyuncs.com",
}


def region_to_endpoint(region_id: str) -> str:
    """将 region_id 转换为 CMS endpoint"""
    if region_id in REGION_ENDPOINT_MAP:
        return REGION_ENDPOINT_MAP[region_id]
    return f"cms.{region_id}.aliyuncs.com"


# ============================================================================
# 数据模型
# ============================================================================

@dataclass
class LogStoreInfo:
    """Logstore 存储信息"""
    region: str
    project: str
    store: str
    domain: str
    name: str
    
    def __str__(self) -> str:
        return f"LogStoreInfo(region={self.region}, project={self.project}, store={self.store})"


@dataclass
class EntityDataSet:
    """实体数据集"""
    data_set_id: str = ""
    type: str = ""
    domain: str = ""
    name: str = ""
    fields_mapping: Dict[str, Any] = field(default_factory=dict)
    filterable_fields: List[str] = field(default_factory=list)
    fields: List[Dict[str, str]] = field(default_factory=list)
    storage_info: List[Dict[str, str]] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntityDataSet":
        target_keys = ["fields_mapping", "filterable_fields", "fields", "storage_info"]
        for k in target_keys:
            if k in data.keys():
                try:
                    if isinstance(data[k], str):
                        data[k] = json.loads(data[k])
                except:
                    pass

        return cls(
            data_set_id=data.get("data_set_id", ""),
            type=data.get("type", ""),
            domain=data.get("domain", ""),
            name=data.get("name", ""),
            fields_mapping=data.get("fields_mapping", {}),
            filterable_fields=data.get("filterable_fields", []),
            fields=data.get("fields", []),
            storage_info=data.get("storage_info", []),
        )
    
    @classmethod
    def from_df(cls, df: pd.DataFrame) -> List["EntityDataSet"]:
        if df is None or df.empty:
            return []
        return [cls.from_dict(row) for row in df.to_dict(orient='records')]
    
    def get_storage_info(self) -> Dict[str, str]:
        if self.storage_info is None or len(self.storage_info) == 0:
            return {}
        one_storage_info = self.storage_info[0]
        if "config" in one_storage_info.keys():
            result = one_storage_info["config"]
            if 'sls_project' in result.keys():
                result["project"] = result["sls_project"]
            if 'sls_metricstore' in result.keys(): 
                result["store"] = result["sls_metricstore"]
            if 'sls_logstore' in result.keys():
                result["store"] = result["sls_logstore"]
            return result
        else:
            return {}


# ============================================================================
# CMS 客户端
# ============================================================================

class SimpleCMSClient:
    """简化版 CMS 客户端"""
    
    def __init__(
        self,
        access_key_id: str,
        access_key_secret: str,
        region_id: str,
        endpoint: Optional[str] = None,
    ):
        self.access_key_id = access_key_id
        self.access_key_secret = access_key_secret
        self.region_id = region_id
        self.endpoint = endpoint or region_to_endpoint(region_id)
        self._client = None
    
    def _get_client(self) -> Cms20240330Client:
        if self._client is None:
            config = open_api_models.Config(
                access_key_id=self.access_key_id,
                access_key_secret=self.access_key_secret,
                endpoint=self.endpoint,
            )
            self._client = Cms20240330Client(config)
        return self._client
    
    def get_query_data(
        self,
        workspace: str,
        spl_query: str,
        from_: int,
        to: int,
    ) -> Optional[Dict[str, Any]]:
        """执行 SPL 查询"""
        try:
            client = self._get_client()
            request = cms_models.QueryDataRequest(
                workspace=workspace,
                query=spl_query,
                from_=from_,
                to=to,
            )
            runtime = util_models.RuntimeOptions()
            response = client.query_data_with_options(request, runtime)
            
            if response.body:
                return {
                    "data": response.body.data,
                    "header": response.body.header,
                }
            return None
        except Exception as e:
            logger.error(f"CMS 查询失败: {e}")
            return None


# ============================================================================
# SLS 客户端
# ============================================================================

class SimpleSLSClient:
    """简化版 SLS 客户端"""
    
    def __init__(
        self,
        access_key_id: str,
        access_key_secret: str,
        region_id: str,
    ):
        self.access_key_id = access_key_id
        self.access_key_secret = access_key_secret
        self.region_id = region_id
        self.endpoint = f"{region_id}.log.aliyuncs.com"
        self._client = None
    
    def _get_client(self) -> Sls20201230Client:
        if self._client is None:
            config = open_api_models.Config(
                access_key_id=self.access_key_id,
                access_key_secret=self.access_key_secret,
                endpoint=self.endpoint,
            )
            self._client = Sls20201230Client(config)
        return self._client
    
    def get_logs(
        self,
        project: str,
        logstore: str,
        query: str,
        start_time: int,
        end_time: int,
        line: int = 100,
    ) -> List[Dict[str, Any]]:
        """查询日志"""
        logger.info(f"SLS 查询: project={project}, logstore={logstore}, query={query}")
        
        max_retry_count = 3
        retry_delay_seconds = 10
        
        for i in range(max_retry_count):
            try:
                client = self._get_client()
                request = sls_models.GetLogsRequest(
                    query=query,
                    from_=start_time,
                    to=end_time,
                    line=line,
                )
                headers: Dict[str, str] = {}
                runtime = util_models.RuntimeOptions()
                response = client.get_logs_with_options(project, logstore, request, headers, runtime)
                
                logs = []
                if response.body:
                    body_list = response.body
                    if isinstance(body_list, list):
                        for log_item in body_list:
                            if hasattr(log_item, 'to_map'):
                                logs.append(log_item.to_map())
                            elif isinstance(log_item, dict):
                                logs.append(log_item)
                    elif hasattr(body_list, 'to_map'):
                        logs = [body_list.to_map()]
                
                logger.info(f"SLS 查询成功，返回 {len(logs)} 条日志")
                return logs
                
            except Exception as e:
                error_code = getattr(e, 'code', None) or getattr(e, 'error_code', 'UnknownError')
                if str(error_code) in ["ParameterInvalid", "InvalidQuery", "InvalidArgument"]:
                    logger.error(f"SLS 查询失败（参数错误）: {e}")
                    return []
                else:
                    logger.warning(f"SLS 查询失败，重试 {i + 1}/{max_retry_count}: {e}")
                    if i < max_retry_count - 1:
                        time.sleep(retry_delay_seconds)
        
        logger.error("SLS 查询失败，已达到最大重试次数")
        return []


# ============================================================================
# 核心功能
# ============================================================================

def fetch_logstore_info(
    cms_client: SimpleCMSClient,
    workspace: str,
    entity_domain: str,
    entity_type: str,
    entity_id: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> List[LogStoreInfo]:
    """
    通过 CMS API 获取实体类型对应的 logstore 配置
    
    查询语句示例：
    .entity_set with(domain='apm', name='apm.service') | entity-call list_data_set(['log_set'], true)
    
    返回的 storage_info 格式：
    [{"domain":"apm","type":"sls_logstore","name":"apm.log_set.agent.storage",
      "config":{"region":"cn-hongkong","project":"proj-xtrace-xxx","store":"logstore-agent"}}]
    """
    logger.info(f"获取 log_set 信息: workspace={workspace}, entity_type={entity_type}")
    
    if entity_id is not None and len(entity_id.strip()) > 0:
        spl_query = Template(dedent("""
            .entity_set with(domain='{{ entity_domain }}', name='{{ entity_type }}', ids=['{{ entity_id }}']) | entity-call list_data_set(['log_set'], true)
        """)).render(
            entity_domain=entity_domain,
            entity_type=entity_type,
            entity_id=entity_id,
        )
    else:
        spl_query = Template(dedent("""
            .entity_set with(domain='{{ entity_domain }}', name='{{ entity_type }}') | entity-call list_data_set(['log_set'], true)
        """)).render(
            entity_domain=entity_domain,
            entity_type=entity_type,
        )
    
    spl_query = spl_query.strip()
    logger.info(f"SPL 查询: {spl_query}")
    
    start_unixtime = convert_to_unixtime(start_time if start_time else "now()-1h")
    end_unixtime = convert_to_unixtime(end_time if end_time else "now()")
    
    response = cms_client.get_query_data(
        workspace=workspace,
        spl_query=spl_query,
        from_=start_unixtime,
        to=end_unixtime,
    )
    
    if response is None:
        raise ValueError(f"无法获取 {entity_type} 的 log_set 信息")
    
    data, header = response.get("data"), response.get("header")
    df = pd.DataFrame(data, columns=header)
    
    if df.empty:
        raise ValueError(f"未找到 {entity_type} 的 log_set 信息")
    
    log_set_list = EntityDataSet.from_df(df)
    logger.info(f"解析到 {len(log_set_list)} 个 log_set")
    
    seen = set()
    result: List[LogStoreInfo] = []
    
    for log_set in log_set_list:
        storage_info = log_set.get_storage_info()
        if storage_info:
            region = storage_info.get("region", "")
            project = storage_info.get("project") or storage_info.get("sls_project", "")
            store = storage_info.get("store") or storage_info.get("sls_logstore", "")
            
            if project and store:
                key = (project, store)
                if key not in seen:
                    seen.add(key)
                    info = LogStoreInfo(
                        region=region,
                        project=project,
                        store=store,
                        domain=log_set.domain,
                        name=log_set.name,
                    )
                    result.append(info)
                    logger.info(f"发现 logstore: {log_set.name} -> {info}")
    
    if not result:
        raise ValueError(f"API 返回的 log_set 中没有有效的 storage_info")
    
    logger.info(f"{entity_type} 共有 {len(result)} 个不同的 logstore")
    return result


def fetch_trace_logs(
    sls_client: SimpleSLSClient,
    project: str,
    logstore: str,
    query: str,
    start_time: int,
    end_time: int,
    limit: int = 1000,
) -> List[Dict[str, Any]]:
    """从 logstore 拉取日志数据"""
    logger.info(f"查询日志: project={project}, logstore={logstore}")
    logger.info(f"  query: {query}")
    logger.info(f"  time_range: [{start_time}, {end_time}]")
    logger.info(f"  limit: {limit}")
    
    logs = sls_client.get_logs(
        project=project,
        logstore=logstore,
        query=query,
        start_time=start_time,
        end_time=end_time,
        line=limit,
    )
    
    logger.info(f"查询完成，返回 {len(logs)} 条日志")
    return logs


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="从 logstore 拉取 trace 数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
环境变量:
  ALIBABA_CLOUD_ACCESS_KEY_ID      阿里云 Access Key ID
  ALIBABA_CLOUD_ACCESS_KEY_SECRET  阿里云 Access Key Secret

示例用法:
  # 设置环境变量
  export ALIBABA_CLOUD_ACCESS_KEY_ID=your_access_key_id
  export ALIBABA_CLOUD_ACCESS_KEY_SECRET=your_access_key_secret
  
  # 使用默认配置（rca-benchmark workspace）
  python trace_log_fetcher.py --entity-type apm.service
  
  # 指定时间范围
  python trace_log_fetcher.py --entity-type apm.service --start-time "now()-2h" --end-time "now()"
  
  # 指定 workspace
  python trace_log_fetcher.py --workspace inner-playground --entity-type apm.service
  
  # 自定义查询（获取全部数据用 *）
  python trace_log_fetcher.py --entity-type apm.service --query "*"
  
  # 保存结果到文件
  python trace_log_fetcher.py --entity-type apm.service --output trace_logs.json
        """
    )
    
    parser.add_argument(
        "--access-key-id",
        default=os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_ID"),
        help="阿里云 Access Key ID（或设置 ALIBABA_CLOUD_ACCESS_KEY_ID 环境变量）"
    )
    parser.add_argument(
        "--access-key-secret",
        default=os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_SECRET"),
        help="阿里云 Access Key Secret（或设置 ALIBABA_CLOUD_ACCESS_KEY_SECRET 环境变量）"
    )
    parser.add_argument(
        "--workspace",
        default="rca-benchmark",
        help="工作空间名称（默认: rca-benchmark）"
    )
    parser.add_argument(
        "--region-id",
        default="cn-hongkong",
        help="区域 ID（默认: cn-hongkong）"
    )
    parser.add_argument(
        "--entity-domain",
        default="apm",
        help="实体领域（默认: apm）"
    )
    parser.add_argument(
        "--entity-type",
        required=True,
        help="实体类型（如: apm.service）"
    )
    parser.add_argument(
        "--entity-id",
        default=None,
        help="实体 ID（可选）"
    )
    parser.add_argument(
        "--start-time",
        default="now()-1h",
        help="开始时间（默认: now()-1h）"
    )
    parser.add_argument(
        "--end-time",
        default="now()",
        help="结束时间（默认: now()）"
    )
    parser.add_argument(
        "--query",
        default="*",
        help="查询语句（默认: * 获取全部数据）"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="返回结果数量限制（默认: 1000）"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出文件路径（可选，默认打印到控制台）"
    )
    parser.add_argument(
        "--logstore-index",
        type=int,
        default=0,
        help="使用第几个 logstore（默认: 0，即第一个）"
    )
    
    args = parser.parse_args()
    
    if not args.access_key_id or not args.access_key_secret:
        logger.error("请提供 Access Key ID 和 Access Key Secret")
        logger.error("可以通过命令行参数或环境变量设置:")
        logger.error("  --access-key-id / ALIBABA_CLOUD_ACCESS_KEY_ID")
        logger.error("  --access-key-secret / ALIBABA_CLOUD_ACCESS_KEY_SECRET")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("Trace 日志数据拉取脚本")
    logger.info("=" * 60)
    
    cms_client = SimpleCMSClient(
        access_key_id=args.access_key_id,
        access_key_secret=args.access_key_secret,
        region_id=args.region_id,
    )
    
    logstore_list = fetch_logstore_info(
        cms_client=cms_client,
        workspace=args.workspace,
        entity_domain=args.entity_domain,
        entity_type=args.entity_type,
        entity_id=args.entity_id,
        start_time=args.start_time,
        end_time=args.end_time,
    )
    
    if not logstore_list:
        logger.error("未找到任何 logstore 配置")
        sys.exit(1)
    
    logger.info(f"\n发现 {len(logstore_list)} 个 logstore:")
    for i, info in enumerate(logstore_list):
        logger.info(f"  [{i}] {info}")
    
    if args.logstore_index >= len(logstore_list):
        logger.error(f"logstore 索引 {args.logstore_index} 超出范围 [0, {len(logstore_list) - 1}]")
        sys.exit(1)
    
    selected_logstore = logstore_list[args.logstore_index]
    logger.info(f"\n选择 logstore [{args.logstore_index}]: {selected_logstore}")
    
    sls_region = selected_logstore.region or args.region_id
    sls_client = SimpleSLSClient(
        access_key_id=args.access_key_id,
        access_key_secret=args.access_key_secret,
        region_id=sls_region,
    )
    
    start_unixtime = convert_to_unixtime(args.start_time)
    end_unixtime = convert_to_unixtime(args.end_time)
    
    logs = fetch_trace_logs(
        sls_client=sls_client,
        project=selected_logstore.project,
        logstore=selected_logstore.store,
        query=args.query,
        start_time=start_unixtime,
        end_time=end_unixtime,
        limit=args.limit,
    )
    
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
        logger.info(f"结果已保存到: {output_path}")
    else:
        logger.info("\n" + "=" * 60)
        logger.info("查询结果:")
        logger.info("=" * 60)
        for i, log in enumerate(logs[:10]):
            logger.info(f"\n--- 日志 {i + 1} ---")
            logger.info(json.dumps(log, ensure_ascii=False, indent=2))
        
        if len(logs) > 10:
            logger.info(f"\n... 还有 {len(logs) - 10} 条日志未显示 ...")
    
    logger.info(f"\n统计信息:")
    logger.info(f"  总日志数: {len(logs)}")
    logger.info(f"  logstore: {selected_logstore.project}/{selected_logstore.store}")
    logger.info(f"  时间范围: [{args.start_time}, {args.end_time}]")


if __name__ == "__main__":
    main()

