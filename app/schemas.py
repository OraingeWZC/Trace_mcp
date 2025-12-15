# Trace_mcp/app/schemas.py
from typing import Any, Dict, Literal
from pydantic import BaseModel, Field


class ToolRequest(BaseModel):
    """
    内部请求模型：
    - op: 算子名（即 tools 目录下哪一类工具）
    - stage: 调用哪个脚本，需包含 train / preprocess / test
    - extra_args: 会被转成命令行参数传给脚本
    """
    op: Literal["aiops_svnd", "aiops_sv", "tratoporca", 'check_svnd'] 
    stage: Literal["train", "preprocess", "test", 'debug'] = "train"
    extra_args: Dict[str, Any] = Field(default_factory=dict)


class ScriptResult(BaseModel):
    """子进程执行结果"""
    exit_code: int
    log: str
