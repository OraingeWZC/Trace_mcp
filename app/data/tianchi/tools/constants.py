#!/usr/bin/env python


"""Global constants for tools configuration.

These constants specify the project, region, and workspace
parameters across all tools in the system.
"""

# Prefer environment variables (typically from .env) over hard-coded defaults.
#
# NOTE:
# - We try to load app/data/tianchi/.env regardless of current working directory.
# - load_dotenv is optional; if python-dotenv isn't installed, we fall back to os.environ only.
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

if load_dotenv is not None:
    _dotenv_path = Path(__file__).resolve().parents[1] / ".env"
    if _dotenv_path.exists():
        # override=False: if user already exported env vars, respect them.
        load_dotenv(dotenv_path=_dotenv_path, override=False)


_DEFAULT_PROJECT_NAME = "proj-xtrace-a46b97cfdc1332238f714864c014a1b-cn-qingdao"
_DEFAULT_REGION_ID = "cn-qingdao"
_DEFAULT_WORKSPACE_NAME = "tianchi-workspace"

# Keep legacy names for backward compatibility with existing scripts/tools imports.
PROJECT_NAME = (
    os.environ.get("SLS_PROJECT_NAME")
    or os.environ.get("PROJECT_NAME")
    or _DEFAULT_PROJECT_NAME
)

# REGION_ID is a region code string like "cn-qingdao".
# Unify with SLS_REGION by default; allow override via REGION_ID / CMS_REGION_ID if needed.
REGION_ID = (
    os.environ.get("CMS_REGION_ID")
    or os.environ.get("REGION_ID")
    or os.environ.get("SLS_REGION")
    or _DEFAULT_REGION_ID
)

# Keep using WORKSPACE_NAME as requested. If not set, fall back to the previous default.
WORKSPACE_NAME = os.environ.get("WORKSPACE_NAME") or _DEFAULT_WORKSPACE_NAME


ALIBABA_CLOUD_ACCESS_KEY_ID = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_ID")
ALIBABA_CLOUD_ACCESS_KEY_SECRET = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_SECRET")
# 注意：不要在代码里给 RoleArn 写死默认值。
# 是否走 STS AssumeRole 应完全由环境变量决定；未配置则使用 AK/SK 直连。
ALIBABA_CLOUD_ROLE_ARN = os.environ.get("ALIBABA_CLOUD_ROLE_ARN", "")
ALIBABA_CLOUD_ROLE_SESSION_NAME = os.environ.get(
    "ALIBABA_CLOUD_ROLE_SESSION_NAME", "aiops-rca-session"
)
