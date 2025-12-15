# app/handler.py
import subprocess
import sys
import os
import json
from pathlib import Path
from typing import Tuple, List, Dict, Any
from app.schemas import ToolRequest, ScriptResult

def _build_command(req: ToolRequest, script_name: str, work_dir: Path) -> Tuple[List[str], Dict[str, str]]:
    script_full_path = work_dir / script_name
    
    # 1. 基础命令：添加 "-u" 参数强制禁用缓冲
    cmd = [sys.executable, "-u", str(script_full_path)]
    
    # 2. 参数处理 (保持不变)
    for k, v in req.extra_args.items():
        if v is None: continue
        cli_key = "--" + k.replace("_", "-")
        cmd.append(cli_key)
        cmd.append(str(v))
        
    # 3. 环境变量增强
    env = os.environ.copy()
    env["PYTHONPATH"] = str(work_dir.parent.parent)
    
    # [核心修复 1] 防止 Intel MKL 库冲突导致的死锁
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    # [核心修复 2] 强制 PyTorch 使用同步 CUDA 加载 (调试用)
    env["CUDA_LAUNCH_BLOCKING"] = "1"
    
    # 路径注入 (保持您之前的正确逻辑)
    python_root = Path(sys.executable).parent
    library_bin = python_root / "Library" / "bin"
    scripts_dir = python_root / "Scripts"
    current_path = env.get("PATH", "")
    env["PATH"] = f"{library_bin};{scripts_dir};{python_root};{current_path}"
    
    return cmd, env

def run_script(req: ToolRequest) -> ScriptResult:
    # 1. 确定脚本路径
    diag_type = "svnd" if "svnd" in req.op else "sv"
    # 定位到 app/tools/trace_svnd_diag
    script_work_dir = (Path(__file__).parent / "tools" / f"trace_{diag_type}_diag").resolve()
    script_name = f"{req.stage}_{req.op}.py"
    
    log_file_path = script_work_dir / "mcp_debug_log.txt"
    
    # 2. 构建命令
    try:
        cmd, env = _build_command(req, script_name, script_work_dir)
    except Exception as e:
        return ScriptResult(exit_code=-1, log=f"Command Build Error: {str(e)}")

    exit_code = -1

    # 3. 执行命令 (读写分离，防止死锁)
    try:
        # 写入模式打开，启动进程
        with open(log_file_path, "w", encoding="utf-8") as f_log:
            f_log.write(f"=== Starting Command ===\n{cmd}\n\nEnvironment PATH prefix:\n{env['PATH'][:300]}...\n\n")
            f_log.flush() # 确保头部写入
            
            result = subprocess.run(
                cmd,
                cwd=script_work_dir,
                env=env,
                stdout=f_log,              # 标准输出重定向到文件
                stderr=subprocess.STDOUT,  # 错误输出也重定向到文件
                text=True
            )
            exit_code = result.returncode
            
    except Exception as e:
        # 启动失败记录
        with open(log_file_path, "a", encoding="utf-8") as f_err:
            f_err.write(f"\n\n[MCP Handler Exception]: {str(e)}")
        return ScriptResult(exit_code=-1, log=f"Execution Failed: {str(e)}")

    # 4. 读取日志 (此时文件已关闭，安全读取)
    try:
        with open(log_file_path, "r", encoding="utf-8") as fr:
            combined_log = fr.read()
    except Exception as e:
        combined_log = f"Command finished with code {exit_code}, but failed to read log: {str(e)}"
        
    return ScriptResult(exit_code=exit_code, log=combined_log)