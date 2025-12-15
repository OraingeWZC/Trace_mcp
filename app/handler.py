# app/handler.py
import subprocess
import sys
import os
from pathlib import Path
from typing import Tuple, List, Dict
from app.schemas import ToolRequest, ScriptResult

def _build_command(req: ToolRequest, target: str, work_dir: Path, is_module: bool = False) -> Tuple[List[str], Dict[str, str]]:
    """
    构建运行命令
    :param target: 脚本名(xxx.py) 或 模块名(pkg.mod)
    :param is_module: 是否使用 python -m 模式
    """
    # 1. 基础命令
    cmd = [sys.executable, "-u"]
    if is_module:
        cmd.extend(["-m", target])
    else:
        script_full_path = work_dir / target
        cmd.append(str(script_full_path))
    
    # 2. 参数转换 (字典转命令行)
    for k, v in req.extra_args.items():
        if v is None: continue
        # 处理布尔开关
        if isinstance(v, bool):
            if v: cmd.append(f"--{k.replace('_', '-')}")
        else:
            cmd.append(f"--{k.replace('_', '-')}")
            cmd.append(str(v))
        
    # 3. 环境变量与路径设置
    env = os.environ.copy()
    
    # [核心修改] 针对 TraTopoRca 设置特定的 PYTHONPATH
    if req.op == "tratoporca":
        # TraTopoRca 的代码引用了 tracegnn 包，所以必须把 TraTopoRca 目录加入路径
        env["PYTHONPATH"] = str(work_dir)  
    else:
        # 旧工具保持原样 (指向项目根目录)
        env["PYTHONPATH"] = str(work_dir.parent.parent)
    
    # [防死锁配置] 之前提到的 Windows 修复补丁
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["CUDA_LAUNCH_BLOCKING"] = "1"
    
    # PATH 注入 (保持原样)
    python_root = Path(sys.executable).parent
    library_bin = python_root / "Library" / "bin"
    scripts_dir = python_root / "Scripts"
    current_path = env.get("PATH", "")
    env["PATH"] = f"{library_bin};{scripts_dir};{python_root};{current_path}"
    
    return cmd, env

def run_script(req: ToolRequest) -> ScriptResult:
    # 1. 路由逻辑：根据 op 决定去哪里找代码
    is_module = False
    
    if req.op == "tratoporca":
        # === 新工具逻辑 ===
        # 定位到 app/tools/TraTopoRca
        script_work_dir = (Path(__file__).parent / "tools" / "TraTopoRca").resolve()
        is_module = True
        
        if req.stage == "train":
            # 对应: python -m tracegnn.models.gtrace.mymodel_main
            script_name = "tracegnn.models.gtrace.mymodel_main"
        elif req.stage == "test":
            # 对应: python -m tracegnn.models.gtrace.mymodel_test
            script_name = "tracegnn.models.gtrace.mymodel_test"
        else:
            return ScriptResult(exit_code=-1, log=f"Unknown stage {req.stage} for tratoporca")
            
    else:
        # === 旧工具逻辑 (sv, svnd) ===
        diag_type = "svnd" if "svnd" in req.op else "sv"
        script_work_dir = (Path(__file__).parent / "tools" / f"trace_{diag_type}_diag").resolve()
        script_name = f"{req.stage}_{req.op}.py"
        is_module = False
    
    log_file_path = script_work_dir / "mcp_debug_log.txt"
    
    # 2. 构建命令
    try:
        cmd, env = _build_command(req, script_name, script_work_dir, is_module=is_module)
    except Exception as e:
        return ScriptResult(exit_code=-1, log=f"Command Build Error: {str(e)}")

    # 3. 执行
    exit_code = -1
    try:
        with open(log_file_path, "w", encoding="utf-8") as f_log:
            f_log.write(f"=== Starting Command ===\n{' '.join(cmd)}\n\nWorkDir: {script_work_dir}\n\n")
            f_log.flush()
            
            result = subprocess.run(
                cmd,
                cwd=script_work_dir,
                env=env,
                stdout=f_log,
                stderr=subprocess.STDOUT,
                text=True
            )
            exit_code = result.returncode
            
    except Exception as e:
        with open(log_file_path, "a", encoding="utf-8") as f_err:
            f_err.write(f"\n\n[MCP Handler Exception]: {str(e)}")
        return ScriptResult(exit_code=-1, log=f"Execution Failed: {str(e)}")

    # 4. 读取日志
    try:
        with open(log_file_path, "r", encoding="utf-8") as fr:
            combined_log = fr.read()
    except Exception as e:
        combined_log = f"Command finished with code {exit_code}, but failed to read log: {str(e)}"
        
    return ScriptResult(exit_code=exit_code, log=combined_log)