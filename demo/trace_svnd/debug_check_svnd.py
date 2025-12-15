# app/tools/trace_svnd_diag/debug_check.py
import sys
import os

# 1. 最基础的存活确认
print("=== DEBUG PROBE START ===", flush=True)
print(f"PID: {os.getpid()}", flush=True)
print(f"Python Executable: {sys.executable}", flush=True)
print(f"CWD: {os.getcwd()}", flush=True)

# 2. 检查环境变量 (确认 PATH 是否注入成功)
print("\n[Env Check]", flush=True)
path_env = os.environ.get("PATH", "")
if "Library\\bin" in path_env:
    print("SUCCESS: 'Library\\bin' found in PATH.", flush=True)
else:
    print("WARNING: 'Library\\bin' NOT found in PATH!", flush=True)

# 3. 逐步尝试加载库 (这是最容易卡死的地方)
print("\n[Import Check]", flush=True)

try:
    print("-> Importing numpy...", end="", flush=True)
    import numpy
    print(" OK", flush=True)
except Exception as e:
    print(f" FAIL: {e}", flush=True)

try:
    print("-> Importing torch (Critical)...", end="", flush=True)
    import torch
    print(" OK", flush=True)
    print(f"   Torch Version: {torch.__version__}", flush=True)
    print(f"   CUDA Available: {torch.cuda.is_available()}", flush=True)
except ImportError:
    print(" FAIL: Torch not found.", flush=True)
except Exception as e:
    print(f" FAIL: {e}", flush=True)

print("\n=== DEBUG PROBE FINISHED ===", flush=True)