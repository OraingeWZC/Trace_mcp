# Trace_mcp/app/main.py
from typing import Any, Dict, Annotated, Optional
from mcp.server.fastmcp import FastMCP
from app.schemas import ToolRequest
from app.handler import run_script
from pydantic import Field

mcp = FastMCP("trace-tools")

@mcp.tool()
def check_env() -> Dict[str, Any]:
    """
    环境诊断探针。
    用于检测子进程是否能正常启动、PATH是否正确、PyTorch是否能加载。
    """
    # 构造一个假的 ToolRequest，欺骗 handler 去运行我们的探针脚本
    # op="svnd" 会让 handler 定位到 trace_svnd_diag 目录
    # stage="debug", op="check" -> 对应文件名 debug_check.py
    req = ToolRequest(op="check_svnd", stage="debug", extra_args={})    
    # 临时修改 handler 里的逻辑让它能找到 debug_check.py
    # 或者我们更简单点，直接利用 handler 现有的逻辑：
    # handler 会寻找 {req.stage}_{req.op}.py
    # 所以文件名必须叫 debug_check.py (stage=debug, op=check)
    
    result = run_script(req)
    return {"exit_code": result.exit_code, "log": result.log}

@mcp.tool()
def ping() -> str:
    """
    基础连接测试工具。
    不加载 PyTorch，不启动子进程，仅用于验证 MCP 通信链路是否通畅。
    """
    return "pong"

# ================= Group 1: Trace + Topology Fusion (SVND) =================
# 关键词：Trace拓扑融合、节点物理信息、上下文、复杂诊断

@mcp.tool()
def preprocess_aiops_svnd(**kwargs: Any) -> Dict[str, Any]:
    """
    [Trace拓扑融合] 数据预处理：融合 Trace 调用链与物理部署拓扑信息。
    适用于需要结合 Service、Node 节点故障以及时间窗口上下文的复杂诊断场景。
    对应脚本：trace_svnd_diag/make_aiops_svnd.py
    """
    req = ToolRequest(op="aiops_svnd", stage="preprocess", extra_args=kwargs)
    result = run_script(req)
    if result.exit_code != 0:
        raise RuntimeError(f"Fusion Preprocess failed:\n{result.log}")
    return {"exit_code": result.exit_code, "log": result.log}

@mcp.tool()
def train_aiops_svnd(**kwargs: Any) -> Dict[str, Any]:
    """
    [Trace拓扑融合] 模型训练：融合 Trace 调用链与物理部署拓扑信息。
    对应脚本：trace_svnd_diag/train_aiops_svnd.py
    """
    req = ToolRequest(op="aiops_svnd", stage="train", extra_args=kwargs)
    result = run_script(req)
    if result.exit_code != 0:
        raise RuntimeError(f"Fusion Training failed:\n{result.log}")
    return {"exit_code": result.exit_code, "log": result.log}

@mcp.tool()
def test_aiops_svnd(
    # 使用 Annotated[类型, Field(description="描述")]
    model_path: Annotated[str, Field(description="模型权重文件路径 (.pt)")] = "dataset/aiops_svnd/1019/aiops_nodectx_multihead.pt",
    data_root: Annotated[str, Field(description="数据集目录")] = "dataset/aiops_svnd",
    batch_size: Annotated[int, Field(description="批大小 (Batch Size)")] = 128,
    seed: Annotated[int, Field(description="随机种子")] = 2025,
    device: Annotated[str, Field(description="运行设备 ('cuda' 或 'cpu')")] = "cuda",
    limit: Annotated[int, Field(description="测试样本数量限制 (强烈建议设置以避免超时，如 50)")] = 50
) -> Dict[str, Any]:
    """
    [Trace拓扑融合] 模型测试：融合 Trace 调用链与物理部署拓扑信息。
    对应脚本：trace_svnd_diag/test_aiops_svnd.py

    Args:
        model_path (str): 模型权重文件路径
        data_root (str): 数据集目录
        limit (int): 测试样本数量限制 (强烈建议设置以避免超时)
    """
    extra_args = {
        "model_path": model_path,
        "data_root": data_root,
        "batch_size": batch_size,
        "seed": seed,
        "device": device,
        "limit": limit
    }
    req = ToolRequest(op="aiops_svnd", stage="test", extra_args=extra_args)
    result = run_script(req)
    if result.exit_code != 0:
        raise RuntimeError(f"Fusion Test failed:\n{result.log}")
    return {"exit_code": result.exit_code, "log": result.log}


# ================= Group 2: Single Trace (SV) =================
# 关键词：单Trace、纯结构、轻量级

@mcp.tool()
def preprocess_aiops_sv(**kwargs: Any) -> Dict[str, Any]:
    """
    [单Trace] 数据预处理：仅基于 Trace 调用链结构。
    适用于不依赖物理节点(Node)信息的轻量级诊断场景，仅关注 Trace 内部结构异常。
    对应脚本：trace_sv_diag/make_aiops_sv.py
    """
    req = ToolRequest(op="aiops_sv", stage="preprocess", extra_args=kwargs)
    result = run_script(req)
    if result.exit_code != 0:
        raise RuntimeError(f"SingleTrace Preprocess failed:\n{result.log}")
    return {"exit_code": result.exit_code, "log": result.log}

@mcp.tool()
def train_aiops_sv(**kwargs: Any) -> Dict[str, Any]:
    """
    [单Trace] 模型训练：仅基于 Trace 调用链结构。
    对应脚本：trace_sv_diag/train_aiops_sv.py
    
    Args:
        data_root (str): 数据目录
        save_dir (str): 保存目录
        task (str): 'fine' 或 'superfine' 
    """
    req = ToolRequest(op="aiops_sv", stage="train", extra_args=kwargs)
    result = run_script(req)
    if result.exit_code != 0:
        raise RuntimeError(f"SingleTrace Training failed:\n{result.log}")
    return {"exit_code": result.exit_code, "log": result.log}

@mcp.tool()
def test_aiops_sv(
    model_path: str = "dataset/aiops_sv/aiops_superfine_cls.pth",
    data_root: str = "dataset/aiops_sv",
    task: str = "superfine",
    batch: int = 64,
    seed: int = 2025,
    device: str = "cuda",
    min_type_support: int = 150,
    run_name: str = "trace_only",
    limit: int = 100
) -> Dict[str, Any]:
    """
    [单Trace] 模型测试：仅基于 Trace 调用链结构进行故障分类评估。
    对应脚本：trace_sv_diag/test_aiops_sv.py
    
    Args:
        model_path (str): 模型权重文件路径 (.pth)，例如 'dataset/aiops_sv/aiops_superfine_cls.pth'
        data_root (str): 数据集目录 (包含 test.jsonl 与 vocab.json)
        task (str): 任务类型，可选 'fine' 或 'superfine', 默认 'superfine'
        batch (int): 批大小 (Batch Size)，默认 64
        seed (int): 随机种子，默认 2025
        device (str): 运行设备，'cuda' 或 'cpu'
        min_type_support (int): 过滤小样本类别的阈值 (默认 150)，样本数少于此值的类别将不参与详细报告
        run_name (str): 运行名称，决定输出结果保存的子目录名
        limit (int): 测试样本数量限制 (可选，用于快速验证/避免超时)
    """
    # 将参数打包，handler.py 会自动将下划线(data_root)转换为短横线参数(--data-root)
    extra_args = {
        "model_path": model_path,
        "data_root": data_root,
        "task": task,
        "batch": batch,
        "seed": seed,
        "device": device,
        "min_type_support": min_type_support,
        "run_name": run_name,
        "limit": limit
    }
    
    req = ToolRequest(op="aiops_sv", stage="test", extra_args=extra_args)
    result = run_script(req)
    
    if result.exit_code != 0:
        raise RuntimeError(f"SV Test failed:\n{result.log}")
        
    return {"exit_code": result.exit_code, "log": result.log}

# ================= Group 3: TraTopoRca (GTrace) =================

@mcp.tool()
def train_tracerca(
    dataset: str = "dataset_demo",
    batch_size: int = 32,
    max_epochs: int = 10,
    device: str = "cuda",
    seed: int = 1234,
    model_path: str = "save/model_rerank.pth"
) -> Dict[str, Any]:
    """
    [TraceRca] 训练工具：运行 tracegnn 图神经网络训练。
    对应命令: python -m tracegnn.models.gtrace.mymodel_main
    
    Args:
        dataset: 数据集名称 (默认: dataset_demo)
        max_epochs: 训练轮数 (测试时建议设为 1)
        model_path: 模型保存路径
    """
    extra_args = {
        "dataset": dataset,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "device": device,
        "seed": seed,
        "model_path": model_path
    }
    # op="tratoporca", stage="train" -> 触发 handler 里的模块运行逻辑
    req = ToolRequest(op="tratoporca", stage="train", extra_args=extra_args)
    result = run_script(req)
    
    if result.exit_code != 0:
        raise RuntimeError(f"TraTopoRca Training failed:\n{result.log}")
    return {"exit_code": result.exit_code, "log": result.log}


@mcp.tool()
def test_tracerca(
    model: Optional[str] = None,
    test_dataset: Optional[str] = None,
    limit: Optional[str] = None,
    report_dir: str = "reports_mcp"
) -> Dict[str, Any]:
    """
    [TraceRca] 评估工具：运行 mymodel_test.py 进行推理和根因分析。
    
    Args:
        model: 模型路径 (.pth)，不填则使用 config 默认
        test_dataset: 测试子集
        limit: 限制测试数量 (例如 10)，用于快速验证
        report_dir: 报告输出目录
    """
    # 在这里定义那些不需要用户输入的“隐藏参数”
    extra_args = {
        # === 用户输入的参数 ===
        "model": model,
        "test_dataset": test_dataset,
        "limit": limit,
        "report_dir": report_dir,
        
        # === 隐藏的固定参数 (写死在这里) ===
        "batch_size": 32,      # 默认批次大小
        "export_debug": True,  # 默认开启调试输出
        # "dataset": "dataset_demo"  # 如果你想硬编码数据集，可以把这行解开
    }
    
    req = ToolRequest(op="tratoporca", stage="test", extra_args=extra_args)
    result = run_script(req)
    
    if result.exit_code != 0:
        raise RuntimeError(f"TraTopoRca test failed:\n{result.log}")
    return {"exit_code": result.exit_code, "log": result.log}

if __name__ == "__main__":
    # mcp.run(transport="sse")
    mcp.run()