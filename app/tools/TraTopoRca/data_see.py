import warnings
warnings.filterwarnings("ignore")

import mltk
import torch
from tracegnn.models.gtrace.config import ExpConfig
from tracegnn.models.gtrace.dataset import TrainDataset, TestDataset
from tracegnn.data.trace_graph import TraceGraph
from tracegnn.data.trace_graph_db import TraceGraphDB
import dgl
import numpy as np
import os
import struct

def analyze_train_dataset():
    print("=== 分析训练数据集 ===")
    # 使用 Experiment 上下文管理器
    with mltk.Experiment(ExpConfig) as exp:
        config = exp.config
        
        # 创建训练数据集实例
        train_dataset = TrainDataset(config, valid=False)
        
        # print(f"训练数据集大小: {len(train_dataset)}")
        
        # 获取前几个样本进行分析
        num_samples = min(1, len(train_dataset))
        for idx in range(num_samples):
            print(f"\n==================== 样本 {idx} ====================")
            graph = train_dataset[idx]

            print(graph)
            
            print(f"图节点数量: {graph.num_nodes()}")
            print(f"图边数量: {graph.num_edges()}")
            
            # 打印节点特征信息
            print("\n--- 芯片特征信息 ---")
            print(f"operation_id: {graph.ndata['operation_id']}")
            print(f"service_id: {graph.ndata['service_id']}")
            print(f"status_id: {graph.ndata['status_id']}")
            print(f"node_depth: {graph.ndata['node_depth']}")
            print(f"span_count: {graph.ndata['span_count']}")
            
            # 特别关注latency相关特征
            print("\n--- Latency相关特征 ---")
            latency = graph.ndata['latency']
            
            print(f"start_time: {graph.ndata['start_time']}")
            
            # 打印每个节点的详细信息
            print("\n--- 每个节点的详细信息 ---")
            for i in range(graph.num_nodes()):
                print(f"节点 {i}:")
                print(f"  operation_id: {graph.ndata['operation_id'][i].item()}")
                print(f"  service_id: {graph.ndata['service_id'][i].item()}")
                print(f"  status_id: {graph.ndata['status_id'][i].item()}")
                print(f"  node_depth: {graph.ndata['node_depth'][i].item()}")
                print(f"  span_count: {graph.ndata['span_count'][i].item()}")
                print(f"  latency (avg_latency): {graph.ndata['latency'][i].item():.4f}")
                print(f"  start_time: {graph.ndata['start_time'][i].item()}")
            
            # 分析图结构
            print("\n--- 图结构信息 ---")
            src, dst = graph.edges()
            print(f"边信息 (源节点 -> 目标节点):")
            for i in range(len(src)):
                print(f"  {src[i].item()} -> {dst[i].item()}")
            
            # 计算入度和出度
            in_degrees = graph.in_degrees()
            out_degrees = graph.out_degrees()
            print(f"节点入度: {in_degrees}")
            print(f"节点出度: {out_degrees}")
            
            # 查找根节点（入度为0的节点）
            root_nodes = torch.where(in_degrees == 0)[0]
            print(f"根节点: {root_nodes.tolist()}")

            print(f"图级属性 anomaly: {graph.anomaly}")
            
            print("=" * 50)

def analyze_test_dataset():
    print("\n\n=== 分析测试数据集 ===")
    # 使用 Experiment 上下文管理器
    with mltk.Experiment(ExpConfig) as exp:
        config = exp.config
        
        # 创建测试数据集实例
        try:
            test_dataset = TestDataset(config)
            print(f"测试数据集大小: {len(test_dataset)}")
            
            # 获取前几个样本进行分析
            num_samples = min(3, len(test_dataset))
            for idx in range(num_samples):
                print(f"\n==================== 测试样本 {idx} ====================")
                graph, anomaly, root_cause, fault_category = test_dataset[idx]
                
                print(f"图节点数量: {graph.num_nodes()}")
                print(f"图边数量: {graph.num_edges()}")
                print(f"Anomaly: {anomaly}")
                print(f"Root cause: {root_cause}")
                print(f"Fault category: {fault_category}")
                
                # 分析图的节点特征
                print("\n--- 图节点特征信息 ---")
                print(f"operation_id: {graph.ndata['operation_id']}")
                print(f"service_id: {graph.ndata['service_id']}")
                print(f"status_id: {graph.ndata['status_id']}")
                print(f"latency: {graph.ndata['latency']}")
                
                print("=" * 50)
        except Exception as e:
            print(f"无法加载测试数据集: {e}")


# ... existing code ...
def analyze_anomaly_distribution():
    """
    统计训练集、验证集和测试集中anomaly的分布情况
    """
    print("=== 统计各数据集中的anomaly分布情况 ===")
    
    with mltk.Experiment(ExpConfig) as exp:
        config = exp.config
        
        # 统计训练集
        print("\n--- 训练集统计 ---")
        train_dataset = TrainDataset(config, valid=False)
        train_total = len(train_dataset)
        
        # 直接从数据库中统计anomaly，避免加载完整图数据
        train_anomaly_count = 0
        for i in range(train_total):
            # 直接从数据库获取原始字节数据
            graph_bytes = train_dataset.train_db.bytes_db.get(i)
            # 反序列化为TraceGraph对象
            trace_graph = TraceGraph.from_bytes(graph_bytes)
            # 获取anomaly属性
            if trace_graph.anomaly != 0:
                train_anomaly_count += 1
                
        print(f"训练集总数: {train_total}")
        print(f"包含异常的图数量: {train_anomaly_count}")
        print(f"异常比例: {train_anomaly_count/train_total*100:.2f}%" if train_total > 0 else "N/A")
        
        # 统计验证集
        print("\n--- 验证集统计 ---")
        val_dataset = TrainDataset(config, valid=True)
        val_total = len(val_dataset)
        
        # 直接从数据库中统计anomaly，避免加载完整图数据
        val_anomaly_count = 0
        for i in range(val_total):
            # 直接从数据库获取原始字节数据
            graph_bytes = val_dataset.train_db.bytes_db.get(i)
            # 反序列化为TraceGraph对象
            trace_graph = TraceGraph.from_bytes(graph_bytes)
            # 获取anomaly属性
            if trace_graph.anomaly != 0:
                val_anomaly_count += 1
                
        print(f"验证集总数: {val_total}")
        print(f"包含异常的图数量: {val_anomaly_count}")
        print(f"异常比例: {val_anomaly_count/val_total*100:.2f}%" if val_total > 0 else "N/A")
        
        # 统计测试集
        print("\n--- 测试集统计 ---")
        try:
            test_dataset = TestDataset(config)
            test_total = len(test_dataset)
            
            # 直接从数据库中统计anomaly，避免加载完整图数据
            test_anomaly_count = 0
            for i in range(test_total):
                # 直接从数据库获取原始字节数据
                graph_bytes = test_dataset.test_db.bytes_db.get(i)
                # 反序列化为TraceGraph对象
                trace_graph = TraceGraph.from_bytes(graph_bytes)
                # 获取anomaly属性
                if trace_graph.anomaly != 0:
                    test_anomaly_count += 1
                    
            print(f"测试集总数: {test_total}")
            print(f"包含异常的图数量: {test_anomaly_count}")
            print(f"异常比例: {test_anomaly_count/test_total*100:.2f}%" if test_total > 0 else "N/A")
        except Exception as e:
            print(f"无法加载测试数据集: {e}")
def view_test_bytes_db():
    """
    查看dataset_d/processed/test/_bytes.db中的数据形式
    """
    print("=== 查看测试数据集_bytes.db数据形式 ===")
    
    try:
        # 构建测试数据集_bytes.db的路径
        db_path = os.path.join("dataset_d", "processed", "test", "_bytes.db")
        
        if not os.path.exists(db_path):
            print(f"错误: 找不到文件 {db_path}")
            return
            
        print(f"数据库文件路径: {db_path}")
        print(f"文件大小: {os.path.getsize(db_path)} 字节")
        
        # 直接从TestDataset获取数据库访问
        print("\n--- 使用TestDataset读取数据 ---\n")
        
        # 导入mltk
        import mltk
        
        with mltk.Experiment(ExpConfig) as exp:
            config = exp.config
            try:
                test_dataset = TestDataset(config)
                
                # 检查数据库大小
                db_size = len(test_dataset)
                print(f"数据库中包含的样本数量: {db_size}")
                
                # 查看TestDataset的结构
                print("\n--- TestDataset结构信息 ---")
                print(f"TestDataset类: {test_dataset.__class__.__name__}")
                print(f"TestDataset属性: {[attr for attr in dir(test_dataset) if not attr.startswith('_')]}")
                
                # 读取前5个样本
                num_samples = min(5, db_size)
                print(f"\n--- 查看前{num_samples}个样本的数据形式 ---\n")
                
                for i in range(num_samples):
                    print(f"\n样本 {i}:\n")
                    
                    # 尝试直接获取样本
                    try:
                        sample = test_dataset[i]
                        print(f"  样本类型: {type(sample).__name__}")
                        
                        # 检查样本是元组还是单个对象
                        if isinstance(sample, tuple):
                            print(f"  元组长度: {len(sample)}")
                            for j, item in enumerate(sample):
                                print(f"  元组元素 {j}: {type(item).__name__}")
                                
                            # 尝试获取graph对象
                            if len(sample) > 0:
                                graph = sample[0]
                                print(f"\n  图对象信息:")
                                print(f"  图类型: {type(graph).__name__}")
                                print(f"  图属性: {[attr for attr in dir(graph) if not attr.startswith('_')]}")
                                
                                # 检查是否有ndata属性
                                if hasattr(graph, 'ndata'):
                                    print(f"  ndata键: {list(graph.ndata.keys())}")
                                
                                # 检查是否有边
                                if hasattr(graph, 'edges'):
                                    try:
                                        src, dst = graph.edges()
                                        print(f"  边数量: {len(src)}")
                                    except:
                                        print("  无法获取边信息")
                        else:
                            # 单个TraceGraph对象
                            print(f"  对象属性: {[attr for attr in dir(sample) if not attr.startswith('_')]}")
                            
                    except Exception as e:
                        print(f"  读取样本失败: {e}")
                        import traceback
                        traceback.print_exc()
                        
                # 尝试直接访问bytes_db
                if hasattr(test_dataset, 'test_db') and hasattr(test_dataset.test_db, 'bytes_db'):
                    print("\n--- 直接访问bytes_db ---")
                    bytes_db = test_dataset.test_db.bytes_db
                    print(f"bytes_db类型: {type(bytes_db).__name__}")
                    
                    # 读取前3个原始字节数据
                    try:
                        for i in range(min(3, db_size)):
                            try:
                                graph_bytes = bytes_db.get(i)
                                print(f"\n样本 {i} 原始字节数据:")
                                print(f"  大小: {len(graph_bytes)} 字节")
                                print(f"  前20字节(十六进制): {graph_bytes[:20].hex()}")
                                print(f"  前100字节(字符): {graph_bytes[:100] if len(graph_bytes)>=100 else graph_bytes}")
                            except Exception as e:
                                print(f"  读取样本 {i} 原始数据失败: {e}")
                    except Exception as e:
                        print(f"  访问bytes_db失败: {e}")
                        
            except Exception as e:
                print(f"无法创建TestDataset实例: {e}")
                import traceback
                traceback.print_exc()
                
        print("\n=== 数据形式查看完成 ===")
        
    except Exception as e:
        print(f"查看数据时出错: {e}")
        import traceback
        traceback.print_exc()

# ... existing code ...
# ... existing code ...


def show_test_dataset_details():
    """
    展示测试数据集中各图的trace_id、node_count、max_depth、anomaly、root_cause及fault_category字段内容
    """
    print("=== 展示测试数据集详细字段信息 ===")
    
    try:
        import mltk
        from tracegnn.models.gtrace.config import ExpConfig
        from tracegnn.models.gtrace.dataset import TestDataset
        from tracegnn.data.trace_graph import TraceGraph
        import os
        
        with mltk.Experiment(ExpConfig) as exp:
            config = exp.config
            try:
                test_dataset = TestDataset(config)
                db_size = len(test_dataset)
                print(f"测试数据集样本数量: {db_size}")
                print("\n--- 数据集详细字段信息 ---")
                print(f"{'索引':<6}{'trace_id':<25}{'node_count':<12}{'max_depth':<10}{'anomaly':<8}{'root_cause':<15}{'fault_category':<15}{'root_cause_type':<15}{'fault_type':<15}")
                print("-" * 120)
                
                # 统计变量
                total_anomaly = 0
                none_root_cause_in_anomaly = 0
                
                # 读取所有样本的信息
                for i in range(min(50, db_size)):  # 限制最多显示50个样本
                    try:
                        # 直接从bytes_db获取原始数据，避免构建完整图对象
                        if hasattr(test_dataset, 'test_db') and hasattr(test_dataset.test_db, 'bytes_db'):
                            graph_bytes = test_dataset.test_db.bytes_db.get(i)
                            trace_graph = TraceGraph.from_bytes(graph_bytes)
                            
                            # 获取所需字段
                            trace_id = str(trace_graph.trace_id) if trace_graph.trace_id else "None"
                            node_count = trace_graph.node_count if trace_graph.node_count else "None"
                            max_depth = trace_graph.max_depth if trace_graph.max_depth else "None"
                            anomaly = trace_graph.anomaly
                            root_cause = trace_graph.root_cause if trace_graph.root_cause is not None else "None"
                            fault_category = trace_graph.fault_category if trace_graph.fault_category is not None else "None"
                            
                            # 统计异常样本和root_cause为None的异常样本
                            if anomaly != 0:
                                total_anomaly += 1
                                if trace_graph.root_cause is None:
                                    none_root_cause_in_anomaly += 1
                            
                            # 添加类型信息用于调试
                            root_cause_type = str(type(trace_graph.root_cause)) if trace_graph.root_cause is not None else "NoneType"
                            fault_type = str(type(trace_graph.fault_category)) if trace_graph.fault_category is not None else "NoneType"
                            
                            # 打印信息
                            print(f"{i:<6}{trace_id:<25}{node_count:<12}{max_depth:<10}{anomaly:<8}{root_cause:<15}{fault_category:<15}{root_cause_type:<15}{fault_type:<15}")
                        else:
                            # 尝试直接获取样本
                            sample = test_dataset[i]
                            if isinstance(sample, tuple):
                                if len(sample) >= 4:
                                    _, anomaly, root_cause, fault_category = sample
                                    trace_id = "N/A"
                                    node_count = "N/A"
                                    max_depth = "N/A"
                                    
                                    # 统计异常样本和root_cause为None的异常样本
                                    if anomaly != 0:
                                        total_anomaly += 1
                                        if root_cause is None:
                                            none_root_cause_in_anomaly += 1
                                    
                                    # 添加类型信息用于调试
                                    root_cause_type = str(type(root_cause)) if root_cause is not None else "NoneType"
                                    fault_type = str(type(fault_category)) if fault_category is not None else "NoneType"
                                    
                                    if hasattr(sample[0], 'num_nodes'):
                                        node_count = sample[0].num_nodes()
                                    
                                    print(f"{i:<6}{trace_id:<25}{node_count:<12}{max_depth:<10}{anomaly:<8}{root_cause:<15}{fault_category:<15}{root_cause_type:<15}{fault_type:<15}")
                    except Exception as e:
                        print(f"{i:<6}读取失败: {str(e)[:50]}...")
                        import traceback
                        traceback.print_exc()
                        
                # 统计异常分布
                print("\n--- 异常分布统计 ---")
                anomaly_count = 0
                for i in range(db_size):
                    try:
                        if hasattr(test_dataset, 'test_db') and hasattr(test_dataset.test_db, 'bytes_db'):
                            graph_bytes = test_dataset.test_db.bytes_db.get(i)
                            trace_graph = TraceGraph.from_bytes(graph_bytes)
                            if trace_graph.anomaly != 0:
                                anomaly_count += 1
                                # 统计root_cause为None的异常样本
                                if trace_graph.root_cause is None:
                                    none_root_cause_in_anomaly += 1
                        else:
                            # 处理直接访问数据集的情况
                            sample = test_dataset[i]
                            if isinstance(sample, tuple) and len(sample) >= 4:
                                _, anomaly, root_cause, _ = sample
                                if anomaly != 0:
                                    anomaly_count += 1
                                    if root_cause is None:
                                        none_root_cause_in_anomaly += 1
                    except:
                        continue
                
                print(f"总样本数: {db_size}")
                print(f"异常样本数: {anomaly_count}")
                print(f"异常比例: {anomaly_count/db_size*100:.2f}%" if db_size > 0 else "N/A")
                print(f"异常样本中root_cause为None的数量: {none_root_cause_in_anomaly}")
                print(f"异常样本中root_cause为None的比例: {none_root_cause_in_anomaly/anomaly_count*100:.2f}%" if anomaly_count > 0 else "N/A")
                
            except Exception as e:
                print(f"无法加载测试数据集: {e}")
                
    except Exception as e:
        print(f"查看数据时出错: {e}")


def show_selected_traces():
    """
    展示3条正常trace和3条异常trace的详细信息
    """
    print("=== 展示精选trace信息 ===")
    
    try:
        import mltk
        from tracegnn.models.gtrace.config import ExpConfig
        from tracegnn.models.gtrace.dataset import TestDataset
        from tracegnn.data.trace_graph import TraceGraph
        import os
        
        with mltk.Experiment(ExpConfig) as exp:
            config = exp.config
            try:
                test_dataset = TestDataset(config)
                db_size = len(test_dataset)
                print(f"测试数据集样本数量: {db_size}")
                
                # 用于存储正常和异常样本
                normal_traces = []
                anomaly_traces = []
                
                # 收集样本
                for i in range(db_size):
                    try:
                        if hasattr(test_dataset, 'test_db') and hasattr(test_dataset.test_db, 'bytes_db'):
                            graph_bytes = test_dataset.test_db.bytes_db.get(i)
                            trace_graph = TraceGraph.from_bytes(graph_bytes)
                            
                            # 保存样本信息
                            sample_info = {
                                'index': i,
                                'trace_id': str(trace_graph.trace_id) if trace_graph.trace_id else "None",
                                'node_count': trace_graph.node_count if trace_graph.node_count else "None",
                                'max_depth': trace_graph.max_depth if trace_graph.max_depth else "None",
                                'anomaly': trace_graph.anomaly,
                                'root_cause': trace_graph.root_cause if trace_graph.root_cause else "None",
                                'fault_category': trace_graph.fault_category if trace_graph.fault_category else "None",
                                'trace_graph': trace_graph
                            }
                            
                            if trace_graph.anomaly != 0 and len(anomaly_traces) < 3:
                                anomaly_traces.append(sample_info)
                            elif trace_graph.anomaly == 0 and len(normal_traces) < 3:
                                normal_traces.append(sample_info)
                            
                            # 如果已经收集了足够的样本，提前结束
                            if len(normal_traces) >= 3 and len(anomaly_traces) >= 3:
                                break
                    except Exception as e:
                        continue
                
                # 打印正常样本
                print("\n--- 正常Trace样本 (3个) ---\n")
                if normal_traces:
                    print(f"{'索引':<6}{'trace_id':<25}{'node_count':<12}{'max_depth':<10}{'anomaly':<8}{'root_cause':<15}{'fault_category':<15}")
                    print("-" * 85)
                    for sample in normal_traces:
                        print(f"{sample['index']:<6}{sample['trace_id']:<25}{sample['node_count']:<12}{sample['max_depth']:<10}{sample['anomaly']:<8}{sample['root_cause']:<15}{sample['fault_category']:<15}")
                else:
                    print("未找到正常Trace样本")
                
                # 打印异常样本
                print("\n--- 异常Trace样本 (3个) ---\n")
                if anomaly_traces:
                    print(f"{'索引':<6}{'trace_id':<25}{'node_count':<12}{'max_depth':<10}{'anomaly':<8}{'root_cause':<15}{'fault_category':<15}")
                    print("-" * 85)
                    for sample in anomaly_traces:
                        print(f"{sample['index']:<6}{sample['trace_id']:<25}{sample['node_count']:<12}{sample['max_depth']:<10}{sample['anomaly']:<8}{sample['root_cause']:<15}{sample['fault_category']:<15}")
                else:
                    print("未找到异常Trace样本")
                
                # 统计异常分布
                print("\n--- 异常分布统计 ---\n")
                anomaly_count = 0
                for i in range(db_size):
                    try:
                        if hasattr(test_dataset, 'test_db') and hasattr(test_dataset.test_db, 'bytes_db'):
                            graph_bytes = test_dataset.test_db.bytes_db.get(i)
                            trace_graph = TraceGraph.from_bytes(graph_bytes)
                            if trace_graph.anomaly != 0:
                                anomaly_count += 1
                    except:
                        continue
                
                print(f"总样本数: {db_size}")
                print(f"异常样本数: {anomaly_count}")
                print(f"异常比例: {(anomaly_count/db_size*100):.2f}%" if db_size > 0 else "N/A")
                
            except Exception as e:
                print(f"无法加载测试数据集: {e}")
                
    except Exception as e:
        print(f"查看数据时出错: {e}")

def analyze_train_dataset():
    print("=== 分析训练数据集 ===")
    # 使用 Experiment 上下文管理器
    with mltk.Experiment(ExpConfig) as exp:
        config = exp.config
        
        # 创建训练数据集实例
        train_dataset = TrainDataset(config, valid=False)
        
        print(f"训练数据集大小: {len(train_dataset)}")
        
        # 获取前几个样本进行分析
        num_samples = min(3, len(train_dataset))
        for idx in range(num_samples):
            print(f"\n==================== 样本 {idx} ====================")
            # 直接从数据库获取原始字节数据
            graph_bytes = train_dataset.train_db.bytes_db.get(idx)
            # 反序列化为TraceGraph对象
            trace_graph = TraceGraph.from_bytes(graph_bytes)
            
            print(f"Trace ID: {trace_graph.trace_id}")
            print(f"节点数量: {trace_graph.node_count}")
            print(f"最大深度: {trace_graph.max_depth}")
            print(f"是否异常: {trace_graph.anomaly}")
            if trace_graph.anomaly:
                print(f"根因: {trace_graph.root_cause}")
                print(f"故障类别: {trace_graph.fault_category}")
            
            # 转换为DGL图
            dgl_graph = train_dataset[idx]
            
            print(f"DGL图节点数量: {dgl_graph.num_nodes()}")
            print(f"DGL图边数量: {dgl_graph.num_edges()}")
            
            # 打印节点特征信息
            print("\n--- 节点特征信息 ---")
            print(f"operation_id: {dgl_graph.ndata['operation_id']}")
            print(f"service_id: {dgl_graph.ndata['service_id']}")
            print(f"status_id: {dgl_graph.ndata['status_id']}")
            print(f"node_depth: {dgl_graph.ndata['node_depth']}")
            print(f"span_count: {dgl_graph.ndata['span_count']}")
            
            # 特别关注latency相关特征
            print("\n--- Latency相关特征 ---")
            print(f"latency: {dgl_graph.ndata['latency']}")
            print(f"start_time: {dgl_graph.ndata['start_time']}")
            
            # 打印每个节点的详细信息
            print("\n--- 每个节点的详细信息 ---")
            for i in range(dgl_graph.num_nodes()):
                print(f"节点 {i}:")
                print(f"  operation_id: {dgl_graph.ndata['operation_id'][i].item()}")
                print(f"  service_id: {dgl_graph.ndata['service_id'][i].item()}")
                print(f"  status_id: {dgl_graph.ndata['status_id'][i].item()}")
                print(f"  node_depth: {dgl_graph.ndata['node_depth'][i].item()}")
                print(f"  span_count: {dgl_graph.ndata['span_count'][i].item()}")
                print(f"  latency (avg_latency): {dgl_graph.ndata['latency'][i].item():.4f}")
                print(f"  start_time: {dgl_graph.ndata['start_time'][i].item()}")
            
            # 分析图结构
            print("\n--- 图结构信息 ---")
            src, dst = dgl_graph.edges()
            print(f"边信息 (源节点 -> 目标节点):")
            for i in range(len(src)):
                print(f"  {src[i].item()} -> {dst[i].item()}")
            
            # 计算入度和出度
            in_degrees = dgl_graph.in_degrees()
            out_degrees = dgl_graph.out_degrees()
            print(f"节点入度: {in_degrees}")
            print(f"节点出度: {out_degrees}")
            
            # 查找根节点（入度为0的节点）
            root_nodes = torch.where(in_degrees == 0)[0]
            print(f"根节点: {root_nodes.tolist()}")

            print("=" * 50)

def main():
    """
    主函数，提供命令行接口
    """
    print("=== tracezly 数据查看工具 ===")
    print("1. 展示测试数据集详细字段信息")
    print("2. 展示异常trace信息")
    print("3. 展示3条正常trace和3条异常trace")
    print("4. 展示训练集数据集详细字段信息")
    
    choice = input("请选择操作 (默认: 3): ").strip()
    
    if choice == "1":
        show_test_dataset_details()
    elif choice == "2":
        analyze_anomaly_distribution()
    elif choice == "3":
        show_selected_traces()
    elif choice == "4":
        analyze_train_dataset()

if __name__ == "__main__":
    main()