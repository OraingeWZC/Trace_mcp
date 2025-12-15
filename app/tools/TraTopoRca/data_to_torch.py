import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import random
import os
import sqlite3
import shutil
from tracegnn.data.trace_graph import df_to_trace_graphs, TraceGraphIDManager
from tracegnn.data.trace_graph import SERVICE_ID_YAML_FILE, OPERATION_ID_YAML_FILE, STATUS_ID_YAML_FILE, FAULT_CATEGORY_YAML_FILE
from tracegnn.data.trace_graph_db import TraceGraphDB, BytesSqliteDB
from dataprocess import convert_csv_to_db
from tracegnn.utils.id_assign import IDAssign


# 修复后的函数：更灵活地加载CSV文件
def flexible_load_trace_csv(input_path: str) -> pd.DataFrame:
    """
    更灵活地加载CSV文件，避免数据类型转换错误
    """
    try:
        # 尝试直接加载而不指定严格的数据类型
        df = pd.read_csv(input_path)
        
        # 确保必要的列存在
        required_columns = ['TraceID', 'SpanID', 'ParentID', 'OperationName', 'ServiceName', 'Duration']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"缺少必要的列: {col}")
                
        # 尝试将特定列转换为合适的类型
        if 'Duration' in df.columns:
            try:
                # 先尝试转换为float
                df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')
            except:
                pass
                
        if 'StartTimeMs' in df.columns:
            try:
                df['StartTime'] = pd.to_numeric(df['StartTimeMs'], errors='coerce')
            except:
                pass
        
        if 'Anomaly' in df.columns:
            try:
                df['Anomaly'] = df['Anomaly'].astype(bool)
            except:
                pass
                
        return df
    except Exception as e:
        print(f"加载CSV文件时出错: {e}")
        raise

# 删除指定目录下的yml映射文件
def remove_yml_files(directory: str):
    """
    删除指定目录下的yml映射文件
    """
    yml_files = ['service_id.yml', 'operation_id.yml', 'status_id.yml', 'fault_category.yml']
    for yml_file in yml_files:
        file_path = os.path.join(directory, yml_file)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"已删除多余的映射文件: {file_path}")

# 扩展TraceGraphIDManager类，添加FaultCategory映射支持
class EnhancedTraceGraphIDManager(TraceGraphIDManager):
    __slots__ = TraceGraphIDManager.__slots__
    
    def __init__(self, root_dir: str):
        super().__init__(root_dir)
    
    def __enter__(self):
        super().__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
    
    def dump_to(self, output_dir: str):
        super().dump_to(output_dir)

# 处理测试数据集的RootCause和FaultCategory
def process_test_dataset_fault_mapping(test_csv_path: str, id_manager: EnhancedTraceGraphIDManager):
    """
    处理测试数据集的RootCause和FaultCategory，创建映射并修改DataFrame中的值
    """
    # 加载测试数据集
    print(f"正在加载测试数据集: {test_csv_path}")
    test_df = flexible_load_trace_csv(test_csv_path)
    
    # 统计信息
    total_anomalies = 0
    processed_root_causes = 0
    processed_fault_categories = 0
    
    # 创建用于存储转换后的根因和故障类别的列
    if 'RootCauseID' not in test_df.columns:
        test_df['RootCauseID'] = None
    if 'FaultCategoryID' not in test_df.columns:
        test_df['FaultCategoryID'] = None
    
    # 按TraceID分组处理
    for trace_id, trace_group in test_df.groupby('TraceID'):
        # 获取该trace的异常信息
        is_anomaly = trace_group['Anomaly'].iloc[0] if 'Anomaly' in trace_group.columns else False
        if is_anomaly:
            total_anomalies += 1
            
            # 取本 trace 的 RootCause 与 FaultCategory 文本
            rc_text = str(trace_group['RootCause'].iloc[0]).strip() if 'RootCause' in trace_group.columns else ''
            fc_text = str(trace_group['FaultCategory'].iloc[0]).strip() if 'FaultCategory' in trace_group.columns else ''
            fc_lower = fc_text.lower()

            # 基于 FaultCategory 判断映射表
            mapped_id = None
            if rc_text:
                if fc_lower.startswith('node'):
                    # 节点级：对照 host_id.yml（只查不增） 把_换成-
                    rc_text = rc_text.replace('_', '-')
                    mapped_id = id_manager.host_id.get(rc_text)
                else:
                    # 服务级：对照 service_id.yml（可按需裁剪 pod- 前缀）
                    rc_svc = rc_text.split('-')[0] if '-' in rc_text else rc_text
                    mapped_id = id_manager.service_id.get(rc_svc)

            if mapped_id is not None:
                test_df.loc[test_df['TraceID'] == trace_id, 'RootCause'] = mapped_id
                processed_root_causes += 1
            else:
                # 留空，后续数值化会置 0
                test_df.loc[test_df['TraceID'] == trace_id, 'RootCause'] = ''
            
            # 处理FaultCategory
            if 'FaultCategory' in trace_group.columns:
                fault_category = trace_group['FaultCategory'].iloc[0]
                if pd.notna(fault_category) and fault_category.strip():
                    # 为FaultCategory创建ID映射
                    fault_category_id = id_manager.fault_category.get_or_assign(fault_category)
                    # 更新DataFrame中的FaultCategory为fault_category_id
                    test_df.loc[test_df['TraceID'] == trace_id, 'FaultCategory'] = fault_category_id
                    processed_fault_categories += 1
    
    # 打印统计信息
    print(f"处理完成！")
    print(f"- 总异常Trace数量: {total_anomalies}")
    print(f"- 成功处理RootCause的数量: {processed_root_causes}")
    print(f"- 成功处理FaultCategory的数量: {processed_fault_categories}")
    
    return test_df




# 第二步：创建目录 dataset_c/processed
print("\n第二步：创建目录 dataset/dataset_08_09_10_11/processed")
os.makedirs('dataset/dataset_08_09_10_11/processed', exist_ok=True)

# 第三步：为train_set.csv和test_set.csv建立统一的yml映射文件
print("\n第三步：建立统一的yml映射文件")

# 加载训练集和测试集
print("正在加载训练集和测试集...")
train_df = flexible_load_trace_csv('dataset/dataset_08_09_10_11/raw/train.csv')
val_df = flexible_load_trace_csv('dataset/dataset_08_09_10_11/raw/val.csv')
test_df = flexible_load_trace_csv('dataset/dataset_08_09_10_11/raw/test.csv')

# 合并数据集用于创建统一的ID映射
combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

# 先创建一个临时目录用于生成ID映射
temp_dir = 'dataset/dataset_08_09_10_11/temp_id_generation'
os.makedirs(temp_dir, exist_ok=True)

# 创建增强的ID管理器并生成ID映射
print("正在生成统一的映射文件...")
id_manager = EnhancedTraceGraphIDManager(temp_dir)

with id_manager:
    for i, row in enumerate(combined_df.itertuples()):
        id_manager.service_id.get_or_assign(row.ServiceName or '')
        id_manager.operation_id.get_or_assign(row.OperationName or '')
        id_manager.status_id.get_or_assign(row.StatusCode or '')

# 保存映射文件到 dataset/dataset_08_09_10_11/processed 目录
id_manager.dump_to('dataset/dataset_08_09_10_11/processed')
print("映射文件已保存到 dataset/dataset_08_09_10_11/processed 目录")

# 第四步：将train_set.csv转换为db文件，保存在train文件夹下
print("\n第四步：将train.csv转换为db文件")
# 创建修改版的convert_csv_to_db函数，使其使用我们的flexible_load_trace_csv函数
def modified_convert_csv_to_db(csv_file: str, output_dir: str, id_manager=None, min_node_count: int = 2, max_node_count: int = 100, processed_df=None):
    """
    修改版的CSV转换为DB函数，使用更灵活的CSV加载方式
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载CSV文件或使用传入的已处理DataFrame
    if processed_df is None:
        print(f"正在加载CSV文件: {csv_file}")
        df = flexible_load_trace_csv(csv_file)
        print(f"成功加载 {len(df)} 行数据")
    else:
        print(f"使用已处理的DataFrame，包含 {len(processed_df)} 行数据")
        df = processed_df
        
        # 确保RootCause和FaultCategory是整数类型
        if 'RootCause' in df.columns:
            # 将非NaN值转换为整数
            df['RootCause'] = pd.to_numeric(df['RootCause'], errors='coerce').fillna(0).astype(int)
        if 'FaultCategory' in df.columns:
            # 将非NaN值转换为整数
            df['FaultCategory'] = pd.to_numeric(df['FaultCategory'], errors='coerce').fillna(0).astype(int)
    
    # 使用传入的ID管理器或创建新的
    if id_manager is None:
        id_manager = TraceGraphIDManager(output_dir)
    
    # 将DataFrame转换为TraceGraph列表
    print("正在将数据转换为图结构...")
    trace_graphs = df_to_trace_graphs(
        df=df,
        id_manager=id_manager,
        min_node_count=min_node_count,
        max_node_count=max_node_count,
        summary_file=os.path.join(output_dir, 'conversion_summary.txt'),
        merge_spans=False
    )

    if trace_graphs is None:
        trace_graphs = []
    print(f"成功转换 {len(trace_graphs)} 个图")

    # 修复：确保数据库文件存在
    db_file_name = '_bytes.db'
    db_path = os.path.join(output_dir, db_file_name)
    
    # 如果数据库文件不存在，先创建一个空的数据库文件
    if not os.path.exists(db_path):
        print(f"创建新的数据库文件: {db_path}")
        conn = sqlite3.connect(db_path)
        conn.close()
    
    # 现在创建数据库连接
    db = TraceGraphDB(BytesSqliteDB(output_dir, write=True))
    
    # 将图数据写入数据库
    print(f"正在写入数据库到: {db_path}")
    try:
        with db.write_batch():
            for graph in trace_graphs:
                # 确保root_cause和fault_category是整数类型
                if hasattr(graph, 'root_cause'):
                    if graph.root_cause is None:
                        graph.root_cause = 0
                    elif not isinstance(graph.root_cause, int):
                        try:
                            graph.root_cause = int(graph.root_cause)
                        except:
                            graph.root_cause = 0
                
                if hasattr(graph, 'fault_category'):
                    if graph.fault_category is None:
                        graph.fault_category = 0
                    elif not isinstance(graph.fault_category, int):
                        try:
                            graph.fault_category = int(graph.fault_category)
                        except:
                            graph.fault_category = 0
                
                db.add(graph)
        db.commit()
        print(f"成功写入 {len(trace_graphs)} 条记录到数据库")
    finally:
        db.close()
    
    return id_manager

if __name__ == '__main__':
    
    # 重新创建一个指向最终目录的增强版ID管理器
    final_id_manager = EnhancedTraceGraphIDManager('dataset/dataset_08_09_10_11/processed')

    # 使用修改版的函数处理训练集，传入统一的ID管理器
    train_id_manager = modified_convert_csv_to_db(
        csv_file='dataset/dataset_08_09_10_11/raw/train.csv',
        output_dir='dataset/dataset_08_09_10_11/processed/train',
        id_manager=final_id_manager,  # 传入统一的ID管理器
        min_node_count=2,
        max_node_count=100
    )

    # 第五步：将test.csv转换为db文件，保存在test文件夹下
    print("\n第五步：将val.csv转换为db文件")

    # 使用修改版的函数处理测试集，传入统一的ID管理器
    modified_convert_csv_to_db(
        csv_file='dataset/dataset_08_09_10_11/raw/val.csv',
        output_dir='dataset/dataset_08_09_10_11/processed/val',
        id_manager=final_id_manager,
        min_node_count=2,
        max_node_count=100
    )

    # 第六步：专门处理测试数据集的FaultCategory和RootCause
    print("\n第六步：处理测试数据集的RootCause和FaultCategory...")
    processed_test_df = process_test_dataset_fault_mapping(
        test_csv_path='dataset/dataset_08_09_10_11/raw/test.csv',
        id_manager=final_id_manager
    )

    # 第七步：将处理后的test.csv转换为db文件，保存在test文件夹下
    print("\n第七步：将处理后的test.csv转换为db文件")

    # 使用修改版的函数处理测试集，传入统一的ID管理器和处理后的DataFrame
    modified_convert_csv_to_db(
        csv_file='dataset/dataset_08_09_10_11/raw/test.csv',
        output_dir='dataset/dataset_08_09_10_11/processed/test',
        id_manager=final_id_manager,
        min_node_count=2,
        max_node_count=100,
        processed_df=processed_test_df
    )

    # 再次保存ID映射文件到主目录，包含新的FaultCategory映射
    print("\n保存最终的ID映射文件到主目录...")
    final_id_manager.dump_to('dataset/dataset_08_09_10_11/processed')
    print("已完成包含FaultCategory的ID映射文件保存")

    # 删除train和test文件夹下的yml映射文件，确保只保留主目录下的统一映射
    print("\n清理多余的映射文件...")
    remove_yml_files('dataset/dataset_08_09_10_11/processed/train')
    remove_yml_files('dataset/dataset_08_09_10_11/processed/test')

    # 现在所有操作都完成了，安全地清理临时目录
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"已清理临时目录: {temp_dir}")

    print("\n所有任务已完成！")