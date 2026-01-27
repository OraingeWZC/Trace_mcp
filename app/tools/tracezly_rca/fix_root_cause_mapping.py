"""
自动修正数据集中的 root_cause 标注
将故障注入时的 service_id 映射到运行时的 service_id
"""
import os
import shutil
from loguru import logger
from tracegnn.data.bytes_db import BytesSqliteDB
from tracegnn.data.trace_graph_db import TraceGraphDB
from tracegnn.data.trace_graph import TraceGraph

# 从分析脚本得出的映射关系
ROOT_CAUSE_MAPPING = {
    18: 13,  # 'shipping.memory' → 'shipping'
    20: 10,  # 'currency.networkLatency' → 'currency'
    21: 4,   # 'cart.Failure' → 'cart'
    22: 4,   # 'cart.memory' → 'cart'
    23: 15,  # 'payment.Failure' → 'payment'
    24: 12,  # 'checkout.podKiller' → 'checkout'
    25: 15,  # 'payment.cpu' → 'payment'
    26: 1,   # 'frontend.cpu' → 'frontend'
    27: 15,  # 'payment.memory' → 'payment'
    28: 4,   # 'cart.cpu' → 'cart'
    29: 4,   # 'cart.networkLatency' → 'cart'
    30: 13,  # 'shipping.networkLatency' → 'shipping'
    31: 14,  # 'quote.networkLatency' → 'quote'
    32: 15,  # 'payment.networkLatency' → 'payment'
    33: 1,   # 'frontend.networkLatency' → 'frontend'
    34: 1,   # 'frontend.podKiller' → 'frontend'
    35: 10,  # 'currency.memory' → 'currency'
    36: 10,  # 'currency.cpu' → 'currency'
}

def fix_dataset(input_path, output_path, mapping):
    """修正数据集中的 root_cause"""

    logger.info(f"输入数据集: {input_path}")
    logger.info(f"输出数据集: {output_path}")
    logger.info(f"映射规则: {len(mapping)} 条")

    # 统计
    total_traces = 0
    anomalous_traces = 0
    fixed_count = 0
    already_correct = 0
    no_mapping = 0

    # 创建输出目录
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    # 复制数据库文件到输出路径（作为模板）
    import shutil
    output_db_file = os.path.join(output_path, "_bytes.db")
    if not os.path.exists(output_db_file):
        input_db_file = os.path.join(input_path, "_bytes.db")
        shutil.copy2(input_db_file, output_db_file)
        logger.info(f"创建输出数据库模板")

    # 打开数据库
    with TraceGraphDB(BytesSqliteDB(input_path)) as input_db, \
         TraceGraphDB(BytesSqliteDB(output_path, write=True)) as output_db:

        logger.info(f"开始处理 {len(input_db)} 个 trace...")

        for i in range(len(input_db)):
            graph: TraceGraph = input_db.get(i)
            total_traces += 1

            if graph.anomaly > 0:
                anomalous_traces += 1

                if graph.root_cause is not None:
                    old_rc = graph.root_cause

                    if old_rc in mapping:
                        # 需要修正
                        new_rc = mapping[old_rc]
                        graph.root_cause = new_rc
                        fixed_count += 1

                        if i < 5:  # 只打印前5个
                            logger.debug(f"Trace {i}: root_cause {old_rc} → {new_rc}")
                    else:
                        # 检查是否已经是正确的值
                        if old_rc in mapping.values():
                            already_correct += 1
                        else:
                            no_mapping += 1
                            if no_mapping <= 3:
                                logger.warning(f"Trace {i}: root_cause={old_rc} 没有映射规则")

            # 保存到输出数据库
            output_db.put(graph)

            if (i + 1) % 100 == 0:
                logger.info(f"已处理 {i + 1}/{len(input_db)} 个 trace...")

    logger.info(f"\n{'='*80}")
    logger.info(f"修正完成!")
    logger.info(f"  总 trace 数: {total_traces}")
    logger.info(f"  异常 trace: {anomalous_traces}")
    logger.info(f"  已修正: {fixed_count}")
    logger.info(f"  本来就正确: {already_correct}")
    logger.info(f"  无映射规则: {no_mapping}")
    logger.info(f"{'='*80}")

    return {
        'total': total_traces,
        'fixed': fixed_count,
        'already_correct': already_correct,
        'no_mapping': no_mapping
    }

def main():
    # 配置
    dataset_root = "dataset/tianchi2"
    processed_dir = os.path.join(dataset_root, "processed")

    datasets_to_fix = ['test', 'train', 'val']

    for dataset_name in datasets_to_fix:
        input_path = os.path.join(processed_dir, dataset_name)
        output_path = os.path.join(processed_dir, f"{dataset_name}_fixed")

        if not os.path.exists(input_path):
            logger.warning(f"数据集不存在: {input_path}，跳过")
            continue

        logger.info(f"\n{'='*80}")
        logger.info(f"处理数据集: {dataset_name}")
        logger.info(f"{'='*80}")

        # 数据库文件路径
        input_db_file = os.path.join(input_path, "_bytes.db")
        output_db_file = os.path.join(output_path, "_bytes.db")
        backup_db_file = os.path.join(processed_dir, f"{dataset_name}_bytes.db.backup")

        # 备份原始数据
        if not os.path.exists(backup_db_file):
            logger.info(f"备份原始数据: {backup_db_file}")
            shutil.copy2(input_db_file, backup_db_file)
        else:
            logger.info(f"备份文件已存在，跳过备份")

        # 修正数据
        result = fix_dataset(input_path, output_path, ROOT_CAUSE_MAPPING)

        # 替换原文件
        if result['fixed'] > 0:
            logger.info(f"替换原数据库文件...")
            os.remove(input_db_file)
            shutil.move(output_db_file, input_db_file)
            logger.info(f"✓ {dataset_name} 数据集已更新")

            # 清理临时目录
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
        else:
            logger.info(f"无需修正 {dataset_name} 数据集")
            # 清理临时目录
            if os.path.exists(output_path):
                shutil.rmtree(output_path)

    logger.info(f"\n{'='*80}")
    logger.info(f"所有数据集处理完成！")
    logger.info(f"备份文件保存在: {processed_dir}/*_bytes.db.backup")
    logger.info(f"如需恢复，将备份文件重命名为 _bytes.db 并复制回对应目录")
    logger.info(f"{'='*80}")

if __name__ == "__main__":
    main()
