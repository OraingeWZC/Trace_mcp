#!/usr/bin/env python3
"""
分析异常检测F1与根因定位Top5准确率的关系
"""
import re
import matplotlib.pyplot as plt
import numpy as np

def parse_log(log_file):
    """解析日志，提取F1和根因定位指标"""
    epochs = []
    f1_scores = []
    top5_acc = []
    top1_acc = []
    best_f1_scores = []  # best_fscore
    precision_scores = []
    recall_scores = []
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    epoch = -1
    for i, line in enumerate(lines):
        if '-------------> Epoch' in line:
            epoch = int(re.search(r'Epoch (\d+)', line).group(1))
        
        # F1和AUC
        if 'fscore' in line and 'auc' in line and epoch >= 0:
            f1_match = re.search(r"'fscore': ([\d\.]+)", line)
            best_f1_match = re.search(r"'best_fscore': ([\d\.]+)", line)
            precision_match = re.search(r"'precision': ([\d\.]+)", line)
            recall_match = re.search(r"'recall': ([\d\.]+)", line)
            
            if f1_match:
                f1_scores.append(float(f1_match.group(1)))
            if best_f1_match:
                best_f1_scores.append(float(best_f1_match.group(1)))
            if precision_match:
                precision_scores.append(float(precision_match.group(1)))
            if recall_match:
                recall_scores.append(float(recall_match.group(1)))
            
            # 确保epochs对齐
            if len(epochs) < len(f1_scores):
                epochs.append(epoch)
        
        # Top1和Top5准确率
        if 'Top1根因定位准确率' in line and epoch >= 0:
            top1_match = re.search(r'Top1根因定位准确率: ([\d\.]+)', line)
            top5_match = re.search(r'Top5根因定位准确率: ([\d\.]+)', line)
            if top1_match and top5_match:
                top1_acc.append(float(top1_match.group(1)))
                top5_acc.append(float(top5_match.group(1)))
    
    # 对齐数据长度
    min_len = min(len(f1_scores), len(top5_acc), len(epochs))
    return {
        'epochs': epochs[:min_len],
        'f1_scores': f1_scores[:min_len],
        'top5_acc': top5_acc[:min_len],
        'top1_acc': top1_acc[:min_len],
        'best_f1_scores': best_f1_scores[:min_len] if len(best_f1_scores) >= min_len else [],
        'precision': precision_scores[:min_len] if len(precision_scores) >= min_len else [],
        'recall': recall_scores[:min_len] if len(recall_scores) >= min_len else []
    }

def analyze_correlation(data):
    """分析F1和Top5准确率的关系"""
    f1 = np.array(data['f1_scores'])
    top5 = np.array(data['top5_acc'])
    
    # 计算相关系数
    correlation = np.corrcoef(f1, top5)[0, 1]
    
    print("=" * 80)
    print("异常检测F1 vs 根因定位Top5准确率 - 关系分析")
    print("=" * 80)
    
    print(f"\n【统计数据】")
    print(f"F1 Score范围: [{f1.min():.4f}, {f1.max():.4f}], 均值: {f1.mean():.4f}")
    print(f"Top5准确率范围: [{top5.min():.4f}, {top5.max():.4f}], 均值: {top5.mean():.4f}")
    print(f"相关系数: {correlation:.4f}")
    
    if correlation > 0.5:
        print("✅ 正相关：F1高时，Top5也倾向于高")
    elif correlation < -0.5:
        print("⚠️  负相关：F1高时，Top5倾向于低")
    else:
        print("➡️  弱相关：F1和Top5关系不明显")
    
    # 分析趋势
    print(f"\n【趋势分析】")
    
    # 找出F1最高的几个epoch
    top_f1_indices = np.argsort(f1)[-5:][::-1]
    print("F1最高的5个epoch:")
    for idx in top_f1_indices:
        print(f"  Epoch {data['epochs'][idx]}: F1={f1[idx]:.4f}, Top5={top5[idx]:.4f}")
    
    # 找出Top5最高的几个epoch
    top_top5_indices = np.argsort(top5)[-5:][::-1]
    print("\nTop5最高的5个epoch:")
    for idx in top_top5_indices:
        print(f"  Epoch {data['epochs'][idx]}: F1={f1[idx]:.4f}, Top5={top5[idx]:.4f}")
    
    # 检查是否存在冲突
    print(f"\n【冲突分析】")
    
    # 找出F1高但Top5低的epoch
    f1_threshold = np.percentile(f1, 75)  # 75分位数
    top5_threshold = np.percentile(top5, 25)  # 25分位数
    
    conflict_cases = []
    for i in range(len(f1)):
        if f1[i] > f1_threshold and top5[i] < top5_threshold:
            conflict_cases.append((i, f1[i], top5[i]))
    
    if conflict_cases:
        print(f"⚠️  发现冲突：F1高(>{f1_threshold:.4f})但Top5低(<{top5_threshold:.4f})的epoch:")
        for idx, f1_val, top5_val in conflict_cases[:5]:
            print(f"  Epoch {data['epochs'][idx]}: F1={f1_val:.4f}, Top5={top5_val:.4f}")
    else:
        print("✅ 未发现明显的冲突情况")
    
    return correlation

def plot_analysis(data, save_path='f1_vs_rootcause_analysis.png'):
    """绘制分析图表"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = np.array(data['epochs'])
    f1 = np.array(data['f1_scores'])
    top5 = np.array(data['top5_acc'])
    top1 = np.array(data['top1_acc'])
    
    # 1. F1 vs Top5 散点图
    ax1 = axes[0, 0]
    scatter = ax1.scatter(f1, top5, c=epochs, cmap='viridis', alpha=0.6, s=50)
    ax1.set_xlabel('F1 Score')
    ax1.set_ylabel('Top5 Root Cause Accuracy')
    ax1.set_title('F1 Score vs Top5 Root Cause Accuracy')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Epoch')
    
    # 添加趋势线
    z = np.polyfit(f1, top5, 1)
    p = np.poly1d(z)
    ax1.plot(f1, p(f1), "r--", alpha=0.5, label=f'Trend line (slope={z[0]:.2f})')
    ax1.legend()
    
    # 2. 随时间变化的趋势
    ax2 = axes[0, 1]
    ax2_twin = ax2.twinx()
    
    line1 = ax2.plot(epochs, f1, 'b-o', label='F1 Score', markersize=4)
    line2 = ax2_twin.plot(epochs, top5, 'r-s', label='Top5 Accuracy', markersize=4)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score', color='b')
    ax2_twin.set_ylabel('Top5 Root Cause Accuracy', color='r')
    ax2.set_title('F1 and Top5 Over Time')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2_twin.tick_params(axis='y', labelcolor='r')
    ax2.grid(True, alpha=0.3)
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left')
    
    # 3. Top1 vs Top5
    ax3 = axes[1, 0]
    ax3.plot(epochs, top1, 'g-^', label='Top1 Accuracy', markersize=4)
    ax3.plot(epochs, top5, 'orange', marker='s', label='Top5 Accuracy', markersize=4)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Root Cause Localization Accuracy')
    ax3.set_title('Top1 vs Top5 Root Cause Accuracy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. F1与Top5的差值分析
    ax4 = axes[1, 1]
    # 归一化到相同尺度
    f1_norm = (f1 - f1.min()) / (f1.max() - f1.min())
    top5_norm = (top5 - top5.min()) / (top5.max() - top5.min())
    diff = top5_norm - f1_norm
    
    ax4.plot(epochs, diff, 'purple', marker='o', markersize=4)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.fill_between(epochs, 0, diff, where=(diff > 0), alpha=0.3, color='green', label='Top5相对更好')
    ax4.fill_between(epochs, 0, diff, where=(diff < 0), alpha=0.3, color='red', label='F1相对更好')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Normalized Difference (Top5 - F1)')
    ax4.set_title('Relative Performance: Top5 vs F1')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 图表已保存至: {save_path}")
    plt.close()

if __name__ == '__main__':
    log_file = 'output.log'
    print(f"正在分析: {log_file}")
    
    data = parse_log(log_file)
    
    if len(data['epochs']) == 0:
        print("❌ 未找到有效数据")
        exit(1)
    
    correlation = analyze_correlation(data)
    plot_analysis(data)
    
    print("\n" + "=" * 80)
    print("分析完成！")

