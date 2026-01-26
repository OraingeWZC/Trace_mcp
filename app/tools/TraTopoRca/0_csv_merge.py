import pandas as pd
import sys

def merge_csv_files(file1_path, file2_path, output_path="merged_output.csv"):
    """
    合并两个列结构相同的CSV文件（纵向拼接）
    
    参数:
        file1_path: 第一个CSV文件的路径
        file2_path: 第二个CSV文件的路径
        output_path: 合并后输出文件的路径，默认是merged_output.csv
    """
    try:
        # 读取第一个CSV文件
        df1 = pd.read_csv(file1_path)
        # 读取第二个CSV文件
        df2 = pd.read_csv(file2_path)
        
        # 检查两个CSV的列是否完全一致（顺序也检查）
        if list(df1.columns) != list(df2.columns):
            raise ValueError(
                f"两个CSV文件的列结构不一致！\n"
                f"文件1的列：{list(df1.columns)}\n"
                f"文件2的列：{list(df2.columns)}"
            )
        
        # 合并两个DataFrame（纵向拼接）
        merged_df = pd.concat([df1, df2], ignore_index=True)
        
        # 保存合并后的文件
        merged_df.to_csv(output_path, index=False)
        
        print(f"✅ 合并成功！")
        print(f"📄 源文件1行数：{len(df1)}")
        print(f"📄 源文件2行数：{len(df2)}")
        print(f"📄 合并后文件行数：{len(merged_df)}")
        print(f"💾 输出文件路径：{output_path}")
        
    except FileNotFoundError as e:
        print(f"❌ 错误：找不到文件 - {e.filename}")
    except ValueError as e:
        print(f"❌ 错误：{e}")
    except Exception as e:
        print(f"❌ 未知错误：{e}")

if __name__ == "__main__":
    # 方式1：直接在脚本里指定文件路径（适合新手）
    file1 = "/root/wzc/Trace_mcp/app/dataset/tianchi/data/NormalData/normal_metrics_9e4_1618.csv"
    file2 = "/root/wzc/Trace_mcp/app/dataset/tianchi/data/NormalData/normal_metrics_1e5_2022.csv"
    output = "/root/wzc/Trace_mcp/app/dataset/tianchi/data/NormalData/normal_metrics_2e5_1622.csv"
    merge_csv_files(file1, file2, output)

    # file1 = "dataset/tianchi/normal_metrics_1e5_30s.csv"
    # file2 = "dataset/tianchi/all_metrics_30s.csv"
    # output = "dataset/tianchi/all_metrics_30s.csv"
    # merge_csv_files(file1, file2, output)
    
    # 方式2：通过命令行参数传入（更灵活）
    # if len(sys.argv) != 4:
    #     print("📖 使用方法：")
    #     print("   方式1（脚本内指定）：修改脚本里的file1/file2/output后运行")
    #     print("   方式2（命令行）：python 脚本名.py 第一个文件.csv 第二个文件.csv 输出文件.csv")
    #     sys.exit(1)
    
    # # 从命令行获取参数
    # file1_path = sys.argv[1]
    # file2_path = sys.argv[2]
    # output_path = sys.argv[3]
    
    # # 执行合并
    # merge_csv_files(file1_path, file2_path, output_path)