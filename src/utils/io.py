import csv
import os
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)

def save_x_values_to_csv(x_values, name, filename="x_values_results.csv", save_dir="results"):
    """
    将 x_values 数据保存到 CSV 文件。

    参数:
        x_values (dict): 包含每种网络类型对应的 y=0 时 x 值列表。
        name (str): 文件名前缀。
        filename (str): 保存的 CSV 文件名，默认是 "x_values_results.csv"。
        save_dir (str): 保存目录，默认为 "results"。
    """
    # 按照固定的顺序保存，确保一致性
    fixed_order = ["Baseline", "RCP", "Bi-level", "Bimodal+Random", "Random+RL", "GA+RL", "Onion+Ra", "Onion+RL", "SOLO+Ra", "SOLO+RL", "ROMEN+Ra", "ROMEN+RL"]
    
    # 只保存存在于 x_values 中的列，但保持固定顺序
    headers = [h for h in fixed_order if h in x_values]
    # 添加剩余的列（如果有其他方法）
    for k in x_values.keys():
        if k not in headers:
            headers.append(k)
    
    # 获取数据的最大长度（因为可能某些列表长度不同）
    max_length = 0
    if x_values:
        max_length = max(len(values) for values in x_values.values())
    
    # 创建一个二维列表，按列对齐数据，不足的部分填充为空值
    rows = []
    for i in range(max_length):
        row = [x_values[header][i] if i < len(x_values[header]) else "" for header in headers]
        rows.append(row)
    
    # 确保 save_dir 目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    path = os.path.join(save_dir, name + "_" + filename)
    # 写入 CSV 文件
    with open(path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        # 写入表头
        writer.writerow(headers)
        # 写入数据
        writer.writerows(rows)
    
    logger.info(f"x_values data saved to file: {path}")


def save_statistics_to_csv(x_values, save_dir, filename="statistics.csv"):
    """
    计算并保存平均值和标准差到 CSV。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    path = os.path.join(save_dir, filename)
    abs_path = os.path.abspath(path)
    
    headers = ["Method", "Mean", "Std"]
    rows = []
    
    # 同样使用固定顺序，确保与绘图一致
    fixed_order = ["Baseline", "RCP", "Bi-level", "Bimodal+Random", "Random+RL", "GA+RL", "Onion+Ra", "Onion+RL", "SOLO+Ra", "SOLO+RL", "ROMEN+Ra", "ROMEN+RL"]
    
    # 处理固定顺序的方法
    for method in fixed_order:
        if method in x_values:
            values = x_values[method]
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                rows.append([method, f"{mean_val:.4f}", f"{std_val:.4f}"])
            else:
                rows.append([method, "N/A", "N/A"])
                
    # 处理其他可能的方法
    for method, values in x_values.items():
        if method not in fixed_order:
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                rows.append([method, f"{mean_val:.4f}", f"{std_val:.4f}"])
            else:
                rows.append([method, "N/A", "N/A"])
            
    with open(path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)
        
    logger.info(f"Statistics saved to file: {abs_path}")

def save_comprehensive_metrics(metrics_summary, save_dir, attack_mode):
    """
    保存综合指标 (CSA, H_C, WCP, R) 到 CSV 文件。
    计算所有迭代的平均值。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    filename = "comprehensive_metrics.csv"
    path = os.path.join(save_dir, filename)
    
    # 定义表头
    # 如果能获取到 Random 和 Degree 的 R，我们可以计算加权 R
    # 目前我们只有当前 attack_mode 的 R
    # 我们将列命名为 "R (Current Attack)"
    # 如果用户严格要求 "0.5*random + 0.5*degree"，我们需要额外数据。
    # 这里我们暂且记录当前的 R。
    
    headers = ["Method", "Nodes", "Edges", "Top-10 Degrees", "CSA", "Control Entropy (H_C)", "Weighted Control Potential (WCP)", f"Robustness (R - {attack_mode})"]
    
    rows = []
    
    # 按照固定顺序输出
    fixed_order = ["Baseline", "RCP", "Bi-level", "Bimodal+Random", "Random+RL", "GA+RL", "Onion+Ra", "Onion+RL", "SOLO+Ra", "SOLO+RL", "ROMEN+Ra", "ROMEN+RL", "BimodalRL", "UNITY"]
    
    # 处理每个方法
    for method in fixed_order:
        if method in metrics_summary:
            data_list = metrics_summary[method]
            if not data_list:
                continue
                
            # 计算平均值
            avg_csa = np.mean([d['CSA'] for d in data_list])
            avg_hc = np.mean([d['H_C'] for d in data_list])
            avg_wcp = np.mean([d['WCP'] for d in data_list])
            avg_r = np.mean([d['R'] for d in data_list])
            
            # 获取拓扑属性（假设每次迭代拓扑结构相似，取最后一次的值）
            last_entry = data_list[-1]
            nodes = last_entry.get('Nodes', 0)
            edges = last_entry.get('Edges', 0)
            top10 = last_entry.get('Top10_Degrees', "[]")
            
            rows.append([
                method,
                nodes,
                edges,
                top10,
                f"{avg_csa:.4f}",
                f"{avg_hc:.4f}",
                f"{avg_wcp:.4f}",
                f"{avg_r:.4f}"
            ])
            
    # 处理其他可能的方法
    for method, data_list in metrics_summary.items():
        if method not in fixed_order:
             if not data_list: continue
             avg_csa = np.mean([d['CSA'] for d in data_list])
             avg_hc = np.mean([d['H_C'] for d in data_list])
             avg_wcp = np.mean([d['WCP'] for d in data_list])
             avg_r = np.mean([d['R'] for d in data_list])
             
             last_entry = data_list[-1]
             nodes = last_entry.get('Nodes', 0)
             edges = last_entry.get('Edges', 0)
             top10 = last_entry.get('Top10_Degrees', "[]")
             
             rows.append([method, nodes, edges, top10, f"{avg_csa:.4f}", f"{avg_hc:.4f}", f"{avg_wcp:.4f}", f"{avg_r:.4f}"])
            
    with open(path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)
        
    logger.info(f"Comprehensive metrics saved to: {path}")

def save_execution_times(execution_times, save_dir):
    """
    保存算法运行时间统计到 CSV 文件。
    
    参数:
        execution_times (dict): key是阶段名称, value是时间列表(每次迭代的时间)。
        save_dir (str): 保存目录。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    path = os.path.join(save_dir, "execution_times.csv")
    
    headers = ["Stage", "Mean Time (s)", "Total Time (s)", "Min Time (s)", "Max Time (s)", "Iterations"]
    rows = []
    
    for stage, times in execution_times.items():
        if not times:
            continue
            
        mean_time = np.mean(times)
        total_time = np.sum(times)
        min_time = np.min(times)
        max_time = np.max(times)
        count = len(times)
        
        rows.append([
            stage,
            f"{mean_time:.4f}",
            f"{total_time:.4f}",
            f"{min_time:.4f}",
            f"{max_time:.4f}",
            count
        ])
        
    with open(path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)
        
    logger.info(f"Execution times saved to: {path}")
