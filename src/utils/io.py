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
    
    # 同样使用固定顺序，确保与绘图一致（已删除 Random+RL, Bimodal+Random, SOLO+Ra, SOLO+RL）
    fixed_order = ["Baseline", "RCP", "Bi-level", "GA+RL", "Onion+Ra", "Onion+RL", "ROMEN+Ra", "ROMEN+RL", "BimodalRL", "UNITY+Ra", "UNITY+RL"]
    
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
    headers = ["Method", "Nodes", "Edges", "Top-10 Degrees", "CSA", "Control Entropy (H_C)", "Weighted Control Potential (WCP)", f"Robustness (R - {attack_mode})"]
    
    rows = []
    
    # 按照固定顺序输出（已删除 Random+RL, Bimodal+Random, SOLO+Ra, SOLO+RL）
    fixed_order = ["Baseline", "RCP", "Bi-level", "GA+RL", "Onion+Ra", "Onion+RL", "ROMEN+Ra", "ROMEN+RL", "BimodalRL", "UNITY+Ra", "UNITY+RL"]
    
    # 处理每个方法
    for method in fixed_order:
        if method in metrics_summary:
            data_list = metrics_summary[method]
            if not data_list:
                continue
            
            # 计算初始指标的平均值（从曲线的第一个点）
            avg_csa = np.mean([d['CSA_Curve'][0][0] if d['CSA_Curve'] else 0 for d in data_list])
            avg_hc = np.mean([d['CCE_Curve'][0][0] if d['CCE_Curve'] else 0 for d in data_list])
            avg_wcp = np.mean([d['WCP_Curve'][0][0] if d['WCP_Curve'] else 0 for d in data_list])
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
             avg_csa = np.mean([d['CSA_Curve'][0][0] if d['CSA_Curve'] else 0 for d in data_list])
             avg_hc = np.mean([d['CCE_Curve'][0][0] if d['CCE_Curve'] else 0 for d in data_list])
             avg_wcp = np.mean([d['WCP_Curve'][0][0] if d['WCP_Curve'] else 0 for d in data_list])
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

def save_curve_data(curves_data, save_dir, metric_type, attack_mode):
    """
    保存曲线数据（不同batch_size的x, y值）到CSV文件。
    
    参数:
        curves_data (dict): {method: [(x_list, y_list, r_value), ...]} 每个方法对应多个batch的曲线数据
        save_dir (str): 保存目录
        metric_type (str): 指标类型
        attack_mode (str): 攻击模式
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 固定顺序，只保存有数据的方法
    fixed_order = ["Baseline", "RCP", "Bi-level", "GA+RL", "Onion+Ra", "Onion+RL", "ROMEN+Ra", "ROMEN+RL", "BimodalRL", "UNITY+Ra", "UNITY+RL"]
    
    # 为每个方法保存曲线数据
    for method in fixed_order:
        if method not in curves_data or not curves_data[method]:
            continue
        
        # 保存每个batch的曲线数据
        for batch_idx, (x_list, y_list, r_value) in enumerate(curves_data[method]):
            filename = f"{method}_batch{batch_idx}_{attack_mode}_{metric_type}.csv"
            path = os.path.join(save_dir, filename)
            
            with open(path, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["x_val", "y_val", "r_value"])
                for x, y in zip(x_list, y_list):
                    writer.writerow([f"{x:.6f}", f"{y:.6f}", f"{r_value:.6f}"])
    
    logger.info(f"Curve data saved to: {save_dir}")

def save_collapse_point_data(collapse_points, save_dir, metric_type, attack_mode):
    """
    保存崩溃点数据（每个batch_size下下降到20%时的移除节点比例）到CSV文件。
    
    参数:
        collapse_points (dict): {method: [collapse_point_values]} 每个方法对应多个batch的崩溃点
        save_dir (str): 保存目录
        metric_type (str): 指标类型
        attack_mode (str): 攻击模式
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    path = os.path.join(save_dir, f"collapse_points_{attack_mode}_{metric_type}.csv")
    
    # 固定顺序，只保存有数据的方法
    fixed_order = ["Baseline", "RCP", "Bi-level", "GA+RL", "Onion+Ra", "Onion+RL", "ROMEN+Ra", "ROMEN+RL", "BimodalRL", "UNITY+Ra", "UNITY+RL"]
    
    # 计算最大batch数量
    max_batches = max(len(v) for v in collapse_points.values()) if collapse_points else 0
    headers = ["Method"] + [f"Batch_{i}" for i in range(max_batches)]
    rows = []
    
    for method in fixed_order:
        if method not in collapse_points or not collapse_points[method]:
            continue
        
        row = [method] + [f"{val:.6f}" for val in collapse_points[method]]
        rows.append(row)
    
    if rows:
        with open(path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(rows)
        
        logger.info(f"Collapse point data saved to: {path}")

def save_area_data(area_values, save_dir, metric_type, attack_mode):
    """
    保存面积数据（曲线下面积值）到CSV文件，用于比较不同算法的平均性能。
    
    参数:
        area_values (dict): {method: [area_values]} 每个方法对应多个batch的面积值
        save_dir (str): 保存目录
        metric_type (str): 指标类型
        attack_mode (str): 攻击模式
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    path = os.path.join(save_dir, f"area_values_{attack_mode}_{metric_type}.csv")
    
    # 固定顺序，只保存有数据的方法
    fixed_order = ["Baseline", "RCP", "Bi-level", "GA+RL", "Onion+Ra", "Onion+RL", "ROMEN+Ra", "ROMEN+RL", "BimodalRL", "UNITY+Ra", "UNITY+RL"]
    
    # 计算最大batch数量
    max_batches = max(len(v) for v in area_values.values()) if area_values else 0
    headers = ["Method", "Mean_Area", "Std_Area"] + [f"Batch_{i}" for i in range(max_batches)]
    rows = []
    
    for method in fixed_order:
        if method not in area_values or not area_values[method]:
            continue
        
        values = area_values[method]
        mean_area = np.mean(values)
        std_area = np.std(values)
        
        row = [method, f"{mean_area:.6f}", f"{std_area:.6f}"] + [f"{val:.6f}" for val in values]
        rows.append(row)
    
    if rows:
        with open(path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(rows)
        
        logger.info(f"Area data saved to: {path}")
