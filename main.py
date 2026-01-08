# -*- coding: utf-8 -*-
import argparse
import networkx as nx
import os
import random
import logging
from tqdm import tqdm

from src.topology.generators import load_graph, construct_random, construct_ba
from src.topology.reconstruction import construct_Two, construct_Two_MaxK1
from src.topology.optimization import construct_Three, construct_onion, construct_solo, construct_romen, construct_unity
from src.topology.bimodal_rl import construct_bimodal_rl
from src.controller.manager import ControllerManager
from src.visualization.result_plotter import plot_curve, plot_bar_chart, plot_box_chart
from src.utils.io import save_x_values_to_csv, save_statistics_to_csv, save_comprehensive_metrics, save_execution_times
from src.utils.logger import setup_logger, get_logger
import time

logger = get_logger(__name__)

def print_topology_info(name, G):
    """
    打印详细的拓扑信息，包括度分布统计。
    
    Args:
        name (str): 拓扑名称。
        G (nx.Graph): 网络图对象。
    """
    if G is None:
        logger.info(f"\n--- {name} : None ---")
        return
        
    degrees = sorted([d for n, d in G.degree()], reverse=True)
    
    info_str = f"\n--- {name} Topology Info ---\n"
    info_str += f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}\n"
    
    if len(degrees) > 0:
        avg_degree = sum(degrees) / len(degrees)
        info_str += f"Degree Stats: Max={max(degrees)}, Min={min(degrees)}, Avg={avg_degree:.4f}\n"
        
        # 打印度序列（如果太长则缩略显示）
        if len(degrees) <= 50:
            info_str += f"Degree Sequence: {degrees}"
        else:
            info_str += f"Degree Sequence (Top 20): {degrees[:20]} ...\n"
            info_str += f"Degree Sequence (Bottom 20): {degrees[-20:]}"
            
    logger.info(info_str)

def parse_arguments():
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(description='SDN Topology Resilience Simulation')
    parser.add_argument('--dataset', type=str, default='GtsCe', help='Dataset name (e.g., GtsCe, Chinanet)')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic BA network')
    parser.add_argument('--nodes', type=int, default=100, help='Number of nodes for synthetic network')
    parser.add_argument('--rate', type=float, default=0.1, help='Controller placement rate')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of simulation runs')
    parser.add_argument('--attack', type=str, default='random', 
                        choices=['hybrid', 'random', 'target', 'degree', 'pagerank', 'betweenness', 'eigenvector', 'bruteforce'], 
                        help='Attack mode')
    parser.add_argument('--metric', type=str, default='gcc',
                        choices=['gcc', 'efficiency', 'coverage'],
                        help='Metric to evaluate resilience')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    return parser.parse_args()

def load_or_generate_graph(args):
    """
    从 GML 文件加载图或根据参数生成合成图。
    
    Args:
        args (argparse.Namespace): 解析后的参数。
        
    Returns:
        tuple: (G, dataset_name)
    """
    dataset_name = args.dataset
    is_synthetic = args.synthetic
    scale = args.nodes

    logger.info(f"\nProcessing Topology: {dataset_name}...")
    
    G = None
    if is_synthetic:
        try:
            logger.info(f"Generating BA Scale-Free Network (N={scale}, m=3)...")
            G = nx.barabasi_albert_graph(scale, 3)
            # 添加边的权重属性
            for u, v in G.edges():
                G[u][v]['weight'] = 1.0
            dataset_name = f"BA_{scale}"
        except Exception as e:
            logger.error(f"Failed to generate synthetic network: {e}")
            return None, None
    else:
        gml_path = os.path.join('data', 'testdata', f'{dataset_name}.gml')
        if not os.path.exists(gml_path):
             # 如果 testdata 中未找到，尝试在 raw 中查找
             gml_path = os.path.join('data', 'raw', f'{dataset_name}.gml')
             
        G, pos = load_graph(gml_path)
    
    return G, dataset_name

def setup_experiment_dir(dataset_name, attack_mode, metric_type):
    """
    确定实验 ID 并创建结果目录。
    
    Args:
        dataset_name (str): 数据集名称。
        attack_mode (str): 攻击模式。
        metric_type (str): 指标类型。
        
    Returns:
        tuple: (experiment_dir, experiment_id)
    """
    base_dir = os.path.join('results', dataset_name, attack_mode, metric_type)
    experiment_id = 1
    if os.path.exists(base_dir):
        subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        numeric_subdirs = []
        for d in subdirs:
            try:
                numeric_subdirs.append(int(d))
            except ValueError:
                pass
        if numeric_subdirs:
            experiment_id = max(numeric_subdirs) + 1
    
    experiment_dir = os.path.join(base_dir, str(experiment_id))
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    # 在此目录下也设置日志记录到文件
    setup_logger(log_dir=experiment_dir, log_filename="experiment.log")

    logger.info(f"Experiment ID: {experiment_id}")
    logger.info(f"Saving results to: {experiment_dir}")
    
    return experiment_dir, experiment_id

def run_simulation_batch(G, args, experiment_dir, experiment_id, dataset_name):
    """
    运行批量仿真。
    
    Args:
        G (nx.Graph): 原始图。
        args (argparse.Namespace): 参数。
        experiment_dir (str): 结果保存目录。
        experiment_id (int): 实验 ID。
        dataset_name (str): 数据集名称。
        
    Returns:
        tuple: (x_values, r_values)
    """
    node_num = G.number_of_nodes()
    edge_num = G.number_of_edges()
    cover_rate = args.rate
    batch_size = args.batch_size
    current_attack_mode = args.attack
    metric_type = args.metric
    
    logger.info(f"Nodes: {node_num}, Edges: {edge_num}")
    
    x_values = {
        "Baseline": [],
        "RCP": [],
        "Bi-level": [],
        "Bimodal+Random": [],
        "Random+RL": [],
        "GA+RL": [],
        "Onion+Ra": [],
        "Onion+RL": [],
        "SOLO+Ra": [],
        "SOLO+RL": [],
        "ROMEN+Ra": [],
        "ROMEN+RL": [],
        "BimodalRL": [],
        "UNITY+Ra": [],
        "UNITY+RL": []
    }
    
    r_values = {
        "Baseline": [],
        "RCP": [],
        "Bi-level": [],
        "Bimodal+Random": [],
        "Random+RL": [],
        "GA+RL": [],
        "Onion+Ra": [],
        "Onion+RL": [],
        "SOLO+Ra": [],
        "SOLO+RL": [],
        "ROMEN+Ra": [],
        "ROMEN+RL": [],
        "BimodalRL": [],
        "UNITY+Ra": [],
        "UNITY+RL": []
    }
    
    plot_flag = True
    
    global_metrics_summary = {}
    
    logger.info(f"Starting {batch_size} simulations...")
    
    # 记录运行时间
    execution_times = {}

    for ijk in tqdm(range(batch_size), desc=f"Simulating {dataset_name}"):
        
        iter_start_time = time.time()
        
        # 生成对比拓扑
        start_time = time.time()
        G1 = construct_random(node_num, edge_num)
        execution_times.setdefault("Construct_Random", []).append(time.time() - start_time)
        
        start_time = time.time()
        G2 = construct_ba(node_num, edge_num)
        execution_times.setdefault("Construct_BA", []).append(time.time() - start_time)
        
        start_time = time.time()
        # G3 = construct_Two(G, cover_rate)
        G3 = construct_Two_MaxK1(G)
        execution_times.setdefault("Construct_Two", []).append(time.time() - start_time)
        
        start_time = time.time()
        G4 = construct_Three(G, cover_rate)
        execution_times.setdefault("Construct_Three", []).append(time.time() - start_time)
        
        start_time = time.time()
        G5 = construct_onion(G, max_iter=2000)
        execution_times.setdefault("Construct_Onion", []).append(time.time() - start_time)
        
        start_time = time.time()
        G6 = construct_solo(G)
        execution_times.setdefault("Construct_SOLO", []).append(time.time() - start_time)
        
        start_time = time.time()
        G7 = construct_romen(G)
        execution_times.setdefault("Construct_ROMEN", []).append(time.time() - start_time)
        
        start_time = time.time()
        G8 = construct_bimodal_rl(G)
        execution_times.setdefault("Construct_BimodalRL", []).append(time.time() - start_time)

        start_time = time.time()
        G9 = construct_unity(G)
        execution_times.setdefault("Construct_UNITY", []).append(time.time() - start_time)
        
        # 将拓扑组织成字典
        all_G = {
            "G": G,   # 原始
            "G1": G1, # 随机
            "G2": G2, # 无标度
            "G3": G3, # 双模态
            "G4": G4, # GA 优化
            "G5": G5, # ONION 优化
            "G6": G6, # SOLO 优化
            "G7": G7, # ROMEN 优化
            "G8": G8, # BimodalRL 优化
            "G9": G9  # UNITY 优化
        }
        
        # 第一次迭代时打印拓扑信息
        if ijk == 0:
            print_topology_info("Original (Baseline/RCP)", G)
            print_topology_info("Random (G1 - 随机部署)", G1)
            print_topology_info("Scale-Free (G2)", G2)
            print_topology_info("Bimodal (G3 - 双层构造/RL-RCP)", G3)
            print_topology_info("GA Optimized (G4 - 混合指标)", G4)
            print_topology_info("ONION Optimized (G5 - ONION)", G5)
            print_topology_info("SOLO Optimized (G6 - SOLO)", G6)
            print_topology_info("ROMEN Optimized (G7 - ROMEN)", G7)
            print_topology_info("BimodalRL Optimized (G8 - BimodalRL)", G8)
            print_topology_info("UNITY Optimized (G9 - UNITY)", G9)
        
        manager = ControllerManager(all_G, cover_rate, node_num, dataset_name, experiment_id)
        
        # 记录仿真时间（包含攻击和评估）
        sim_start_time = time.time()
        x_values = manager._run_specific_attack(current_attack_mode, x_values, r_values, ijk, plot_flag, metric_type, save_dir=experiment_dir)
        execution_times.setdefault("Simulation_Run", []).append(time.time() - sim_start_time)
        
        execution_times.setdefault("Total_Iteration", []).append(time.time() - iter_start_time)

        # 额外保存每次迭代的 nn:x_val 数据
        
        # 额外保存每次迭代的 nn:x_val 数据
        if hasattr(manager, 'metrics_summary'):
            # 创建 extradata 目录
            extra_data_dir = os.path.join(experiment_dir, "extradata")
            if not os.path.exists(extra_data_dir):
                os.makedirs(extra_data_dir)
            
            # 保存当前迭代的曲线数据
            import pandas as pd
            for name, metrics_list in manager.metrics_summary.items():
                # metrics_list 中应该只有一项（当前迭代的）
                current_metric = metrics_list[-1]
                x_curve = current_metric.get('X_Curve', [])
                y_curve = current_metric.get('Y_Curve', [])
                
                if x_curve and y_curve:
                    df = pd.DataFrame({'x_val': x_curve, 'nn': y_curve})
                    # 文件名格式: method_iter_attack.csv
                    # 由于这里是在 batch 循环中，我们需要 ijk
                    filename = f"{name}_iter{ijk}_{current_attack_mode}.csv"
                    filepath = os.path.join(extra_data_dir, filename)
                    df.to_csv(filepath, index=False) 
        
        # 收集本轮的综合指标
        if hasattr(manager, 'metrics_summary'):
            for name, metrics_list in manager.metrics_summary.items():
                if name not in global_metrics_summary:
                    global_metrics_summary[name] = []
                global_metrics_summary[name].extend(metrics_list)

    save_comprehensive_metrics(global_metrics_summary, experiment_dir, current_attack_mode)
    
    # 保存运行时间记录
    if execution_times:
        save_execution_times(execution_times, experiment_dir)
        
    return x_values, r_values

def save_and_plot_results(x_values, r_values, dataset_name, attack_mode, experiment_id, experiment_dir):
    """
    保存结果到 CSV 并生成绘图。
    
    Args:
        x_values (dict): 崩溃点数据。
        r_values (dict): 鲁棒性指标数据。
        dataset_name (str): 数据集名称。
        attack_mode (str): 攻击模式。
        experiment_id (int): 实验 ID。
        experiment_dir (str): 文件保存目录。
    """
    # 保存 x_values 和 r_values
    save_x_values_to_csv(x_values, f"{dataset_name}_{attack_mode}", save_dir=experiment_dir)
    save_x_values_to_csv(r_values, f"{dataset_name}_{attack_mode}", filename="r_values_results.csv", save_dir=experiment_dir)

    # 保存统计数据
    save_statistics_to_csv(x_values, experiment_dir)
    save_statistics_to_csv(r_values, experiment_dir, filename="r_statistics.csv")

    try:
        # 1. 绘制崩溃点 (X values)
        # plot_curve 内部已经调用了 plot_bar_chart 和 plot_box_chart 用于 x_values，且文件名固定
        # 但为了更清晰，我们可以显式调用，或者保留 plot_curve 作为向后兼容
        plot_bar_chart(x_values, "Proportion of Removed Nodes at Collapse (f)", experiment_dir, "collapse_point_bar.png")
        plot_box_chart(x_values, "Proportion of Removed Nodes at Collapse (f)", experiment_dir, "collapse_point_box.png")
        
        # 2. 绘制鲁棒性 (R values)
        plot_bar_chart(r_values, "Robustness (R)", experiment_dir, "robustness_bar.png")
        plot_box_chart(r_values, "Robustness (R)", experiment_dir, "robustness_box.png")
        
    except Exception as e:
        logger.error(f"Plotting failed: {e}")

def main():
    # 1. 解析参数
    args = parse_arguments()
    
    # 设置初始日志记录器（输出到控制台）
    # 强制控制台输出为 INFO 级别，即使用户设置了 debug
    log_level = logging.INFO 
    # 如果用户想看 debug，可以在 setup_logger 内部针对文件 handler 进行特殊处理，
    # 但这里我们只控制 "初始" 阶段的控制台输出。
    # 实际上 setup_logger 的 level 参数同时控制控制台和文件。
    # 用户要求：控制台只输出 INFO，关闭 DEBUG。
    
    # 我们可以传递一个 console_level 参数给 setup_logger
    setup_logger(log_dir="logs", log_filename="latest_run.log", level=logging.DEBUG if args.debug else logging.INFO, console_level=logging.INFO)
    
    # 2. 加载或生成图
    G, dataset_name = load_or_generate_graph(args)
    
    if G is None or G.number_of_nodes() == 0:
        logger.warning(f"Skipping {dataset_name}: Load/Generate failed or empty graph.")
        return
        
    # 3. 设置实验目录（并切换日志记录到该目录下的文件）
    experiment_dir, experiment_id = setup_experiment_dir(dataset_name, args.attack, args.metric)
    
    # 4. 运行仿真
    x_values, r_values = run_simulation_batch(G, args, experiment_dir, experiment_id, dataset_name)
    
    # 5. 保存并绘制结果
    save_and_plot_results(x_values, r_values, dataset_name, args.attack, experiment_id, experiment_dir)
    
    logger.info("Task completed.")

if __name__ == "__main__":
    main()
