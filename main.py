# -*- coding: utf-8 -*-
import argparse
import networkx as nx
import os
import random
import logging
from tqdm import tqdm

from src.topology.generators import load_graph, construct_random, construct_ba
from src.topology.reconstruction import construct_Two, construct_Two_MaxK1, create_bimodal_network_exact
from src.topology.optimization import construct_Three, construct_onion, construct_solo, construct_romen, construct_unity
from src.topology.bimodal_rl import construct_bimodal_rl
from src.controller.manager import ControllerManager
from src.visualization.result_plotter import plot_curve, plot_bar_chart, plot_box_chart, plot_curves_for_batches, plot_collapse_point_charts
from src.utils.io import save_x_values_to_csv, save_statistics_to_csv, save_comprehensive_metrics, save_execution_times, save_curve_data, save_collapse_point_data, save_area_data
from src.utils.logger import setup_logger, get_logger
from src.metrics.metrics import calculate_r_value_interpolated
import time
import numpy as np

logger = get_logger(__name__)

def print_topology_info(name, G):
    """
    打印详细的拓扑信息，包括度分布统计。
    
    Args:
        name (str): 拓扑名称。
        G (nx.Graph): 网络图对象。
    """
    if G is None:
        logger.info(f"{name}: None")
        return
        
    # 精简输出：只显示节点数和边数
    logger.info(f"{name}: Nodes={G.number_of_nodes()}, Edges={G.number_of_edges()}")

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

    # 精简日志：只输出关键信息
    # logger.info(f"\nProcessing Topology: {dataset_name}...")
    
    G = None
    if is_synthetic:
        try:
            # logger.info(f"Generating BA Scale-Free Network (N={scale}, m=3)...")
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

def setup_experiment_dir(dataset_name, attack_mode):
    """
    确定实验 ID 并创建结果目录。
    
    Args:
        dataset_name (str): 数据集名称。
        attack_mode (str): 攻击模式。
        
    Returns:
        tuple: (experiment_id)
    """
    base_dir = os.path.join('results', dataset_name, attack_mode)
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
    
    # 创建实验主目录
    experiment_dir = os.path.join(base_dir, str(experiment_id))
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    
    # 创建日志目录（在实验ID目录下）
    log_dir = os.path.join(experiment_dir, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 在此目录下也设置日志记录到文件
    setup_logger(log_dir=log_dir, log_filename="experiment.log")

    logger.info(f"Experiment ID: {experiment_id}")
    logger.info(f"Saving results to: {experiment_dir}")
    
    return experiment_id

def run_simulation_batch(G, args, experiment_id, dataset_name):
    """
    运行批量仿真。
    
    Args:
        G (nx.Graph): 原始图。
        args (argparse.Namespace): 参数。
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
    
    # 定义所有指标类型
    metric_types = ['lcc', 'csa', 'cce', 'wcp']
    
    # 创建实验主目录：results/{dataset}/{attack_mode}/{experiment_id}/
    experiment_base_dir = os.path.join('results', dataset_name, current_attack_mode, str(experiment_id))
    if not os.path.exists(experiment_base_dir):
        os.makedirs(experiment_base_dir)
    
    # 为每个指标类型在实验目录下创建子目录，每个指标下再分为三个文件夹
    metric_dirs = {}
    metric_subdirs = {}
    for metric_type in metric_types:
        metric_dir = os.path.join(experiment_base_dir, metric_type)
        if not os.path.exists(metric_dir):
            os.makedirs(metric_dir)
        metric_dirs[metric_type] = metric_dir
        
        # 为每个指标创建三个子文件夹
        curves_dir = os.path.join(metric_dir, "curves")
        collapse_point_dir = os.path.join(metric_dir, "collapse_point")
        area_dir = os.path.join(metric_dir, "area")
        
        for subdir in [curves_dir, collapse_point_dir, area_dir]:
            if not os.path.exists(subdir):
                os.makedirs(subdir)
        
        metric_subdirs[metric_type] = {
            'curves': curves_dir,
            'collapse_point': collapse_point_dir,
            'area': area_dir
        }
    
    # 创建extradata目录（用于保存每次迭代的详细曲线）
    extra_data_dir = os.path.join(experiment_base_dir, "extradata")
    if not os.path.exists(extra_data_dir):
        os.makedirs(extra_data_dir)
    
    # 共同目录（用于保存运行时间等共同指标）- 保持在attack_mode级别
    common_dir = os.path.join('results', dataset_name, current_attack_mode)
    if not os.path.exists(common_dir):
        os.makedirs(common_dir)
    
    # 精简日志：只输出关键信息
    # logger.info(f"Nodes: {node_num}, Edges: {edge_num}")
    
    # 为每个指标类型初始化 x_values 和 r_values
    x_values_by_metric = {}
    r_values_by_metric = {}
    # 用于保存曲线数据、崩溃点数据和面积数据
    curves_data_by_metric = {}  
    collapse_points_by_metric = {}
    area_values_by_metric = {}
    
    for metric_type in metric_types:
        x_values_by_metric[metric_type] = {
            "Baseline": [],
            "RCP": [],
            "Bi-level": [],
            "GA+RL": [],
            "Onion+RL": [],
            "ROMEN+RL": [],
            "BimodalRL": [],
            "UNITY+RL": []
        }
        
        r_values_by_metric[metric_type] = {
            "Baseline": [],
            "RCP": [],
            "Bi-level": [],
            "GA+RL": [],
            "Onion+RL": [],
            "ROMEN+RL": [],
            "BimodalRL": [],
            "UNITY+RL": []
        }
        
        # 初始化曲线数据、崩溃点数据和面积数据
        curves_data_by_metric[metric_type] = {}
        collapse_points_by_metric[metric_type] = {}
        area_values_by_metric[metric_type] = {}
    
    plot_flag = True
    
    global_metrics_summary = {}
    
    # 精简日志：只在开始时输出一次
    if batch_size > 0:
        logger.info(f"Starting experiment: {dataset_name}, attack={current_attack_mode}, batch_size={batch_size}")
    
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
        # G3 = construct_Two_MaxK1(G)
        G3 = create_bimodal_network_exact(G)
        execution_times.setdefault("Construct_Two", []).append(time.time() - start_time)
        
        start_time = time.time()
        G4 = construct_Three(G, cover_rate)
        execution_times.setdefault("Construct_Three", []).append(time.time() - start_time)
        
        start_time = time.time()
        # 根据图规模动态调整迭代次数
        node_count = G.number_of_nodes()
        if node_count > 200:
            max_iter_onion = 600  # 大图使用较少的迭代次数（已有早停机制）
        else:
            max_iter_onion = 800
        G5 = construct_onion(G, max_iter=max_iter_onion)
        execution_times.setdefault("Construct_Onion", []).append(time.time() - start_time)
        
        # start_time = time.time()
        # G6 = construct_solo(G)
        # execution_times.setdefault("Construct_SOLO", []).append(time.time() - start_time)
        
        start_time = time.time()
        G7 = construct_romen(G)
        execution_times.setdefault("Construct_ROMEN", []).append(time.time() - start_time)
        
        start_time = time.time()
        G8 = construct_bimodal_rl(G)
        execution_times.setdefault("Construct_BimodalRL", []).append(time.time() - start_time)
        
        # 检查G8是否等于G（如果相同，BimodalRL和Baseline会使用相同的图）
        if ijk == 0:
            if G8.number_of_nodes() == G.number_of_nodes() and G8.number_of_edges() == G.number_of_edges():
                edges_G = set(G.edges())
                edges_G8 = set(G8.edges())
                if edges_G == edges_G8:
                    logger.warning("=" * 60)
                    logger.warning("CRITICAL WARNING: G8 (BimodalRL) is IDENTICAL to G (Baseline)!")
                    logger.warning("This means BimodalRL optimization failed or returned the original graph.")
                    logger.warning("Baseline and BimodalRL will use the SAME graph structure.")
                    logger.warning("=" * 60)
                else:
                    logger.info(f"G8 has same size as G but different edges. Common edges: {len(edges_G & edges_G8)}/{len(edges_G)}")
            else:
                logger.info(f"G8 differs from G: G({G.number_of_nodes()}, {G.number_of_edges()}) vs G8({G8.number_of_nodes()}, {G8.number_of_edges()})")

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
            # "G6": G6, # SOLO 优化
            "G7": G7, # ROMEN 优化
            "G8": G8, # BimodalRL 优化
            "G9": G9  # UNITY 优化
        }
        
        # 精简日志：第一次迭代时只打印关键拓扑信息
        if ijk == 0:
            # 只打印原始拓扑和核心方法（Bi-level）的拓扑信息
            print_topology_info("Original (Baseline/RCP)", G)
            print_topology_info("Bimodal (G3 - Bi-level)", G3)
        
        manager = ControllerManager(all_G, cover_rate, node_num, dataset_name, experiment_id)
        
        # 记录仿真时间（包含攻击和评估）
        sim_start_time = time.time()
        
        # 运行攻击仿真，为所有指标类型计算数据
        manager._run_specific_attack_multi_metric(
            current_attack_mode, 
            x_values_by_metric, 
            r_values_by_metric, 
            ijk, 
            plot_flag, 
            metric_types,
            metric_dirs
        )
        
        # 收集当前迭代的曲线数据、崩溃点数据和面积数据
        if hasattr(manager, 'metrics_summary'):
            for name, metrics_list in manager.metrics_summary.items():
                if not metrics_list:
                    continue
                
                current_metric = metrics_list[-1]  # 当前迭代的数据
                
                for metric_type in metric_types:
                    curve_key = f'{metric_type.upper()}_Curve'
                    curve_data = current_metric.get(curve_key, [])
                    
                    if curve_data:
                        x_curve = [point[1] for point in curve_data]
                        y_curve = [point[0] for point in curve_data]
                        
                        # 计算面积（R值）- 使用统一x网格插值方法
                        # 解决多层网络拆解中不同方法x采样点不同的问题
                        r_val = calculate_r_value_interpolated(x_curve, y_curve, num_points=101)
                        
                        # 保存曲线数据
                        if name not in curves_data_by_metric[metric_type]:
                            curves_data_by_metric[metric_type][name] = []
                        curves_data_by_metric[metric_type][name].append((x_curve, y_curve, r_val))
                        
                        # 保存面积数据
                        if name not in area_values_by_metric[metric_type]:
                            area_values_by_metric[metric_type][name] = []
                        area_values_by_metric[metric_type][name].append(r_val)
                        
                        # 计算崩溃点（下降到20%时的移除节点比例）
                        if len(y_curve) > 0:
                            initial_value = y_curve[0] if y_curve[0] > 0 else 1.0
                            threshold = initial_value * 0.2
                            
                            collapse_point = None
                            for i, y_val in enumerate(y_curve):
                                if y_val <= threshold:
                                    collapse_point = x_curve[i]
                                    break
                            
                            if collapse_point is None and len(x_curve) > 0:
                                collapse_point = x_curve[-1]
                            
                            # 保存崩溃点数据
                            if name not in collapse_points_by_metric[metric_type]:
                                collapse_points_by_metric[metric_type][name] = []
                            collapse_points_by_metric[metric_type][name].append(collapse_point if collapse_point is not None else 1.0)
        
        execution_times.setdefault("Simulation_Run", []).append(time.time() - sim_start_time)
        execution_times.setdefault("Total_Iteration", []).append(time.time() - iter_start_time)

        # 额外保存每次迭代的曲线数据
        if hasattr(manager, 'metrics_summary'):
            # 保存当前迭代的曲线数据
            import pandas as pd
            for name, metrics_list in manager.metrics_summary.items():
                # metrics_list 中应该只有一项（当前迭代的）
                current_metric = metrics_list[-1]
                
                # 保存所有指标的曲线数据
                for metric_type in metric_types:
                    curve_key = f'{metric_type.upper()}_Curve'
                    curve_data = current_metric.get(curve_key, [])
                    
                    if curve_data:
                        x_curve = [point[1] for point in curve_data]
                        y_curve = [point[0] for point in curve_data]
                        
                        df = pd.DataFrame({'x_val': x_curve, 'y_val': y_curve})
                        # 文件名格式: method_iter_attack_metric.csv
                        filename = f"{name}_iter{ijk}_{current_attack_mode}_{metric_type}.csv"
                        filepath = os.path.join(extra_data_dir, filename)
                        df.to_csv(filepath, index=False)
        
        # 收集本轮的综合指标
        if hasattr(manager, 'metrics_summary'):
            for name, metrics_list in manager.metrics_summary.items():
                if name not in global_metrics_summary:
                    global_metrics_summary[name] = []
                global_metrics_summary[name].extend(metrics_list)

    # 保存综合指标到实验主目录
    save_comprehensive_metrics(global_metrics_summary, experiment_base_dir, current_attack_mode)
    
    # 保存运行时间记录到共同目录
    if execution_times:
        save_execution_times(execution_times, common_dir)
    
    # 为每个指标类型保存结果到新的目录结构
    for metric_type in metric_types:
        subdirs = metric_subdirs[metric_type]
        
        # 1. 保存曲线数据到curves文件夹
        if curves_data_by_metric[metric_type]:
            save_curve_data(
                curves_data_by_metric[metric_type],
                subdirs['curves'],
                metric_type,
                current_attack_mode
            )
            
            # 绘制前5个batch的折线图（如果batch_size<5就全画）
            plot_curves_for_batches(
                curves_data_by_metric[metric_type],
                subdirs['curves'],
                metric_type,
                current_attack_mode,
                batch_size,
                dataset_name
            )
        
        # 2. 保存崩溃点数据到collapse_point文件夹
        if collapse_points_by_metric[metric_type]:
            save_collapse_point_data(
                collapse_points_by_metric[metric_type],
                subdirs['collapse_point'],
                metric_type,
                current_attack_mode
            )
            
            # 绘制崩溃点的bar图和箱线图
            plot_collapse_point_charts(
                collapse_points_by_metric[metric_type],
                subdirs['collapse_point'],
                metric_type,
                current_attack_mode
            )
        
        # 3. 保存面积数据到area文件夹
        if area_values_by_metric[metric_type]:
            save_area_data(
                area_values_by_metric[metric_type],
                subdirs['area'],
                metric_type,
                current_attack_mode
            )
        
    return x_values_by_metric['lcc'], r_values_by_metric['lcc']

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
    experiment_id = setup_experiment_dir(dataset_name, args.attack)
    
    # 4. 运行仿真
    x_values, r_values = run_simulation_batch(G, args, experiment_id, dataset_name)
    
    logger.info("Task completed.")

if __name__ == "__main__":
    main()
