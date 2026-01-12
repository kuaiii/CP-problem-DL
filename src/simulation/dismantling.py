# -*- coding: utf-8 -*-
import networkx as nx
import random
import logging
import numpy as np
from itertools import combinations
from tqdm import tqdm
from src.metrics.metrics import coverage_compute2, weight_efficiency_compute, max_component_size_compute, calculate_lcc, calculate_csa, calculate_control_entropy, calculate_wcp
from src.utils.logger import get_logger

# 尝试导入 GPU 工具
try:
    from src.utils.gpu_utils import (
        is_gpu_available, to_tensor, to_numpy, TORCH_AVAILABLE, CUDA_AVAILABLE, DEVICE
    )
    if TORCH_AVAILABLE:
        import torch
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    
    def is_gpu_available():
        return False

logger = get_logger(__name__)

# -------------------------------------------------------
# New Dismantling Strategies (Static)
# -------------------------------------------------------

def bruteforce_dismantler(G, stop_condition=None, max_k=None):
    """
    使用暴力枚举方法找到最优网络拆解序列。
    该方法计算复杂度极高，仅适用于非常小的网络。
    
    Args:
        G (nx.Graph): 网络图。
        stop_condition (float, optional): 停止条件（如GCC大小），默认为网络大小的10%。
        max_k (int, optional): 最大移除节点数，默认为网络大小。
    
    Returns:
        list: 拆解节点序列。
    """
    network_size = G.number_of_nodes()
    if stop_condition is None:
        stop_condition = max(1, int(0.1 * network_size))
    if max_k is None:
        max_k = network_size
    
    max_k = min(max_k, 10)
    nodes = list(G.nodes())
    best_combination = None
    
    for k in range(1, max_k + 1):
        best_score = network_size
        for combination in combinations(range(network_size), k):
            temp_G = G.copy()
            temp_G.remove_nodes_from([nodes[i] for i in combination])
            
            if temp_G.number_of_nodes() > 0:
                components = list(nx.connected_components(temp_G))
                largest_cc_size = len(max(components, key=len)) if components else 0
            else:
                largest_cc_size = 0
            
            if largest_cc_size < best_score:
                best_score = largest_cc_size
                best_combination = [nodes[i] for i in combination]
        
        if best_score <= stop_condition:
            break
    
    if best_combination is None:
        return []
    
    dismantling_sequence = list(best_combination)
    remaining_nodes = [node for node in nodes if node not in dismantling_sequence]
    remaining_nodes.sort(key=lambda x: G.degree(x), reverse=True)
    dismantling_sequence.extend(remaining_nodes)
    return dismantling_sequence

def degree_dismantling(G):
    """
    基于度中心性（静态）的拆解策略。
    一次性计算所有节点的度，并按降序排列。
    
    支持 GPU 加速计算
    
    Args:
        G (nx.Graph): 网络图。
        
    Returns:
        list: 拆解节点序列。
    """
    nodes = list(G.nodes())
    
    if is_gpu_available() and len(nodes) > 100:
        # GPU 加速版本：使用 GPU 进行排序
        degrees = np.array([G.degree(n) for n in nodes], dtype=np.float32)
        degree_tensor = torch.tensor(degrees, device=DEVICE)
        
        # 获取排序索引（降序）
        _, sorted_indices = torch.sort(degree_tensor, descending=True)
        sorted_indices = sorted_indices.cpu().numpy()
        
        return [nodes[i] for i in sorted_indices]
    else:
        # CPU 版本
        sorted_nodes = sorted(nodes, key=G.degree, reverse=True)
        return list(sorted_nodes)

def random_dismantling(G):
    """
    随机拆解策略。
    随机打乱节点顺序。
    
    Args:
        G (nx.Graph): 网络图。
        
    Returns:
        list: 拆解节点序列。
    """
    random_nodes = list(G.nodes())
    random.shuffle(random_nodes)
    return random_nodes

def pagerank_dismantling(G):
    """
    基于 PageRank 的拆解策略。
    
    Args:
        G (nx.Graph): 网络图。
        
    Returns:
        list: 拆解节点序列。
    """
    pagerank = nx.pagerank(G)
    sorted_nodes = sorted(G.nodes(), key=lambda x: pagerank[x], reverse=True)
    return list(sorted_nodes)

def betweenness_dismantling(G):
    """
    基于介数中心性（Betweenness Centrality）的拆解策略。
    
    Args:
        G (nx.Graph): 网络图。
        
    Returns:
        list: 拆解节点序列。
    """
    betweenness = nx.betweenness_centrality(G)
    sorted_nodes = sorted(G.nodes(), key=lambda x: betweenness[x], reverse=True)
    return list(sorted_nodes)

def eigenvector_dismantling(G):
    """
    基于特征向量中心性（Eigenvector Centrality）的拆解策略。
    
    Args:
        G (nx.Graph): 网络图。
        
    Returns:
        list: 拆解节点序列。
    """
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
        sorted_nodes = sorted(G.nodes(), key=lambda x: eigenvector[x], reverse=True)
        return list(sorted_nodes)
    except:
        # 如果不收敛，回退到度中心性
        return degree_dismantling(G)

# -------------------------------------------------------
# Adaptive Dismantling (To preserve original behavior)
# -------------------------------------------------------

def adaptive_degree_dismantling(G):
    """
    自适应度拆解策略（Target Attack）。
    每移除一个节点后，重新计算剩余节点的度，选择度最大的节点进行下一轮移除。
    
    注意：此函数返回的是攻击序列，但为了计算序列，它实际上在临时图上模拟了拆解过程。
    
    Args:
        G (nx.Graph): 网络图。
        
    Returns:
        list: 拆解节点序列。
    """
    G_temp = G.copy()
    sequence = []
    while G_temp.number_of_nodes() > 0:
        # 计算当前图的度中心性
        node_centrality = nx.degree_centrality(G_temp)
        if not node_centrality:
             break
        # 选择度最大的节点
        node = max(node_centrality, key=node_centrality.get)
        sequence.append(node)
        
        # 移除节点及其边
        if G_temp.has_node(node):
            edges = list(G_temp.edges(node))
            G_temp.remove_edges_from(edges)
            G_temp.remove_node(node)
            
    return sequence

def adaptive_hybrid_dismantling(G):
    """
    自适应混合拆解策略（Hybrid Attack）。
    每一步有 20% 的概率选择度最大的节点，80% 的概率随机选择节点。
    
    Args:
        G (nx.Graph): 网络图。
        
    Returns:
        list: 拆解节点序列。
    """
    G_temp = G.copy()
    sequence = []
    while G_temp.number_of_nodes() > 0:
        random_number = random.random()
        if random_number < 0.2:
            node_centrality = nx.degree_centrality(G_temp)
            node = max(node_centrality, key=node_centrality.get)
        else:
            node = random.choice(list(G_temp.nodes()))
        sequence.append(node)
        
        # 移除节点及其边
        if G_temp.has_node(node):
            edges = list(G_temp.edges(node))
            G_temp.remove_edges_from(edges)
            G_temp.remove_node(node)
            
    return sequence

# -------------------------------------------------------
# Main Simulation Logic
# -------------------------------------------------------

def get_dismantling_sequence(G, method):
    """
    根据指定的方法名称获取拆解序列。
    
    Args:
        G (nx.Graph): 网络图。
        method (str): 拆解方法名称。
        
    Returns:
        list: 拆解节点序列。
    """
    logger.info(f"Generating dismantling sequence using method: {method}")
    if method == "bruteforce":
        return bruteforce_dismantler(G)
    elif method == "degree" or method == "target": 
        # 'target' 通常意味着自适应度拆解（Target Attack）
        if method == "target":
             return adaptive_degree_dismantling(G)
        return degree_dismantling(G)
    elif method == "random": 
        return random_dismantling(G)
    elif method == "pagerank":
        return pagerank_dismantling(G)
    elif method == "betweenness":
        return betweenness_dismantling(G)
    elif method == "eigenvector":
        return eigenvector_dismantling(G)
    elif method == "hybrid": 
        return adaptive_hybrid_dismantling(G)
    else:
        # 默认为随机
        logger.warning(f"Unknown method '{method}', defaulting to random.")
        return random_dismantling(G)

def simulate_dismantling(G, sequence, nodes_num, centers):
    """
    模拟网络拆解过程。
    
    过程：
    1. 按照 sequence 顺序移除节点。
    2. 检查并移除级联失效的节点（即所在的连通分量中没有控制器的节点）。
    3. 计算当前网络的性能指标（LCC, CSA, CCE, WCP）。
    4. 记录指标随移除比例变化的曲线。
    
    Args:
        G (nx.Graph): 网络图（会被修改）。
        sequence (list): 节点移除序列。
        nodes_num (int): 初始节点总数（用于归一化）。
        centers (set): 控制器节点集合。
        
    Returns:
        tuple: (metrics_dict, x_when_lcc_20_percent)
            - metrics_dict (dict): 包含所有指标的曲线数据
                - 'lcc': [(lcc_value, remove_ratio), ...]
                - 'csa': [(csa_value, remove_ratio), ...]
                - 'cce': [(cce_value, remove_ratio), ...]
                - 'wcp': [(wcp_value, remove_ratio), ...]
            - x_when_lcc_20_percent (float): LCC降到20%时的移除比例。
    """
    # 计算初始指标（在任何攻击之前）
    centers_list = list(centers)
    
    # 计算初始 LCC（包含控制器的最大连通分量）
    initial_lcc = calculate_lcc(G, centers_list)
    initial_lcc_normalized = initial_lcc / nodes_num if nodes_num > 0 else 0
    
    # 计算其他初始指标
    initial_csa = calculate_csa(G, centers_list)
    initial_cce = calculate_control_entropy(G, centers_list)
    initial_wcp = calculate_wcp(G, centers_list)
    
    # 初始化指标记录（使用实际计算的初始值）
    metrics_dict = {
        'lcc': [(initial_lcc_normalized, 0.0)],
        'csa': [(initial_csa, 0.0)],
        'cce': [(initial_cce, 0.0)],
        'wcp': [(initial_wcp, 0.0)]
    }
    
    x_when_lcc_20_percent = None
    removed_count = 0
    
    logger.debug(f"Starting simulation. Initial nodes: {G.number_of_nodes()}, Centers: {len(centers)}, Initial LCC: {initial_lcc_normalized:.4f}")
    
    # 遍历攻击序列
    for node in sequence:
        # 如果节点已经不在图中（可能在之前的级联失效中被移除），跳过
        if not G.has_node(node):
            continue
        
        # 1. 攻击移除
        edges = list(G.edges(node))
        G.remove_edges_from(edges)
        G.remove_node(node)
        removed_count += 1
        
        # 判断被移除的节点是否是控制器
        if node in centers:
            centers.remove(node)
            logger.debug(f"Controller {node} removed by attack.")
            
        # 2. 级联失效逻辑 (Cascade Failure)
        # 关键修复：任何节点移除后都可能导致网络分裂，
        # 需要检查所有连通分量，移除没有控制器的分量
        if G.number_of_nodes() > 0 and len(centers) > 0:
            components = list(nx.connected_components(G))
            nodes_to_cascade_remove = []
            
            for comp in components:
                # 检查该分量是否包含控制器
                if centers.isdisjoint(comp):
                    # 该分量没有控制器，级联失效
                    nodes_to_cascade_remove.extend(list(comp))
            
            if nodes_to_cascade_remove:
                logger.debug(f"Cascade failure triggered. Removing {len(nodes_to_cascade_remove)} nodes from uncontrolled components.")
                for inode in nodes_to_cascade_remove:
                    if G.has_node(inode):
                        edges = list(G.edges(inode))
                        G.remove_edges_from(edges)
                        G.remove_node(inode)
                        # 注意：不增加 removed_count，因为这些是级联失效，不是直接攻击

        # 3. 计算指标并记录
        current_nodes = G.number_of_nodes()
        x_val = (nodes_num - current_nodes) / nodes_num
        
        if current_nodes == 0:
            # 网络完全崩溃
            metrics_dict['lcc'].append((0.0, x_val))
            metrics_dict['csa'].append((0.0, x_val))
            metrics_dict['cce'].append((0.0, x_val))
            metrics_dict['wcp'].append((0.0, x_val))
            if x_when_lcc_20_percent is None:
                x_when_lcc_20_percent = x_val
            logger.debug("Network fully collapsed.")
            break
        else:
            # 预先计算共同变量以避免重复计算
            centers_list = list(centers)
            components = list(nx.connected_components(G))
            centers_set = set(centers_list)
            
            # 计算LCC（极大连通子图占比）
            lcc_value = calculate_lcc(G, centers_list)
            lcc_normalized = lcc_value / nodes_num
            
            # 计算CSA（控制供给可用性）
            csa_value = calculate_csa(G, centers_list)
            
            # 计算CCE（控制熵）
            cce_value = calculate_control_entropy(G, centers_list)
            
            # 计算WCP（加权控制势能）
            wcp_value = calculate_wcp(G, centers_list)
            
            # 记录指标
            metrics_dict['lcc'].append((lcc_normalized, x_val))
            metrics_dict['csa'].append((csa_value, x_val))
            metrics_dict['cce'].append((cce_value, x_val))
            metrics_dict['wcp'].append((wcp_value, x_val))
            
            # 终止条件: LCC下降到20%以下
            if lcc_normalized <= 0.2:
                if x_when_lcc_20_percent is None:
                    x_when_lcc_20_percent = x_val
                logger.debug(f"LCC dropped below 20% at x={x_val:.4f}")
                
                # 为了画图完整性，填充后续点为0（确保x值递增，避免突然上升）
                if x_val < 1.0:
                    # 确保填充点的x值大于当前x值
                    next_x = min(x_val + 0.01, 1.0)  # 使用0.01而不是0.001，确保有足够间隔
                    metrics_dict['lcc'].append((0.0, next_x))
                    metrics_dict['lcc'].append((0.0, 1.0))
                    metrics_dict['csa'].append((0.0, next_x))
                    metrics_dict['csa'].append((0.0, 1.0))
                    metrics_dict['cce'].append((0.0, next_x))
                    metrics_dict['cce'].append((0.0, 1.0))
                    metrics_dict['wcp'].append((0.0, next_x))
                    metrics_dict['wcp'].append((0.0, 1.0))
                    
                break
                
    return metrics_dict, x_when_lcc_20_percent

def node_attack(G, nodes_num, centers, flag):
    """
    为了向后兼容的包装函数。
    1. 根据 flag 生成拆解序列。
    2. 运行仿真。
    """
    #1. 获取序列
    # 我们传递副本给 get_sequence，因为自适应方法会修改图
    sequence = get_dismantling_sequence(G.copy(), flag)
    
    logger.info(f"Starting node attack simulation. Mode: {flag}")
    
    #2. 运行仿真
    # 使用原始 G（仿真会修改它）和原始控制器集合（仿真会修改集合）
    centers_set = set(centers)
    metrics_dict, x_when_lcc_20_percent = simulate_dismantling(G, sequence, nodes_num, centers_set)
    
    logger.info(f"Attack completed. Collapse point (LCC 20%): {x_when_lcc_20_percent}")
    
    return metrics_dict, x_when_lcc_20_percent
