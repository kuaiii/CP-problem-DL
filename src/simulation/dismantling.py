# -*- coding: utf-8 -*-
import networkx as nx
import random
import logging
from itertools import combinations
from tqdm import tqdm
from src.metrics.metrics import coverage_compute2, weight_efficiency_compute, max_component_size_compute
from src.utils.logger import get_logger

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
    
    Args:
        G (nx.Graph): 网络图。
        
    Returns:
        list: 拆解节点序列。
    """
    sorted_nodes = sorted(G.nodes(), key=G.degree, reverse=True)
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

def simulate_dismantling(G, sequence, nodes_num, centers, metric_type="gcc"):
    """
    模拟网络拆解过程。
    
    过程：
    1. 按照 sequence 顺序移除节点。
    2. 检查并移除级联失效的节点（即所在的连通分量中没有控制器的节点）。
    3. 计算当前网络的性能指标。
    4. 记录指标随移除比例变化的曲线。
    
    Args:
        G (nx.Graph): 网络图（会被修改）。
        sequence (list): 节点移除序列。
        nodes_num (int): 初始节点总数（用于归一化）。
        centers (set): 控制器节点集合。
        metric_type (str): 评估指标类型 ("gcc", "efficiency", "coverage")。
        
    Returns:
        tuple: (cover_nodes, x_when_y_zero, attack_record)
            - cover_nodes (list): [(metric_val, remove_ratio), ...]
            - x_when_y_zero (float): 指标降为0时的移除比例。
            - attack_record (dict): 记录被攻击节点的度。
    """
    from src.metrics.metrics import get_metric_function
    metric_func = get_metric_function(metric_type)
    
    cover_nodes = [(1.0, 0.0)] # 初始状态：性能1.0，移除比例0.0
    x_when_y_zero = None
    attack_record = {}
    
    removed_count = 0
    
    logger.debug(f"Starting simulation. Initial nodes: {G.number_of_nodes()}, Centers: {len(centers)}")
    
    # 遍历攻击序列
    for node in sequence:
        # 如果节点已经不在图中（可能在之前的级联失效中被移除），跳过
        if not G.has_node(node):
            continue
        
        # 1. 攻击移除
        # 记录被攻击节点的度（用于分析攻击倾向）
        attack_record[node] = G.degree(node)
        
        # 显式移除边和节点
        edges = list(G.edges(node))
        G.remove_edges_from(edges)
        G.remove_node(node)
        removed_count += 1
        
        # 判断被移除的节点是否是控制器
        is_controller_removed = False
        if node in centers:
            centers.remove(node)
            is_controller_removed = True
            logger.debug(f"Controller {node} removed by attack.")
            
        # 2. 级联失效逻辑 (Cascade Failure)
        # 逻辑修改：只有当控制器被移除时，才检查连通分量的控制器覆盖情况
        if is_controller_removed:
            # 原理：如果一个连通分量中没有任何控制器，则该分量中的所有节点失效（被移除）。
            components = list(nx.connected_components(G))
            nodes_to_cascade_remove = []
            
            for comp in components:
                # isdisjoint: 如果两个集合没有共同元素，返回 True
                if centers.isdisjoint(comp):
                    # 该分量中没有控制器，标记为级联移除
                    nodes_to_cascade_remove.extend(list(comp))
            
            if nodes_to_cascade_remove:
                logger.debug(f"Cascade failure triggered. Removing {len(nodes_to_cascade_remove)} nodes.")
                for inode in nodes_to_cascade_remove:
                    if G.has_node(inode):
                        edges = list(G.edges(inode))
                        G.remove_edges_from(edges)
                        G.remove_node(inode)
                        removed_count += 1 # 级联移除也算作网络损失
                        # 理论上这些节点里不含控制器，但为了安全再次检查
                        if inode in centers:
                            centers.remove(inode)
                            logger.warning(f"Unexpected: Controller {inode} found in cascade removal set!")

        # 3. 计算指标并记录
        current_nodes = G.number_of_nodes()
        # 计算当前的移除比例（包括攻击移除和级联移除）
        x_val = (nodes_num - current_nodes) / nodes_num
        
        if current_nodes == 0:
            cover_nodes.append((0.0, x_val))
            if x_when_y_zero is None:
                x_when_y_zero = x_val
            logger.debug("Network fully collapsed.")
            break
        else:
            # 计算指标
            metric_value = metric_func(G, list(centers))
            
            # 归一化指标
            if metric_type == "efficiency":
                nn = metric_value # Efficiency 通常已经在 0-1 之间，或者由函数内部处理
            else:
                nn = metric_value / nodes_num
            
            cover_nodes.append((nn, x_val))
            
            # 终止条件: 指标下降到 0.2 以下（或者接近 0）
            # 原本是 1e-6，现在改为 0.2 作为统计记录的阈值，但我们选择在这里直接结束，视为瓦解。
            if nn <= 0.2:
                if x_when_y_zero is None:
                    x_when_y_zero = x_val
                logger.debug(f"Metric dropped below 0.2 at x={x_val:.4f}")
                
                # 为了 R 值计算和画图完整性，填充后续点为 0
                if x_val < 1.0:
                    cover_nodes.append((0.0, x_val + 0.001)) # 稍微偏移一点避免重合
                    cover_nodes.append((0.0, 1.0))
                    
                break
                
    return cover_nodes, x_when_y_zero, attack_record

def node_attack(G, nodes_num, centers, flag, metric_type="gcc"):
    """
    为了向后兼容的包装函数。
    1. 根据 flag 生成拆解序列。
    2. 运行仿真。
    """
    # 1. 获取序列
    # 我们传递副本给 get_sequence，因为自适应方法会修改图
    sequence = get_dismantling_sequence(G.copy(), flag)
    
    logger.info(f"Starting node attack simulation. Mode: {flag}, Metric: {metric_type}")
    
    # 2. 运行仿真
    # 使用原始 G（仿真会修改它）和原始控制器集合（仿真会修改集合）
    centers_set = set(centers)
    cover_nodes, x_when_y_zero, attack_record = simulate_dismantling(G, sequence, nodes_num, centers_set, metric_type)
    
    logger.info(f"Attack completed. Collapse point (x_zero): {x_when_y_zero}")
    
    return cover_nodes, x_when_y_zero, attack_record
