# -*- coding: utf-8 -*-
from collections import defaultdict
import networkx as nx
import random
import itertools
from src.metrics.metrics import get_shortest_dist
from src.utils.gpu_utils import (
    is_gpu_available, to_tensor, to_numpy, 
    compute_shortest_paths_from_sources_gpu, gpu_greedy_selection,
    TORCH_AVAILABLE, CUDA_AVAILABLE, DEVICE
)
from tqdm import tqdm
import numpy as np
from math import ceil

# 如果 PyTorch 可用，导入
if TORCH_AVAILABLE:
    import torch

# Try to import louvain_communities
try:
    from networkx.algorithms.community import louvain_communities
except ImportError:
    try:
        from networkx.community import louvain_communities
    except ImportError:
        louvain_communities = None

# ----------------- K-Median / K-Center Utilities -----------------

def k_median(G, points, is_weight):
    min_value = float('inf')
    index = -1
    for source in points:
        temp = 0
        for target in points:
            if nx.has_path(G, source, target):
                if is_weight == 1:
                    temp += get_shortest_dist(G, source, target)
                else:
                    temp += nx.shortest_path_length(G, source, target)
            else:
                temp += 999999
        if temp < min_value:
            min_value = temp
            index = source
    return index

def k_center(G, points, is_weight):
    min_value = float('inf')
    index = -1
    for source in points:
        temp = 0
        for target in points:
            if nx.has_path(G, source, target):
                if is_weight == 1:
                    dist = get_shortest_dist(G, source, target)
                else:
                    dist = nx.shortest_path_length(G, source, target)
                if dist > temp:
                    temp = dist
            else:
                temp = float('inf')
                break 
        if temp < min_value:
            min_value = temp
            index = source
    return index

def update_centers(G, assignments, strategy, is_weight):
    new_means = defaultdict(list)
    centers = []
    nodes = G.nodes()
    for assignment, point in zip(assignments, nodes):
        if assignment >= 0 :
            new_means[assignment].append(point)

    for points in new_means.values():
        if strategy == 1:
            centers.append(k_median(G, points, is_weight))
        else:
            centers.append(k_center(G, points, is_weight))

    return centers

def assign_points(G, centers, is_weight):
    assignments = []
    nodes = G.nodes()
    
    for point in nodes:
        shortest = float('inf') 
        shortest_index = -1
        
        for i in range(len(centers)):
            if not G.has_node(centers[i]): 
                 continue
                 
            if not nx.has_path(G, point, centers[i]):
                continue
                
            if is_weight == 1:
                val = get_shortest_dist(G, point, centers[i])
            else:
                val = nx.shortest_path_length(G, point, centers[i])
                
            if val < shortest:
                shortest = val
                shortest_index = i
        
        assignments.append(shortest_index)
    return assignments

def generate_k(G, k):
    centers = []
    i = 0
    nodes_num = G.number_of_nodes()
    while i < k:
        node = random.randint(0, nodes_num - 1)
        if node not in centers:
            centers.append(node)
            i += 1
    return centers

def k_means(graph, k, strategy, is_weight):
    k_points = generate_k(graph, k)
    assignments = assign_points(graph, k_points, is_weight)
    old_assignments = None
    new_centers = []
    while assignments != old_assignments:
        new_centers = update_centers(graph, assignments, strategy, is_weight)
        old_assignments = assignments
        assignments = assign_points(graph, new_centers, is_weight)
    return assignments, new_centers

# ----------------- Community Detection -----------------

def GN(G, k):
    comp = nx.algorithms.community.girvan_newman(G)
    limited = itertools.takewhile(lambda c: len(c) <= k, comp)
    k_communities = list()
    for communities in limited:
        k_communities = list(sorted(c) for c in communities)
    assignments = [0 for x in range(G.number_of_nodes())]
    for i, nodes in enumerate(k_communities):
        for node in nodes:
            assignments[node] = i
    return assignments

def improved_GN(G, k):
    for i in range(k):
        edge_betweenness = nx.edge_betweenness_centrality(G)
        edge = max(edge_betweenness, key=edge_betweenness.get)
        G.remove_edge(*edge)
    communities = list(nx.connected_components(G))
    nodes_num = G.number_of_nodes()
    while len(max(communities, key=len)) >= int(nodes_num * 3 / len(communities)):
        subgraph = G.subgraph(list(max(communities, key=len)))
        edge_betweenness = nx.edge_betweenness_centrality(subgraph)
        edge = max(edge_betweenness, key=edge_betweenness.get)
        G.remove_edge(*edge)
        communities = list(nx.connected_components(G))
    communities = sorted(communities)
    assignments = [0] * nodes_num
    for i, nodes in enumerate(communities):
        for node in nodes:
            assignments[node] = i
    return assignments, communities

def weight_improved_GN(G, k):
    for i in range(k):
        edge_betweenness = nx.edge_betweenness_centrality(G, weight='weight')
        edge = max(edge_betweenness, key=edge_betweenness.get)
        G.remove_edge(*edge)
    communities = list(nx.connected_components(G))
    nodes_num = G.number_of_nodes()
    while len(max(communities, key=len)) >= int(nodes_num * 3 / len(communities)):
        subgraph = G.subgraph(list(max(communities, key=len)))
        edge_betweenness = nx.edge_betweenness_centrality(subgraph, weight='weight')
        edge = max(edge_betweenness, key=edge_betweenness.get)
        G.remove_edge(*edge)
        communities = list(nx.connected_components(G))
    communities = sorted(communities)
    assignments = [0] * nodes_num
    for i, nodes in enumerate(communities):
        for node in nodes:
            assignments[node] = i
    return assignments, communities

def louvain_partition(G, weight='weight'):
    if louvain_communities is None:
        print("Louvain not available, falling back to GN")
        return weight_improved_GN(G, 10)

    communities_gen = louvain_communities(G, weight=weight, resolution=1.0, seed=42)
    communities = sorted([list(c) for c in communities_gen], key=len, reverse=True)
    
    assignments = {}
    for i, comm in enumerate(communities):
        for node in comm:
            assignments[node] = i
            
    nodes_num = G.number_of_nodes()
    assignment_list = [0] * nodes_num
    for node in G.nodes():
        if isinstance(node, int) and node < nodes_num:
            assignment_list[node] = assignments.get(node, -1)
             
    return assignment_list, communities

def fast_GN_partition(G, remove_count=None):
    if remove_count is None:
        remove_count = int(G.number_of_edges() * 0.2)
        
    G_copy = G.copy()
    edge_betweenness = nx.edge_betweenness_centrality(G_copy, weight='weight')
    sorted_edges = sorted(edge_betweenness.items(), key=lambda item: item[1], reverse=True)
    
    for i in range(min(remove_count, len(sorted_edges))):
        edge = sorted_edges[i][0]
        if G_copy.has_edge(*edge):
            G_copy.remove_edge(*edge)
            
    communities_gen = nx.connected_components(G_copy)
    communities = sorted([list(c) for c in communities_gen], key=len, reverse=True)
    
    nodes_num = G.number_of_nodes()
    assignments = {}
    for i, comm in enumerate(communities):
        for node in comm:
            assignments[node] = i
            
    assignment_list = [0] * nodes_num
    for node in G.nodes():
        if isinstance(node, int) and node < nodes_num:
            assignment_list[node] = assignments.get(node, -1)
            
    return assignment_list, communities

# ----------------- Optimized Selection -----------------

def k_median_opt(G, points, is_weight):
    if not points:
        return -1
    
    candidates = list(points)
    min_total_dist = float('inf')
    best_center = -1
    
    for source in candidates:
        if is_weight:
            length = nx.single_source_dijkstra_path_length(G, source, weight='weight')
        else:
            length = nx.single_source_shortest_path_length(G, source)
            
        current_dist = 0
        for target in points:
            if target in length:
                current_dist += length[target]
            else:
                current_dist += 999999 
                
        if current_dist < min_total_dist:
            min_total_dist = current_dist
            best_center = source
            
    return best_center

def k_median_vectorized(G, points, controller_num, is_weight):
    """
    优化后的 k-median 算法：使用 GPU 加速矩阵运算和预计算距离
    
    Args:
        G: NetworkX 图对象
        points: 候选节点列表
        controller_num: 要选择的控制器数量
        is_weight: 是否考虑边权重
        
    Returns:
        list: 选中的控制器节点列表
    """
    nodes_list = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes_list)}
    num_nodes = len(nodes_list)
    
    points_list = list(points)
    num_candidates = len(points_list)
    
    # 使用 GPU 或 CPU 预计算距离矩阵
    if is_gpu_available():
        # GPU 加速版本
        dist_matrix = np.full((num_candidates, num_nodes), np.inf, dtype=np.float32)
        
        # 批量计算最短路径
        for i, candidate in enumerate(tqdm(points_list, desc="Pre-calculating Paths (GPU)", leave=False)):
            if is_weight:
                dists = nx.single_source_dijkstra_path_length(G, candidate, weight='weight')
            else:
                dists = nx.single_source_shortest_path_length(G, candidate)
            
            for target_node, d in dists.items():
                if target_node in node_to_idx:
                    dist_matrix[i, node_to_idx[target_node]] = d
        
        # 使用 GPU 进行贪婪选择
        selected_indices = gpu_greedy_selection(dist_matrix, controller_num)
    else:
        # CPU 版本 (原始实现)
        dist_matrix = np.full((num_candidates, num_nodes), np.inf)
        
        for i, candidate in enumerate(tqdm(points_list, desc="Pre-calculating Paths", leave=False)):
            if is_weight:
                dists = nx.single_source_dijkstra_path_length(G, candidate, weight='weight')
            else:
                dists = nx.single_source_shortest_path_length(G, candidate)
            
            for target_node, d in dists.items():
                if target_node in node_to_idx:
                    dist_matrix[i, node_to_idx[target_node]] = d
                    
        current_min_dists = np.full(num_nodes, np.inf)
        
        selected_indices = []
        remaining_indices = set(range(num_candidates))
        
        for _ in tqdm(range(controller_num), desc="Selecting Controllers", leave=False):
            if not remaining_indices:
                break

            rem_indices_list = list(remaining_indices)
            candidate_dists = dist_matrix[rem_indices_list] 
            new_dists = np.minimum(candidate_dists, current_min_dists)
            total_dists = np.sum(new_dists, axis=1)
            
            min_idx_in_rem = np.argmin(total_dists)
            best_candidate_idx = rem_indices_list[min_idx_in_rem]
            
            selected_indices.append(best_candidate_idx)
            remaining_indices.remove(best_candidate_idx)
            current_min_dists = np.minimum(current_min_dists, dist_matrix[best_candidate_idx])
        
    return [points_list[i] for i in selected_indices]

# ----------------- E-RCP Logic -----------------

def find_center(G, communities, β, is_weight):
    sorted_communities = sorted(communities, key=len, reverse=True)
    k = 0
    cover_nodes = 0
    total_nodes = G.number_of_nodes()
    centers = []
    
    for community in sorted_communities:
        if cover_nodes >= total_nodes * β:
            break
        k += 1
        cover_nodes += len(community)
        centers.append(k_median(G, community, is_weight))
        
    center_dict = dict()
    for i, center in enumerate(centers):
        for j, community in enumerate(communities):
            if center in community:
                center_dict[j] = i
    return centers, center_dict

def RCP(G, controller_rate, is_weight):
    # Fixed: Use global rate limit
    total_controllers_budget = ceil(G.number_of_nodes() * controller_rate)
    
    communities = list(nx.connected_components(G))
    
    # Calculate initial allocation
    controllers = []
    
    # Filter out empty communities just in case
    communities = [list(c) for c in communities if len(c) > 0]
    if not communities:
        return []

    # Sort communities by size to prioritize larger ones
    communities.sort(key=len, reverse=True)
    
    allocated_count = 0
    allocations = []
    
    # 1. Minimum allocation: Each component gets at least 1 if budget allows
    for i in range(len(communities)):
        if allocated_count < total_controllers_budget:
            allocations.append(1)
            allocated_count += 1
        else:
            allocations.append(0)
            
    # 2. Proportional allocation for remaining budget
    remaining_budget = total_controllers_budget - allocated_count
    
    if remaining_budget > 0:
        total_nodes = G.number_of_nodes()
        for i, nodes in enumerate(communities):
            if remaining_budget <= 0: break
            
            # Simple proportional share
            share = int((len(nodes) / total_nodes) * total_controllers_budget)
            # Subtract what we already gave (1)
            extra = max(0, share - allocations[i])
            
            take = min(remaining_budget, extra)
            allocations[i] += take
            remaining_budget -= take
            
    # 3. If still budget left (due to rounding), give to largest communities
    if remaining_budget > 0:
        for i in range(len(communities)):
            if remaining_budget <= 0: break
            # Don't give more than community size
            if allocations[i] < len(communities[i]):
                allocations[i] += 1
                remaining_budget -= 1

    # Execute Selection
    for i, nodes in enumerate(communities):
        num_controllers = allocations[i]
        if num_controllers > 0:
            select_controllers = k_median_vectorized(G.subgraph(nodes), nodes, num_controllers, is_weight)
            for node in select_controllers:
                controllers.append(node)
                
    return controllers

# Alias E_RCP to RCP for backward compatibility if needed, though usually E_RCP was the name.
# The user's code referenced E_RCP before, now references RCP.
# I will keep E_RCP as alias.
E_RCP = RCP

# ----------------- Other Strategies -----------------

def random_select_controllers(G, controller_rate):
    """
    选择度最高的节点作为控制器（原名为随机选择，已修改为基于度的选择）
    
    策略：选择度数最高的 rate 比例节点作为控制器
    
    Args:
        G (nx.Graph): 网络拓扑
        controller_rate (float): 控制器比例
        
    Returns:
        list: 选中的控制器节点列表
    """
    total_nodes = G.number_of_nodes()
    total_controllers_budget = ceil(total_nodes * controller_rate)
    
    # 获取所有节点的度数，并按度数降序排序
    node_degrees = [(node, G.degree(node)) for node in G.nodes()]
    node_degrees.sort(key=lambda x: x[1], reverse=True)
    
    # 选择度数最高的 total_controllers_budget 个节点
    controllers = [node for node, degree in node_degrees[:total_controllers_budget]]
    
    return controllers

def get_furthest_distance(G, nodes):
    max_distance = 0
    for node in G.nodes:
        shortest_dist = float('inf')
        for controller in nodes:
            if nx.has_path(G, node, controller):
                dist = nx.shortest_path_length(G, node, controller)
                shortest_dist = min(shortest_dist, dist)
        max_distance = max(max_distance, shortest_dist)
    return max_distance

def select_furthest_nodes(G, points, node_num):
    selected_nodes = []
    remaining_nodes = set(points)
    for _ in range(node_num):
        best_node = None
        best_distance = 0
        for node in remaining_nodes:
            temp_nodes = selected_nodes + [node]
            distance = get_furthest_distance(G, temp_nodes)
            if distance > best_distance:
                best_distance = distance
                best_node = node
        if best_node is not None:
            selected_nodes.append(best_node)
            remaining_nodes.remove(best_node)
    return selected_nodes
