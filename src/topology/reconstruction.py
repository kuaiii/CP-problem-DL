# -*- coding: utf-8 -*-
import networkx as nx
import random
import numpy as np
from math import ceil

def construct_Two(G, controllers_rate):
    """
    Constructs a Bimodal (Two-Peak) Degree Distribution Graph.
    Goal: k1*kmax + k2*kmin = 2*m
          k1 + k2 = n
    Where k1 is the number of hub nodes (controllers), k2 is the number of leaf nodes.
    We fix k1 = ceil(n * controllers_rate) based on budget.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    total_degree = 2 * m
    
    # k1: Number of hub nodes (fixed by budget)
    k1 = ceil(n * controllers_rate)
    if k1 < 1: k1 = 1
    if k1 >= n: k1 = n - 1
    
    # k2: Number of leaf nodes
    k2 = n - k1
    
    # Solve for kmax and kmin
    # k1 * kmax + k2 * kmin = total_degree
    # We want to maximize kmax, so we minimize kmin.
    # Constraint: kmin >= 1
    
    # Theoretical max kmax if kmin = 1
    # k1 * kmax + k2 * 1 = total_degree  => kmax = (total_degree - k2) / k1
    
    # Try to set kmin = 1 first
    kmin = 1
    remaining_degree = total_degree - k2 * kmin
    
    if remaining_degree < k1: 
        # Not enough degree to give kmax >= 1, this means m is very small (sparse graph)
        # Fallback: distribute evenly
        kmax = total_degree // n
        kmin = kmax
    else:
        kmax = remaining_degree // k1
        # It's possible kmax > n-1, so cap it
        if kmax >= n:
            kmax = n - 1
            # Recalculate kmin to absorb the excess
            # k2 * kmin = total_degree - k1 * kmax
            remaining_for_k2 = total_degree - k1 * kmax
            kmin = remaining_for_k2 // k2
            if kmin < 1: kmin = 1 # Should not happen if m is large enough
            
    return _generate_bimodal_graph(n, m, k1, k2, kmax, kmin)

def construct_Two_MaxK1(G):
    """
    Constructs a Bimodal Graph by maximizing k1 (number of high degree nodes).
    Instead of fixing k1 based on controller rate, we find k1 that is maximized
    subject to:
    1. k1 * kmax + k2 * kmin = 2 * m
    2. k1 + k2 = n
    3. kmax > kmin >= 1
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    total_degree = 2 * m
    
    best_k1 = -1
    best_params = None
    
    # Iterate over possible values of kmin
    # kmin must be less than average degree 2m/n
    # because kmax > kmin and k1*kmax + k2*kmin = 2m
    avg_degree = total_degree / n
    limit_kmin = int(avg_degree)
    if limit_kmin < 1: limit_kmin = 1
    
    for k_min in range(limit_kmin, 0, -1):
        # We want to solve for k1, kmax
        # k1 = (2m - n * kmin) / (kmax - kmin)
        # To maximize k1, we need to minimize (kmax - kmin).
        # Smallest integer kmax > kmin is kmin + 1.
        
        numerator = total_degree - n * k_min
        
        # Iterate k_max from k_min + 1 to n - 1
        # Actually we can iterate delta = k_max - k_min
        # k1 = numerator / delta
        # We want largest k1, so we want smallest delta that divides numerator
        
        for delta in range(1, n):
            if numerator % delta == 0:
                k1 = numerator // delta
                k_max = k_min + delta
                
                # Check constraints
                if k_max >= n: continue # Degree too high
                if k1 >= n: continue    # All nodes are hubs, not bimodal
                if k1 <= 0: continue
                
                # Found a valid k1
                if k1 > best_k1:
                    best_k1 = k1
                    best_params = (k1, n - k1, k_max, k_min)
                    
                # Since we iterate delta from 1 upwards, k1 decreases.
                # The first valid k1 we find for this k_min is the maximum for this k_min.
                # So we can break here for this k_min?
                # Yes, for a fixed k_min, k1 is max when delta is min.
                # But we need to compare across different k_min.
                break 

    if best_params is None:
        # Fallback if no perfect bimodal solution found
        # Just use the original logic with some default
        return construct_Two(G, 0.1)
        
    k1, k2, kmax, kmin = best_params
    return _generate_bimodal_graph(n, m, k1, k2, kmax, kmin)

def _generate_bimodal_graph(n, m, k1, k2, kmax, kmin):
    """
    Helper function to generate graph from parameters
    """
    total_degree = 2 * m
    current_sum = k1 * kmax + k2 * kmin
    remainder = total_degree - current_sum
    
    degree_seq = [kmax] * k1 + [kmin] * k2
    
    # Distribute remainder
    for i in range(remainder):
        degree_seq[i % n] += 1
        
    # Ensure even sum
    if sum(degree_seq) % 2 != 0:
        degree_seq[0] += 1
        
    # Generate graph
    try:
        G_multi = nx.configuration_model(degree_seq, seed=42)
        G_simple = nx.Graph(G_multi)
        G_simple.remove_edges_from(nx.selfloop_edges(G_simple))
    except:
        G_simple = nx.expected_degree_graph(degree_seq, selfloops=False)
        
    # Fix edge count
    current_edges = G_simple.number_of_edges()
    
    if current_edges < m:
        non_edges = list(nx.non_edges(G_simple))
        random.seed(42)
        random.shuffle(non_edges)
        needed = m - current_edges
        for u, v in non_edges:
            if needed <= 0: break
            G_simple.add_edge(u, v)
            needed -= 1
            
    elif current_edges > m:
        edges = list(G_simple.edges())
        random.seed(42)
        random.shuffle(edges)
        to_remove = current_edges - m
        for i in range(to_remove):
            G_simple.remove_edge(*edges[i])
            
    return G_simple


def create_bimodal_network_exact(G):
    """
    构造符合最优双峰度分布的无向网络（精准分配低度节点的度）
    
    参数:
        G: 输入图
    
    返回:
        G: networkx.Graph 对象，构造好的双峰网络
    """
    # 1. 初始化随机种子和空图
    N = G.number_of_nodes()
    M = G.number_of_edges()
    
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    
    # 2. 动态计算度序列参数
    total_degree = 2 * M
    avg_degree = total_degree / N
    
    # 根据理论公式计算 k_max
    # k_max = A * N^(2/3)
    # 其中 A = (2 * <k>^2 * (<k>-1)^2 / (2*<k>-1))^(1/3)
    k = avg_degree
    A = (2 * k**2 * (k - 1)**2 / (2 * k - 1))**(1/3)
    k_max = int(A * (N**(2/3)))
    
    # 确保 k_max 在合理范围内
    k_max = min(k_max, N - 1)
    if k_max < int(avg_degree) + 1:
        k_max = int(avg_degree) + 1
    
    num_max_nodes = 1
    remaining_degree = total_degree - k_max * num_max_nodes
    num_min_nodes = N - num_max_nodes
    
    # 计算低度节点的度分配
    # 设 x 个节点度为 k_mid，y 个节点度为 k_min
    # x + y = num_min_nodes
    # x * k_mid + y * k_min = remaining_degree
    
    k_min = max(1, int(avg_degree * 0.5))
    k_mid = int(avg_degree * 1.5)
    
    # 解方程: x * k_mid + (num_min_nodes - x) * k_min = remaining_degree
    # x * (k_mid - k_min) = remaining_degree - num_min_nodes * k_min
    denominator = k_mid - k_min
    if denominator <= 0:
        denominator = 1
    
    x = (remaining_degree - num_min_nodes * k_min) // denominator
    y = num_min_nodes - x
    
    # 确保 x 和 y 在合理范围内
    if x < 0:
        x = 0
        y = num_min_nodes
    if y < 0:
        y = 0
        x = num_min_nodes
    
    # 调整度值以确保总和匹配
    actual_degree = k_max * num_max_nodes + x * k_mid + y * k_min
    remainder = total_degree - actual_degree
    
    # 将余数均匀分配到节点上
    degree_sequence = [k_max] * num_max_nodes + [k_mid] * x + [k_min] * y
    for i in range(remainder):
        degree_sequence[i % N] += 1
    
    # 打乱度序列（避免模式化）
    random.shuffle(degree_sequence)
    
    # 3. 验证度序列的合法性（可图化）
    # 满足Erdős–Gallai条件（networkx会自动校验）
    try:
        # 用配置模型生成网络（保证度序列严格匹配）
        G = nx.configuration_model(degree_sequence, seed=seed)
        # 转为无向简单图（移除自环和重边）
        G = G.to_undirected()
        G.remove_edges_from(nx.selfloop_edges(G))
    except Exception as e:
        print(f"配置模型生成失败，原因：{e}")
        # 备用方案：手动构造
        G = nx.Graph()
        G.add_nodes_from(range(N))
        # 按度序列连接节点
        nodes_by_degree = sorted(range(N), key=lambda i: degree_sequence[i], reverse=True)
        for node in nodes_by_degree:
            target_degree = degree_sequence[node]
            current_degree = G.degree(node)
            need_edges = target_degree - current_degree
            if need_edges <= 0:
                continue
            # 随机选其他节点连接（避免重复/自环）
            candidates = [n for n in range(N) if n != node and not G.has_edge(node, n)]
            add_edges = random.sample(candidates, min(need_edges, len(candidates)))
            for other_node in add_edges:
                G.add_edge(node, other_node)
    
    # 4. 最终校准边数（确保严格等于M）
    current_edges = G.number_of_edges()
    if current_edges < M:
        need_more = M - current_edges
        # 生成所有可能的未连接边
        all_possible = []
        for u in range(N):
            for v in range(u+1, N):
                if not G.has_edge(u, v):
                    all_possible.append((u, v))
        add_edges = random.sample(all_possible, need_more)
        G.add_edges_from(add_edges)
    elif current_edges > M:
        need_less = current_edges - M
        remove_edges = random.sample(list(G.edges()), need_less)
        G.remove_edges_from(remove_edges)
    
    return G

