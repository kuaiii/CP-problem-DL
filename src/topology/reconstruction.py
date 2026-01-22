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


def create_bimodal_network_exact(G, hub_ratio=0.05):
    """
    构造符合最优双峰度分布的无向网络（增强版 - 多hub节点）
    
    改进：
    1. 增加高度节点（hub）数量，使网络更健壮
    2. 优化hub节点之间的连接，形成核心骨干
    3. 保证级联失效时的抗毁性
    
    参数:
        G: 输入图
        hub_ratio: hub节点占总节点的比例，默认0.05 (5%)
    
    返回:
        G: networkx.Graph 对象，构造好的双峰网络
    """
    # 1. 初始化随机种子和空图
    N = G.number_of_nodes()
    M = G.number_of_edges()
    
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    
    # 2. 计算双峰度分布参数
    total_degree = 2 * M
    avg_degree = total_degree / N
    
    # Hub节点数量：至少2个，最多N的hub_ratio比例
    num_hubs = max(2, min(int(N * hub_ratio), N // 5))
    
    # 计算 k_max (hub节点的度)
    # 根据理论公式: k_max = A * N^(2/3)，但考虑多个hub节点
    k = avg_degree
    if k > 1 and (2 * k - 1) > 0:
        A = (2 * k**2 * (k - 1)**2 / (2 * k - 1))**(1/3)
        k_max = int(A * (N**(2/3)) / (num_hubs ** 0.5))  # 调整以适应多hub
    else:
        k_max = int(avg_degree * 3)
    
    # 确保 k_max 在合理范围内
    k_max = min(k_max, N - 1)
    if k_max < int(avg_degree * 2):
        k_max = int(avg_degree * 2)
    
    # 3. 计算叶节点（spoke）的度
    num_spokes = N - num_hubs
    total_hub_degree = k_max * num_hubs
    remaining_degree = total_degree - total_hub_degree
    
    if num_spokes > 0:
        avg_spoke_degree = remaining_degree / num_spokes
        # 叶节点使用较低的度
        k_min = max(1, int(avg_spoke_degree * 0.7))
        k_mid = max(k_min + 1, int(avg_spoke_degree * 1.3))
    else:
        k_min = 1
        k_mid = 2
    
    # 4. 分配叶节点的度
    if num_spokes > 0:
        denominator = k_mid - k_min if k_mid > k_min else 1
        x = (remaining_degree - num_spokes * k_min) // denominator
        y = num_spokes - x
        
        # 确保 x 和 y 在合理范围内
        x = max(0, min(x, num_spokes))
        y = num_spokes - x
    else:
        x = y = 0
    
    # 5. 构造度序列
    # Hub节点在前，叶节点在后
    degree_sequence = [k_max] * num_hubs + [k_mid] * x + [k_min] * y
    
    # 调整度值以确保总和匹配
    actual_degree = sum(degree_sequence)
    remainder = total_degree - actual_degree
    
    # 将余数分配到节点上（优先分配给叶节点，避免hub过度集中）
    if remainder > 0:
        for i in range(remainder):
            idx = num_hubs + (i % max(1, num_spokes))
            if idx < N:
                degree_sequence[idx] += 1
    elif remainder < 0:
        for i in range(-remainder):
            idx = num_hubs + (i % max(1, num_spokes))
            if idx < N and degree_sequence[idx] > 1:
                degree_sequence[idx] -= 1
    
    # 确保总度数为偶数
    if sum(degree_sequence) % 2 != 0:
        # 找到一个非hub节点加1
        for i in range(num_hubs, N):
            degree_sequence[i] += 1
            break
    
    # 6. 保存hub节点索引（在打乱之前）
    hub_indices_before_shuffle = list(range(num_hubs))
    
    # 打乱叶节点部分（保持hub在前面以便后续处理）
    spoke_degrees = degree_sequence[num_hubs:]
    random.shuffle(spoke_degrees)
    degree_sequence = degree_sequence[:num_hubs] + spoke_degrees
    
    # 7. 生成网络
    try:
        G_new = nx.configuration_model(degree_sequence, seed=seed)
        G_new = nx.Graph(G_new)  # 转为简单图
        G_new.remove_edges_from(nx.selfloop_edges(G_new))
    except Exception as e:
        print(f"配置模型生成失败，原因：{e}")
        G_new = nx.Graph()
        G_new.add_nodes_from(range(N))
        
        # 手动构造：优先连接高度节点
        nodes_by_degree = sorted(range(N), key=lambda i: degree_sequence[i], reverse=True)
        for node in nodes_by_degree:
            target_degree = degree_sequence[node]
            current_degree = G_new.degree(node)
            need_edges = target_degree - current_degree
            if need_edges <= 0:
                continue
            candidates = [n for n in range(N) if n != node and not G_new.has_edge(node, n)]
            if candidates:
                add_edges = random.sample(candidates, min(need_edges, len(candidates)))
                for other in add_edges:
                    G_new.add_edge(node, other)
    
    # 8. 确保hub节点之间互相连接（形成核心骨干）
    for i in range(num_hubs):
        for j in range(i + 1, num_hubs):
            if not G_new.has_edge(i, j):
                # 尝试添加边，但要保持总边数
                # 找一条可以移除的边（非hub-hub边）
                removable_edges = [(u, v) for u, v in G_new.edges() 
                                   if not (u < num_hubs and v < num_hubs)]
                if removable_edges:
                    edge_to_remove = random.choice(removable_edges)
                    G_new.remove_edge(*edge_to_remove)
                    G_new.add_edge(i, j)
    
    # 9. 最终校准边数
    current_edges = G_new.number_of_edges()
    if current_edges < M:
        need_more = M - current_edges
        all_possible = [(u, v) for u in range(N) for v in range(u+1, N) 
                        if not G_new.has_edge(u, v)]
        if len(all_possible) >= need_more:
            add_edges = random.sample(all_possible, need_more)
            G_new.add_edges_from(add_edges)
    elif current_edges > M:
        need_less = current_edges - M
        # 优先移除非hub边
        removable = [(u, v) for u, v in G_new.edges() 
                     if not (u < num_hubs or v < num_hubs)]
        if len(removable) >= need_less:
            remove_edges = random.sample(removable, need_less)
        else:
            remove_edges = random.sample(list(G_new.edges()), need_less)
        G_new.remove_edges_from(remove_edges)
    
    return G_new

