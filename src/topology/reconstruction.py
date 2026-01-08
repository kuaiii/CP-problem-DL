# -*- coding: utf-8 -*-
import networkx as nx
import random
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
    
    for k_min in range(1, limit_kmin + 1):
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
