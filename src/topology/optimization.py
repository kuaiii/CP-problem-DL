# -*- coding: utf-8 -*-
import networkx as nx
import logging
from src.utils.logger import get_logger

# 导入拆分后的模块
from src.topology.genetic_optimization import solution
from src.topology.onion_optimization import OnionOptimizer
from src.topology.onion_optimization_v3 import OnionOptimizerV3, optimize_onion_v3
from src.topology.romen_optimization import ROHEMOptimizer
from src.topology.solo_optimization import SOLO
from src.topology.unity_optimization import UNITY

logger = get_logger(__name__)

# ----------------- Helper Functions (保持兼容性) -----------------

def construct_Three(G, controllers_rate):
    """
    构建 GA 优化网络 (G4)
    """
    nodes_num = G.number_of_nodes()
    edges_num = G.number_of_edges()
    best_solution, best_fitness = solution(nodes_num, edges_num)
    G = nx.from_numpy_array(best_solution)
    return G

def construct_onion(G, max_iter=800, use_v3=True):
    """
    构建 ONION 结构的优化网络 (G5)
    
    Args:
        G: 输入图
        max_iter: 最大迭代次数
        use_v3: 是否使用 V3 超高性能版本（默认 True）
            V3 优化:
            1. 逆向重组法 + Union-Find: 复杂度从 O(K(N+M)) 降至 O(M·α(N))
            2. 移除无效哈希缓存
            3. 攻击序列只计算一次（度保持性质）
    """
    if use_v3:
        optimizer = OnionOptimizerV3(G)
        G_onion = optimizer.optimize_network(max_iterations=max_iter, verbose=True)
    else:
        optimizer = OnionOptimizer(G)
        G_onion = optimizer.optimize_network(max_iterations=max_iter, verbose=True)
    return G_onion

def construct_romen(G):
    """
    构建 ROMEN 结构的优化网络 (G6)
    优化版本：减少代数和种群大小
    """
    logger.info("  Starting ROMEN Structure Optimization...")
    optimizer = ROHEMOptimizer(
        population_size=2,  # 减少种群大小
        generations=15,     # 减少代数
        verbose=True
    )
    G_romen = optimizer.optimize_topology(
        initial_graph=G,
        area_size=500,
        comm_distance=200
    )
    return G_romen

def construct_solo(G):
    """
    构建 SOLO 结构的优化网络 (G6)
    """
    logger.info("  Starting SOLO Structure Optimization...")
    solo = SOLO()
    G_solo = solo.generate_matched_topology(G)
    return G_solo

def construct_unity(G):
    """
    构建 UNITY 结构的优化网络 (G9)
    """
    logger.info("  Starting UNITY Structure Optimization...")
    unity = UNITY()
    G_unity = unity.generate_matched_topology(G)
    return G_unity

# construct_romen 已从 romen_optimization 直接导入
