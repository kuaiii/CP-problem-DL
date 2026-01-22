# -*- coding: utf-8 -*-
"""
优化模块 V2 - 高性能版本
统一接口，支持切换新旧版本
"""
import networkx as nx
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 检查是否使用 V2 版本（通过环境变量控制）
USE_V2 = os.environ.get('USE_OPTIMIZATION_V2', '1') == '1'

if USE_V2:
    try:
        from src.topology.genetic_optimization_v2 import solution_v2 as solution
        from src.topology.onion_optimization_v2 import OnionOptimizerV2 as OnionOptimizer
        logger.info("[Optimization] Using V2 (High Performance) version")
    except ImportError as e:
        logger.warning(f"[Optimization] V2 import failed: {e}, falling back to V1")
        from src.topology.genetic_optimization import solution
        from src.topology.onion_optimization import OnionOptimizer
else:
    from src.topology.genetic_optimization import solution
    from src.topology.onion_optimization import OnionOptimizer
    logger.info("[Optimization] Using V1 version")

# 其他优化器保持不变
from src.topology.romen_optimization import ROHEMOptimizer
from src.topology.solo_optimization import SOLO
from src.topology.unity_optimization import UNITY


def construct_Three(G, controllers_rate, use_v2=None):
    """
    构建 GA 优化网络 (G4)
    
    Args:
        G: NetworkX 图对象
        controllers_rate: 控制器比例（兼容参数，实际未使用）
        use_v2: 是否使用 V2 版本（None 表示使用全局设置）
    """
    nodes_num = G.number_of_nodes()
    edges_num = G.number_of_edges()
    
    if use_v2 is None:
        use_v2 = USE_V2
    
    if use_v2:
        try:
            from src.topology.genetic_optimization_v2 import solution_v2
            best_solution, best_fitness = solution_v2(nodes_num, edges_num)
        except ImportError:
            best_solution, best_fitness = solution(nodes_num, edges_num)
    else:
        best_solution, best_fitness = solution(nodes_num, edges_num)
    
    G = nx.from_numpy_array(best_solution)
    return G


def construct_onion(G, max_iter=None, use_v2=None):
    """
    构建 ONION 结构的优化网络 (G5)
    
    Args:
        G: NetworkX 图对象
        max_iter: 最大迭代次数（None 表示自动设置）
        use_v2: 是否使用 V2 版本（None 表示使用全局设置）
    """
    if use_v2 is None:
        use_v2 = USE_V2
    
    if use_v2:
        try:
            from src.topology.onion_optimization_v2 import OnionOptimizerV2
            optimizer = OnionOptimizerV2(G)
            # V2 会自动设置参数
            G_onion = optimizer.optimize_network(max_iterations=max_iter, verbose=True)
        except ImportError:
            from src.topology.onion_optimization import OnionOptimizer
            optimizer = OnionOptimizer(G)
            if max_iter is None:
                max_iter = 600 if G.number_of_nodes() > 200 else 800
            G_onion = optimizer.optimize_network(max_iterations=max_iter, verbose=True)
    else:
        optimizer = OnionOptimizer(G)
        if max_iter is None:
            max_iter = 600 if G.number_of_nodes() > 200 else 800
        G_onion = optimizer.optimize_network(max_iterations=max_iter, verbose=True)
    
    return G_onion


def construct_romen(G):
    """
    构建 ROMEN 结构的优化网络 (G7)
    """
    logger.info("  Starting ROMEN Structure Optimization...")
    optimizer = ROHEMOptimizer(
        population_size=2,
        generations=15,
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
    构建 SOLO 结构的优化网络
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
