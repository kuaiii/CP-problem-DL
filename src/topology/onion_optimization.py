# -*- coding: utf-8 -*-
import networkx as nx
import random
from tqdm import tqdm
from src.utils.logger import get_logger

logger = get_logger(__name__)

class OnionOptimizer:
    def __init__(self, G):
        self.G = G

    def robustness_measure(self, G, attack_type="HDA"):
        """
        计算网络鲁棒性 R (R-index)
        R = 1/N * sum(S(q))
        
        Parameters:
        G: networkx.Graph
        attack_type: str - "HDA" (High Degree Attack) or "RA" (Random Attack)
        
        Returns:
        float: 鲁棒性 R 值
        """
        N = G.number_of_nodes()
        nodes = list(G.nodes())
        
        if attack_type == "HDA":
            # 按度数降序排序
            nodes_sorted = sorted(nodes, key=lambda n: G.degree(n), reverse=True)
        else:
            # 随机排序
            nodes_sorted = random.sample(nodes, len(nodes))
            
        sum_S = 0.0
        G_temp = G.copy()
        
        # 初始最大连通分量大小
        if len(G_temp) > 0:
            largest_cc = len(max(nx.connected_components(G_temp), key=len))
            sum_S += largest_cc
        
        # 逐步移除节点
        for node in nodes_sorted[:-1]: # 最后一个节点移除后大小为0，不需要计算
            G_temp.remove_node(node)
            if len(G_temp) > 0:
                largest_cc = len(max(nx.connected_components(G_temp), key=len))
                sum_S += largest_cc
            else:
                break
                
        return sum_S / (N * N)

    def optimize_network(self, max_iterations=10000, threshold=0.0, tolerance=0.01, verbose=True):
        """
        通过边交换优化网络的鲁棒性 (ONION Structure)
        
        Parameters:
        max_iterations: int - 最大迭代次数
        threshold: float - 接受交换的阈值
        tolerance: float - 收敛容差
        verbose: bool - 是否显示进度
        
        Returns:
        networkx.Graph - 优化后的网络
        """
        G_optimized = self.G.copy()
        R_old = self.robustness_measure(G_optimized, attack_type="HDA")
        
        if verbose:
            logger.info(f"  Initial Robustness: {R_old:.4f}")
            logger.info(f"  Starting ONION Structure Optimization...")
            
        # 跟踪最近的改进
        recent_improvements = []
        last_check_point = 0
        
        # 使用 tqdm 显示进度条
        iterator = range(max_iterations)
        if verbose:
            iterator = tqdm(iterator, desc="Onion Optimization", disable=not verbose)
            
        for i in iterator:
            # 随机选择两条边
            edges = list(G_optimized.edges())
            if len(edges) < 2:
                break
                
            edge1, edge2 = random.sample(edges, 2)
            u, v = edge1
            x, y = edge2
            
            # 避免选择相邻的边和自环 (确保涉及4个不同的节点)
            if len({u, v, x, y}) < 4:
                continue
                
            # 尝试交换: (u,v), (x,y) -> (u,x), (v,y) 或 (u,y), (v,x)
            # 这里随机选择一种重连方式
            if random.random() < 0.5:
                new_edge1 = (u, x)
                new_edge2 = (v, y)
            else:
                new_edge1 = (u, y)
                new_edge2 = (v, x)
                
            # 确保新边不会形成重复边
            if G_optimized.has_edge(*new_edge1) or G_optimized.has_edge(*new_edge2):
                continue
            
            # 临时移除旧边
            G_optimized.remove_edge(u, v)
            G_optimized.remove_edge(x, y)
            
            # 添加新边
            G_optimized.add_edge(*new_edge1)
            G_optimized.add_edge(*new_edge2)
            
            # 计算新的鲁棒性
            R_new = self.robustness_measure(G_optimized, attack_type="HDA")
            
            # 判断是否接受交换 (贪婪策略：只接受改进)
            if R_new > R_old + threshold:
                # 接受交换
                R_old = R_new
                recent_improvements.append(R_new - R_old)
            else:
                # 拒绝交换，恢复原始边
                G_optimized.remove_edge(*new_edge1)
                G_optimized.remove_edge(*new_edge2)
                G_optimized.add_edge(u, v)
                G_optimized.add_edge(x, y)
                recent_improvements.append(0)
            
            # 每1000次迭代检查一次收敛情况
            if i - last_check_point >= 1000:
                last_check_point = i
                recent_improvement = sum(recent_improvements[-1000:] if recent_improvements else [0])
                
                # 如果改进非常小，认为已收敛
                if recent_improvement < tolerance * R_old and i > 2000:
                    if verbose:
                        pass
                    break
                    
        if verbose:
            logger.info(f"  Final Robustness: {R_old:.4f}")
            
        return G_optimized
