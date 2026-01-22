# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
import random
from tqdm import tqdm
from src.utils.logger import get_logger
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

logger = get_logger(__name__)

class OnionOptimizer:
    """
    ONION 结构优化器 - 高性能版本
    通过边交换优化网络的抗攻击鲁棒性
    """
    def __init__(self, G):
        self.G = G
        self._robustness_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def _graph_hash(self, G):
        """计算图的哈希值用于缓存"""
        return hash(tuple(sorted(G.edges())))

    def robustness_measure_fast(self, G, sample_ratio=None):
        """
        快速计算网络鲁棒性 R (采样版本) - 优化版本
        使用scipy.sparse加速连通分量计算
        
        Parameters:
        G: networkx.Graph
        sample_ratio: float - 采样比例 (None时根据图大小自动调整)
        
        Returns:
        float: 鲁棒性 R 值估计
        """
        # 根据图大小动态调整采样比例
        if sample_ratio is None:
            N = G.number_of_nodes()
            if N > 300:
                sample_ratio = 0.15  # 大图：15%采样
            elif N > 100:
                sample_ratio = 0.2   # 中等图：20%采样
            else:
                sample_ratio = 0.3   # 小图：30%采样（计算快，不需要太多采样）
        # 检查缓存
        graph_hash = self._graph_hash(G)
        if graph_hash in self._robustness_cache:
            self._cache_hits += 1
            return self._robustness_cache[graph_hash]
        self._cache_misses += 1
        
        N = G.number_of_nodes()
        if N == 0:
            return 0
        
        nodes = list(G.nodes())
        
        # 按度数降序排序 (HDA)
        degrees = dict(G.degree())
        nodes_sorted = sorted(nodes, key=lambda n: degrees[n], reverse=True)
        
        # 只采样部分节点进行模拟（降低采样比例以加速）
        max_steps = max(3, int(N * sample_ratio))
        
        sum_S = 0.0
        G_temp = G.copy()
        
        # 初始最大连通分量大小（使用scipy.sparse加速）
        if len(G_temp) > 0:
            sparse_adj = nx.to_scipy_sparse_array(G_temp, format='csr')
            n_components, labels = connected_components(csgraph=sparse_adj, directed=False, return_labels=True)
            if n_components > 0:
                counts = np.bincount(labels)
                largest_cc = np.max(counts) if len(counts) > 0 else 0
            else:
                largest_cc = 0
            sum_S += largest_cc
        
        # 逐步移除节点（只移除采样数量）
        for i, node in enumerate(nodes_sorted[:max_steps]):
            G_temp.remove_node(node)
            if len(G_temp) > 0:
                # 使用scipy.sparse加速连通分量计算
                sparse_adj = nx.to_scipy_sparse_array(G_temp, format='csr')
                n_components, labels = connected_components(csgraph=sparse_adj, directed=False, return_labels=True)
                if n_components > 0:
                    counts = np.bincount(labels)
                    largest_cc = np.max(counts) if len(counts) > 0 else 0
                else:
                    largest_cc = len(G_temp)
                sum_S += largest_cc
            else:
                break
        
        # 归一化
        R = sum_S / (N * (max_steps + 1))
        
        # 存入缓存
        if len(self._robustness_cache) < 5000:
            self._robustness_cache[graph_hash] = R
            
        return R

    def optimize_network(self, max_iterations=600, threshold=0.0, tolerance=0.01, verbose=True):
        """
        通过边交换优化网络的鲁棒性 (ONION Structure) - 高性能版本
        
        优化点:
        1. 使用采样版本的鲁棒性计算（scipy.sparse加速）
        2. 批量尝试多次交换，选择最优
        3. 早停机制
        4. 减少迭代次数
        
        Parameters:
        max_iterations: int - 最大迭代次数 (默认600，进一步优化)
        threshold: float - 接受交换的阈值
        tolerance: float - 收敛容差
        verbose: bool - 是否显示进度
        
        Returns:
        networkx.Graph - 优化后的网络
        """
        G_optimized = self.G.copy()
        node_count = G_optimized.number_of_nodes()
        R_old = self.robustness_measure_fast(G_optimized)
        best_R = R_old
        
        if verbose:
            logger.info(f"  Initial Robustness: {R_old:.4f}")
            logger.info(f"  Starting ONION Structure Optimization (Fast, N={node_count})...")
        
        # 早停参数（根据图大小调整）
        # 大图需要更多迭代才能找到好的解，小图可以早停
        if node_count > 200:
            early_stop_patience = 120  # 大图：120次无改进
        else:
            early_stop_patience = 80   # 小图：80次无改进
        no_improvement_count = 0
        
        # 预计算边列表
        edges = list(G_optimized.edges())
        
        # 使用 tqdm 显示进度条
        pbar = tqdm(range(max_iterations), desc="Onion Opt") if verbose else range(max_iterations)
            
        for i in pbar:
            edges = list(G_optimized.edges())
            if len(edges) < 2:
                break
            
            # 批量尝试多次交换，选择最优的
            best_swap = None
            best_R_new = R_old
            
            # 根据图大小动态调整尝试次数：大图减少尝试次数以加速
            num_trials = 2 if node_count > 200 else 3
            # 尝试随机交换
            for _ in range(num_trials):
                edge1, edge2 = random.sample(edges, 2)
                u, v = edge1
                x, y = edge2
                
                # 确保涉及4个不同的节点
                if len({u, v, x, y}) < 4:
                    continue
                
                # 随机选择重连方式
                if random.random() < 0.5:
                    new_edge1 = (u, x)
                    new_edge2 = (v, y)
                else:
                    new_edge1 = (u, y)
                    new_edge2 = (v, x)
                
                # 确保新边不会形成重复边
                if G_optimized.has_edge(*new_edge1) or G_optimized.has_edge(*new_edge2):
                    continue
                
                # 临时执行交换
                G_optimized.remove_edge(u, v)
                G_optimized.remove_edge(x, y)
                G_optimized.add_edge(*new_edge1)
                G_optimized.add_edge(*new_edge2)
                
                # 计算新的鲁棒性
                R_new = self.robustness_measure_fast(G_optimized)
                
                if R_new > best_R_new:
                    best_R_new = R_new
                    best_swap = (edge1, edge2, new_edge1, new_edge2)
                
                # 恢复原状
                G_optimized.remove_edge(*new_edge1)
                G_optimized.remove_edge(*new_edge2)
                G_optimized.add_edge(u, v)
                G_optimized.add_edge(x, y)
            
            # 应用最优交换
            if best_swap and best_R_new > R_old + threshold:
                edge1, edge2, new_edge1, new_edge2 = best_swap
                G_optimized.remove_edge(*edge1)
                G_optimized.remove_edge(*edge2)
                G_optimized.add_edge(*new_edge1)
                G_optimized.add_edge(*new_edge2)
                R_old = best_R_new
                no_improvement_count = 0
                
                if R_old > best_R:
                    best_R = R_old
            else:
                no_improvement_count += 1
            
            # 更新进度条
            if verbose and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({'R': f'{R_old:.4f}', 'no_imp': no_improvement_count})
            
            # 早停检查
            if no_improvement_count >= early_stop_patience:
                if verbose:
                    logger.info(f"  Early stopping at iteration {i}")
                break
                    
        if verbose:
            logger.info(f"  Final Robustness: {R_old:.4f}")
            total = self._cache_hits + self._cache_misses
            if total > 0:
                logger.info(f"  Cache: {self._cache_hits}/{total} hits ({100*self._cache_hits/total:.1f}%)")
            
        return G_optimized
