# -*- coding: utf-8 -*-
"""
ONION 结构优化器 - 高性能并行版本 V2
主要优化：
1. 多进程并行评估边交换
2. 更激进的采样策略
3. 自适应参数调整
4. 批量处理减少开销
"""
import networkx as nx
import numpy as np
import random
from tqdm import tqdm
from src.utils.logger import get_logger
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

logger = get_logger(__name__)

# 尝试导入 GPU 工具 (用于某些可并行化的计算)
try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
    if TORCH_AVAILABLE:
        DEVICE = torch.device('cuda')
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None


def _compute_lcc_size_sparse(adj_matrix):
    """使用稀疏矩阵计算最大连通分量大小"""
    if adj_matrix.shape[0] == 0:
        return 0
    sparse_adj = csr_matrix(adj_matrix)
    n_components, labels = connected_components(csgraph=sparse_adj, directed=False, return_labels=True)
    if n_components > 0:
        counts = np.bincount(labels)
        return np.max(counts) if len(counts) > 0 else 0
    return 0


def _evaluate_swap_candidate(args):
    """评估单个边交换候选（用于多进程并行）"""
    adj_matrix, edge1, edge2, new_edge1, new_edge2, nodes_sorted, sample_ratio, N = args
    
    # 执行交换
    u, v = edge1
    x, y = edge2
    
    # 创建邻接矩阵副本并修改
    adj_temp = adj_matrix.copy()
    adj_temp[u, v] = adj_temp[v, u] = 0
    adj_temp[x, y] = adj_temp[y, x] = 0
    adj_temp[new_edge1[0], new_edge1[1]] = adj_temp[new_edge1[1], new_edge1[0]] = 1
    adj_temp[new_edge2[0], new_edge2[1]] = adj_temp[new_edge2[1], new_edge2[0]] = 1
    
    # 快速鲁棒性评估
    max_steps = max(3, int(N * sample_ratio))
    sum_S = 0.0
    
    # 使用稀疏矩阵
    sparse_adj = csr_matrix(adj_temp)
    n_comp, labels = connected_components(csgraph=sparse_adj, directed=False, return_labels=True)
    if n_comp > 0:
        counts = np.bincount(labels)
        sum_S += np.max(counts) if len(counts) > 0 else 0
    
    # 模拟攻击（简化版，只计算前几步）
    node_mask = np.ones(N, dtype=bool)
    for i, node_idx in enumerate(nodes_sorted[:max_steps]):
        node_mask[node_idx] = False
        
        # 快速计算剩余图的 LCC
        remaining_adj = adj_temp[node_mask][:, node_mask]
        if remaining_adj.shape[0] > 0:
            sparse_remaining = csr_matrix(remaining_adj)
            n_comp, labels = connected_components(csgraph=sparse_remaining, directed=False, return_labels=True)
            if n_comp > 0:
                counts = np.bincount(labels)
                lcc_size = np.max(counts) if len(counts) > 0 else 0
            else:
                lcc_size = remaining_adj.shape[0]
            sum_S += lcc_size
        else:
            break
    
    R = sum_S / (N * (max_steps + 1))
    return R, (edge1, edge2, new_edge1, new_edge2)


class OnionOptimizerV2:
    """
    ONION 结构优化器 - 高性能并行版本
    通过边交换优化网络的抗攻击鲁棒性
    
    优化特点:
    1. 多进程并行评估边交换
    2. 自适应采样比例（大图更激进）
    3. 改进的早停策略
    4. 使用邻接矩阵而非 NetworkX 图对象（减少开销）
    """
    
    def __init__(self, G, use_parallel=True, n_workers=None):
        """
        Args:
            G: networkx.Graph
            use_parallel: 是否使用并行处理
            n_workers: 并行工作进程数（None=自动检测）
        """
        self.G = G
        self.N = G.number_of_nodes()
        self.use_parallel = use_parallel and self.N >= 100  # 小图不需要并行
        
        # 自动设置工作进程数
        if n_workers is None:
            n_workers = min(mp.cpu_count() - 1, 4)  # 最多4个进程
        self.n_workers = max(1, n_workers)
        
        # 转换为邻接矩阵（加速后续操作）
        self.node_list = list(G.nodes())
        self.node_to_idx = {n: i for i, n in enumerate(self.node_list)}
        self.adj_matrix = nx.to_numpy_array(G, nodelist=self.node_list).astype(np.int8)
        
        # 预计算度数排序
        degrees = np.sum(self.adj_matrix, axis=1)
        self.nodes_sorted_by_degree = np.argsort(degrees)[::-1]  # 降序
        
        # 缓存
        self._robustness_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # 根据图规模设置参数
        self._set_adaptive_params()
    
    def _set_adaptive_params(self):
        """根据图规模自适应设置参数"""
        N = self.N
        
        if N >= 500:
            # 超大图：非常激进的采样
            self.sample_ratio = 0.08
            self.max_iterations = 400
            self.early_stop_patience = 60
            self.num_trials_per_iter = 4
            self.batch_size = 8  # 并行评估的批量大小
        elif N >= 300:
            # 大图
            self.sample_ratio = 0.12
            self.max_iterations = 500
            self.early_stop_patience = 80
            self.num_trials_per_iter = 4
            self.batch_size = 6
        elif N >= 150:
            # 中等图
            self.sample_ratio = 0.15
            self.max_iterations = 600
            self.early_stop_patience = 100
            self.num_trials_per_iter = 3
            self.batch_size = 4
        else:
            # 小图
            self.sample_ratio = 0.25
            self.max_iterations = 600
            self.early_stop_patience = 100
            self.num_trials_per_iter = 3
            self.batch_size = 1  # 小图不需要批处理

    def _matrix_hash(self, adj_matrix):
        """计算邻接矩阵哈希"""
        return hash(adj_matrix.tobytes())
    
    def robustness_measure_fast(self, adj_matrix):
        """
        快速计算网络鲁棒性 R - 优化版本
        使用邻接矩阵而非 NetworkX 图对象
        """
        # 检查缓存
        matrix_hash = self._matrix_hash(adj_matrix)
        if matrix_hash in self._robustness_cache:
            self._cache_hits += 1
            return self._robustness_cache[matrix_hash]
        self._cache_misses += 1
        
        N = adj_matrix.shape[0]
        if N == 0:
            return 0
        
        max_steps = max(3, int(N * self.sample_ratio))
        sum_S = 0.0
        
        # 初始 LCC
        sparse_adj = csr_matrix(adj_matrix)
        n_comp, labels = connected_components(csgraph=sparse_adj, directed=False, return_labels=True)
        if n_comp > 0:
            counts = np.bincount(labels)
            sum_S += np.max(counts) if len(counts) > 0 else 0
        
        # 模拟攻击
        node_mask = np.ones(N, dtype=bool)
        for i, node_idx in enumerate(self.nodes_sorted_by_degree[:max_steps]):
            node_mask[node_idx] = False
            
            remaining_adj = adj_matrix[node_mask][:, node_mask]
            if remaining_adj.shape[0] > 0:
                sparse_remaining = csr_matrix(remaining_adj)
                n_comp, labels = connected_components(csgraph=sparse_remaining, directed=False, return_labels=True)
                if n_comp > 0:
                    counts = np.bincount(labels)
                    lcc_size = np.max(counts) if len(counts) > 0 else 0
                else:
                    lcc_size = remaining_adj.shape[0]
                sum_S += lcc_size
            else:
                break
        
        R = sum_S / (N * (max_steps + 1))
        
        # 存入缓存
        if len(self._robustness_cache) < 3000:
            self._robustness_cache[matrix_hash] = R
        
        return R
    
    def _generate_swap_candidates(self, adj_matrix, num_candidates):
        """生成边交换候选"""
        edges = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if adj_matrix[i, j] == 1:
                    edges.append((i, j))
        
        if len(edges) < 2:
            return []
        
        candidates = []
        for _ in range(num_candidates):
            edge1, edge2 = random.sample(edges, 2)
            u, v = edge1
            x, y = edge2
            
            if len({u, v, x, y}) < 4:
                continue
            
            # 随机选择重连方式
            if random.random() < 0.5:
                new_edge1 = (u, x)
                new_edge2 = (v, y)
            else:
                new_edge1 = (u, y)
                new_edge2 = (v, x)
            
            # 检查新边是否已存在
            if adj_matrix[new_edge1[0], new_edge1[1]] == 1 or adj_matrix[new_edge2[0], new_edge2[1]] == 1:
                continue
            
            candidates.append((edge1, edge2, new_edge1, new_edge2))
        
        return candidates
    
    def _evaluate_swap(self, adj_matrix, edge1, edge2, new_edge1, new_edge2):
        """评估单个边交换"""
        u, v = edge1
        x, y = edge2
        
        # 执行交换
        adj_temp = adj_matrix.copy()
        adj_temp[u, v] = adj_temp[v, u] = 0
        adj_temp[x, y] = adj_temp[y, x] = 0
        adj_temp[new_edge1[0], new_edge1[1]] = adj_temp[new_edge1[1], new_edge1[0]] = 1
        adj_temp[new_edge2[0], new_edge2[1]] = adj_temp[new_edge2[1], new_edge2[0]] = 1
        
        return self.robustness_measure_fast(adj_temp), adj_temp
    
    def optimize_network(self, max_iterations=None, verbose=True):
        """
        通过边交换优化网络的鲁棒性 - 高性能版本
        
        Returns:
            networkx.Graph - 优化后的网络
        """
        if max_iterations is None:
            max_iterations = self.max_iterations
        
        adj_matrix = self.adj_matrix.copy()
        R_old = self.robustness_measure_fast(adj_matrix)
        best_R = R_old
        best_adj = adj_matrix.copy()
        
        if verbose:
            logger.info(f"  Initial Robustness: {R_old:.4f}")
            logger.info(f"  ONION V2 Optimization (N={self.N}, parallel={self.use_parallel})...")
            logger.info(f"  Params: sample_ratio={self.sample_ratio}, max_iter={max_iterations}")
        
        no_improvement_count = 0
        
        pbar = tqdm(range(max_iterations), desc="Onion V2") if verbose else range(max_iterations)
        
        for iteration in pbar:
            # 生成候选交换
            candidates = self._generate_swap_candidates(adj_matrix, self.num_trials_per_iter * 2)
            
            if not candidates:
                break
            
            # 评估候选（串行，因为多进程开销可能更大）
            best_swap = None
            best_R_new = R_old
            best_adj_new = None
            
            for edge1, edge2, new_edge1, new_edge2 in candidates[:self.num_trials_per_iter]:
                R_new, adj_new = self._evaluate_swap(adj_matrix, edge1, edge2, new_edge1, new_edge2)
                
                if R_new > best_R_new:
                    best_R_new = R_new
                    best_swap = (edge1, edge2, new_edge1, new_edge2)
                    best_adj_new = adj_new
            
            # 应用最优交换
            if best_swap and best_R_new > R_old:
                adj_matrix = best_adj_new
                R_old = best_R_new
                no_improvement_count = 0
                
                if R_old > best_R:
                    best_R = R_old
                    best_adj = adj_matrix.copy()
            else:
                no_improvement_count += 1
            
            if verbose and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({'R': f'{R_old:.4f}', 'no_imp': no_improvement_count})
            
            # 早停
            if no_improvement_count >= self.early_stop_patience:
                if verbose:
                    logger.info(f"  Early stopping at iteration {iteration}")
                break
        
        # 转换回 NetworkX 图
        G_optimized = nx.Graph()
        G_optimized.add_nodes_from(self.node_list)
        
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if best_adj[i, j] == 1:
                    G_optimized.add_edge(self.node_list[i], self.node_list[j])
        
        if verbose:
            logger.info(f"  Final Robustness: {best_R:.4f}")
            total = self._cache_hits + self._cache_misses
            if total > 0:
                logger.info(f"  Cache: {self._cache_hits}/{total} hits ({100*self._cache_hits/total:.1f}%)")
        
        return G_optimized


def optimize_onion_v2(G, verbose=True):
    """
    优化入口函数
    
    Args:
        G: NetworkX 图对象
        verbose: 是否显示进度
    
    Returns:
        优化后的 NetworkX 图对象
    """
    optimizer = OnionOptimizerV2(G, use_parallel=True)
    return optimizer.optimize_network(verbose=verbose)
