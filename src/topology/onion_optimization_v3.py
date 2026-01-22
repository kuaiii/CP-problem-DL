# -*- coding: utf-8 -*-
"""
ONION 结构优化器 - 超高性能版本 V3

核心优化:
1. 【颠覆性优化】逆向重组法 + Union-Find: 复杂度从 O(K(N+M)) 降至 O(M·α(N))
2. 【移除哈希缓存】缓存命中率极低，计算哈希开销大于收益
3. 【度数排序一次性计算】边交换是度保持操作，攻击序列永远不变

理论分析:
- 原始方法: 每次移除节点后调用 connected_components，复杂度 O(K(N+M))
- 逆向重组法: 从空图开始按逆序添加节点和边，使用并查集维护 LCC
- 并查集单次操作复杂度: O(α(N)) ≈ O(1)（反阿克曼函数）
- 总复杂度: O(M·α(N)) ≈ O(M)

作者: 优化版本
"""
import networkx as nx
import numpy as np
import random
from tqdm import tqdm
from src.utils.logger import get_logger

logger = get_logger(__name__)


class UnionFind:
    """
    并查集 (Disjoint Set Union) 数据结构 - 支持增量节点添加
    
    支持:
    - 路径压缩 (Path Compression)
    - 按秩合并 (Union by Rank)
    - 动态维护各连通分量大小
    - 追踪最大连通分量大小
    - 增量添加节点（用于逆向重组法）
    
    时间复杂度:
    - find: O(α(N)) 均摊，其中 α 是反阿克曼函数，实际上接近 O(1)
    - union: O(α(N)) 均摊
    """
    
    __slots__ = ['parent', 'rank', 'size', 'max_component_size', 'num_components', 'active']
    
    def __init__(self, n: int, initially_active: set = None):
        """
        初始化并查集
        
        Args:
            n: 节点总数
            initially_active: 初始活跃的节点集合（默认为空，所有节点都不活跃）
        """
        self.parent = list(range(n))  # 每个节点初始时是自己的父节点
        self.rank = [0] * n           # 用于按秩合并
        self.size = [0] * n           # 每个连通分量的大小（未激活节点为0）
        self.max_component_size = 0   # 最大连通分量大小
        self.num_components = 0       # 活跃连通分量数量
        self.active = [False] * n     # 节点是否活跃
        
        # 激活初始节点
        if initially_active:
            for node in initially_active:
                self.activate_node(node)
    
    def activate_node(self, x: int):
        """
        激活一个节点（使其成为独立的连通分量）
        
        Args:
            x: 节点索引
        """
        if not self.active[x]:
            self.active[x] = True
            self.size[x] = 1
            self.parent[x] = x  # 重置为自己
            self.rank[x] = 0
            self.num_components += 1
            if self.max_component_size < 1:
                self.max_component_size = 1
    
    def find(self, x: int) -> int:
        """
        查找节点 x 所属的连通分量的根节点（带路径压缩）
        
        Args:
            x: 节点索引
            
        Returns:
            根节点索引
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 路径压缩
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        """
        合并节点 x 和 y 所在的连通分量
        
        Args:
            x, y: 要合并的两个节点（必须都是活跃的）
            
        Returns:
            bool: 是否真正发生了合并（如果已在同一分量则返回 False）
        """
        if not self.active[x] or not self.active[y]:
            return False
        
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False  # 已经在同一连通分量
        
        # 按秩合并：将较小的树挂到较大的树下
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
        
        self.parent[root_y] = root_x
        self.size[root_x] += self.size[root_y]
        self.size[root_y] = 0  # 清空被合并节点的大小
        
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1
        
        # 更新最大连通分量大小
        if self.size[root_x] > self.max_component_size:
            self.max_component_size = self.size[root_x]
        
        self.num_components -= 1
        return True
    
    def get_component_size(self, x: int) -> int:
        """获取节点 x 所在连通分量的大小"""
        if not self.active[x]:
            return 0
        return self.size[self.find(x)]
    
    def get_max_component_size(self) -> int:
        """获取最大连通分量的大小"""
        return self.max_component_size
    
    def recalculate_max_component_size(self):
        """重新计算最大连通分量大小（在节点激活后可能需要）"""
        self.max_component_size = 0
        for i, active in enumerate(self.active):
            if active and self.parent[i] == i:  # 是根节点
                if self.size[i] > self.max_component_size:
                    self.max_component_size = self.size[i]


class OnionOptimizerV3:
    """
    ONION 结构优化器 - 超高性能版本 V3
    
    核心优化:
    1. 逆向重组法 + Union-Find 计算鲁棒性
    2. 移除无效的哈希缓存
    3. 攻击序列只计算一次（度保持性质）
    
    性能提升:
    - 鲁棒性计算: 从 O(K(N+M)) 降至 O(M·α(N)) ≈ O(M)
    - 消除缓存计算开销: 节省 O(M log M) 的哈希计算
    - 消除重复排序: 节省每次调用的 O(N log N) 排序
    """
    
    def __init__(self, G: nx.Graph):
        """
        初始化优化器
        
        Args:
            G: NetworkX 图对象
        """
        self.G = G
        self.N = G.number_of_nodes()
        
        # 建立节点索引映射（支持任意节点标签）
        self.node_list = list(G.nodes())
        self.node_to_idx = {node: idx for idx, node in enumerate(self.node_list)}
        self.idx_to_node = {idx: node for idx, node in enumerate(self.node_list)}
        
        # 【优化3】预计算攻击序列（按度数降序）
        # 边交换是度保持操作，攻击序列永远不变
        degrees = dict(G.degree())
        self.attack_sequence = sorted(
            range(self.N),
            key=lambda idx: degrees[self.node_list[idx]],
            reverse=True  # 高度数优先被攻击
        )
        
        # 预计算初始边集合（使用索引）
        self._update_edge_index_set(G)
        
        # 自适应参数设置
        self._set_adaptive_params()
    
    def _update_edge_index_set(self, G: nx.Graph):
        """更新边的索引集合"""
        self.edge_set = set()
        for u, v in G.edges():
            idx_u = self.node_to_idx[u]
            idx_v = self.node_to_idx[v]
            # 使用有序元组确保唯一性
            self.edge_set.add((min(idx_u, idx_v), max(idx_u, idx_v)))
        self.edge_list = list(self.edge_set)
    
    def _set_adaptive_params(self):
        """根据图规模自适应设置参数"""
        N = self.N
        
        if N >= 500:
            # 超大图
            self.sample_ratio = 0.10
            self.max_iterations = 500
            self.early_stop_patience = 80
            self.num_trials = 3
        elif N >= 300:
            # 大图
            self.sample_ratio = 0.15
            self.max_iterations = 600
            self.early_stop_patience = 100
            self.num_trials = 3
        elif N >= 150:
            # 中等图
            self.sample_ratio = 0.20
            self.max_iterations = 700
            self.early_stop_patience = 100
            self.num_trials = 3
        else:
            # 小图：可以更精确计算
            self.sample_ratio = 0.30
            self.max_iterations = 800
            self.early_stop_patience = 100
            self.num_trials = 4
    
    def robustness_measure_reverse_union_find(self, edge_set: set) -> float:
        """
        【核心优化】使用逆向重组法 + Union-Find 计算鲁棒性
        
        原理:
        - HDA攻击顺序: 高度数节点 → 低度数节点
        - 逆向模拟: 从空图开始，按 "低度数 → 高度数" 顺序添加节点
        - 使用并查集动态维护最大连通分量大小
        
        复杂度分析:
        - 原始方法（每步调用 connected_components）: O(K × (N + M))
        - 本方法: O(N + M × α(N)) ≈ O(N + M)
        
        Args:
            edge_set: 边的集合，每条边为 (min_idx, max_idx) 元组
            
        Returns:
            float: 鲁棒性 R 值 (归一化到 0-1)
        """
        N = self.N
        if N == 0:
            return 0.0
        
        # 计算采样步数
        K = max(3, int(N * self.sample_ratio))
        
        # 攻击序列（按度数降序，即先攻击高度数节点）
        # 我们只关心前 K 个被攻击的节点
        attack_order = self.attack_sequence[:K]
        attack_set = set(attack_order)
        
        # 逆向添加序列：从最后被攻击的开始添加
        reverse_order = list(reversed(attack_order))
        
        # 还有一些节点从未被攻击（第 K 到 N-1 个），它们始终存在
        surviving_nodes = set(range(N)) - attack_set
        
        # 预处理每个节点的邻接边（基于索引）
        adj_list = [[] for _ in range(N)]
        for u, v in edge_set:
            adj_list[u].append(v)
            adj_list[v].append(u)
        
        # === 阶段1: 初始化并查集，激活 surviving_nodes ===
        uf = UnionFind(N, initially_active=surviving_nodes)
        
        # 建立 surviving_nodes 之间的连接
        for node in surviving_nodes:
            for neighbor in adj_list[node]:
                if neighbor in surviving_nodes:
                    uf.union(node, neighbor)
        
        # 记录每一步的 LCC 大小（逆向，从攻击结束时开始）
        # lcc_sizes[i] = 逆向添加了 i 个节点后的 LCC 大小
        lcc_sizes = [0] * (K + 1)
        
        # 初始状态（攻击完成后，只剩 surviving_nodes）
        lcc_sizes[0] = uf.get_max_component_size()
        
        # === 阶段2: 逆向添加节点，记录每步的 LCC ===
        for step, node in enumerate(reverse_order, start=1):
            # 激活这个节点
            uf.activate_node(node)
            
            # 将它与所有已激活的邻居合并
            for neighbor in adj_list[node]:
                if uf.active[neighbor]:
                    uf.union(node, neighbor)
            
            # 记录当前 LCC 大小
            lcc_sizes[step] = uf.get_max_component_size()
        
        # === 阶段3: 转换为正向视角并计算鲁棒性 ===
        # lcc_sizes[i] = 逆向添加了 i 个节点后的 LCC
        # 正向视角:
        # - 正向第 0 步（未攻击）= 逆向第 K 步 = lcc_sizes[K]
        # - 正向第 1 步（移除1个）= 逆向第 K-1 步 = lcc_sizes[K-1]
        # - ...
        # - 正向第 K 步（移除K个）= 逆向第 0 步 = lcc_sizes[0]
        
        # 计算鲁棒性 R = sum(LCC_i) / (N × (K+1))
        sum_lcc = sum(lcc_sizes)  # lcc_sizes[0] + ... + lcc_sizes[K]
        
        R = sum_lcc / (N * (K + 1))
        return R
    
    def _try_swap(self, edge_set: set, edge1: tuple, edge2: tuple) -> tuple:
        """
        尝试一次边交换并计算新的鲁棒性
        
        边交换: 移除 (u,v) 和 (x,y)，添加 (u,x) 和 (v,y) 或 (u,y) 和 (v,x)
        
        Args:
            edge_set: 当前边集合
            edge1, edge2: 要交换的两条边
            
        Returns:
            (R_new, new_edge_set, swap_info) 或 (None, None, None) 如果交换无效
        """
        u, v = edge1
        x, y = edge2
        
        # 确保 4 个不同的节点
        if len({u, v, x, y}) < 4:
            return None, None, None
        
        # 随机选择重连方式
        if random.random() < 0.5:
            new_edge1 = (min(u, x), max(u, x))
            new_edge2 = (min(v, y), max(v, y))
        else:
            new_edge1 = (min(u, y), max(u, y))
            new_edge2 = (min(v, x), max(v, x))
        
        # 检查新边是否已存在
        if new_edge1 in edge_set or new_edge2 in edge_set:
            return None, None, None
        
        # 创建新的边集合
        new_edge_set = edge_set.copy()
        new_edge_set.discard(edge1)
        new_edge_set.discard(edge2)
        new_edge_set.add(new_edge1)
        new_edge_set.add(new_edge2)
        
        # 计算新的鲁棒性
        R_new = self.robustness_measure_reverse_union_find(new_edge_set)
        
        return R_new, new_edge_set, (edge1, edge2, new_edge1, new_edge2)
    
    def optimize_network(self, max_iterations: int = None, threshold: float = 0.0, 
                        verbose: bool = True) -> nx.Graph:
        """
        通过边交换优化网络的鲁棒性
        
        Args:
            max_iterations: 最大迭代次数（None 使用自适应值）
            threshold: 接受交换的最小改善阈值
            verbose: 是否显示进度
            
        Returns:
            nx.Graph: 优化后的网络
        """
        if max_iterations is None:
            max_iterations = self.max_iterations
        
        # 使用边集合进行优化（比邻接矩阵更高效）
        current_edge_set = self.edge_set.copy()
        current_edge_list = list(current_edge_set)
        
        # 计算初始鲁棒性
        R_old = self.robustness_measure_reverse_union_find(current_edge_set)
        best_R = R_old
        best_edge_set = current_edge_set.copy()
        
        if verbose:
            logger.info(f"  Initial Robustness: {R_old:.4f}")
            logger.info(f"  ONION V3 Optimization (N={self.N}, M={len(current_edge_set)})...")
            logger.info(f"  Using Reverse Union-Find method (complexity: O(M·α(N)) per evaluation)")
        
        no_improvement_count = 0
        total_evaluations = 0
        
        pbar = tqdm(range(max_iterations), desc="Onion V3") if verbose else range(max_iterations)
        
        for iteration in pbar:
            if len(current_edge_list) < 2:
                break
            
            # 尝试多次边交换
            best_swap_result = None
            best_R_new = R_old
            best_new_edge_set = None
            
            for _ in range(self.num_trials):
                # 随机选择两条边
                edge1, edge2 = random.sample(current_edge_list, 2)
                
                R_new, new_edge_set, swap_info = self._try_swap(
                    current_edge_set, edge1, edge2
                )
                total_evaluations += 1
                
                if R_new is not None and R_new > best_R_new:
                    best_R_new = R_new
                    best_new_edge_set = new_edge_set
                    best_swap_result = swap_info
            
            # 应用最优交换
            if best_swap_result and best_R_new > R_old + threshold:
                current_edge_set = best_new_edge_set
                current_edge_list = list(current_edge_set)
                R_old = best_R_new
                no_improvement_count = 0
                
                if R_old > best_R:
                    best_R = R_old
                    best_edge_set = current_edge_set.copy()
            else:
                no_improvement_count += 1
            
            # 更新进度条
            if verbose and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({
                    'R': f'{R_old:.4f}',
                    'best': f'{best_R:.4f}',
                    'no_imp': no_improvement_count
                })
            
            # 早停检查
            if no_improvement_count >= self.early_stop_patience:
                if verbose:
                    logger.info(f"  Early stopping at iteration {iteration}")
                break
        
        # 从边集合构建优化后的图
        G_optimized = nx.Graph()
        G_optimized.add_nodes_from(self.node_list)
        for idx_u, idx_v in best_edge_set:
            G_optimized.add_edge(self.idx_to_node[idx_u], self.idx_to_node[idx_v])
        
        if verbose:
            logger.info(f"  Final Robustness: {best_R:.4f}")
            logger.info(f"  Improvement: {best_R - self.robustness_measure_reverse_union_find(self.edge_set):.4f}")
            logger.info(f"  Total evaluations: {total_evaluations}")
        
        return G_optimized


def optimize_onion_v3(G: nx.Graph, verbose: bool = True) -> nx.Graph:
    """
    ONION V3 优化入口函数
    
    Args:
        G: NetworkX 图对象
        verbose: 是否显示进度
        
    Returns:
        优化后的 NetworkX 图对象
    """
    optimizer = OnionOptimizerV3(G)
    return optimizer.optimize_network(verbose=verbose)


# ============ 性能对比测试工具 ============

def benchmark_robustness_methods(G: nx.Graph, num_runs: int = 5, sample_ratio: float = 0.25):
    """
    对比不同鲁棒性计算方法的性能
    
    Args:
        G: 测试图
        num_runs: 运行次数
        sample_ratio: 采样比例（确保两个方法使用相同比例以公平比较）
    """
    import time
    from scipy.sparse.csgraph import connected_components
    
    N = G.number_of_nodes()
    M = G.number_of_edges()
    K = max(3, int(N * sample_ratio))
    
    print(f"\n=== 鲁棒性计算方法性能对比 ===")
    print(f"图规模: N={N}, M={M}")
    print(f"采样比例: {sample_ratio}, 采样步数 K={K}")
    print(f"运行次数: {num_runs}")
    print()
    
    # 预计算攻击序列
    degrees = dict(G.degree())
    nodes_sorted = sorted(G.nodes(), key=lambda n: degrees[n], reverse=True)
    
    # 方法1: 原始方法（每步调用 connected_components）
    def method_original():
        G_temp = G.copy()
        sum_S = 0.0
        
        # 初始 LCC
        if len(G_temp) > 0:
            sparse_adj = nx.to_scipy_sparse_array(G_temp, format='csr')
            n_comp, labels = connected_components(csgraph=sparse_adj, directed=False, return_labels=True)
            if n_comp > 0:
                counts = np.bincount(labels)
                sum_S += np.max(counts)
        
        # 逐步移除节点
        for node in nodes_sorted[:K]:
            G_temp.remove_node(node)
            if len(G_temp) > 0:
                sparse_adj = nx.to_scipy_sparse_array(G_temp, format='csr')
                n_comp, labels = connected_components(csgraph=sparse_adj, directed=False, return_labels=True)
                if n_comp > 0:
                    counts = np.bincount(labels)
                    sum_S += np.max(counts)
        
        return sum_S / (N * (K + 1))
    
    # 方法2: V3 逆向重组法（使用相同的采样比例）
    optimizer = OnionOptimizerV3(G)
    optimizer.sample_ratio = sample_ratio  # 确保相同采样比例
    
    def method_v3():
        return optimizer.robustness_measure_reverse_union_find(optimizer.edge_set)
    
    # 测试方法1
    times_original = []
    result_original = None
    for _ in range(num_runs):
        start = time.perf_counter()
        result_original = method_original()
        times_original.append(time.perf_counter() - start)
    
    # 测试方法2
    times_v3 = []
    result_v3 = None
    for _ in range(num_runs):
        start = time.perf_counter()
        result_v3 = method_v3()
        times_v3.append(time.perf_counter() - start)
    
    avg_original = np.mean(times_original) * 1000
    avg_v3 = np.mean(times_v3) * 1000
    speedup = avg_original / avg_v3 if avg_v3 > 0 else float('inf')
    
    print(f"方法1 (原始 connected_components):")
    print(f"  平均时间: {avg_original:.2f} ms")
    print(f"  结果: R = {result_original:.4f}")
    print()
    print(f"方法2 (V3 逆向重组 Union-Find):")
    print(f"  平均时间: {avg_v3:.2f} ms")
    print(f"  结果: R = {result_v3:.4f}")
    print()
    print(f"加速比: {speedup:.1f}x")
    print(f"结果差异: {abs(result_original - result_v3):.6f}")


if __name__ == "__main__":
    # 测试代码
    print("Testing OnionOptimizerV3...")
    
    # 创建测试图
    G = nx.barabasi_albert_graph(100, 3, seed=42)
    print(f"Test graph: N={G.number_of_nodes()}, M={G.number_of_edges()}")
    
    # 运行性能对比
    benchmark_robustness_methods(G, num_runs=10)
    
    # 测试优化
    print("\n=== 测试 ONION V3 优化 ===")
    G_opt = optimize_onion_v3(G, verbose=True)
    print(f"Optimized graph: N={G_opt.number_of_nodes()}, M={G_opt.number_of_edges()}")
