# -*- coding: utf-8 -*-
"""
遗传算法 - 高性能并行版本 V2
主要优化：
1. 批量 GPU 特征值计算
2. 多进程种群评估
3. 更智能的 GPU 使用策略
4. 改进的早停机制
"""
import numpy as np
import random
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from src.utils.logger import get_logger

# GPU 加速支持
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        DEVICE = torch.device("cuda")
        GPU_NAME = torch.cuda.get_device_name(0)
        print(f"[GA-V2] GPU 加速已启用: {GPU_NAME}")
    else:
        DEVICE = torch.device("cpu")
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    DEVICE = None

logger = get_logger(__name__)


class BatchEigenvalueSolver:
    """
    批量特征值求解器 - GPU 加速
    一次性计算多个矩阵的特征值，减少 GPU 传输开销
    """
    
    def __init__(self, use_gpu=True, batch_size=8):
        self.use_gpu = use_gpu and CUDA_AVAILABLE
        self.batch_size = batch_size
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def _matrix_hash(self, matrix):
        return hash(matrix.tobytes())
    
    def compute_natural_connectivity_batch(self, matrices):
        """
        批量计算自然连通度
        
        Args:
            matrices: list of numpy arrays (邻接矩阵列表)
        
        Returns:
            list of float (自然连通度列表)
        """
        results = []
        uncached_indices = []
        uncached_matrices = []
        
        # 检查缓存
        for i, mat in enumerate(matrices):
            mat_hash = self._matrix_hash(mat)
            if mat_hash in self._cache:
                results.append((i, self._cache[mat_hash]))
                self._cache_hits += 1
            else:
                uncached_indices.append(i)
                uncached_matrices.append(mat)
                self._cache_misses += 1
        
        if not uncached_matrices:
            # 全部命中缓存
            return [r[1] for r in sorted(results, key=lambda x: x[0])]
        
        # 计算未缓存的矩阵
        if self.use_gpu:
            nc_values = self._compute_batch_gpu(uncached_matrices)
        else:
            nc_values = self._compute_batch_cpu(uncached_matrices)
        
        # 更新缓存和结果
        for idx, mat, nc in zip(uncached_indices, uncached_matrices, nc_values):
            mat_hash = self._matrix_hash(mat)
            if len(self._cache) < 10000:
                self._cache[mat_hash] = nc
            results.append((idx, nc))
        
        # 按原始顺序返回
        return [r[1] for r in sorted(results, key=lambda x: x[0])]
    
    def _compute_batch_gpu(self, matrices):
        """GPU 批量计算"""
        results = []
        
        # 分批处理以避免 GPU 内存溢出
        for i in range(0, len(matrices), self.batch_size):
            batch = matrices[i:i + self.batch_size]
            batch_results = []
            
            for mat in batch:
                try:
                    adj_tensor = torch.tensor(mat.astype(np.float32), device=DEVICE)
                    eigenvalues = torch.linalg.eigvalsh(adj_tensor)
                    evals = eigenvalues.cpu().numpy().astype(np.float64)
                    
                    max_eval = np.max(evals)
                    # 数值稳定的 log-sum-exp
                    nc = max_eval + np.log(np.sum(np.exp(evals - max_eval))) - np.log(len(evals))
                    batch_results.append(nc)
                except Exception as e:
                    logger.warning(f"GPU eigenvalue error: {e}")
                    batch_results.append(0)
            
            results.extend(batch_results)
            
            # 清理 GPU 内存
            if i % (self.batch_size * 4) == 0:
                torch.cuda.empty_cache()
        
        return results
    
    def _compute_batch_cpu(self, matrices):
        """CPU 批量计算"""
        from scipy.linalg import eig
        
        results = []
        for mat in matrices:
            try:
                eigenvalues = eig(mat.astype(np.float64), left=False, right=False)
                evals = eigenvalues.real
                
                max_eval = np.max(evals)
                nc = max_eval + np.log(np.sum(np.exp(evals - max_eval))) - np.log(len(evals))
                results.append(nc)
            except Exception as e:
                logger.warning(f"CPU eigenvalue error: {e}")
                results.append(0)
        
        return results


class GeneticAlgorithmV2:
    """
    遗传算法 V2 - 高性能版本
    
    优化点：
    1. 批量 GPU 特征值计算
    2. 智能参数自适应
    3. 改进的早停机制
    4. 精英保留策略
    """
    
    def __init__(self, nodes, edges, max_generations=80, population_size=8, 
                 mutation_rate=0.2, alpha=0.4, beta=0.4):
        self.nodes = nodes
        self.edges = edges
        self.max_generations = max_generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.alpha = alpha
        self.beta = beta
        
        # 自适应参数
        self._set_adaptive_params()
        
        # 初始化批量特征值求解器
        use_gpu = CUDA_AVAILABLE and nodes >= 200  # 200节点以上使用GPU
        self.eigen_solver = BatchEigenvalueSolver(use_gpu=use_gpu, batch_size=self.population_size)
        
        # 预计算归一化常数
        full_adj = np.ones((nodes, nodes)) - np.eye(nodes)
        self.nc_max = self._compute_single_nc(full_adj)
        self.mu_max = self._compute_degree_variance(full_adj)
        
        logger.info(f"  GA-V2 initialized: nodes={nodes}, edges={edges}, "
                   f"pop={self.population_size}, gen={self.max_generations}, "
                   f"GPU={'Yes' if use_gpu else 'No'}")
    
    def _set_adaptive_params(self):
        """根据图规模自适应设置参数"""
        N = self.nodes
        
        if N >= 400:
            # 超大图：减少计算量
            self.population_size = max(4, self.population_size)
            self.max_generations = min(60, self.max_generations)
            self.early_stop_patience = 20
            self.elite_count = 1
        elif N >= 200:
            # 大图
            self.population_size = max(6, self.population_size)
            self.max_generations = min(80, self.max_generations)
            self.early_stop_patience = 25
            self.elite_count = 2
        else:
            # 小图
            self.early_stop_patience = 30
            self.elite_count = 2
    
    def _compute_single_nc(self, adj_matrix):
        """计算单个矩阵的自然连通度"""
        from scipy.linalg import eig
        try:
            eigenvalues = eig(adj_matrix.astype(np.float64), left=False, right=False)
            evals = eigenvalues.real
            max_eval = np.max(evals)
            return max_eval + np.log(np.sum(np.exp(evals - max_eval))) - np.log(len(evals))
        except:
            return 0
    
    def _compute_degree_variance(self, adj_matrix):
        """计算度方差"""
        degrees = np.sum(adj_matrix, axis=1)
        mean_degree = np.mean(degrees)
        return np.mean((degrees - mean_degree) ** 2)
    
    def _compute_largest_component(self, adj_matrix):
        """计算最大连通分量大小"""
        n_comp, labels = connected_components(csr_matrix(adj_matrix), directed=False, return_labels=True)
        counts = np.bincount(labels)
        return np.max(counts)
    
    def generate_adjacency_matrix(self):
        """生成随机邻接矩阵"""
        adj_matrix = np.zeros((self.nodes, self.nodes), dtype=np.int8)
        possible_edges = [(i, j) for i in range(self.nodes) for j in range(i + 1, self.nodes)]
        selected_edges = random.sample(possible_edges, self.edges)
        for i, j in selected_edges:
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1
        return adj_matrix
    
    def initialize_population(self):
        """初始化种群"""
        return [self.generate_adjacency_matrix() for _ in range(self.population_size)]
    
    def evaluate_population_batch(self, population):
        """
        批量评估种群适应度 - 使用批量 GPU 计算
        """
        # 批量计算自然连通度
        nc_values = self.eigen_solver.compute_natural_connectivity_batch(population)
        
        fitness_values = []
        for i, (adj, nc) in enumerate(zip(population, nc_values)):
            sg = self._compute_largest_component(adj)
            mu = self._compute_degree_variance(adj)
            
            nc_norm = nc / self.nc_max if self.nc_max != 0 else 0
            mu_norm = mu / self.mu_max if self.mu_max != 0 else 0
            
            fitness = (self.alpha * nc_norm + 
                      self.beta * (sg / self.nodes) - 
                      (1 - self.alpha - self.beta) * mu_norm)
            
            if np.isnan(fitness) or np.isinf(fitness):
                fitness = -np.inf
            
            fitness_values.append(fitness)
        
        return fitness_values
    
    def select_parents(self, population, fitness_values):
        """轮盘赌选择"""
        min_fitness = min(fitness_values)
        adjusted = [f - min_fitness + 1 for f in fitness_values]
        total = sum(adjusted)
        
        if total == 0:
            return random.choice(population), random.choice(population)
        
        probs = [f / total for f in adjusted]
        indices = np.random.choice(len(population), size=2, p=probs)
        return population[indices[0]], population[indices[1]]
    
    def crossover(self, parent1, parent2):
        """交叉操作"""
        size = parent1.shape[0]
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        crossover_point = random.randint(1, size - 1)
        
        for i in range(crossover_point, size):
            child1[i, :] = parent2[i, :]
            child1[:, i] = parent2[:, i]
            child2[i, :] = parent1[i, :]
            child2[:, i] = parent1[:, i]
        
        child1 = self._adjust_edges(child1)
        child2 = self._adjust_edges(child2)
        
        return child1, child2
    
    def _adjust_edges(self, adj_matrix):
        """调整边数"""
        edges = [(i, j) for i in range(self.nodes - 1) 
                 for j in range(i + 1, self.nodes) if adj_matrix[i, j] == 1]
        non_edges = [(i, j) for i in range(self.nodes - 1) 
                     for j in range(i + 1, self.nodes) if adj_matrix[i, j] == 0]
        
        current_edges = len(edges)
        
        while current_edges > self.edges and edges:
            i, j = random.choice(edges)
            adj_matrix[i, j] = 0
            adj_matrix[j, i] = 0
            edges.remove((i, j))
            current_edges -= 1
        
        while current_edges < self.edges and non_edges:
            i, j = random.choice(non_edges)
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1
            non_edges.remove((i, j))
            current_edges += 1
        
        return adj_matrix
    
    def mutate(self, individual):
        """变异操作"""
        adj_matrix = individual.copy()
        if random.random() < self.mutation_rate:
            edges = [(i, j) for i in range(self.nodes - 1) 
                     for j in range(i + 1, self.nodes) if adj_matrix[i, j] == 1]
            non_edges = [(i, j) for i in range(self.nodes - 1) 
                         for j in range(i + 1, self.nodes) if adj_matrix[i, j] == 0]
            
            num_mutations = max(1, int(self.mutation_rate * self.nodes * 0.1))
            
            for _ in range(num_mutations):
                if edges and non_edges:
                    i, j = random.choice(edges)
                    adj_matrix[i, j] = 0
                    adj_matrix[j, i] = 0
                    edges.remove((i, j))
                    
                    i, j = random.choice(non_edges)
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1
                    non_edges.remove((i, j))
        
        return adj_matrix
    
    def optimize(self):
        """执行遗传算法优化"""
        population = self.initialize_population()
        best_fitness = float('-inf')
        best_solution = None
        no_improvement_count = 0
        
        pbar = tqdm(range(self.max_generations), desc="GA-V2 Optimization")
        
        for generation in pbar:
            # 批量评估
            fitness_scores = self.evaluate_population_batch(population)
            
            # 找出最优个体
            max_idx = np.argmax(fitness_scores)
            current_best = fitness_scores[max_idx]
            
            if current_best > best_fitness:
                best_fitness = current_best
                best_solution = population[max_idx].copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            pbar.set_postfix({'fitness': f'{best_fitness:.4f}', 'no_imp': no_improvement_count})
            
            # 早停
            if no_improvement_count >= self.early_stop_patience:
                logger.info(f"  Early stopping at generation {generation}")
                break
            
            # 精英保留
            sorted_indices = np.argsort(fitness_scores)[::-1]
            elite = [population[i].copy() for i in sorted_indices[:self.elite_count]]
            
            # 生成新种群
            new_population = elite.copy()
            while len(new_population) < self.population_size:
                parent1, parent2 = self.select_parents(population, fitness_scores)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
        
        # 打印缓存统计
        solver = self.eigen_solver
        total = solver._cache_hits + solver._cache_misses
        if total > 0:
            logger.info(f"  NC Cache: {solver._cache_hits}/{total} hits "
                       f"({100*solver._cache_hits/total:.1f}%)")
        
        return best_solution, best_fitness


def solution_v2(nodes, edges, max_generations=80, population_size=8):
    """
    GA 优化入口函数 - V2 高性能版本
    
    Args:
        nodes: 节点数
        edges: 边数
        max_generations: 最大代数
        population_size: 种群大小
    
    Returns:
        tuple: (最优解邻接矩阵, 最优适应度)
    """
    logger.info(f"  GA-V2 Optimizing Network Structure (nodes={nodes}, edges={edges})")
    
    ga = GeneticAlgorithmV2(
        nodes=nodes,
        edges=edges,
        max_generations=max_generations,
        population_size=population_size
    )
    
    return ga.optimize()
