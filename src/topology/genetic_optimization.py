# -*- coding: utf-8 -*-
import numpy as np
import random
from scipy.sparse import csr_matrix
from scipy.linalg import eig
import logging
from src.utils.logger import get_logger

# GPU 加速支持
try:
    import torch
    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"[GA] GPU 加速已启用: {torch.cuda.get_device_name(0)}")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None

logger = get_logger(__name__)

class GeneticAlgorithm:
    """
    遗传算法实现，用于网络结构优化。
    """
    def __init__(self, population_size, evaluator, max_generations, mutation_rate, nodes, edges):
        self.population_size = population_size
        self.evaluator = evaluator
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.nodes = nodes
        self.edges = edges
        
    def generate_adjacency_matrix(self):
        """
        生成随机邻接矩阵。
        """
        adj_matrix = np.zeros((self.nodes, self.nodes), dtype=int)
        possible_edges = [(i, j) for i in range(self.nodes) for j in range(i + 1, self.nodes)]
        selected_edges = random.sample(possible_edges, self.edges)
        for i, j in selected_edges:
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1
        return adj_matrix
    
    def initialize_population(self):
        """
        初始化种群。
        """
        population = []
        for _ in range(self.population_size):
            adj_matrix = self.generate_adjacency_matrix()
            population.append(adj_matrix)
        return population
    
    def evaluate_population(self, population, alpha=0.3, beta=0.3):
        """
        评估种群中个体的适应度 - 优化版本。
        """
        # 串行评估（已足够快，批量GPU评估需要更多实现）
        fitness_values = [self.evaluator.evaluate_network(individual) for individual in population]
        # 处理无效的适应度值
        fitness_values = [f if not (np.isnan(f) or np.isinf(f)) else -np.inf for f in fitness_values]
        return fitness_values
     
    def select_parents(self, population, fitness_values):
        """
        根据适应度选择父母个体。
        """
        min_fitness = min(fitness_values)
        if min_fitness < 0:
            fitness_values = [f - min_fitness + 1 for f in fitness_values]
        
        total_fitness = sum(fitness_values)
        if total_fitness == 0:
            return random.choice(population), random.choice(population)

        probabilities = [f / total_fitness for f in fitness_values]
        if np.any(np.isnan(probabilities)):
            raise ValueError("Probabilities contain NaN values. Check fitness calculation.")
        
        parents_indices = np.random.choice(len(population), size=2, p=probabilities)
        return population[parents_indices[0]], population[parents_indices[1]]
    
    def crossover(self, parent1, parent2):
        """
        交叉操作。
        """
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
        """
        调整边的数量以符合约束。
        """
        edges = [(i, j) for i in range(self.nodes-1) for j in range(i + 1, self.nodes) if adj_matrix[i, j] == 1]
        non_edges = [(i, j) for i in range(self.nodes-1) for j in range(i + 1, self.nodes) if adj_matrix[i, j] == 0]

        current_edges = len(edges)

        while current_edges > self.edges:
            i, j = random.choice(edges)
            adj_matrix[i, j] = 0
            adj_matrix[j, i] = 0
            edges.remove((i, j))
            current_edges -= 1

        while current_edges < self.edges:
            i, j = random.choice(non_edges)
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1
            non_edges.remove((i, j))
            current_edges += 1
        return adj_matrix

    def mutate(self, individual):
        """
        变异操作。
        """
        adj_matrix = individual.copy()
        if random.random() < self.mutation_rate:
            edges = [(i, j) for i in range(self.nodes-1) for j in range(i + 1, self.nodes) if adj_matrix[i, j] == 1]
            non_edges = [(i, j) for i in range(self.nodes-1) for j in range(i + 1, self.nodes) if adj_matrix[i, j] == 0]

            num_mutations = int(self.mutation_rate * self.nodes)
            
            for _ in range(num_mutations):
                if len(edges) > 0 and len(non_edges) > 0:
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
        """
        执行遗传算法优化 - 带早停和进度显示。
        """
        from tqdm import tqdm
        
        population = self.initialize_population()
        best_fitness = float('-inf')
        best_solution = None
        
        # 早停参数
        no_improvement_count = 0
        early_stop_patience = 30  # 连续30代无改进则停止
        
        pbar = tqdm(range(self.max_generations), desc="GA Optimization")

        for generation in pbar:
            fitness_scores = self.evaluate_population(population)
            
            max_fitness_idx = np.argmax(fitness_scores)
            current_best = fitness_scores[max_fitness_idx]
            
            if current_best > best_fitness:
                best_fitness = current_best
                best_solution = population[max_fitness_idx].copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # 更新进度条
            pbar.set_postfix({'fitness': f'{best_fitness:.4f}', 'no_imp': no_improvement_count})
            
            # 早停检查
            if no_improvement_count >= early_stop_patience:
                logger.info(f"Early stopping at generation {generation}")
                break
            
            new_population = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = self.select_parents(population, fitness_scores)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])

            population = new_population

        return best_solution, best_fitness

class ResilienceNetworkOptimizer():
    """
    网络鲁棒性优化器（计算适应度）- GPU 加速版本。
    """
    def __init__(self, N, L, alpha, beta):
        self.nodes = N
        self.edges = L
        self.alpha = alpha
        self.beta = beta
        # 智能GPU选择：对于小图（<200节点），CPU可能更快（避免传输开销）
        # 对于大图（>=200节点），GPU加速效果明显
        # GPU对于中等图（200-400节点）可能因为传输开销反而变慢
        gpu_available = TORCH_AVAILABLE and torch.cuda.is_available()
        if gpu_available:
            # 只对较大图使用GPU（>=300节点），小图用CPU更快
            # 可以根据实际测试调整阈值
            self.use_gpu = N >= 300  # 只对300节点以上的图使用GPU
            if self.use_gpu:
                logger.info(f"  Using GPU for fitness evaluation (N={N})")
            else:
                logger.info(f"  Using CPU for fitness evaluation (N={N}, GPU overhead too high for small graphs)")
        else:
            self.use_gpu = False
        
        # 缓存机制
        self._nc_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        full_adj = np.ones((self.nodes, self.nodes)) - np.eye(self.nodes)
        self.nc_max = self.calculate_natural_connectivity(full_adj)
        self.mu_max = self.calculate_degree_variance(full_adj)
    
    def _matrix_hash(self, adj_matrix):
        """计算邻接矩阵的哈希值用于缓存"""
        # 使用稀疏表示的哈希
        return hash(adj_matrix.tobytes())
    
    def calculate_natural_connectivity(self, adj_matrix):
        """
        计算自然连通度 (Natural Connectivity) - GPU 加速版本。
        使用缓存避免重复计算。
        """
        # 检查缓存
        matrix_hash = self._matrix_hash(adj_matrix)
        if matrix_hash in self._nc_cache:
            self._cache_hits += 1
            return self._nc_cache[matrix_hash]
        
        self._cache_misses += 1
        
        adj_matrix = adj_matrix.astype(np.float64)
        try:
            if self.use_gpu:
                # GPU 加速版本（仅用于大图，>=300节点）
                # 使用float32减少传输时间，提高GPU利用率
                adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32, device=DEVICE)
                eigenvalues = torch.linalg.eigvalsh(adj_tensor)  # 使用对称矩阵特化算法
                evals = eigenvalues.cpu().numpy().astype(np.float64)  # 转回float64用于后续计算
            else:
                # CPU 版本（对小图更快，避免GPU传输开销）
                eigenvalues = eig(adj_matrix, left=False, right=False)
                evals = eigenvalues.real

            max_eval = np.max(evals)
            # 数值稳定的 log-sum-exp
            nc = max_eval + np.log(np.sum(np.exp(evals - max_eval))) - np.log(len(evals))
            
            # 存入缓存（限制缓存大小）
            if len(self._nc_cache) < 10000:
                self._nc_cache[matrix_hash] = nc
            
            return nc
        except Exception as e:
            logger.error(f"Error calculating NC: {e}")
            return 0
    
    def calculate_largest_component(self, adj_matrix):
        """
        计算最大连通分量大小。
        """
        from scipy.sparse.csgraph import connected_components
        n_components, labels = connected_components(csr_matrix(adj_matrix), directed=False, return_labels=True)
        counts = np.bincount(labels)
        return np.max(counts)
    
    def calculate_degree_variance(self, adj_matrix):
        """
        计算度方差 - 向量化版本（CPU优化）。
        度方差计算很快，GPU传输开销大于收益，统一使用CPU。
        """
        # 即使使用GPU，度方差计算也用CPU（计算量小，GPU传输开销大）
        degrees = np.sum(adj_matrix, axis=1)
        mean_degree = np.mean(degrees)
        degree_variance = np.mean((degrees - mean_degree) ** 2)
        return degree_variance
    
    def evaluate_network(self, adj_matrix):
        """
        计算网络适应度。
        """
        sg = self.calculate_largest_component(adj_matrix)
        mu = self.calculate_degree_variance(adj_matrix)
        nc = self.calculate_natural_connectivity(adj_matrix)
        
        nc_max = self.nc_max
        mu_max = self.mu_max

        if nc_max == 0: nc_max = 1e-9
        if mu_max == 0: mu_max = 1e-9

        fitness = (self.alpha * (nc/nc_max) + 
                self.beta * (sg/self.nodes) - 
                (1-self.alpha-self.beta) * (mu/mu_max))
        return fitness

def solution(nodes, edges, max_generations=80, population_size=6):
    """
    GA 优化入口函数 - 高性能版本。
    
    Args:
        nodes: 节点数
        edges: 边数
        max_generations: 最大代数 (默认80，配合早停)
        population_size: 种群大小 (默认6)
    """
    mutation_rate = 0.2
    alpha = 0.4
    beta = 0.4
    logger.info(f"  Optimizing Network Structure using GA (nodes={nodes}, edges={edges})")
    evaluator = ResilienceNetworkOptimizer(
        N=nodes,
        L=edges,
        alpha=alpha,
        beta=beta
    )
    ga = GeneticAlgorithm(
        population_size=population_size,
        evaluator=evaluator,
        max_generations=max_generations,
        mutation_rate=mutation_rate,
        nodes=nodes,
        edges=edges
    )
    best_solution, best_fitness = ga.optimize()
    
    # 打印缓存统计
    if hasattr(evaluator, '_cache_hits'):
        total = evaluator._cache_hits + evaluator._cache_misses
        if total > 0:
            logger.info(f"  NC Cache: {evaluator._cache_hits}/{total} hits ({100*evaluator._cache_hits/total:.1f}%)")
    
    return best_solution, best_fitness
