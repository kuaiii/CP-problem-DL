# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
import random
import math
from src.utils.logger import get_logger

logger = get_logger(__name__)

class SOLO:
    def __init__(self, area_diameter: float = 500, comm_range: float = 200, epsilon: float = 1e-9):
        """
        初始化SOLO算法
        """
        self.area_diameter = area_diameter
        self.comm_range = comm_range
        self.epsilon = epsilon
        
    def _initialize_network(self, N: int) -> nx.Graph:
        """
        初始化网络拓扑
        """
        G = nx.Graph()
        
        # 在指定区域内随机生成节点位置
        for i in range(N):
            # 在圆形区域内随机生成坐标
            r = self.area_diameter/2 * math.sqrt(random.random())
            theta = random.random() * 2 * math.pi
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            
            # 添加节点及其位置
            G.add_node(i, pos=(x, y))
            
        return G
    
    def _get_potential_neighbors(self, G: nx.Graph, node_idx: int) -> tuple[list[int], list[int]]:
        """
        获取节点的潜在邻居列表及其度数
        """
        neighbor_list = []
        degree_list = []
        
        node_pos = G.nodes[node_idx]['pos']
        
        for j in G.nodes():
            # 跳过自身和已经连接的节点
            if j == node_idx or G.has_edge(node_idx, j):
                continue
                
            j_pos = G.nodes[j]['pos']
            
            # 计算两节点间距离
            distance = math.sqrt((node_pos[0] - j_pos[0])**2 + (node_pos[1] - j_pos[1])**2)
            
            # 如果在通信范围内，添加到潜在邻居列表
            if distance <= self.comm_range:
                neighbor_list.append(j)
                degree_list.append(G.degree(j))
                
        return neighbor_list, degree_list
    
    def _mapping_operator(self, degree_list: list[int]) -> list[float]:
        """
        映射操作符：将节点度数映射为连接概率
        实现度平衡连接机制
        """
        if not degree_list:
            return []
            
        # 为所有度加上epsilon，避免除零错误
        adjusted_degrees = [d + self.epsilon for d in degree_list]
        
        # 计算反比概率（度越小，概率越大）
        inverse_degrees = [1.0 / d for d in adjusted_degrees]
        sum_inverse = sum(inverse_degrees)
        
        # 归一化得到概率分布
        probability_list = [inv_d / sum_inverse for inv_d in inverse_degrees]
        
        return probability_list
    
    def _roulette_selection(self, probability_list: list[float]) -> int:
        """
        轮盘赌选择算法
        """
        if not probability_list:
            return None
            
        # 计算累积概率
        cum_prob = np.cumsum(probability_list)
        
        # 生成随机数
        r = random.random()
        
        # 轮盘赌选择
        for i, prob in enumerate(cum_prob):
            if r <= prob:
                return i
                
        return len(probability_list) - 1
    
    def generate_matched_topology(self, input_graph: nx.Graph) -> nx.Graph:
        """
        生成与输入网络节点数和边数完全一致的鲁棒拓扑
        """
        # 获取输入网络的核心参数
        N = input_graph.number_of_nodes()  # 节点数严格匹配
        target_edges = input_graph.number_of_edges()  # 目标边数严格匹配
        
        # 初始化网络（复制输入网络的节点位置以保持部署一致性，增强对比公平性）
        matched_graph = nx.Graph()
        for node in input_graph.nodes():
            # 若输入网络有位置信息则复用，否则随机生成（与论文中部署场景一致）
            if 'pos' in input_graph.nodes[node]:
                matched_graph.add_node(node, pos=input_graph.nodes[node]['pos'])
            else:
                # 按论文设定的圆形区域生成位置
                r = self.area_diameter/2 * math.sqrt(random.random())
                theta = random.random() * 2 * math.pi
                x = r * math.cos(theta)
                y = r * math.sin(theta)
                matched_graph.add_node(node, pos=(x, y))
        
        # 执行度平衡连接，直到达到目标边数
        edges_added = 0
        max_attempts = target_edges * 10
        attempts = 0
        
        while edges_added < target_edges and attempts < max_attempts:
            attempts += 1
            # 遍历所有节点，依次尝试添加边
            for i in range(N):
                if edges_added >= target_edges:
                    break
                # 获取节点i的潜在邻居（未连接且在通信范围内）
                neighbor_list, degree_list = self._get_potential_neighbors(matched_graph, i)
                if not neighbor_list:
                    continue
                # 应用映射操作符计算连接概率
                probability_list = self._mapping_operator(degree_list)
                # 轮盘赌选择目标节点
                selected_idx = self._roulette_selection(probability_list)
                if selected_idx is not None:
                    k = neighbor_list[selected_idx]
                    if not matched_graph.has_edge(i, k):
                        matched_graph.add_edge(i, k)
                        edges_added += 1
        return matched_graph
