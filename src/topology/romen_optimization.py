# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
import random
import copy
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
from src.utils.logger import get_logger

logger = get_logger(__name__)

class NetworkTopology:
    """优化的网络拓扑类"""
    
    def __init__(self, num_nodes=None, density=None, area_size=None, comm_distance=None, 
                 graph=None, positions=None):
        if graph is not None:
            # 从现有图初始化
            self.graph = graph.copy()
            self.num_nodes = graph.number_of_nodes()
            # 确保节点是整数索引
            mapping = {n: i for i, n in enumerate(self.graph.nodes())}
            self.graph = nx.relabel_nodes(self.graph, mapping)
            
            self.positions = positions if positions else self._generate_positions_from_graph()
            # 估算参数
            self.density = max(1, int(np.mean([graph.degree(n) for n in graph.nodes()])))
            self.area_size = area_size if area_size else 1000
            self.comm_distance = comm_distance if comm_distance else self._estimate_comm_distance()
        else:
            # 新建拓扑
            self.num_nodes = num_nodes
            self.density = density
            self.area_size = area_size
            self.comm_distance = comm_distance
            self.positions = {}
            self.graph = None
            
        self._valid_connections_cache = None
        self._distance_matrix = None
        
    def _generate_positions_from_graph(self):
        """从图结构生成节点位置"""
        positions = {}
        # 使用spring layout生成位置
        pos = nx.spring_layout(self.graph, scale=500)
        for node, (x, y) in pos.items():
            positions[node] = (x + 500, y + 500)  # 确保坐标为正
        return positions
    
    def _estimate_comm_distance(self):
        """估算通信距离"""
        if not self.positions or len(self.positions) < 2:
            return 200
        
        distances = []
        for edge in self.graph.edges():
            node1, node2 = edge
            if node1 in self.positions and node2 in self.positions:
                pos1 = self.positions[node1]
                pos2 = self.positions[node2]
                dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                distances.append(dist)
        
        if distances:
            # 设置为现有边距离的1.2倍作为通信距离
            return max(distances) * 1.2
        return 200
    
    def _compute_distance_matrix(self):
        """预计算距离矩阵"""
        self._distance_matrix = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                pos1 = self.positions[i]
                pos2 = self.positions[j]
                distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                self._distance_matrix[i, j] = distance
                self._distance_matrix[j, i] = distance
    
    def _is_within_distance(self, node1, node2):
        """检查两个节点是否在通信距离内"""
        return self._distance_matrix[node1, node2] <= self.comm_distance
    
    def _update_valid_connections_cache(self):
        """更新有效连接缓存 - 优化版本"""
        self._valid_connections_cache = []
        edges = list(self.graph.edges())
        max_connections = 5000  # 限制最大有效连接数
         
        # 随机采样边对
        for _ in range(min(max_connections, len(edges)**2)):
            edge1, edge2 = random.sample(edges, 2)
            if len(set(edge1 + edge2)) == 4:  # 确保四个不同节点
                self._valid_connections_cache.append((edge1, edge2))
        
        # logger.info(f"优化有效连接数: {len(self._valid_connections_cache)}")
    
    def get_valid_connections(self):
        """获取有效连接（使用缓存）"""
        if self._valid_connections_cache is None:
            self._update_valid_connections_cache()
        return self._valid_connections_cache
    
    def get_adjacency_matrix(self):
        """获取邻接矩阵"""
        return nx.adjacency_matrix(self.graph).todense()
    
    def get_state_vector(self):
        """将拓扑转换为状态向量 - Vectorized optimization"""
        # Create upper triangle mask
        mask = np.triu_indices(self.num_nodes, k=1)
        
        # Get adjacency matrix (dense)
        # Using nx.to_numpy_array is faster than nx.adjacency_matrix(..).todense() usually
        adj = nx.to_numpy_array(self.graph)
        
        # Get distance matrix
        dist = self._distance_matrix
        
        # Logic: 1.0 if edge exists, else 1.0 / (dist + 1e-6)
        # We can calculate the 'else' part for all, then overwrite where edge exists
        
        # Base vector: 1.0 / (dist + 1e-6)
        # Note: dist might have 0s on diagonal but we only care about upper triangle k=1
        inv_dist = 1.0 / (dist + 1e-6)
        
        # Where adj is 1, value is 1.0
        # state_matrix = np.where(adj > 0, 1.0, inv_dist)
        # Simplified:
        state_matrix = np.where(adj > 0, 1.0, inv_dist)
        
        # Extract upper triangle
        state = state_matrix[mask]
        
        return state
    
    def _rollback_graph(self, edges_backup):
        """回滚图到备份状态"""
        # 清空当前图的所有边
        self.graph.clear_edges()
        # 恢复备份的边
        self.graph.add_edges_from(edges_backup)
    
    def edge_swap(self, edge_pair1, edge_pair2):
        """执行边交换操作"""
        i, j = edge_pair1
        m, n = edge_pair2
        
        # 记录原始状态用于验证和可能的回滚
        original_edge_count = self.graph.number_of_edges()
        original_node_count = self.graph.number_of_nodes()
        
        # 1. 基本有效性检查
        if not (self.graph.has_edge(i, j) and self.graph.has_edge(m, n)):
            return False
            
        # 2. 确保是四个不同的节点
        if len(set([i, j, m, n])) != 4:
            return False
        
        # 3. 定义两种可能的边交换方式
        swap_option1 = [(i, m), (j, n)]
        swap_option2 = [(i, n), (j, m)]
        
        # 4. 检查两种交换方式的有效性
        def is_valid_swap_option(new_edges):
            """检查一种交换方式是否有效"""
            for edge in new_edges:
                node1, node2 = edge
                # 检查是否在通信距离内
                if not self._is_within_distance(node1, node2):
                    return False
                # 检查是否会创建自环
                if node1 == node2:
                    return False
                # 关键：检查新边是否已存在（这会导致边数减少）
                if self.graph.has_edge(node1, node2):
                    return False
            return True
        
        # 5. 选择有效的交换方式
        valid_options = []
        if is_valid_swap_option(swap_option1):
            valid_options.append(swap_option1)
        if is_valid_swap_option(swap_option2):
            valid_options.append(swap_option2)
        
        # 6. 如果没有有效的交换方式，返回失败
        if not valid_options:
            return False
        
        # 7. 随机选择一个有效的交换方式
        selected_option = random.choice(valid_options)
        
        try:
            # 8. 创建图的备份（用于回滚）
            edges_backup = list(self.graph.edges())
            
            # 9. 执行边交换
            # 首先移除原有的两条边
            self.graph.remove_edge(i, j)
            self.graph.remove_edge(m, n)
            
            # 然后添加新的两条边
            for edge in selected_option:
                node1, node2 = edge
                self.graph.add_edge(node1, node2)
            
            # 10. 验证交换结果
            final_edge_count = self.graph.number_of_edges()
            final_node_count = self.graph.number_of_nodes()
            
            # 严格验证边数和节点数
            if final_edge_count != original_edge_count or final_node_count != original_node_count:
                # 回滚操作
                self._rollback_graph(edges_backup)
                return False
            
            # 11. 验证图的基本性质
            if not self._verify_graph_properties():
                self._rollback_graph(edges_backup)
                return False
            
            # 12. 成功：更新有效连接缓存
            self._valid_connections_cache = None
            
            return True
            
        except Exception as e:
            logger.error(f"边交换过程中发生异常: {e}")
            # 尝试回滚
            try:
                self._rollback_graph(edges_backup)
            except:
                logger.critical("回滚失败，图状态可能不一致")
            return False

    def _verify_graph_properties(self):
        """验证图的基本性质"""
        try:
            # 检查节点数
            if self.graph.number_of_nodes() != self.num_nodes:
                return False
            
            # 检查是否有自环
            if nx.number_of_selfloops(self.graph) > 0:
                return False
            
            return True
        except Exception as e:
            logger.error(f"图性质验证时发生异常: {e}")
            return False
    
    def calculate_robustness_RG(self, sample_ratio=0.3):
        """
        计算鲁棒性指标RG - 采样优化版本。
        不需要模拟移除所有节点，只采样一部分来估算。
        """
        graph_copy = self.graph.copy()
        total_robustness = 0
        steps = 0
        
        # 只采样 sample_ratio 比例的节点进行模拟
        max_steps = max(5, int(self.num_nodes * sample_ratio))
        
        while graph_copy.number_of_nodes() > 0 and steps < max_steps:
            # 计算当前最大连通分量
            if graph_copy.number_of_edges() > 0:
                components = list(nx.connected_components(graph_copy))
                max_component_size = max(len(c) for c in components) if components else 0
            else:
                max_component_size = 1 if graph_copy.number_of_nodes() == 1 else 0
            
            total_robustness += max_component_size / self.num_nodes
            steps += 1
            
            # 移除度数最高的节点
            if graph_copy.number_of_nodes() > 0:
                degrees = dict(graph_copy.degree())
                if degrees:
                    highest_degree_node = max(degrees, key=degrees.get)
                    graph_copy.remove_node(highest_degree_node)
                else:
                    break
        
        # 归一化：使用实际步数而不是总节点数
        return total_robustness / (steps + 1) if steps > 0 else 0

class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        if batch_size == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return (np.array(states), np.array(actions), 
                np.array(rewards), np.array(next_states))
    
    def size(self):
        return len(self.buffer)

class ActorNetwork(nn.Module):
    """演员网络（策略网络）"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

class CriticNetwork(nn.Module):
    """评价网络"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class ValueNetwork(nn.Module):
    """价值网络"""
    
    def __init__(self, state_dim, hidden_dim=128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class SACAgent:
    """SAC智能体"""
    
    def __init__(self, state_dim, action_dim, lr=3e-4, alpha=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.critic1 = CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic2 = CriticNetwork(state_dim, action_dim).to(self.device)
        self.value = ValueNetwork(state_dim).to(self.device)
        self.target_value = ValueNetwork(state_dim).to(self.device)
        
        self.hard_update(self.target_value, self.value)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)
        
        self.alpha = alpha
        self.replay_buffer = ReplayBuffer()
        
    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )
        
    def select_action(self, state):
        try:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action, _ = self.actor.sample(state)
            
            noise = np.random.normal(0, 0.2, size=action.shape)
            noisy_action = action.cpu().data.numpy().flatten() + noise
            return np.clip(noisy_action, -1, 1)
        except Exception as e:
            logger.warning(f"Action selection error: {e}, returning random action")
            return np.random.uniform(-1, 1, size=(2,))
    
    def update(self, batch_size=64, gamma=0.99, tau=0.005):
        if self.replay_buffer.size() < batch_size:
            return {"actor_loss": 0, "critic_loss": 0, "value_loss": 0}
        
        try:
            states, actions, rewards, next_states = self.replay_buffer.sample(batch_size)
            
            if len(states) == 0:
                return {"actor_loss": 0, "critic_loss": 0, "value_loss": 0}
            
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.FloatTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            
            if states.shape[0] == 0:
                return {"actor_loss": 0, "critic_loss": 0, "value_loss": 0}
            
            with torch.no_grad():
                next_state_actions, next_state_log_pi = self.actor.sample(next_states)
                target_q1 = self.critic1(next_states, next_state_actions)
                target_q2 = self.critic2(next_states, next_state_actions)
                target_q = torch.min(target_q1, target_q2) - self.alpha * next_state_log_pi
                target_value = rewards + gamma * target_q
            
            current_value = self.value(states)
            value_loss = F.mse_loss(current_value, target_value)
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=1.0)
            self.value_optimizer.step()
            
            current_q1 = self.critic1(states, actions)
            current_q2 = self.critic2(states, actions)
            critic1_loss = F.mse_loss(current_q1, target_value)
            critic2_loss = F.mse_loss(current_q2, target_value)
            
            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=1.0)
            self.critic1_optimizer.step()
            
            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)
            self.critic2_optimizer.step()
            
            state_actions, log_pi = self.actor.sample(states)
            q1_new = self.critic1(states, state_actions)
            q2_new = self.critic2(states, state_actions)
            q_new = torch.min(q1_new, q2_new)
            actor_loss = (self.alpha * log_pi - q_new).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()
            
            self.soft_update(self.target_value, self.value, tau)
            
            return {
                "actor_loss": actor_loss.item(),
                "critic_loss": (critic1_loss + critic2_loss).item(),
                "value_loss": value_loss.item()
            }
            
        except Exception as e:
            logger.warning(f"Update error: {e}")
            return {"actor_loss": 0, "critic_loss": 0, "value_loss": 0}

class ROHEMOptimizer:
    """ROMEM优化器"""
    
    def __init__(self, population_size=3, generations=50, verbose=True):
        self.population_size = population_size
        self.generations = generations
        self.verbose = verbose
        
        self.crossover_prob = 0.8
        self.mutation_prob = 0.9
        
        self.robustness_history = []
        self.fitness_history = []
        self.loss_history = []
        
    def optimize_topology(self, initial_graph, positions=None, area_size=1000, 
                         comm_distance=None, max_time=None):
        start_time = time.time()
        
        if self.verbose:
            logger.info("=== Starting ROMEM Topology Optimization ===")
        
        try:
            topology = NetworkTopology(
                graph=initial_graph,
                positions=positions,
                area_size=area_size,
                comm_distance=comm_distance
            )
            
            topology._compute_distance_matrix()
            
            initial_robustness = topology.calculate_robustness_RG()
            
            state_dim = topology.num_nodes * (topology.num_nodes - 1) // 2
            action_dim = 2
            
            master_agent = SACAgent(state_dim, action_dim)
            population = []
            for i in range(self.population_size):
                agent = SACAgent(state_dim, action_dim)
                population.append(agent)
            
            best_robustness = initial_robustness
            best_topology = copy.deepcopy(topology)
            
            pbar = tqdm(range(self.generations), desc="Optimizing") if self.verbose else range(self.generations)
            
            for generation in pbar:
                if max_time and (time.time() - start_time) > max_time:
                    break
                
                try:
                    all_experiences = []
                    fitness_scores = []
                    
                    master_experiences, master_fitness, master_topology, master_swaps = \
                        self._topology_interaction(master_agent, topology, steps=15)
                    all_experiences.extend(master_experiences)
                    
                    for i, agent in enumerate(population):
                        experiences, fitness, agent_topology, swaps = \
                            self._topology_interaction(agent, topology, steps=10)
                        all_experiences.extend(experiences)
                        fitness_scores.append((i, fitness, agent_topology))
                        
                        for exp in experiences[-5:]:
                            agent.replay_buffer.push(*exp)
                    
                    for exp in all_experiences[-20:]:
                        master_agent.replay_buffer.push(*exp)
                    
                    if generation % 30 == 0 and len(fitness_scores) >= 2:
                        fitness_scores.sort(key=lambda x: x[1], reverse=True)
                        
                        if random.random() < self.crossover_prob:
                            parent1_idx, parent2_idx = fitness_scores[0][0], fitness_scores[1][0]
                            parent1 = population[parent1_idx]
                            parent2 = population[parent2_idx]
                            
                            offspring = self._distillation_crossover(parent1, parent2)
                            worst_idx = fitness_scores[-1][0]
                            population[worst_idx] = offspring
                        
                        if random.random() < 0.2:
                            mutation_idx = random.randint(0, len(population) - 1)
                            population[mutation_idx] = self._gaussian_mutation(population[mutation_idx])
                    
                    if generation % 3 == 0 and master_agent.replay_buffer.size() > 16:
                        master_agent.update(batch_size=16)
                    
                    if generation % 10 == 0:
                        for agent in population:
                            if agent.replay_buffer.size() > 8:
                                agent.update(batch_size=8)
                    
                    current_best_fitness = -float('inf')
                    current_best_topology = None
                    
                    for _, fitness, topology_candidate in fitness_scores:
                        if fitness > current_best_fitness:
                            current_best_fitness = fitness
                            current_best_topology = topology_candidate
                    
                    if current_best_topology:
                        try:
                            current_robustness = current_best_topology.calculate_robustness_RG()
                            if current_robustness > best_robustness:
                                best_robustness = current_robustness
                                best_topology = copy.deepcopy(current_best_topology)
                                topology = copy.deepcopy(current_best_topology)
                        except Exception as e:
                            logger.warning(f"Robustness evaluation error: {e}")
                    
                    self.robustness_history.append(best_robustness)
                    
                    if self.verbose and hasattr(pbar, 'set_postfix'):
                        pbar.set_postfix({
                            'Robustness': f'{best_robustness:.4f}'
                        })
                
                except Exception as e:
                    logger.error(f"Generation {generation} error: {e}")
                    continue
            
            return best_topology.graph
            
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            return initial_graph
    
    def _topology_interaction(self, agent, topology, steps=20):
        current_topology = copy.deepcopy(topology)
        fitness = 0
        experiences = []
        successful_swaps = 0
        
        try:
            initial_robustness = current_topology.calculate_robustness_RG()
        except:
            initial_robustness = 0.1
        
        for step in range(steps):
            try:
                current_state = current_topology.get_state_vector()
                action = agent.select_action(current_state)
                
                edge1, edge2 = self._map_action_to_edges(action, current_topology)
                if edge1 is None or edge2 is None:
                    continue
                
                topology_copy = copy.deepcopy(current_topology)
                swap_success = topology_copy.edge_swap(edge1, edge2)
                
                if not swap_success:
                    experiences.append((current_state, action, -1, current_state))
                    fitness -= 1
                    continue
                
                next_state = topology_copy.get_state_vector()
                next_robustness = topology_copy.calculate_robustness_RG()
                current_robustness = current_topology.calculate_robustness_RG()
                
                if next_robustness > current_robustness:
                    reward = 10
                    accept = True
                elif next_robustness == current_robustness:
                    reward = 0
                    accept = (random.random() < 0.3)
                else:
                    reward = -1
                    accept = (random.random() < 0.1)
                
                if swap_success and accept:
                    experiences.append((current_state, action, reward, next_state))
                    current_topology = topology_copy
                    successful_swaps += 1
                else:
                    experiences.append((current_state, action, -1, current_state))
                
                fitness += reward
                
            except Exception as e:
                continue
        
        return experiences, fitness, current_topology, successful_swaps
    
    def _map_action_to_edges(self, action, topology):
        try:
            valid_connections = topology.get_valid_connections()
            if not valid_connections:
                return None, None
            
            action_normalized = (action + 1) / 2
            index1 = int(action_normalized[0] * len(valid_connections))
            index2 = int(action_normalized[1] * len(valid_connections))
            
            index1 = np.clip(index1, 0, len(valid_connections) - 1)
            index2 = np.clip(index2, 0, len(valid_connections) - 1)
            
            if index1 == index2:
                index2 = (index2 + 1) % len(valid_connections)
            
            connection1 = valid_connections[index1]
            connection2 = valid_connections[index2]
            
            return connection1[0], connection1[1]
        except:
            return None, None
    
    def _distillation_crossover(self, parent1, parent2):
        try:
            state_dim = parent1.actor.fc1.in_features
            action_dim = parent1.actor.mean.out_features
            offspring = SACAgent(state_dim, action_dim)
            
            offspring.actor.load_state_dict(copy.deepcopy(parent1.actor.state_dict()))
            return offspring
        except:
            return SACAgent(parent1.actor.fc1.in_features, parent1.actor.mean.out_features)
    
    def _gaussian_mutation(self, agent):
        try:
            state_dim = agent.actor.fc1.in_features
            action_dim = agent.actor.mean.out_features
            mutated_agent = SACAgent(state_dim, action_dim)
            mutated_agent.actor.load_state_dict(copy.deepcopy(agent.actor.state_dict()))
            
            with torch.no_grad():
                for param in mutated_agent.actor.parameters():
                    if len(param.shape) > 1:
                        mask = torch.rand_like(param) < 0.03
                        noise = torch.randn_like(param) * 0.05
                        param.data = param.data + noise * mask.float()
            
            return mutated_agent
        except:
            return agent

def construct_romen(G, generations=20):
    """
    构建 ROMEN 结构的优化网络 (G6)
    优化版本：减少代数，加快收敛。
    """
    logger.info("  Starting ROMEN Structure Optimization (Optimized)...")
    optimizer = ROHEMOptimizer(
        population_size=2,  # 减少种群大小
        generations=generations,
        verbose=True
    )
    G_romen = optimizer.optimize_topology(
        initial_graph=G,
        area_size=500,
        comm_distance=200
    )
    return G_romen
