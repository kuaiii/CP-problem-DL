# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
import random
import torch
import copy
from scipy.stats import kurtosis, skew
import logging
from src.utils.logger import get_logger
from src.topology.romen_optimization import SACAgent, NetworkTopology, ROHEMOptimizer
from src.controller.rl_optimizer import predict, calculate_reward, load_or_train_model

logger = get_logger(__name__)

class BimodalRLOptimizer(ROHEMOptimizer):
    """
    Extends ROHEMOptimizer to include Bimodal Structure Reward and Controller Deployment.
    """
    def __init__(self, population_size=5, generations=100, verbose=True, 
                 bimodal_weight=0.8, robustness_weight=1.0, 
                 controller_model_path='models/rl_agent.pth',
                 dynamic_weight_decay=True):
        super().__init__(population_size, generations, verbose)
        self.initial_bimodal_weight = bimodal_weight
        self.bimodal_weight = bimodal_weight
        self.robustness_weight = robustness_weight
        self.controller_model_path = controller_model_path
        self.controller_model = None
        self.dynamic_weight_decay = dynamic_weight_decay
        
    def _calculate_bimodality_score(self, G):
        """
        Calculate a score representing how close the degree distribution is to a bimodal distribution.
        Uses Sarazin's Bimodality Coefficient (BC).
        BC = (gamma^2 + 1) / kappa
        where gamma is skewness and kappa is kurtosis.
        BC > 5/9 (approx 0.555) implies bimodality.
        """
        degrees = [d for n, d in G.degree()]
        if not degrees:
            return 0
        
        # Add small noise to avoid singular matrices in calculations if all degrees are same
        degrees = np.array(degrees) + np.random.normal(0, 0.01, len(degrees))
        
        n = len(degrees)
        if n < 3:
            return 0
            
        s = skew(degrees)
        k = kurtosis(degrees, fisher=False) # Pearson's kurtosis (normal = 3)
        
        # BC calculation
        bc = (s**2 + 1) / k
        
        # We want to maximize BC. 
        # Normalize or clip? BC is in [0, 1].
        return bc

    def _calculate_combined_fitness(self, topology):
        """
        Calculate combined fitness: Robustness (with controllers) + Bimodality.
        """
        G = topology.graph
        
        # 1. Bimodality Score
        bimodality_score = self._calculate_bimodality_score(G)
        
        # 2. Robustness with Controllers
        # We need to deploy controllers first.
        # Use the pre-trained RL agent for fast deployment or just Degree/Random if model not available.
        if self.controller_model is None:
             self.controller_model = load_or_train_model(self.controller_model_path)
        
        # Assume budget is 10% of nodes or 5%? 
        # In main.py, cover_rate is used. We'll assume a fixed small budget for optimization loop speed.
        # Say 5 controllers or 5%?
        k = max(1, int(G.number_of_nodes() * 0.05))
        
        try:
            centers = predict(G, k, self.controller_model)
        except Exception as e:
            # Fallback to High Degree if RL fails
            degrees = dict(G.degree())
            centers = sorted(degrees, key=degrees.get, reverse=True)[:k]
            
        # Calculate Robustness (R) of the controlled network
        # calculate_reward returns a weighted sum. We might just want R here.
        # But calculate_reward with mode='robustness' gives R.
        robustness_score = calculate_reward(G, centers, mode='robustness')
        
        # Combined Reward
        # We might want to normalize them. R is [0,1], BC is [0,1].
        total_score = self.robustness_weight * robustness_score + self.bimodal_weight * bimodality_score
        
        return total_score, robustness_score, bimodality_score

    def optimize_topology(self, initial_graph, positions=None, area_size=1000, 
                         comm_distance=None, max_time=None):
        """
        Override to implement dynamic weighting schedule
        """
        # Call the parent method but intercept the loop? 
        # Actually ROHEMOptimizer.optimize_topology is monolithic. 
        # We need to copy-paste and modify or just accept static weights if we don't want to duplicate code.
        # However, to implement "Training" improvement, we really should duplicate and modify the loop to support dynamic weights.
        # Or simpler: We just update the self.bimodal_weight inside _topology_interaction based on some global counter?
        # But _topology_interaction doesn't know about current generation.
        
        # Let's fully reimplement optimize_topology here to support dynamic weighting properly.
        import time
        from tqdm import tqdm
        
        start_time = time.time()
        
        if self.verbose:
            logger.info("=== Starting Bimodal RL Topology Optimization (Enhanced) ===")
        
        try:
            topology = NetworkTopology(
                graph=initial_graph,
                positions=positions,
                area_size=area_size,
                comm_distance=comm_distance
            )
            
            topology._compute_distance_matrix()
            
            # Initial evaluation
            initial_score, initial_r, initial_b = self._calculate_combined_fitness(topology)
            
            state_dim = topology.num_nodes * (topology.num_nodes - 1) // 2
            action_dim = 2
            
            master_agent = SACAgent(state_dim, action_dim)
            population = []
            for i in range(self.population_size):
                agent = SACAgent(state_dim, action_dim)
                population.append(agent)
            
            best_fitness = initial_score
            best_topology = copy.deepcopy(topology)
            
            pbar = tqdm(range(self.generations), desc="Optimizing") if self.verbose else range(self.generations)
            
            for generation in pbar:
                # Dynamic Weight Update
                if self.dynamic_weight_decay:
                    progress = generation / self.generations
                    # Decay bimodal weight from initial to 0.1
                    # self.bimodal_weight = self.initial_bimodal_weight * (1.0 - progress)
                    # Use a softer decay, keep some guidance
                    self.bimodal_weight = max(0.1, self.initial_bimodal_weight * (1.0 - 0.8 * progress))
                
                if max_time and (time.time() - start_time) > max_time:
                    break
                
                try:
                    all_experiences = []
                    fitness_scores = []
                    
                    # Master agent interaction (more steps)
                    master_experiences, master_fitness, master_topology, master_swaps = \
                        self._topology_interaction(master_agent, topology, steps=20) # Increased steps
                    all_experiences.extend(master_experiences)
                    
                    # Population interaction
                    for i, agent in enumerate(population):
                        experiences, fitness, agent_topology, swaps = \
                            self._topology_interaction(agent, topology, steps=15) # Increased steps
                        all_experiences.extend(experiences)
                        fitness_scores.append((i, fitness, agent_topology))
                        
                        for exp in experiences[-5:]:
                            agent.replay_buffer.push(*exp)
                    
                    for exp in all_experiences[-20:]:
                        master_agent.replay_buffer.push(*exp)
                    
                    # Evolution: Selection & Crossover
                    if generation % 10 == 0 and len(fitness_scores) >= 2: # More frequent evolution
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
                    
                    # RL Update
                    if generation % 2 == 0 and master_agent.replay_buffer.size() > 32: # More frequent updates
                        master_agent.update(batch_size=32)
                    
                    if generation % 5 == 0:
                        for agent in population:
                            if agent.replay_buffer.size() > 16:
                                agent.update(batch_size=16)
                    
                    # Track Best Solution
                    current_best_fitness = -float('inf')
                    current_best_topology = None
                    
                    for _, fitness, topology_candidate in fitness_scores:
                        if fitness > current_best_fitness:
                            current_best_fitness = fitness
                            current_best_topology = topology_candidate
                    
                    if current_best_topology:
                        try:
                            # Evaluate real robustness (pure R) for tracking
                            current_score, current_r, current_b = self._calculate_combined_fitness(current_best_topology)
                            
                            # We track based on TOTAL score (including bimodality guidance)
                            # But we might want to ensure Robustness is actually improving
                            if current_score > best_fitness:
                                best_fitness = current_score
                                best_topology = copy.deepcopy(current_best_topology)
                                topology = copy.deepcopy(current_best_topology)
                        except Exception as e:
                            logger.warning(f"Evaluation error: {e}")
                    
                    self.robustness_history.append(best_fitness)
                    
                    if self.verbose and hasattr(pbar, 'set_postfix'):
                        pbar.set_postfix({
                            'Score': f'{best_fitness:.2f}',
                            'W_Bi': f'{self.bimodal_weight:.2f}'
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
            current_score, _, _ = self._calculate_combined_fitness(current_topology)
        except:
            current_score = 0
        
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
                    experiences.append((current_state, action, -0.1, current_state)) # Small penalty for invalid move
                    fitness -= 0.1
                    continue
                
                next_state = topology_copy.get_state_vector()
                
                # Calculate new combined score
                next_score, r_score, b_score = self._calculate_combined_fitness(topology_copy)
                
                # Reward Scaling: 
                # Changes in R are usually small (0.01), so multiply by large factor
                reward = (next_score - current_score) * 100 
                
                # Bonus for achieving high bimodality
                if b_score > 0.555:
                    reward += 1.0
                
                if next_score > current_score:
                    accept = True
                elif next_score == current_score:
                    accept = (random.random() < 0.3)
                else:
                    accept = (random.random() < 0.1)
                
                if swap_success and accept:
                    experiences.append((current_state, action, reward, next_state))
                    current_topology = topology_copy
                    current_score = next_score
                    successful_swaps += 1
                else:
                    experiences.append((current_state, action, -0.01, current_state)) # Small penalty for rejected move
                
                fitness += reward
                
            except Exception as e:
                # logger.error(f"Step error: {e}")
                continue
        
        return experiences, fitness, current_topology, successful_swaps

def construct_bimodal_rl(G, generations=100):
    """
    Construct a network using Bimodal RL Optimization.
    """
    logger.info("  Starting Bimodal RL Structure Optimization (Enhanced)...")
    optimizer = BimodalRLOptimizer(
        population_size=5,
        generations=generations,
        verbose=True,
        bimodal_weight=1.0, # Start with strong guidance
        robustness_weight=2.0, # Robustness is the ultimate goal
        dynamic_weight_decay=True
    )
    G_bimodal = optimizer.optimize_topology(
        initial_graph=G,
        area_size=500,
        comm_distance=200
    )
    return G_bimodal
