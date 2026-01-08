# -*- coding: utf-8 -*-
import numpy as np
from math import ceil
from src.controller.strategies import random_select_controllers, RCP
from src.controller.rl_optimizer import train_and_select
from src.simulation.dismantling import node_attack
from src.visualization.result_plotter import plot_different_topo
from src.utils.logger import get_logger
from src.metrics.metrics import calculate_csa, calculate_control_entropy, calculate_wcp

logger = get_logger(__name__)

class ControllerManager:
    def __init__(self, all_G, rate, nodes_num, dataset_name, experiment_id):
        """
        初始化控制器管理器，用于管理不同拓扑结构和控制器部署策略的仿真实验。

        Args:
            all_G (dict): 包含所有待评估拓扑的字典，键为拓扑名称（如 'G', 'G1'），值为 networkx 图对象。
            rate (float): 控制器部署比例（例如 0.1 表示 10% 的节点作为控制器）。
            nodes_num (int): 网络中的总节点数。
            dataset_name (str): 数据集名称，用于结果保存路径。
            experiment_id (int): 实验 ID，用于结果保存路径的区分。
        """
        # 解包拓扑字典
        self.all_G = all_G
        self.G = all_G.get('G')    # 原始网络
        self.G1 = all_G.get('G1')  # 随机重构网络
        self.G2 = all_G.get('G2')  # 无标度重构网络 (预留)
        self.G3 = all_G.get('G3')  # 双峰网络 (Bimodal)
        self.G4 = all_G.get('G4')  # GA 优化后的网络
        self.G5 = all_G.get('G5')  # ONION 优化后的网络 (G5)
        self.G6 = all_G.get('G6')  # SOLO 优化后的网络 (G6)
        self.G7 = all_G.get('G7')  # ROMEN 优化后的网络 (G7)
        self.G8 = all_G.get('G8')  # BimodalRL 优化后的网络 (G8)
        self.G9 = all_G.get('G9')  # UNITY 优化后的网络 (G9)
        
        self.rate = rate
        self.nodes_num = nodes_num
        self.dataset_name = dataset_name
        self.experiment_id = experiment_id
        
        # 统一计算控制器数量上限，确保所有方法使用相同的资源约束
        self.controller_num = ceil(nodes_num * rate)
        
        # 定义绘图用的颜色、线型和标记，扩展至 12 种以支持新增方法 (Onion+Ra, Onion+RL, SOLO+Ra, SOLO+RL)
        # 新增颜色: #BCBD22 (Olive), #17BECF (Cyan), #FF7F0E (Orange), #2CA02C (Green), #D62728 (Red), #9467BD (Purple)
        # 现在总共 15 种方法 (含 UNITY+Ra, UNITY+RL)
        self.colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974', '#64B5CD', '#8C8C8C', '#E377C2', '#BCBD22', '#17BECF', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B']
        self.linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.']
        self.markers = ['o', 's', '^', 'D', 'v', '+', '*', 'x', 'p', 'h', '1', '2', '3', '4', '8']

    def run_simulation(self, mode, x_values, r_values, iteration_idx, metric_type="gcc", plot_flag=True):
        """
        运行单次仿真实验，根据指定的攻击模式调用相应的处理流程。

        Args:
            mode (str): 攻击模式。
            x_values (dict): 用于存储各方法在完全崩溃时的移除节点比例 (y=0 时的 x 值)。
            r_values (dict): 用于存储各方法的鲁棒性 R 值 (曲线下面积)。
            iteration_idx (int): 当前仿真的迭代轮次索引。
            metric_type (str): 评估指标类型 ('gcc', 'efficiency', 'coverage')。
            plot_flag (bool): 是否生成绘图文件。

        Returns:
            dict: 更新后的 x_values 字典。
        """
        # Note: plot_title was missing in original code, inferring from usage in _run_specific_attack
        # Here we don't call this method directly in main.py anymore, main.py calls _run_specific_attack.
        # But for completeness:
        return self._run_specific_attack(mode, x_values, r_values, iteration_idx, plot_flag, metric_type)

    def _process_method(self, G_target, centers, attack_mode, name, X, Y, labels, x_values, r_values, metric_type):
        """
        处理单个部署策略的评估：执行攻击仿真、计算指标并记录数据。
        """
        # 1. 打印调试信息
        logger.info(f"\n--- Processing Strategy: {name} ---")
        if G_target:
            logger.info(f"Topology: Nodes={G_target.number_of_nodes()}, Edges={G_target.number_of_edges()}")
        else:
            logger.error(f"Topology for {name} is None!")
            return

        logger.info(f"Controllers ({len(centers)}): {list(centers)}")
        
        # --- 新增指标计算 ---
        try:
            csa = calculate_csa(G_target, centers, alpha=0.5)
            hc = calculate_control_entropy(G_target, centers)
            wcp = calculate_wcp(G_target, centers, beta=1.0)
        except Exception as e:
            logger.error(f"Error calculating new metrics for {name}: {e}")
            csa, hc, wcp = 0.0, 0.0, 0.0
            
        # 将新指标存储到 manager 实例中，以便稍后保存
        if not hasattr(self, 'metrics_summary'):
            self.metrics_summary = {}
            
        if name not in self.metrics_summary:
            self.metrics_summary[name] = []
            
        # 计算拓扑属性 (仅对第一次出现的拓扑计算，避免重复)
        # 简单起见，每次都算，反正很快
        num_nodes = G_target.number_of_nodes()
        num_edges = G_target.number_of_edges()
        
        # 计算 Top-10 度分布
        degrees = sorted([d for n, d in G_target.degree()], reverse=True)
        top_10_degrees = degrees[:10] if len(degrees) >= 10 else degrees
        top_10_str = str(top_10_degrees).replace(',', ';') # 避免CSV冲突
            
        # 2. 执行节点攻击仿真
        # xy_res 包含每一步攻击后的 (GCC大小, 移除比例) 元组
        # x_zero 是网络完全崩溃（GCC=0）时的移除比例
        xy_res, x_zero, _ = node_attack(G_target.copy(), self.nodes_num, set(centers), attack_mode, metric_type)
        
        # 解包仿真结果
        x_res, y_res = [xx[1] for xx in xy_res], [xx[0] for xx in xy_res]
        
        # 3. 计算鲁棒性指标 R (Robustness)
        if len(x_res) > 1:
            # 使用梯形法则计算曲线下面积
            r_val = np.trapz(y_res, x_res)
        else:
            r_val = 0.0
            
        logger.info(f"Result: r_val={r_val:.4f}, collapse_point={x_zero}")
        
        # 暂时只记录单次迭代的值
        self.metrics_summary[name].append({
            'CSA': csa,
            'H_C': hc,
            'WCP': wcp,
            'R': r_val,
            'Nodes': num_nodes,
            'Edges': num_edges,
            'Top10_Degrees': top_10_str,
            'X_Curve': x_res,  # 保存原始曲线数据
            'Y_Curve': y_res   # 保存原始曲线数据
        })

        # 4. 存储绘图数据
        X.append(x_res)
        Y.append(y_res)
        labels.append(f"{name} (R={r_val:.4f})")
        
        # 5. 存储统计数据
        if name not in x_values: x_values[name] = []
        if name not in r_values: r_values[name] = []
            
        x_values[name].append(x_zero)
        r_values[name].append(r_val)

    def _run_specific_attack(self, attack_mode_str, x_values, r_values, ijk, plot_flag, metric_type, save_dir=None):
        """
        执行特定攻击模式下的所有对比方案仿真。
        
        Args:
            attack_mode_str (str): 攻击模式名称。
            x_values (dict): 统计数据容器。
            r_values (dict): 统计数据容器。
            ijk (int): 当前迭代次数。
            plot_flag (bool): 是否绘图。
            metric_type (str): 指标类型。
            save_dir (str, optional): 保存路径。
        """
        labels = []
        X, Y = [], []
        
        attack_code = attack_mode_str
            
        logger.info(f"\n[Iteration {ijk}] {attack_mode_str} Simulation (Metric: {metric_type})")
        
        # 1. Baseline: 原始拓扑 + 随机控制器
        ori_random_centers = random_select_controllers(self.G, self.rate)
        self._process_method(self.G, ori_random_centers, attack_code, "Baseline", X, Y, labels, x_values, r_values, metric_type)
        
        # 2. RCP: 原始拓扑 + E-RCP
        ori_ECA_centers = RCP(self.G, self.rate, 1)
        self._process_method(self.G, ori_ECA_centers, attack_code, "RCP", X, Y, labels, x_values, r_values, metric_type)
        
        # 3. 随机部署: 随机重构(G1) + 随机控制器
        # ran_random_centers = random_select_controllers(self.G1, self.rate)
        # self._process_method(self.G1, ran_random_centers, attack_code, "Random", X, Y, labels, x_values, r_values, metric_type)
        
        # 4. 混合指标: GA优化拓扑(G4) + 随机控制器
        # ran_random_centers = random_select_controllers(self.G4, self.rate)
        # self._process_method(self.G4, ran_random_centers, attack_code, "Hybrid", X, Y, labels, x_values, r_values, metric_type)
        
        # 5. RL-RCP (双层构造): 双峰拓扑(G3) + RL选择 (本文核心方法)
        rl_centers = train_and_select(self.G3, self.controller_num, metric_type="robustness")
        self._process_method(self.G3, rl_centers, attack_code, "Bi-level", X, Y, labels, x_values, r_values, metric_type)
        
        # 6. Bimodal + Random: 双峰拓扑(G3) + 随机选择 (验证拓扑结构的有效性)
        random_bimodal_centers = random_select_controllers(self.G3, self.rate)
        self._process_method(self.G3, random_bimodal_centers, attack_code, "Bimodal+Random", X, Y, labels, x_values, r_values, metric_type)
        
        # 7. Random + RL (新对比): 随机重构(G1) + RL选择 (验证RL算法的有效性)
        random_rl_centers = train_and_select(self.G1, self.controller_num, metric_type="robustness")
        self._process_method(self.G1, random_rl_centers, attack_code, "Random+RL", X, Y, labels, x_values, r_values, metric_type)
        
        # 8. GA + RL (新对比): GA优化(G4) + RL选择 (验证组合效果)
        ran_rl_centers = train_and_select(self.G4, self.controller_num, metric_type="robustness")
        self._process_method(self.G4, ran_rl_centers, attack_code, "GA+RL", X, Y, labels, x_values, r_values, metric_type)
        
        # 9. Onion+Ra (重命名): Onion优化(G5) + 随机部署
        if self.G5:
             onion_random_centers = random_select_controllers(self.G5, self.rate)
             self._process_method(self.G5, onion_random_centers, attack_code, "Onion+Ra", X, Y, labels, x_values, r_values, metric_type)

        # 10. Onion+RL (新对比): Onion优化(G5) + RL选择
        if self.G5:
             onion_rl_centers = train_and_select(self.G5, self.controller_num, metric_type="robustness")
             self._process_method(self.G5, onion_rl_centers, attack_code, "Onion+RL", X, Y, labels, x_values, r_values, metric_type)

        # 11. SOLO+Ra (新对比): SOLO优化(G6) + 随机部署
        if self.G6:
             solo_random_centers = random_select_controllers(self.G6, self.rate)
             self._process_method(self.G6, solo_random_centers, attack_code, "SOLO+Ra", X, Y, labels, x_values, r_values, metric_type)

        # 12. SOLO+RL (新对比): SOLO优化(G6) + RL选择
        if self.G6:
             solo_rl_centers = train_and_select(self.G6, self.controller_num, metric_type="robustness")
             self._process_method(self.G6, solo_rl_centers, attack_code, "SOLO+RL", X, Y, labels, x_values, r_values, metric_type)
 
        # 13. ROMEN+Ra (新对比): ROMEN优化(G7) + 随机部署
        if hasattr(self, 'G7') and self.G7:
             romen_random_centers = random_select_controllers(self.G7, self.rate)
             self._process_method(self.G7, romen_random_centers, attack_code, "ROMEN+Ra", X, Y, labels, x_values, r_values, metric_type)

        # 14. ROMEN+RL (新对比): ROMEN优化(G7) + RL选择
        if hasattr(self, 'G7') and self.G7:
             romen_rl_centers = train_and_select(self.G7, self.controller_num, metric_type="robustness")
             self._process_method(self.G7, romen_rl_centers, attack_code, "ROMEN+RL", X, Y, labels, x_values, r_values, metric_type)

        # 15. BimodalRL (新方法): BimodalRL优化(G8) + RL选择
        if hasattr(self, 'G8') and self.G8:
             bimodal_rl_centers = train_and_select(self.G8, self.controller_num, metric_type="robustness")
             self._process_method(self.G8, bimodal_rl_centers, attack_code, "BimodalRL", X, Y, labels, x_values, r_values, metric_type)

        # 16. UNITY+Ra (新方法): UNITY优化(G9) + 随机部署
        if hasattr(self, 'G9') and self.G9:
             unity_random_centers = random_select_controllers(self.G9, self.rate)
             self._process_method(self.G9, unity_random_centers, attack_code, "UNITY+Ra", X, Y, labels, x_values, r_values, metric_type)

        # 17. UNITY+RL (新方法): UNITY优化(G9) + RL选择
        if hasattr(self, 'G9') and self.G9:
             unity_rl_centers = train_and_select(self.G9, self.controller_num, metric_type="robustness")
             self._process_method(self.G9, unity_rl_centers, attack_code, "UNITY+RL", X, Y, labels, x_values, r_values, metric_type)
 
        if plot_flag and save_dir:
            plot_title = attack_mode_str
            logger.debug(f"Calling plot_different_topo. len(X)={len(X)}, save_dir={save_dir}")
            try:
                plot_different_topo(len(X), X, Y, self.colors, self.linestyles, self.markers, labels, ijk, plot_title, save_dir)
            except Exception as e:
                logger.error(f"Plotting failed: {e}")
        return x_values
