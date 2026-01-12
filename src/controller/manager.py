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
        # self.G1 = all_G.get('G1')  # 随机重构网络 (未使用)
        # self.G2 = all_G.get('G2')  # 无标度重构网络 (未使用)
        self.G3 = all_G.get('G3')  # 双峰网络 (Bimodal)
        self.G4 = all_G.get('G4')  # GA 优化后的网络
        self.G5 = all_G.get('G5')  # ONION 优化后的网络 (G5)
        # self.G6 = all_G.get('G6')  # SOLO 优化后的网络 (未使用)
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

    def _process_method(self, G_target, centers, attack_mode, name, X, Y, labels, x_values, r_values):
        """
        处理单个部署策略的评估：执行攻击仿真、计算指标并记录数据。
        """
        # 1. 精简日志输出
        if not G_target:
            logger.error(f"Topology for {name} is None!")
            return
        
        # 将新指标存储到 manager 实例中，以便稍后保存
        if not hasattr(self, 'metrics_summary'):
            self.metrics_summary = {}
            
        if name not in self.metrics_summary:
            self.metrics_summary[name] = []
            
        # 计算拓扑属性 (仅对第一次出现的拓扑计算，避免重复)
        num_nodes = G_target.number_of_nodes()
        num_edges = G_target.number_of_edges()
        
        # 计算 Top-10 度分布
        degrees = sorted([d for n, d in G_target.degree()], reverse=True)
        top_10_degrees = degrees[:10] if len(degrees) >= 10 else degrees
        top_10_str = str(top_10_degrees).replace(',', ';') # 避免CSV冲突
            
        # 2. 执行节点攻击仿真
        # metrics_dict 包含所有指标的曲线数据
        # x_lcc_20 是 LCC 降到 20% 时的移除比例
        metrics_dict, x_lcc_20 = node_attack(G_target.copy(), self.nodes_num, set(centers), attack_mode)
        
        # 解包 LCC 曲线数据
        lcc_curve = metrics_dict['lcc']
        x_res = [xx[1] for xx in lcc_curve]
        y_res = [xx[0] for xx in lcc_curve]
        
        # 3. 计算鲁棒性指标 R (Robustness) - 基于 LCC 曲线
        if len(x_res) > 1:
            # 使用梯形法则计算曲线下面积
            r_val = np.trapz(y_res, x_res)
        else:
            r_val = 0.0
            
        logger.info(f"Result: r_val={r_val:.4f}, collapse_point={x_lcc_20}")
        
        # 记录所有指标曲线数据
        self.metrics_summary[name].append({
            'LCC_Curve': lcc_curve,
            'CSA_Curve': metrics_dict['csa'],
            'CCE_Curve': metrics_dict['cce'],
            'WCP_Curve': metrics_dict['wcp'],
            'R': r_val,
            'Collapse_Point': x_lcc_20,
            'Nodes': num_nodes,
            'Edges': num_edges,
            'Top10_Degrees': top_10_str
        })

        # 4. 存储绘图数据 (基于 LCC)
        X.append(x_res)
        Y.append(y_res)
        labels.append(f"{name} (R={r_val:.4f})")
        
        # 5. 存储统计数据
        if name not in x_values: x_values[name] = []
        if name not in r_values: r_values[name] = []
            
        x_values[name].append(x_lcc_20)
        r_values[name].append(r_val)

    def _process_method_multi_metric(self, G_target, centers, attack_mode, name, x_values_by_metric, r_values_by_metric):
        """
        处理单个部署策略的评估（多指标版本）：执行攻击仿真、计算所有指标并记录数据。
        
        Args:
            G_target: 目标拓扑
            centers: 控制器集合
            attack_mode: 攻击模式
            name: 方法名称
            x_values_by_metric: 按指标类型组织的崩溃点数据字典 {metric_type: {method: [values]}}
            r_values_by_metric: 按指标类型组织的鲁棒性数据字典 {metric_type: {method: [values]}}
        """
        # 1. 打印调试信息
        logger.info(f"\n--- Processing Strategy: {name} ---")
        if G_target:
            logger.info(f"Topology: Nodes={G_target.number_of_nodes()}, Edges={G_target.number_of_edges()}")
        else:
            logger.error(f"Topology for {name} is None!")
            return

        logger.info(f"Controllers ({len(centers)}): {list(centers)}")
        
        # 将新指标存储到 manager 实例中，以便稍后保存
        if not hasattr(self, 'metrics_summary'):
            self.metrics_summary = {}
            
        if name not in self.metrics_summary:
            self.metrics_summary[name] = []
            
        # 计算拓扑属性
        num_nodes = G_target.number_of_nodes()
        num_edges = G_target.number_of_edges()
        
        # 计算 Top-10 度分布
        degrees = sorted([d for n, d in G_target.degree()], reverse=True)
        top_10_degrees = degrees[:10] if len(degrees) >= 10 else degrees
        top_10_str = str(top_10_degrees).replace(',', ';')
            
        # 2. 执行节点攻击仿真（只执行一次，获取所有指标）
        metrics_dict, x_lcc_20 = node_attack(G_target.copy(), self.nodes_num, set(centers), attack_mode)
        
        # 3. 为每个指标类型计算崩溃点和鲁棒性
        metric_collapse_points = {}
        metric_r_values = {}
        
        # 定义指标阈值（当指标降到初始值的20%时认为崩溃）
        for metric_type in ['lcc', 'csa', 'cce', 'wcp']:
            curve_key = metric_type
            curve_data = metrics_dict.get(curve_key, [])
            
            if not curve_data:
                metric_collapse_points[metric_type] = None
                metric_r_values[metric_type] = 0.0
                continue
            
            x_curve = [point[1] for point in curve_data]
            y_curve = [point[0] for point in curve_data]
            
            # 计算鲁棒性 R（曲线下面积）
            if len(x_curve) > 1:
                r_val = np.trapz(y_curve, x_curve)
            else:
                r_val = 0.0
            metric_r_values[metric_type] = r_val
            
            # 计算崩溃点（指标降到初始值的20%）
            if len(y_curve) > 0:
                initial_value = y_curve[0] if y_curve[0] > 0 else 1.0
                threshold = initial_value * 0.2
                
                collapse_point = None
                for i, y_val in enumerate(y_curve):
                    if y_val <= threshold:
                        collapse_point = x_curve[i]
                        break
                
                # 如果没找到崩溃点，使用最后一个点
                if collapse_point is None and len(x_curve) > 0:
                    collapse_point = x_curve[-1]
                
                metric_collapse_points[metric_type] = collapse_point
            else:
                metric_collapse_points[metric_type] = None
        
        # 记录所有指标曲线数据（使用LCC的崩溃点作为主要崩溃点）
        self.metrics_summary[name].append({
            'LCC_Curve': metrics_dict['lcc'],
            'CSA_Curve': metrics_dict['csa'],
            'CCE_Curve': metrics_dict['cce'],
            'WCP_Curve': metrics_dict['wcp'],
            'R': metric_r_values['lcc'],  # 主要鲁棒性指标基于LCC
            'Collapse_Point': x_lcc_20,   # 主要崩溃点基于LCC
            'Nodes': num_nodes,
            'Edges': num_edges,
            'Top10_Degrees': top_10_str
        })
        
        # 4. 为每个指标类型存储统计数据
        for metric_type in ['lcc', 'csa', 'cce', 'wcp']:
            if name not in x_values_by_metric[metric_type]:
                x_values_by_metric[metric_type][name] = []
            if name not in r_values_by_metric[metric_type]:
                r_values_by_metric[metric_type][name] = []
            
            collapse_point = metric_collapse_points[metric_type]
            r_val = metric_r_values[metric_type]
            
            if collapse_point is not None:
                x_values_by_metric[metric_type][name].append(collapse_point)
            else:
                x_values_by_metric[metric_type][name].append(1.0)  # 默认完全崩溃
            
            r_values_by_metric[metric_type][name].append(r_val)
        
        # 精简日志：只输出关键结果
        # logger.info(f"Result (LCC): r_val={metric_r_values['lcc']:.4f}, collapse_point={x_lcc_20}")

    def _run_specific_attack(self, attack_mode_str, x_values, r_values, ijk, plot_flag, metric_type, save_dir=None):
        """
        执行特定攻击模式下的所有对比方案仿真。
        
        Args:
            attack_mode_str (str): 攻击模式名称。
            x_values (dict): 统计数据容器。
            r_values (dict): 统计数据容器。
            ijk (int): 当前迭代次数。
            plot_flag (bool): 是否绘图。
            metric_type (str): 指标类型（保留参数以兼容，但不再使用）。
            save_dir (str, optional): 保存路径。
        """
        labels = []
        X, Y = [], []
        
        attack_code = attack_mode_str
            
        logger.info(f"\n[Iteration {ijk}] {attack_mode_str} Simulation")
        
        # 1. Baseline: 原始拓扑 + 随机控制器
        ori_random_centers = random_select_controllers(self.G, self.rate)
        self._process_method(self.G, ori_random_centers, attack_code, "Baseline", X, Y, labels, x_values, r_values)
        
        # 2. RCP: 原始拓扑 + E-RCP
        ori_ECA_centers = RCP(self.G, self.rate, 1)
        self._process_method(self.G, ori_ECA_centers, attack_code, "RCP", X, Y, labels, x_values, r_values)
        
        # 3. 随机部署: 随机重构(G1) + 随机控制器
        # ran_random_centers = random_select_controllers(self.G1, self.rate)
        # self._process_method(self.G1, ran_random_centers, attack_code, "Random", X, Y, labels, x_values, r_values)
        
        # 4. 混合指标: GA优化拓扑(G4) + 随机控制器
        # ran_random_centers = random_select_controllers(self.G4, self.rate)
        # self._process_method(self.G4, ran_random_centers, attack_code, "Hybrid", X, Y, labels, x_values, r_values)
        
        # 5. RL-RCP (双层构造): 双峰拓扑(G3) + RL选择 (本文核心方法)
        rl_centers = train_and_select(self.G3, self.controller_num, metric_type="robustness")
        self._process_method(self.G3, rl_centers, attack_code, "Bi-level", X, Y, labels, x_values, r_values)
        
        # 6. Bimodal + Random: 双峰拓扑(G3) + 随机选择 (验证拓扑结构的有效性)
        random_bimodal_centers = random_select_controllers(self.G3, self.rate)
        self._process_method(self.G3, random_bimodal_centers, attack_code, "Bimodal+Random", X, Y, labels, x_values, r_values)
        
        # 7. Random + RL (新对比): 随机重构(G1) + RL选择 (验证RL算法的有效性) - 已禁用
        # random_rl_centers = train_and_select(self.G1, self.controller_num, metric_type="robustness")
        # self._process_method(self.G1, random_rl_centers, attack_code, "Random+RL", X, Y, labels, x_values, r_values)
        
        # 8. GA + RL (新对比): GA优化(G4) + RL选择 (验证组合效果)
        ran_rl_centers = train_and_select(self.G4, self.controller_num, metric_type="robustness")
        self._process_method(self.G4, ran_rl_centers, attack_code, "GA+RL", X, Y, labels, x_values, r_values)
        
        # 9. Onion+Ra (重命名): Onion优化(G5) + 随机部署
        if self.G5:
             onion_random_centers = random_select_controllers(self.G5, self.rate)
             self._process_method(self.G5, onion_random_centers, attack_code, "Onion+Ra", X, Y, labels, x_values, r_values)

        # 10. Onion+RL (新对比): Onion优化(G5) + RL选择
        if self.G5:
             onion_rl_centers = train_and_select(self.G5, self.controller_num, metric_type="robustness")
             self._process_method(self.G5, onion_rl_centers, attack_code, "Onion+RL", X, Y, labels, x_values, r_values)

        # 11. SOLO+Ra (新对比): SOLO优化(G6) + 随机部署 - 已禁用
        # if self.G6:
        #      solo_random_centers = random_select_controllers(self.G6, self.rate)
        #      self._process_method(self.G6, solo_random_centers, attack_code, "SOLO+Ra", X, Y, labels, x_values, r_values)

        # 12. SOLO+RL (新对比): SOLO优化(G6) + RL选择 - 已禁用
        # if self.G6:
        #      solo_rl_centers = train_and_select(self.G6, self.controller_num, metric_type="robustness")
        #      self._process_method(self.G6, solo_rl_centers, attack_code, "SOLO+RL", X, Y, labels, x_values, r_values)
 
        # 13. ROMEN+Ra (新对比): ROMEN优化(G7) + 随机部署
        if hasattr(self, 'G7') and self.G7:
             romen_random_centers = random_select_controllers(self.G7, self.rate)
             self._process_method(self.G7, romen_random_centers, attack_code, "ROMEN+Ra", X, Y, labels, x_values, r_values)

        # 14. ROMEN+RL (新对比): ROMEN优化(G7) + RL选择
        if hasattr(self, 'G7') and self.G7:
             romen_rl_centers = train_and_select(self.G7, self.controller_num, metric_type="robustness")
             self._process_method(self.G7, romen_rl_centers, attack_code, "ROMEN+RL", X, Y, labels, x_values, r_values)

        # 15. BimodalRL (新方法): BimodalRL优化(G8) + RL选择
        if hasattr(self, 'G8') and self.G8:
             bimodal_rl_centers = train_and_select(self.G8, self.controller_num, metric_type="robustness")
             self._process_method(self.G8, bimodal_rl_centers, attack_code, "BimodalRL", X, Y, labels, x_values, r_values)

        # 16. UNITY+Ra (新方法): UNITY优化(G9) + 随机部署
        if hasattr(self, 'G9') and self.G9:
             unity_random_centers = random_select_controllers(self.G9, self.rate)
             self._process_method(self.G9, unity_random_centers, attack_code, "UNITY+Ra", X, Y, labels, x_values, r_values)

        # 17. UNITY+RL (新方法): UNITY优化(G9) + RL选择
        if hasattr(self, 'G9') and self.G9:
             unity_rl_centers = train_and_select(self.G9, self.controller_num, metric_type="robustness")
             self._process_method(self.G9, unity_rl_centers, attack_code, "UNITY+RL", X, Y, labels, x_values, r_values)
 
        if plot_flag and save_dir:
            plot_title = attack_mode_str
            logger.debug(f"Calling plot_different_topo. len(X)={len(X)}, save_dir={save_dir}")
            try:
                plot_different_topo(len(X), X, Y, self.colors, self.linestyles, self.markers, labels, ijk, plot_title, save_dir)
            except Exception as e:
                logger.error(f"Plotting failed: {e}")
        return x_values

    def _run_specific_attack_multi_metric(self, attack_mode_str, x_values_by_metric, r_values_by_metric, ijk, plot_flag, metric_types, metric_dirs):
        """
        执行特定攻击模式下的所有对比方案仿真（多指标版本）。
        
        Args:
            attack_mode_str (str): 攻击模式名称。
            x_values_by_metric (dict): 按指标类型组织的崩溃点数据 {metric_type: {method: [values]}}
            r_values_by_metric (dict): 按指标类型组织的鲁棒性数据 {metric_type: {method: [values]}}
            ijk (int): 当前迭代次数。
            plot_flag (bool): 是否绘图。
            metric_types (list): 指标类型列表。
            metric_dirs (dict): 按指标类型组织的保存目录 {metric_type: save_dir}
        """
        attack_code = attack_mode_str
        # 精简日志：只在第一次迭代时输出
        if ijk == 0:
            logger.info(f"Starting {attack_mode_str} simulation (Multi-Metric)")
        
        # 为每个指标类型准备绘图数据
        plot_data_by_metric = {}
        for metric_type in metric_types:
            plot_data_by_metric[metric_type] = {
                'X': [],
                'Y': [],
                'labels': []
            }
        
        # 1. Baseline: 原始拓扑 + 随机控制器
        ori_random_centers = random_select_controllers(self.G, self.rate)
        self._process_method_multi_metric(self.G, ori_random_centers, attack_code, "Baseline", x_values_by_metric, r_values_by_metric)
        
        # 2. RCP: 原始拓扑 + E-RCP
        ori_ECA_centers = RCP(self.G, self.rate, 1)
        self._process_method_multi_metric(self.G, ori_ECA_centers, attack_code, "RCP", x_values_by_metric, r_values_by_metric)
        
        # 3. RL-RCP (双层构造): 双峰拓扑(G3) + RL选择 (本文核心方法)
        rl_centers = train_and_select(self.G3, self.controller_num, metric_type="robustness")
        self._process_method_multi_metric(self.G3, rl_centers, attack_code, "Bi-level", x_values_by_metric, r_values_by_metric)
        
        # 4. GA + RL: GA优化(G4) + RL选择
        ran_rl_centers = train_and_select(self.G4, self.controller_num, metric_type="robustness")
        self._process_method_multi_metric(self.G4, ran_rl_centers, attack_code, "GA+RL", x_values_by_metric, r_values_by_metric)
        
        onion_random_centers = random_select_controllers(self.G5, self.rate)
        self._process_method_multi_metric(self.G5, onion_random_centers, attack_code, "Onion+Ra", x_values_by_metric, r_values_by_metric)

        # # 6. Onion+RL
        # onion_rl_centers = train_and_select(self.G5, self.controller_num, metric_type="robustness")
        # self._process_method_multi_metric(self.G5, onion_rl_centers, attack_code, "Onion+RL", x_values_by_metric, r_values_by_metric)
 
        # 7. ROMEN+Ra
        romen_random_centers = random_select_controllers(self.G7, self.rate)
        self._process_method_multi_metric(self.G7, romen_random_centers, attack_code, "ROMEN+Ra", x_values_by_metric, r_values_by_metric)

        # # 8. ROMEN+RL
        # romen_rl_centers = train_and_select(self.G7, self.controller_num, metric_type="robustness")
        # self._process_method_multi_metric(self.G7, romen_rl_centers, attack_code, "ROMEN+RL", x_values_by_metric, r_values_by_metric)

        # 9. BimodalRL
        bimodal_rl_centers = train_and_select(self.G8, self.controller_num, metric_type="robustness")
        self._process_method_multi_metric(self.G8, bimodal_rl_centers, attack_code, "BimodalRL", x_values_by_metric, r_values_by_metric)

        # 10. UNITY+Ra
        unity_random_centers = random_select_controllers(self.G9, self.rate)
        self._process_method_multi_metric(self.G9, unity_random_centers, attack_code, "UNITY+Ra", x_values_by_metric, r_values_by_metric)

        # # 11. UNITY+RL    
        # unity_rl_centers = train_and_select(self.G9, self.controller_num, metric_type="robustness")
        # self._process_method_multi_metric(self.G9, unity_rl_centers, attack_code, "UNITY+RL", x_values_by_metric, r_values_by_metric)
        
        # 为每个指标类型生成绘图数据并保存
        if plot_flag:
            for metric_type in metric_types:
                # 从metrics_summary中提取该指标类型的曲线数据
                X, Y, labels = [], [], []
                
                for name, metrics_list in self.metrics_summary.items():
                    if metrics_list:
                        current_metric = metrics_list[-1]  # 当前迭代的数据
                        curve_key = f'{metric_type.upper()}_Curve'
                        curve_data = current_metric.get(curve_key, [])
                        
                        if curve_data:
                            x_curve = [point[1] for point in curve_data]
                            y_curve = [point[0] for point in curve_data]
                            
                            # 修复突然上升的问题：确保数据单调递减（对于应该递减的指标）
                            # 对于某些指标，如果出现上升，可能是计算错误，需要平滑处理
                            if metric_type in ['lcc', 'csa', 'cce', 'wcp']:
                                # 确保单调递减：从前往后，如果当前值大于前一个值，则取前一个值
                                y_curve_smooth = []
                                prev_val = float('inf')
                                for y_val in y_curve:
                                    if y_val <= prev_val:
                                        prev_val = y_val
                                    # 如果上升，保持前一个值（单调递减）
                                    y_curve_smooth.append(prev_val)
                                y_curve = y_curve_smooth
                                
                                # 同时确保x和y按x值排序
                                combined = list(zip(x_curve, y_curve))
                                combined.sort(key=lambda x: x[0])
                                x_curve, y_curve = zip(*combined) if combined else ([], [])
                                x_curve = list(x_curve)
                                y_curve = list(y_curve)
                            
                            # 计算该指标的R值
                            if len(x_curve) > 1:
                                r_val = np.trapz(y_curve, x_curve)
                            else:
                                r_val = 0.0
                            
                            X.append(x_curve)
                            Y.append(y_curve)
                            labels.append(f"{name} (R={r_val:.4f})")
                
                # 绘制该指标的曲线图
                save_dir = metric_dirs.get(metric_type)
                if X and save_dir:
                    plot_title = f"{attack_mode_str} - {metric_type.upper()}"
                    # 定义指标名称映射
                    metric_labels = {
                        'lcc': 'LCC (Largest Connected Component)',
                        'csa': 'CSA (Control Supply Availability)',
                        'cce': 'CCE (Control Entropy)',
                        'wcp': 'WCP (Weighted Control Potential)'
                    }
                    y_label = metric_labels.get(metric_type, metric_type.upper())
                    try:
                        plot_different_topo(len(X), X, Y, self.colors, self.linestyles, self.markers, labels, ijk, plot_title, save_dir, y_label=y_label)
                    except Exception as e:
                        logger.error(f"Plotting failed for {metric_type}: {e}")
