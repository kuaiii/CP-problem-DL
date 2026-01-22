# -*- coding: utf-8 -*-
"""
LCC (Largest Connected Component with Controllers) 计算详细示例
展示10节点网络的LCC计算过程
"""
import networkx as nx
import matplotlib.pyplot as plt

def calculate_lcc_detailed(G, centers, step_name=""):
    """
    详细计算LCC，并打印每一步的过程
    """
    print(f"\n{'='*60}")
    print(f"步骤: {step_name}")
    print(f"{'='*60}")
    
    print(f"\n当前网络状态:")
    print(f"  节点数: {G.number_of_nodes()}")
    print(f"  边数: {G.number_of_edges()}")
    print(f"  节点列表: {list(G.nodes())}")
    print(f"  边列表: {list(G.edges())}")
    print(f"  控制器: {centers}")
    
    # 步骤1: 检查图是否为空
    if G.number_of_nodes() == 0:
        print(f"\n结果: 图为空，LCC = 0")
        return 0
    
    # 步骤2: 找到所有连通分量
    components = list(nx.connected_components(G))
    print(f"\n步骤1: 找到所有连通分量")
    print(f"  连通分量数量: {len(components)}")
    for i, comp in enumerate(components):
        print(f"  分量 {i+1}: {sorted(comp)} (大小: {len(comp)})")
    
    if not components:
        print(f"\n结果: 没有连通分量，LCC = 0")
        return 0
    
    # 步骤3: 检查是否有控制器
    centers_set = set(centers) if centers else set()
    print(f"\n步骤2: 检查控制器")
    print(f"  控制器集合: {centers_set}")
    
    if not centers_set:
        print(f"\n结果: 没有控制器，网络完全失控，LCC = 0")
        return 0
    
    # 步骤4: 找出包含控制器的连通分量
    print(f"\n步骤3: 找出包含控制器的连通分量")
    controlled_components = []
    for i, comp in enumerate(components):
        has_controller = not centers_set.isdisjoint(comp)
        controllers_in_comp = centers_set.intersection(comp)
        print(f"  分量 {i+1} {sorted(comp)}: 包含控制器? {has_controller}, 控制器: {sorted(controllers_in_comp)}")
        if has_controller:
            controlled_components.append(comp)
    
    if not controlled_components:
        print(f"\n结果: 没有包含控制器的分量，网络完全失控，LCC = 0")
        return 0
    
    # 步骤5: 找出最大的包含控制器的连通分量
    print(f"\n步骤4: 找出最大的包含控制器的连通分量")
    max_comp = max(controlled_components, key=len)
    lcc_value = len(max_comp)
    
    print(f"  包含控制器的分量: {[sorted(comp) for comp in controlled_components]}")
    print(f"  各分量大小: {[len(comp) for comp in controlled_components]}")
    print(f"  最大分量: {sorted(max_comp)}")
    print(f"  最大分量大小: {lcc_value}")
    
    print(f"\n结果: LCC = {lcc_value}")
    return lcc_value

def example_10_nodes():
    """
    10节点网络的LCC计算示例
    """
    print("\n" + "="*60)
    print("10节点网络LCC计算详细示例")
    print("="*60)
    
    # 创建10节点网络
    # 节点编号: 0-9
    # 网络结构:
    #   0-1-2-3-4  (分量1: 5个节点)
    #   5-6-7      (分量2: 3个节点)
    #   8-9        (分量3: 2个节点)
    
    G = nx.Graph()
    
    # 分量1: 0-1-2-3-4 (链状)
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
    
    # 分量2: 5-6-7 (链状)
    G.add_edges_from([(5, 6), (6, 7)])
    
    # 分量3: 8-9 (单边)
    G.add_edge(8, 9)
    
    print("\n初始网络结构:")
    print("  分量1 (5节点): 0-1-2-3-4")
    print("  分量2 (3节点): 5-6-7")
    print("  分量3 (2节点): 8-9")
    
    # 场景1: 控制器在分量1中
    print("\n" + "="*60)
    print("场景1: 控制器在分量1中 (节点1)")
    print("="*60)
    centers1 = [1]
    lcc1 = calculate_lcc_detailed(G.copy(), centers1, "场景1: 初始状态")
    print(f"\n归一化LCC = {lcc1}/10 = {lcc1/10:.4f}")
    
    # 场景2: 控制器在分量2中
    print("\n" + "="*60)
    print("场景2: 控制器在分量2中 (节点6)")
    print("="*60)
    centers2 = [6]
    lcc2 = calculate_lcc_detailed(G.copy(), centers2, "场景2: 初始状态")
    print(f"\n归一化LCC = {lcc2}/10 = {lcc2/10:.4f}")
    
    # 场景3: 多个控制器
    print("\n" + "="*60)
    print("场景3: 多个控制器 (节点1和节点6)")
    print("="*60)
    centers3 = [1, 6]
    lcc3 = calculate_lcc_detailed(G.copy(), centers3, "场景3: 初始状态")
    print(f"\n归一化LCC = {lcc3}/10 = {lcc3/10:.4f}")
    
    # 场景4: 没有控制器
    print("\n" + "="*60)
    print("场景4: 没有控制器")
    print("="*60)
    centers4 = []
    lcc4 = calculate_lcc_detailed(G.copy(), centers4, "场景4: 初始状态")
    print(f"\n归一化LCC = {lcc4}/10 = {lcc4/10:.4f}")
    
    # 场景5: 攻击过程模拟
    print("\n" + "="*60)
    print("场景5: 攻击过程模拟 (控制器在节点1)")
    print("="*60)
    
    G_attack = G.copy()
    centers_attack = [1]
    
    # 初始状态
    lcc_initial = calculate_lcc_detailed(G_attack.copy(), centers_attack, "攻击前")
    print(f"初始LCC = {lcc_initial}, 归一化 = {lcc_initial/10:.4f}")
    
    # 攻击1: 移除节点0 (非控制器，在分量1中)
    print("\n" + "-"*60)
    print("攻击1: 移除节点0 (非控制器)")
    print("-"*60)
    G_attack.remove_node(0)
    lcc_after_0 = calculate_lcc_detailed(G_attack.copy(), centers_attack, "移除节点0后")
    print(f"LCC = {lcc_after_0}, 归一化 = {lcc_after_0/10:.4f}")
    
    # 攻击2: 移除节点1 (控制器!)
    print("\n" + "-"*60)
    print("攻击2: 移除节点1 (控制器!)")
    print("-"*60)
    G_attack.remove_node(1)
    centers_attack.remove(1)  # 移除控制器
    print("注意: 节点1是控制器，移除后控制器集合变为空")
    lcc_after_1 = calculate_lcc_detailed(G_attack.copy(), centers_attack, "移除节点1(控制器)后")
    print(f"LCC = {lcc_after_1}, 归一化 = {lcc_after_1/10:.4f}")
    print("结果: 由于没有控制器，所有分量都会级联失效，LCC=0")
    
    # 场景6: 级联失效示例
    print("\n" + "="*60)
    print("场景6: 级联失效示例")
    print("="*60)
    
    G_cascade = G.copy()
    centers_cascade = [1]  # 控制器在节点1
    
    print("\n初始状态:")
    lcc_cascade_initial = calculate_lcc_detailed(G_cascade.copy(), centers_cascade, "初始状态")
    
    # 移除节点2，导致分量1分裂
    print("\n" + "-"*60)
    print("移除节点2，导致分量1分裂为: [0,1] 和 [3,4]")
    print("-"*60)
    G_cascade.remove_node(2)
    
    # 检查级联失效
    components_after = list(nx.connected_components(G_cascade))
    print(f"\n移除节点2后的连通分量:")
    for i, comp in enumerate(components_after):
        has_controller = not set(centers_cascade).isdisjoint(comp)
        print(f"  分量 {i+1} {sorted(comp)}: 包含控制器? {has_controller}")
        if not has_controller:
            print(f"    -> 该分量会级联失效!")
    
    lcc_cascade_after = calculate_lcc_detailed(G_cascade.copy(), centers_cascade, "移除节点2后(级联失效前)")
    print(f"\n级联失效前 LCC = {lcc_cascade_after}")
    
    # 模拟级联失效: 移除没有控制器的分量 [3,4]
    print("\n执行级联失效: 移除分量 [3,4] (没有控制器)")
    G_cascade.remove_nodes_from([3, 4])
    lcc_cascade_final = calculate_lcc_detailed(G_cascade.copy(), centers_cascade, "级联失效后")
    print(f"级联失效后 LCC = {lcc_cascade_final}, 归一化 = {lcc_cascade_final/10:.4f}")

if __name__ == "__main__":
    example_10_nodes()
    
    print("\n" + "="*60)
    print("总结: LCC计算步骤")
    print("="*60)
    print("""
LCC (Largest Connected Component with Controllers) 计算步骤:

1. 检查图是否为空
   - 如果图为空，返回 0

2. 找到所有连通分量
   - 使用 nx.connected_components(G) 获取所有连通分量

3. 检查是否有控制器
   - 如果没有控制器，返回 0 (网络完全失控)

4. 找出包含控制器的连通分量
   - 对每个连通分量，检查是否与控制器集合有交集
   - 使用 centers_set.isdisjoint(comp) 检查
   - 如果 isdisjoint 返回 False，说明有交集，该分量包含控制器

5. 找出最大的包含控制器的连通分量
   - 在所有包含控制器的分量中，找出节点数最多的
   - 返回该分量的节点数

关键点:
- LCC只计算包含控制器的连通分量
- 没有控制器的分量会被级联失效移除
- 如果有多个包含控制器的分量，取最大的那个
- 归一化: LCC_normalized = LCC / 初始节点数
    """)
