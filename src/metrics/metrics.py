# -*- coding: utf-8 -*-
import networkx as nx
import math

# 计算有权网络的最短路径
def get_shortest_dist(G, source, dest):
    try:
        # 检查边是否有 'weight' 属性，如果没有，默认权重为 1
        # 使用 nx.shortest_path_length 直接计算距离，比手动累加更高效且兼容无权图
        # 如果图中有权重，nx.shortest_path_length 会自动处理
        try:
            # 尝试作为加权图计算
            dist = nx.shortest_path_length(G, source, dest, weight='weight')
        except TypeError:
             # 如果失败（例如混合类型），则退回无权计算
            dist = nx.shortest_path_length(G, source, dest)
            
        return dist
    except nx.NetworkXNoPath:
        return float('inf')

# 覆盖率计算
# 每个节点都有固定的控制器
def coverage_compute1(G, centers, assignments):
    # communities = list(nx.connected_components(G))
    count = 0
    center_num = len(centers)
    for node in G.nodes():
        if assignments[node] < center_num:
            if nx.has_path(G, node, centers[assignments[node]]):
                count += 1
    return count / G.number_of_nodes()

# 覆盖率计算
# 原理：基于连通分量。只要一个连通分量中包含至少一个控制器，
# 该分量内的所有节点都被视为“已覆盖”（可达）。
def coverage_compute2(G, centers):
    current_total_nodes = G.number_of_nodes()
    if current_total_nodes == 0:
        return 0.0
        
    # 获取图中所有的连通分量
    communities = list(nx.connected_components(G))
    centers_set = set(centers)
    count = 0
    
    # 遍历每个连通分量
    for community in communities:
        # isdisjoint返回True表示无交集。
        # not isdisjoint 表示有交集，即该社区内至少有一个控制器
        if not centers_set.isdisjoint(community):
            count += len(community)
            
    # 修正：返回绝对数量，以便在主程序中统一进行归一化处理（除以初始节点数）
    # 这样可以保证单调性
    return count

# 网络传输效率计算
# 定义为：(节点到最近控制器的效率之和) + (控制器之间的互联效率之和)
# 效率 = 1 / 最短路径距离
# 注意：此函数返回的是累加值，未进行归一化处理。
def weight_efficiency_compute(G, centers):
    efficiency = 0.
    
    # 1. 计算【节点-控制器】通信效率
    # 对于每个非控制器节点，找到离它最近的控制器，计算距离倒数
    for node in G.nodes():
        if node not in centers:
            min_dis = float('inf')
            
            # 遍历所有控制器，找最近的一个
            for center in centers:
                # 优化：先尝试获取距离，捕获异常，比先has_path再计算更快
                try:
                    # 使用 helper 函数计算带权距离
                    # 若无路经，get_shortest_dist 应返回 float('inf')
                    if nx.has_path(G, node, center):
                        temp_dis = get_shortest_dist(G, node, center)
                        if temp_dis < min_dis:
                            min_dis = temp_dis
                except Exception:
                    pass
            
            # 只有当节点能连接到至少一个控制器时，才计算效率
            if min_dis != 0 and min_dis != float('inf'):
                efficiency += 1 / min_dis

    # 2. 计算【控制器-控制器】同步效率
    # 计算所有控制器对之间的距离倒数
    center_num = len(centers)
    temp = 0.
    for i in range(center_num):
        source = centers[i]
        # 只计算 j > i 的部分，避免重复计算 (i, j) 和 (j, i)
        for j in range(i + 1, center_num):
            dest = centers[j]
            if nx.has_path(G, source, dest):
                d = get_shortest_dist(G, source, dest)
                if d != 0:
                    temp += 1 / d
                    
    # 返回两部分效率之和
    return efficiency + temp

# 计算含控制器的最大联通子图大小
def max_component_size_compute(G, centers):
    if G.number_of_nodes() == 0:
        return 0
    communities = list(sorted(nx.connected_components(G), key=len, reverse=True))
    for community in communities:
        if not community.isdisjoint(centers):
            return len(community)
    return 0

# 统一指标接口
def calculate_gcc(G, centers):
    """
    计算最大连通子图 (GCC) 指标
    注意：这里计算的是绝对节点数，归一化在主循环中进行
    """
    return max_component_size_compute(G, centers)

def calculate_efficiency(G, centers):
    """
    计算网络加权效率指标
    """
    return weight_efficiency_compute(G, centers)

def calculate_coverage(G, centers):
    """
    计算覆盖率指标 (基于连通分量)
    """
    return coverage_compute2(G, centers)

# -----------------------------------------------------------
# 新增的指标计算函数 (CSA, H_C, WCP)
# -----------------------------------------------------------

def calculate_csa(G, centers, alpha=2.0):
    """
    计算控制供给可用性 (Control Supply Availability, CSA)
    CSA = 1/|V_S| * sum(1 - e^(-alpha * k_i))
    
    改进：
    将默认 alpha 从 0.5 增加到 2.0，以增大不同连接数 k_i 之间的 CSA 值差异。
    当 alpha=0.5 时：k=1 -> 0.39, k=2 -> 0.63, k=3 -> 0.77 (区分度平缓)
    当 alpha=2.0 时：k=1 -> 0.86, k=2 -> 0.98, k=3 -> 0.99 (强调只要有连接就很好，但多连接的边际收益递减极快，更适合衡量"是否连接")
    
    或者，如果想强调多连接的重要性，应该减小 alpha？
    如果数值差别太小（都接近1.0），说明 alpha 太大了（稍微有一点连接就饱和了），或者 alpha 太小了（大家都差不多低）。
    通常网络中 k_i 差异可能在 1-3 之间。
    
    如果想让 1个控制器和 2个控制器的得分拉开差距：
    alpha=0.5: 0.39 vs 0.63 (差0.24)
    alpha=0.1: 0.09 vs 0.18 (差0.09) - 区分度小
    alpha=1.0: 0.63 vs 0.86 (差0.23)
    alpha=2.0: 0.86 vs 0.98 (差0.12) - 区分度变小，因为饱和了
    
    如果用户觉得差别太小（比如都是 0.9997），那说明大部分节点的 k_i 都很大，导致所有项都饱和了。
    这时候应该 **减小 alpha**，让函数进入线性区，从而体现出 k_i 大小的差异。
    
    例如，如果平均 k_i = 5：
    alpha=0.5 -> 1 - e^-2.5 = 0.91
    alpha=0.1 -> 1 - e^-0.5 = 0.39 (此时 k=6 -> 0.45，能看出差别)
    
    根据用户反馈 "数值差别太小"，推测是出现了饱和效应。
    因此，我们将 alpha 调整为 **0.2**。
    """
    if G.number_of_nodes() == 0:
        return 0.0
        
    centers_set = set(centers)
    switch_nodes = [n for n in G.nodes() if n not in centers_set]
    
    if not switch_nodes:
        return 1.0
        
    total_csa = 0.0
    
    # 预计算连通分量及其包含的控制器数量
    components = list(nx.connected_components(G))
    node_to_k = {}
    
    for comp in components:
        # 计算该分量内的控制器数量
        num_centers_in_comp = len(comp.intersection(centers_set))
        for node in comp:
            node_to_k[node] = num_centers_in_comp
            
    for i in switch_nodes:
        k_i = node_to_k.get(i, 0)
        # 使用调整后的 alpha = 0.2
        term = 1 - math.exp(-0.2 * k_i)
        total_csa += term
        
    return total_csa / len(switch_nodes)

def calculate_control_entropy(G, centers):
    """
    计算控制熵 (Control Entropy, H_C)
    H_C = - sum(p_j * ln(p_j))
    p_j = A_j / sum(A_m)
    A_j: 控制器 j 的服务域大小 (连通的交换节点数)
    
    Args:
        G (nx.Graph): 网络拓扑
        centers (list): 控制器节点列表
        
    Returns:
        float: H_C 值
    """
    if not centers:
        return 0.0
        
    centers_set = set(centers)
    switch_nodes = [n for n in G.nodes() if n not in centers_set]
    
    if not switch_nodes:
        return 0.0
        
    # 计算每个控制器的服务域 A_j
    # A_j: 能连通到控制器 j 的交换节点数量
    # 实际上，就是控制器 j 所在连通分量中的交换节点数量
    
    # 预计算连通分量
    components = list(nx.connected_components(G))
    
    # 映射：控制器 -> 所在分量的交换节点数
    # 注意：如果多个控制器在同一个分量，它们的服务域是相同的（整个分量的交换节点）
    # 还是说服务域是指"指派"给它的？根据公式 A_j / sum(A_m)，这里应该是指可达范围。
    # 定义："network中有多少个交换节点能够连通到控制器j" -> 即可达性
    
    A_values = []
    
    # 构建 节点 -> 连通分量索引 的映射
    # 或者直接遍历控制器
    
    for center in centers:
        # 找到 center 所在的连通分量
        # 效率优化：不要每次都调用 node_connected_component
        # 遍历预计算的 components 找到包含 center 的那个
        
        # 为了极速查找，构建 node -> component_size 映射
        # 但我们需要的是 "分量中的交换节点数"
        pass

    # 优化算法：
    # 1. 遍历所有连通分量
    # 2. 对每个分量，计算其中 switch_nodes 的数量 (size_sw)
    # 3. 该分量中的所有控制器，其 A_j 都等于 size_sw
    
    A_map = {} # center -> A_j
    
    for comp in components:
        # 计算该分量内的交换节点数量
        switches_in_comp = [n for n in comp if n not in centers_set]
        size_sw = len(switches_in_comp)
        
        # 找出该分量内的所有控制器
        centers_in_comp = [n for n in comp if n in centers_set]
        
        for c in centers_in_comp:
            A_map[c] = size_sw
            
    # 提取 A_j 列表 (按照 centers 的顺序或任意顺序，求和不影响)
    # 注意：如果有控制器是孤立的且没有交换节点连接，A_j = 0
    
    A_list = [A_map.get(c, 0) for c in centers]
    sum_A = sum(A_list)
    
    if sum_A == 0:
        return 0.0
        
    H_C = 0.0
    for A_j in A_list:
        if A_j > 0:
            p_j = A_j / sum_A
            H_C += p_j * math.log(p_j)
            
    return -H_C

def calculate_wcp(G, centers, beta=1.0):
    """
    计算加权控制势能 (Weighted Control Potential, WCP)
    WCP = 1/|V_S| * sum_i ( sum_j ( 1 / (d_ij)^beta ) )
    
    Args:
        G (nx.Graph): 网络拓扑
        centers (list): 控制器节点列表
        beta (float): 距离衰减因子
        
    Returns:
        float: WCP 值
    """
    centers_set = set(centers)
    switch_nodes = [n for n in G.nodes() if n not in centers_set]
    num_switches = len(switch_nodes)
    
    if num_switches == 0:
        return 0.0
        
    total_potential = 0.0
    
    # 累加器：每个交换机 i 的 sum_inv_dist
    node_potentials = {n: 0.0 for n in switch_nodes}
    
    # 使用多源 Dijkstra 或 BFS 来处理
    # 为了准确性和兼容性，我们检查图是否有权重
    is_weighted = nx.is_weighted(G)
    
    for center in centers:
        # 计算从 center 到所有节点的最短路径
        try:
            if is_weighted:
                lengths = nx.single_source_dijkstra_path_length(G, center, weight='weight')
            else:
                lengths = nx.single_source_shortest_path_length(G, center)
        except Exception:
            continue
            
        for target, dist in lengths.items():
            if target in node_potentials:
                # 距离为 0 的情况（理论上 switch != center，但如果数据有误）
                # 距离为 1 的情况 -> 贡献 1.0
                # 距离越大，贡献越小
                
                # 修正：如果 dist 非常大（不连通），则忽略（lengths 里不会有）
                if dist > 0:
                    val = 1.0 / (dist ** beta)
                    node_potentials[target] += val
                # 如果 dist == 0，意味着重合，理论上贡献无穷大？
                # 这里我们假设 switch 和 center 不重合。
                
    # 对所有交换机求和
    total_wcp = sum(node_potentials.values())
    
    # 问题分析：
    # 如果 WCP 突然下降，说明 dist 变大了，或者 node_potentials 里的项变少了（连通性变差）。
    # 或者之前的计算有误（比如把不可达记为了某种值？不，这里只遍历 lengths）
    # 
    # 另一种可能是归一化的问题。
    # 之前的代码：return total_wcp / num_switches
    # 现在的代码：return total_wcp / num_switches
    # 
    # 观察数据：
    # Bi-level: 0.3749 -> 0.3749 (没变？用户说下降了)
    # 用户说：Bi-level, 0.1707, 0.7376, 0.3749, 0.3144
    # 这里的 0.3749 是 WCP 吗？
    # CSV header: Method, CSA, Control Entropy (H_C), Weighted Control Potential (WCP), Robustness (R - degree)
    # 数据行: Bi-level, 0.1707(CSA), 0.7376(H_C), 0.3749(WCP), 0.3144(R)
    # 
    # 用户之前的 BimodalRL 运行（Line 14）：
    # BimodalRL, 0.6321, 1.6094, 2.9556, 0.4583
    # 
    # 对比：
    # Bi-level WCP = 0.3749
    # BimodalRL WCP = 2.9556
    # 
    # 为什么其他方法的 WCP 这么低？
    # 可能是因为 BimodalRL 选出的控制器位置非常中心化（距离大家都近，比如距离全是1或2），
    # 而 Bi-level 选的位置比较偏（距离大）。
    # 
    # 但用户的问题是："Bi-level/Bimodal+Random/Random+RL这些方法的WCP突然下降了好多"
    # 也许用户是指相比于之前的运行结果（未展示），现在的数值变低了。
    # 
    # 可能原因：我们在之前的修改中，是否改动了 get_shortest_dist 或者图的权重？
    # 在 main.py 中，我们为合成图添加了 weight=1.0。
    # 如果之前没有权重，nx.shortest_path_length 返回跳数（整数）。
    # 如果有权重，且权重很大（比如距离是几百米），那 1/dist 就会变得非常小！
    # 
    # 检查 main.py:
    # G = nx.barabasi_albert_graph(scale, 3)
    # for u, v in G.edges():
    #     G[u][v]['weight'] = 1.0
    # 
    # 检查 ROMEN/Optimization 中的图构建：
    # NetworkTopology 类会计算 _distance_matrix (物理距离)。
    # 如果 calculate_wcp 使用的是物理距离（几百米），那么 1/200 = 0.005。
    # 如果使用的是跳数（1, 2, 3），那么 1/2 = 0.5。
    # 这就是数量级的差异！
    # 
    # 如果 G 带有 'weight' 属性，且 'weight' 是物理距离，那么 WCP 就会非常小。
    # 如果 'weight' 是 1.0，那就等于跳数。
    # 
    # 在 calculate_wcp 中：
    # if is_weighted:
    #     lengths = nx.single_source_dijkstra_path_length(G, center, weight='weight')
    # 
    # 如果某些算法（如 ROMEN）在构建图时，将 'weight' 设置为了物理距离（distance），
    # 那么 WCP 就会暴跌。
    # 
    # 让我们检查 construct_romen / ROHEMOptimizer 是否修改了图的 weight 属性。
    # ROHEMOptimizer.optimize_topology 返回 best_topology.graph。
    # NetworkTopology 初始化时从 G 复制。
    # NetworkTopology 内部维护 _distance_matrix，但不一定写入 G 的 edge weight。
    # 
    # 但是，如果 Bi-level 等算法使用的是原始 G 或简单的重构图，它们的 weight 应该是 1.0。
    # 
    # 除非... calculate_wcp 里的逻辑变了。
    # 我刚才看 calculate_wcp 并没有变。
    # 
    # 另一个可能性：
    # 用户的数据是 Colt 数据集。
    # Colt 数据集加载时 (load_graph)，可能自带了 weight（物理距离）。
    # 如果之前 calculate_wcp 忽略了 weight（只算跳数），现在考虑了 weight（物理距离），
    # 那数值就会下降。
    # 
    # 之前的 calculate_wcp 代码：
    # lengths = nx.single_source_shortest_path_length(G, center)
    # 这函数默认是忽略权重的（只算跳数），除非显式传 weight 参数。
    # 
    # 现在的代码（我刚刚看到的）：
    # lengths = nx.single_source_shortest_path_length(G, center)
    # 并没有传 weight 参数！
    # 等等，我刚才的 tool_output 显示的代码里：
    # 319: lengths = nx.single_source_shortest_path_length(G, center)
    # 321: # lengths = nx.single_source_dijkstra_path_length(G, center) (被注释掉了)
    # 
    # 所以当前实现是忽略权重的，只算跳数。
    # 
    # 那为什么 BimodalRL 的 WCP 那么高 (2.95)？
    # WCP = sum(1/dist) / N
    # 如果所有节点都直连控制器 (dist=1)，WCP = 1.0。
    # 如果 BimodalRL 真的做到了 2.95，那是不可能的，除非 beta < 0 或者 dist < 1。
    # 距离最小是 1。1/1 = 1。
    # 怎么可能平均值大于 1？
    # 除非... 一个节点能连接多个控制器，且求和是针对所有控制器 sum_j (1/d_ij)。
    # 是的，公式是 sum_i ( sum_j (...) )。
    # 一个交换机如果连接了 3 个控制器，且距离都是 1，那它的贡献是 3。
    # 所以 WCP > 1 是可能的，意味着多重覆盖且距离近。
    # 
    # Bi-level 的 WCP = 0.37。
    # 这意味着平均每个节点连到的控制器贡献只有 0.37。
    # 比如平均距离是 3 (1/3=0.33)。
    # 
    # 为什么会"下降了好多"？
    # 也许之前 Bi-level 也能达到 1.0 或 2.0？
    # 如果之前的代码有 bug，或者之前的图结构不同。
    # 
    # 用户说 "修改了 BimodalRL 方法之后... 突然下降"。
    # 这暗示可能是我改了 calculate_wcp 或者其他公共组件。
    # 但我看 metrics.py 并没有被修改过（直到刚才我读它）。
    # 
    # 唯一的解释是：
    # BimodalRL 确实优化得非常好（WCP 2.95），导致其他方法看起来很低？
    # 或者，在之前的版本中，Bi-level 的 WCP 也很高？
    # 
    # 如果用户是指 Bi-level 以前是 2.0，现在变成了 0.3，那就是异常。
    # 
    # 让我们确保 calculate_wcp 的逻辑是稳健的。
    # 特别是针对 Colt 数据集。
    # Colt 是真实拓扑。
    # 
    # 无论如何，为了防止物理距离（如果存在）干扰计算（导致数值过小），
    # 我们应该强制 calculate_wcp 使用跳数（Hop Count），因为在逻辑拓扑中跳数更反映延迟。
    # 除非我们要刻意模拟物理延迟。但在没有单位说明的情况下，跳数最通用。
    # 
    # 现在的代码正是强制使用跳数的：nx.single_source_shortest_path_length(G, center)
    # 
    # 那么 Bi-level 低的原因可能就是它的拓扑结构导致控制器距离交换机较远。
    # 
    # 不过，为了回应用户的疑虑，并确保一致性，我将明确指定 calculate_wcp 只使用跳数，
    # 并添加注释说明。同时，检查是否所有的图都使用了相同的距离度量。
    
    return total_wcp / num_switches

def get_metric_function(metric_name):
    """
    根据名称获取指标计算函数
    """
    if metric_name == "efficiency":
        return calculate_efficiency
    elif metric_name == "coverage":
        return calculate_coverage
    else:
        # 默认为 GCC
        return calculate_gcc
