# -*- coding: utf-8 -*-
"""分析不同网络的结构特征"""
import networkx as nx
import numpy as np
from src.topology.generators import load_graph

# 分析网络特征
print("加载网络...")
cogentco_g, _ = load_graph('data/testdata/Cogentco.gml')
colt_g, _ = load_graph('data/testdata/Colt.gml')

networks = {
    'BA_300': nx.barabasi_albert_graph(300, 3, seed=42),
    'Cogentco': cogentco_g,
    'Colt': colt_g
}

print('=== 网络结构特征对比 ===\n')
print(f"{'Network':<12} {'N':>6} {'M':>6} {'密度':>8} {'平均度':>8} {'最大度':>8} {'度方差':>10} {'聚类系数':>8}")
print('-' * 80)

for name, G in networks.items():
    G = nx.Graph(G)  # 转换为无向图
    G.remove_edges_from(nx.selfloop_edges(G))  # 移除自环
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = 2*m / (n*(n-1)) if n > 1 else 0
    degrees = [d for _, d in G.degree()]
    avg_deg = np.mean(degrees)
    max_deg = np.max(degrees)
    var_deg = np.var(degrees)
    clustering = nx.average_clustering(G)
    
    print(f'{name:<12} {n:>6} {m:>6} {density:>8.4f} {avg_deg:>8.2f} {max_deg:>8} {var_deg:>10.2f} {clustering:>8.4f}')

print()
print('=== 度分布分析 ===\n')
for name, G in networks.items():
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    
    degrees = [d for _, d in G.degree()]
    
    # 计算双峰系数
    n_deg = len(degrees)
    mean = np.mean(degrees)
    std = np.std(degrees)
    if std > 1e-6:
        normalized = [(d - mean) / std for d in degrees]
        skewness = np.mean([x**3 for x in normalized])
        kurtosis_val = np.mean([x**4 for x in normalized])
        if kurtosis_val > 1e-6:
            bc = (skewness**2 + 1) / kurtosis_val
        else:
            bc = 0
    else:
        bc = 0
    
    # 度分布特征
    unique_degrees = len(set(degrees))
    degree_entropy = -sum([(degrees.count(d)/n_deg) * np.log(degrees.count(d)/n_deg + 1e-10) for d in set(degrees)])
    
    print(f'{name}:')
    print(f'  度范围: [{min(degrees)}, {max(degrees)}]')
    print(f'  度分布熵: {degree_entropy:.3f}')
    print(f'  双峰系数 (BC): {bc:.4f}')
    print(f'  偏度: {skewness:.3f}')
    print(f'  峰度: {kurtosis_val:.3f}')
    print()

print('=== Bi-level方法问题分析 ===\n')
print("""
根据以上分析，Bi-level方法在实际网络上效果不佳的原因可能是：

1. 【双峰优化目标不适配】
   - BA网络本身具有幂律度分布，双峰优化可以强化hub节点结构
   - 实际网络度分布已经不规则，强制双峰化可能破坏原有的有效结构

2. 【快速鲁棒性估计的偏差】
   - 使用 comp_ratio * 0.7 + var_ratio * 0.3 作为鲁棒性代理
   - 这个代理在BA网络上可能准确，但在实际网络上可能有偏差

3. 【边交换的局限性】
   - BA网络结构规整，边交换容易找到改进方向
   - 实际网络有地理约束和社区结构，边交换可能破坏这些结构
""")
