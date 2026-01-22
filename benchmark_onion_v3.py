# -*- coding: utf-8 -*-
"""
ONION V3 性能测试脚本

对比原始版本和 V3 优化版本的性能差异:
1. 单次鲁棒性计算时间
2. 完整优化流程时间
3. 不同图规模下的加速比

核心优化:
- 逆向重组法 + Union-Find: O(K(N+M)) -> O(M·α(N))
- 移除无效哈希缓存
- 攻击序列只计算一次
"""
import time
import numpy as np
import networkx as nx
import pandas as pd
from tqdm import tqdm

from src.topology.onion_optimization import OnionOptimizer
from src.topology.onion_optimization_v3 import OnionOptimizerV3, benchmark_robustness_methods


def benchmark_single_robustness_calculation(graph_sizes=[50, 100, 150, 200, 300], 
                                            avg_degree=4, 
                                            num_runs=5):
    """
    对比单次鲁棒性计算的性能
    """
    print("=" * 60)
    print("单次鲁棒性计算性能对比")
    print("=" * 60)
    
    results = []
    
    for n in graph_sizes:
        m = n * avg_degree // 2  # 大约的边数
        G = nx.barabasi_albert_graph(n, avg_degree // 2, seed=42)
        
        print(f"\n--- 图规模: N={n}, M={G.number_of_edges()} ---")
        
        # 原始版本
        optimizer_v1 = OnionOptimizer(G)
        times_v1 = []
        for _ in range(num_runs):
            # 清除缓存以确保公平比较
            optimizer_v1._robustness_cache.clear()
            start = time.perf_counter()
            r_v1 = optimizer_v1.robustness_measure_fast(G)
            times_v1.append(time.perf_counter() - start)
        avg_v1 = np.mean(times_v1) * 1000
        
        # V3 版本
        optimizer_v3 = OnionOptimizerV3(G)
        times_v3 = []
        for _ in range(num_runs):
            start = time.perf_counter()
            r_v3 = optimizer_v3.robustness_measure_reverse_union_find(optimizer_v3.edge_set)
            times_v3.append(time.perf_counter() - start)
        avg_v3 = np.mean(times_v3) * 1000
        
        speedup = avg_v1 / avg_v3 if avg_v3 > 0 else float('inf')
        
        print(f"  原始版本 (V1): {avg_v1:.3f} ms, R={r_v1:.4f}")
        print(f"  优化版本 (V3): {avg_v3:.3f} ms, R={r_v3:.4f}")
        print(f"  加速比: {speedup:.1f}x")
        print(f"  结果差异: {abs(r_v1 - r_v3):.6f}")
        
        results.append({
            'N': n,
            'M': G.number_of_edges(),
            'V1_time_ms': avg_v1,
            'V3_time_ms': avg_v3,
            'speedup': speedup,
            'R_v1': r_v1,
            'R_v3': r_v3,
            'R_diff': abs(r_v1 - r_v3)
        })
    
    return pd.DataFrame(results)


def benchmark_full_optimization(graph_sizes=[50, 100, 150], 
                                 avg_degree=4, 
                                 max_iterations=100):
    """
    对比完整优化流程的性能
    """
    print("\n" + "=" * 60)
    print("完整优化流程性能对比")
    print(f"最大迭代次数: {max_iterations}")
    print("=" * 60)
    
    results = []
    
    for n in graph_sizes:
        G = nx.barabasi_albert_graph(n, avg_degree // 2, seed=42)
        
        print(f"\n--- 图规模: N={n}, M={G.number_of_edges()} ---")
        
        # 原始版本
        print("  运行原始版本 (V1)...")
        optimizer_v1 = OnionOptimizer(G.copy())
        start = time.perf_counter()
        G_opt_v1 = optimizer_v1.optimize_network(max_iterations=max_iterations, verbose=False)
        time_v1 = time.perf_counter() - start
        
        # V3 版本
        print("  运行优化版本 (V3)...")
        optimizer_v3 = OnionOptimizerV3(G.copy())
        start = time.perf_counter()
        G_opt_v3 = optimizer_v3.optimize_network(max_iterations=max_iterations, verbose=False)
        time_v3 = time.perf_counter() - start
        
        # 计算最终鲁棒性（使用V3方法确保一致性）
        final_optimizer = OnionOptimizerV3(G_opt_v1)
        r_final_v1 = final_optimizer.robustness_measure_reverse_union_find(final_optimizer.edge_set)
        
        final_optimizer = OnionOptimizerV3(G_opt_v3)
        r_final_v3 = final_optimizer.robustness_measure_reverse_union_find(final_optimizer.edge_set)
        
        speedup = time_v1 / time_v3 if time_v3 > 0 else float('inf')
        
        print(f"  原始版本 (V1): {time_v1:.2f}s, 最终 R={r_final_v1:.4f}")
        print(f"  优化版本 (V3): {time_v3:.2f}s, 最终 R={r_final_v3:.4f}")
        print(f"  加速比: {speedup:.1f}x")
        
        results.append({
            'N': n,
            'M': G.number_of_edges(),
            'V1_time_s': time_v1,
            'V3_time_s': time_v3,
            'speedup': speedup,
            'R_v1': r_final_v1,
            'R_v3': r_final_v3
        })
    
    return pd.DataFrame(results)


def benchmark_with_real_graph(gml_path: str, max_iterations: int = 200):
    """
    使用真实网络数据测试
    """
    print("\n" + "=" * 60)
    print(f"真实网络测试: {gml_path}")
    print("=" * 60)
    
    G = nx.read_gml(gml_path)
    G = nx.Graph(G)  # 转换为无向图
    
    # 移除自环
    G.remove_edges_from(nx.selfloop_edges(G))
    
    # 取最大连通分量
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    print(f"图规模: N={n}, M={m}")
    
    # 单次鲁棒性计算对比
    print("\n--- 单次鲁棒性计算 ---")
    
    optimizer_v1 = OnionOptimizer(G)
    times_v1 = []
    for _ in range(5):
        optimizer_v1._robustness_cache.clear()
        start = time.perf_counter()
        r_v1 = optimizer_v1.robustness_measure_fast(G)
        times_v1.append(time.perf_counter() - start)
    avg_v1 = np.mean(times_v1) * 1000
    
    optimizer_v3 = OnionOptimizerV3(G)
    times_v3 = []
    for _ in range(5):
        start = time.perf_counter()
        r_v3 = optimizer_v3.robustness_measure_reverse_union_find(optimizer_v3.edge_set)
        times_v3.append(time.perf_counter() - start)
    avg_v3 = np.mean(times_v3) * 1000
    
    print(f"  原始版本 (V1): {avg_v1:.3f} ms, R={r_v1:.4f}")
    print(f"  优化版本 (V3): {avg_v3:.3f} ms, R={r_v3:.4f}")
    print(f"  加速比: {avg_v1/avg_v3:.1f}x")
    
    # 完整优化对比
    print(f"\n--- 完整优化 (max_iter={max_iterations}) ---")
    
    print("  运行原始版本 (V1)...")
    optimizer_v1 = OnionOptimizer(G.copy())
    start = time.perf_counter()
    G_opt_v1 = optimizer_v1.optimize_network(max_iterations=max_iterations, verbose=True)
    time_v1 = time.perf_counter() - start
    
    print("\n  运行优化版本 (V3)...")
    optimizer_v3 = OnionOptimizerV3(G.copy())
    start = time.perf_counter()
    G_opt_v3 = optimizer_v3.optimize_network(max_iterations=max_iterations, verbose=True)
    time_v3 = time.perf_counter() - start
    
    print(f"\n  原始版本 (V1): {time_v1:.2f}s")
    print(f"  优化版本 (V3): {time_v3:.2f}s")
    print(f"  加速比: {time_v1/time_v3:.1f}x")


def main():
    """主测试函数"""
    print("=" * 60)
    print("ONION V3 性能测试")
    print("=" * 60)
    print("\n核心优化:")
    print("1. 逆向重组法 + Union-Find: O(K(N+M)) -> O(M·α(N))")
    print("2. 移除无效哈希缓存")
    print("3. 攻击序列只计算一次（度保持性质）")
    print()
    
    # 测试1: 单次鲁棒性计算
    df_single = benchmark_single_robustness_calculation(
        graph_sizes=[50, 100, 150, 200, 250, 300],
        num_runs=10
    )
    print("\n单次计算结果汇总:")
    print(df_single.to_string(index=False))
    
    # 测试2: 完整优化流程
    df_full = benchmark_full_optimization(
        graph_sizes=[50, 100, 150, 200],
        max_iterations=150
    )
    print("\n完整优化结果汇总:")
    print(df_full.to_string(index=False))
    
    # 测试3: 使用真实网络（如果存在）
    import os
    test_files = [
        "data/testdata/Cogentco.gml",
        "data/testdata/Colt.gml",
        "data/testdata/GtsCe.gml"
    ]
    
    for path in test_files:
        if os.path.exists(path):
            try:
                benchmark_with_real_graph(path, max_iterations=200)
            except Exception as e:
                print(f"测试 {path} 失败: {e}")


if __name__ == "__main__":
    main()
