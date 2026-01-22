# -*- coding: utf-8 -*-
"""
验证LCC计算代码的正确性
对比实际代码和预期结果
"""
import networkx as nx
from src.metrics.metrics import calculate_lcc

def test_lcc_cases():
    """
    测试各种LCC计算场景
    """
    print("="*60)
    print("LCC计算代码验证测试")
    print("="*60)
    
    # 测试用例1: 简单连通图，控制器在中间
    print("\n测试用例1: 简单链状图 0-1-2-3-4，控制器在节点1")
    G1 = nx.Graph()
    G1.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
    centers1 = [1]
    lcc1 = calculate_lcc(G1, centers1)
    expected1 = 5
    print(f"  结果: LCC = {lcc1}, 期望 = {expected1}, {'✓' if lcc1 == expected1 else '✗'}")
    assert lcc1 == expected1, f"测试失败: 期望 {expected1}, 得到 {lcc1}"
    
    # 测试用例2: 多个连通分量，控制器在其中一个
    print("\n测试用例2: 三个连通分量，控制器在分量1")
    G2 = nx.Graph()
    G2.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])  # 分量1: 5节点
    G2.add_edges_from([(5, 6), (6, 7)])  # 分量2: 3节点
    G2.add_edge(8, 9)  # 分量3: 2节点
    centers2 = [1]
    lcc2 = calculate_lcc(G2, centers2)
    expected2 = 5
    print(f"  结果: LCC = {lcc2}, 期望 = {expected2}, {'✓' if lcc2 == expected2 else '✗'}")
    assert lcc2 == expected2, f"测试失败: 期望 {expected2}, 得到 {lcc2}"
    
    # 测试用例3: 多个控制器，在不同分量中
    print("\n测试用例3: 多个控制器，在不同分量中")
    G3 = nx.Graph()
    G3.add_edges_from([(0, 1), (1, 2)])  # 分量1: 3节点
    G3.add_edges_from([(3, 4), (4, 5)])  # 分量2: 3节点
    centers3 = [1, 4]
    lcc3 = calculate_lcc(G3, centers3)
    expected3 = 3  # 两个分量都包含控制器，取最大的
    print(f"  结果: LCC = {lcc3}, 期望 = {expected3}, {'✓' if lcc3 == expected3 else '✗'}")
    assert lcc3 == expected3, f"测试失败: 期望 {expected3}, 得到 {lcc3}"
    
    # 测试用例4: 没有控制器
    print("\n测试用例4: 没有控制器")
    G4 = nx.Graph()
    G4.add_edges_from([(0, 1), (1, 2)])
    centers4 = []
    lcc4 = calculate_lcc(G4, centers4)
    expected4 = 0
    print(f"  结果: LCC = {lcc4}, 期望 = {expected4}, {'✓' if lcc4 == expected4 else '✗'}")
    assert lcc4 == expected4, f"测试失败: 期望 {expected4}, 得到 {lcc4}"
    
    # 测试用例5: 空图
    print("\n测试用例5: 空图")
    G5 = nx.Graph()
    centers5 = [1]
    lcc5 = calculate_lcc(G5, centers5)
    expected5 = 0
    print(f"  结果: LCC = {lcc5}, 期望 = {expected5}, {'✓' if lcc5 == expected5 else '✗'}")
    assert lcc5 == expected5, f"测试失败: 期望 {expected5}, 得到 {lcc5}"
    
    # 测试用例6: 控制器不在任何连通分量中（理论上不可能，但测试边界情况）
    print("\n测试用例6: 控制器节点不在图中")
    G6 = nx.Graph()
    G6.add_edges_from([(0, 1), (1, 2)])
    centers6 = [99]  # 节点99不在图中
    lcc6 = calculate_lcc(G6, centers6)
    expected6 = 0  # 没有包含控制器的分量
    print(f"  结果: LCC = {lcc6}, 期望 = {expected6}, {'✓' if lcc6 == expected6 else '✗'}")
    assert lcc6 == expected6, f"测试失败: 期望 {expected6}, 得到 {lcc6}"
    
    # 测试用例7: 单个节点，是控制器
    print("\n测试用例7: 单个节点，是控制器")
    G7 = nx.Graph()
    G7.add_node(0)
    centers7 = [0]
    lcc7 = calculate_lcc(G7, centers7)
    expected7 = 1
    print(f"  结果: LCC = {lcc7}, 期望 = {expected7}, {'✓' if lcc7 == expected7 else '✗'}")
    assert lcc7 == expected7, f"测试失败: 期望 {expected7}, 得到 {lcc7}"
    
    # 测试用例8: 单个节点，不是控制器
    print("\n测试用例8: 单个节点，不是控制器")
    G8 = nx.Graph()
    G8.add_node(0)
    centers8 = []
    lcc8 = calculate_lcc(G8, centers8)
    expected8 = 0
    print(f"  结果: LCC = {lcc8}, 期望 = {expected8}, {'✓' if lcc8 == expected8 else '✗'}")
    assert lcc8 == expected8, f"测试失败: 期望 {expected8}, 得到 {lcc8}"
    
    # 测试用例9: 复杂网络，控制器在最大分量中
    print("\n测试用例9: 复杂网络，控制器在最大分量中")
    G9 = nx.Graph()
    # 大分量: 0-1-2-3-4-5 (6节点)
    G9.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
    # 小分量: 6-7 (2节点)
    G9.add_edge(6, 7)
    centers9 = [2]  # 控制器在大分量中
    lcc9 = calculate_lcc(G9, centers9)
    expected9 = 6
    print(f"  结果: LCC = {lcc9}, 期望 = {expected9}, {'✓' if lcc9 == expected9 else '✗'}")
    assert lcc9 == expected9, f"测试失败: 期望 {expected9}, 得到 {lcc9}"
    
    # 测试用例10: 复杂网络，控制器在小分量中
    print("\n测试用例10: 复杂网络，控制器在小分量中")
    G10 = nx.Graph()
    # 大分量: 0-1-2-3-4-5 (6节点)
    G10.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
    # 小分量: 6-7 (2节点)
    G10.add_edge(6, 7)
    centers10 = [6]  # 控制器在小分量中
    lcc10 = calculate_lcc(G10, centers10)
    expected10 = 2  # 只有小分量包含控制器
    print(f"  结果: LCC = {lcc10}, 期望 = {expected10}, {'✓' if lcc10 == expected10 else '✗'}")
    assert lcc10 == expected10, f"测试失败: 期望 {expected10}, 得到 {lcc10}"
    
    print("\n" + "="*60)
    print("所有测试用例通过! ✓")
    print("="*60)

if __name__ == "__main__":
    try:
        test_lcc_cases()
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
