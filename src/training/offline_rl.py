# -*- coding: utf-8 -*-
import sys
import os

# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import glob
import random
import networkx as nx
from src.controller.rl_optimizer import train_offline
from src.topology.generators import load_graph

def generate_synthetic_datasets(num_per_type=1000, min_nodes=10, max_nodes=500, save_dir='data/raw/syn'):
    """
    生成合成网络并保存为 GML 文件。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    graphs = []
    
    print(f"Generating synthetic datasets ({num_per_type} of each type) in {save_dir}...")
    
    # Helper to save graph
    def save_graph(G, prefix, idx):
        try:
            filename = f"{prefix}_{idx}.gml"
            path = os.path.join(save_dir, filename)
            nx.write_gml(G, path)
            return path
        except Exception as e:
            print(f"Error saving {prefix}_{idx}: {e}")
            return None

    # 1. Barabasi-Albert (Scale-Free)
    for i in range(num_per_type):
        n = random.randint(min_nodes, max_nodes)
        m = random.randint(1, min(5, n-1))
        try:
            G = nx.barabasi_albert_graph(n, m)
            if save_graph(G, "BA", i):
                graphs.append(G)
        except:
            pass
            
    # 2. Watts-Strogatz (Small-World)
    for i in range(num_per_type):
        n = random.randint(min_nodes, max_nodes)
        k = random.randint(2, min(10, n-1))
        if k % 2 != 0: k += 1
        p = random.uniform(0.1, 0.5)
        try:
            G = nx.watts_strogatz_graph(n, k, p)
            if save_graph(G, "WS", i):
                graphs.append(G)
        except:
            pass
            
    # 3. Erdos-Renyi (Random)
    for i in range(num_per_type):
        n = random.randint(min_nodes, max_nodes)
        avg_k = random.randint(4, 10)
        p = avg_k / n
        try:
            G = nx.erdos_renyi_graph(n, p)
            if nx.is_connected(G):
                if save_graph(G, "ER", i):
                    graphs.append(G)
            else:
                largest_cc = max(nx.connected_components(G), key=len)
                G_sub = G.subgraph(largest_cc).copy()
                # 重新标记节点以避免 GML 问题
                G_sub = nx.convert_node_labels_to_integers(G_sub)
                if G_sub.number_of_nodes() >= min_nodes:
                    if save_graph(G_sub, "ER", i):
                        graphs.append(G_sub)
        except:
            pass
            
    print(f"Generated and saved {len(graphs)} synthetic graphs.")
    return graphs

def load_raw_datasets(data_dir='data/raw'):
    graphs = []
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found.")
        return graphs
    
    # 递归查找所有 .gml 文件
    files = glob.glob(os.path.join(data_dir, '**/*.gml'), recursive=True)
    print(f"Loading raw datasets from {data_dir} ({len(files)} files found)...")
    
    for f in files:
        try:
            G = nx.read_gml(f, label='id')
            if G.is_multigraph():
                G = nx.Graph(G)
            
            if not nx.is_connected(G):
                largest_cc = max(nx.connected_components(G), key=len)
                G = G.subgraph(largest_cc).copy()
                
            if G.number_of_nodes() > 10:
                graphs.append(G)
        except Exception as e:
            # print(f"Error loading {f}: {e}")
            pass
            
    print(f"Loaded {len(graphs)} raw graphs.")
    return graphs

def main():
    # 1. Generate Synthetic Data and Save
    # 规模 10-500, 每种1000个
    synthetic_graphs = generate_synthetic_datasets(num_per_type=1000, min_nodes=10, max_nodes=500, save_dir='data/raw/syn')
    
    # 2. Load Raw Data (including the ones just generated if they are in data/raw subfolder)
    # 为了避免重复，我们只加载 data/raw 下的其他数据，或者直接使用 synthetic_graphs + 其他
    # load_raw_datasets 现在支持递归查找，所以它会找到刚才生成的 syn 数据
    
    all_graphs = load_raw_datasets(data_dir='data/raw')
    
    # 如果 load_raw_datasets 没找到（例如路径问题），至少使用 synthetic_graphs
    if len(all_graphs) == 0:
        all_graphs = synthetic_graphs
    
    print(f"Total training graphs: {len(all_graphs)}")
    
    if not all_graphs:
        print("No graphs found/generated. Exiting.")
        return

    # 4. Train
    # Train separate models for each objective + combined
    modes = ['combined', 'robustness', 'csa', 'entropy', 'wcp']
    
    for mode in modes:
        print(f"\n{'='*50}")
        print(f"Training model for objective: {mode}")
        print(f"{'='*50}\n")
        
        save_path = f'models/rl_agent_{mode}.pth' if mode != 'combined' else 'models/rl_agent.pth'
        
        # 为了节省时间，如果是特定指标，可以适当减少 epoch 或使用相同的 epoch
        # 这里统一使用 100 epoch
        train_offline(all_graphs, epochs=100, save_path=save_path, mode=mode)
        
    print("\nAll training tasks completed.")

if __name__ == "__main__":
    main()
