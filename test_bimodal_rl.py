
import sys
import os
import networkx as nx
# Add project root to path
sys.path.append(r"e:\项目\02-论文\03-论文计划\17-BIG\code")

try:
    from src.topology.bimodal_rl import construct_bimodal_rl
    print("Import successful!")
    
    # Create a small random graph
    G = nx.erdos_renyi_graph(20, 0.3)
    print(f"Original graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Run optimization (short run)
    G_new = construct_bimodal_rl(G, generations=2)
    print(f"Optimized graph: {G_new.number_of_nodes()} nodes, {G_new.number_of_edges()} edges")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
