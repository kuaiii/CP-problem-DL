import pandas as pd
import os
import sys

def calculate_weighted_r(random_file, degree_file, output_file="weighted_robustness.csv"):
    """
    计算加权鲁棒性 R = 0.5 * R_random + 0.5 * R_degree
    """
    print(f"Reading Random Attack Results: {random_file}")
    print(f"Reading Degree Attack Results: {degree_file}")
    
    if not os.path.exists(random_file):
        print(f"Error: Random file not found: {random_file}")
        return
    if not os.path.exists(degree_file):
        print(f"Error: Degree file not found: {degree_file}")
        return
        
    try:
        df_rand = pd.read_csv(random_file)
        df_deg = pd.read_csv(degree_file)
        
        # 假设第一列是 Method，最后一列是 Robustness
        # 或者是根据列名
        # Random file cols: Method, CSA, H_C, WCP, Robustness (R - random)
        # Degree file cols: Method, CSA, H_C, WCP, Robustness (R - degree)
        
        # 重命名列以便合并
        df_rand = df_rand.rename(columns={df_rand.columns[-1]: 'R_random'})
        df_deg = df_deg.rename(columns={df_deg.columns[-1]: 'R_degree'})
        
        # 提取 R 值列
        # 确保方法顺序一致，或者进行 merge
        df_merged = pd.merge(df_rand[['Method', 'R_random']], df_deg[['Method', 'R_degree']], on='Method')
        
        # 计算加权 R
        df_merged['Weighted_R'] = 0.5 * df_merged['R_random'] + 0.5 * df_merged['R_degree']
        
        # 排序
        df_merged = df_merged.sort_values(by='Weighted_R', ascending=False)
        
        print("\n--- Weighted Robustness Results ---")
        print(df_merged.to_string(index=False))
        
        df_merged.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        
    except Exception as e:
        print(f"Error processing files: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Colt", help="Dataset name")
    args = parser.parse_args()
    
    base_dir = f"results/{args.dataset}"
    
    def get_latest_metrics(attack_type):
        path = os.path.join(base_dir, attack_type, "gcc")
        if not os.path.exists(path):
            return None
        subdirs = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        if not subdirs:
            return None
        latest = max(subdirs, key=os.path.getmtime)
        return os.path.join(latest, "comprehensive_metrics.csv")
    
    rand_file = get_latest_metrics("random")
    deg_file = get_latest_metrics("degree")
    
    if rand_file and deg_file:
        output_filename = f"{args.dataset}_weighted_robustness.csv"
        calculate_weighted_r(rand_file, deg_file, output_file=output_filename)
    else:
        print(f"Could not automatically find result files for {args.dataset}. Please check directories.")
