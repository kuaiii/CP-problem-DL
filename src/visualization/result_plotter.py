# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.pyplot import MultipleLocator
import seaborn as sns
from matplotlib import font_manager
import os
import numpy as np

# 设置中文字体
# 尝试查找系统中的中文字体文件（如 SimHei），如果找到则加载，否则回退到备用字体列表。
# 这是为了解决 Matplotlib 在默认情况下无法显示中文字符的问题。
try:
    font_path = 'C:/Windows/Fonts/simhei.ttf' 
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
    else:
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft YaHei', 'SimHei', 'sans-serif']
except:
    pass
    
# 解决负号显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

def plot_line(x_ran, y_ran, x_tar, y_tar):
    """
    绘制两种攻击模式（随机攻击 vs 定向攻击）下的对比折线图。
    主要用于简单的二元对比展示。

    Args:
        x_ran (list): 随机攻击下的 X 轴数据（移除比例）。
        y_ran (list): 随机攻击下的 Y 轴数据（GCC 大小）。
        x_tar (list): 定向攻击下的 X 轴数据。
        y_tar (list): 定向攻击下的 Y 轴数据。
    """
    plt.figure()
    plt.plot(x_ran, y_ran, marker='.', color='#1f77b4', linestyle='-', label='Random')
    plt.plot(x_tar, y_tar, marker='<', color='#ff7f0e', linestyle='--', label='Target')
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']
    rcParams['axes.unicode_minus'] = False
    plt.xlabel("Proportion of Removed Nodes : f")
    plt.ylabel("Proportion of the Giant Connected Component")
    plt.legend()
    plt.show()

def plot_different_topo(nums, X, Y, colors, linestyles, markers, labels, number, name, save_dir, y_label="Resilience"):
    """
    绘制不同拓扑或部署策略下的鲁棒性对比折线图。
    这是生成 "攻击0.png" 等核心结果图的函数。

    Args:
        nums (int): 需要绘制的曲线数量（对应不同的策略）。
        X (list of lists): 每个策略对应的 X 轴数据列表集合。
        Y (list of lists): 每个策略对应的 Y 轴数据列表集合。
        colors (list): 颜色列表，用于区分不同策略。
        linestyles (list): 线型列表。
        markers (list): 数据点标记列表。
        labels (list): 图例标签列表。
        number (int): 当前迭代编号，用于文件名后缀。
        name (str): 图表标题/文件名前缀（如 "定向攻击"）。
        save_dir (str): 保存目录。
        y_label (str): Y轴标签，默认为"Resilience"。
    """
    # 临时设置默认字体以避免在设置特定中文字体前的警告
    rcParams['font.family'] = 'sans-serif'
    
    # 使用 seaborn 设置绘图风格，使其更加美观
    sns.set_style("whitegrid", {
        'grid.linestyle': '--',
        'grid.alpha': 0.5,
        'axes.facecolor': '#f8f9fa'
    })
    sns.set_context("notebook", font_scale=1.2)
    
    fig, ax = plt.subplots()
    
    # 循环绘制每一条曲线
    for i in range(nums):
        ax.plot(X[i], Y[i],
                color=colors[i],
                linestyle=linestyles[i],
                marker=markers[i],
                label=labels[i],
                markersize=4,  # 减小标记大小
                linewidth=1.5)

    # 设置 X 轴主刻度间隔为 0.2
    x_major_locator = MultipleLocator(0.2)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.set_xlim(0, 1)

    ax.set_xlabel("Proportion of Removed Nodes : f", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    
    # 设置图例（缩小）
    ax.legend(fontsize=8, loc='best', framealpha=0.9)

    # 构建保存路径
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_path = os.path.join(save_dir, name + str(number) + '.png')
    
    # 保存图片，dpi=300 保证清晰度，bbox_inches='tight' 裁剪多余空白
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_bar_chart(data_dict, y_label, save_dir, filename):
    """
    绘制柱状图（带误差棒和散点）。
    """
    from matplotlib.font_manager import FontProperties
    try:
        font = FontProperties(family='SimHei')
    except:
        font = None
        
    sns.set_style("whitegrid", {
        'grid.linestyle': '--',
        'grid.alpha': 0.5,
        'axes.facecolor': '#f8f9fa'
    })
    
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974', '#64B5CD', '#8C8C8C', '#E377C2', '#BCBD22', '#17BECF', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B']
    markers = ['o', 's', '^', 'D', 'v', '+', '*', 'x', 'p', 'h', '1', '2', '3', '4', '8']
    
    # 定义标准的显示顺序，确保柱状图按逻辑排列（已删除 Random+RL, Bimodal+Random, SOLO+Ra, SOLO+RL）
    labels = ["Baseline", "RCP", "Bi-level", "GA+RL", "Onion+Ra", "Onion+RL", "ROMEN+Ra", "ROMEN+RL", "BimodalRL", "UNITY+Ra", "UNITY+RL"]
    
    # 提取数据
    xvalues_list = []
    valid_labels = []
    valid_colors = []
    valid_markers = []
    
    for i, label in enumerate(labels):
        if label in data_dict:
            xvalues_list.append(data_dict[label])
            valid_labels.append(label)
            if i < len(colors):
                valid_colors.append(colors[i])
                valid_markers.append(markers[i])
            else:
                valid_colors.append('#000000')
                valid_markers.append('o')
    
    if not valid_labels:
        return

    plt.figure(figsize=(20, 8)) 
    x_pos = np.arange(len(valid_labels))
    means = [np.mean(values) for values in xvalues_list]
    stds = [np.std(values) for values in xvalues_list]
    
    # 柱状图
    bars = plt.bar(x_pos, means, yerr=stds, 
                  capsize=5, 
                  alpha=0.8, 
                  color=valid_colors,
                  ecolor='black', 
                  width=0.6)
    
    # 数值标签
    for i, bar in enumerate(bars):
        height = means[i]
        plt.text(bar.get_x() + bar.get_width()/2., height + stds[i] + 0.01,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    # 散点图 (Jitter)
    for i, values in enumerate(xvalues_list):
        x_scatter = np.random.normal(x_pos[i], 0.04, size=len(values))
        plt.scatter(x_scatter, values, 
                   alpha=0.6,
                   color='black', # 散点用黑色更明显，或者用 valid_colors[i]
                   marker=valid_markers[i],
                   s=30, zorder=10)

    plt.xlabel("Deployment Strategy", fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.xticks(x_pos, valid_labels, rotation=15, fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(True, axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_box_chart(data_dict, y_label, save_dir, filename):
    """
    绘制箱线图。
    """
    sns.set_style("whitegrid", {
        'grid.linestyle': '--',
        'grid.alpha': 0.5,
        'axes.facecolor': '#f8f9fa'
    })
    
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974', '#64B5CD', '#8C8C8C', '#E377C2', '#BCBD22', '#17BECF', '#FF7F0E']
    labels = ["Baseline", "RCP", "Bi-level", "GA+RL", "Onion+Ra", "Onion+RL", "ROMEN+Ra", "ROMEN+RL", "BimodalRL", "UNITY+Ra", "UNITY+RL"]
    
    plot_data = []
    plot_labels = []
    palette = []
    
    for i, label in enumerate(labels):
        if label in data_dict:
            # Seaborn boxplot 需要列表数据，或者 DataFrame
            # 这里我们构建一个简单的列表结构，稍微麻烦点因为要对应颜色
            plot_data.append(data_dict[label])
            plot_labels.append(label)
            if i < len(colors):
                palette.append(colors[i])
            else:
                palette.append('#000000')

    if not plot_labels:
        return

    plt.figure(figsize=(20, 8))
    
    # 使用 boxplot
    # 注意：boxplot 的 input 如果是 list of lists，可以直接画
    bplot = plt.boxplot(plot_data, 
                        patch_artist=True,  # fill with color
                        labels=plot_labels,
                        medianprops=dict(color="black", linewidth=1.5),
                        flierprops=dict(marker='o', markerfacecolor='red', markersize=6))
    
    # 填充颜色
    for patch, color in zip(bplot['boxes'], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.xlabel("Deployment Strategy", fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.xticks(rotation=15, fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(True, axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_curve(x_values, dataset_name, attack_mode, experiment_id, save_dir=None):
    """
    兼容旧代码的接口，实际调用新函数。
    绘制不同策略下网络完全崩溃点（移除节点比例）的柱状图。
    """
    if save_dir is None:
        save_dir = os.path.join('results', 'pictures', dataset_name, 'control', attack_mode, str(experiment_id))
        
    # 绘制崩溃点 (X values)
    plot_bar_chart(x_values, "Proportion of Removed Nodes at Collapse (f)", save_dir, "collapse_point_bar.png")
    plot_box_chart(x_values, "Proportion of Removed Nodes at Collapse (f)", save_dir, "collapse_point_box.png")
