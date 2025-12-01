# plot_heatmap.py
"""
Script to plot the Hyperparameter Grid Search Heatmap (Figure 6 in paper).
读取 optimization_report.csv 并绘制超参数热力图。
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# --- 配置 ---
# 输入文件路径 (确保你已经运行过 run_optimization.py)
INPUT_CSV = os.path.join("results", "optimization_logs", "optimization_report.csv")
# 输出图片路径
OUTPUT_IMG = os.path.join("results", "optimization_logs", "figure6_heatmap.png")

def plot_heatmap():
    print(f"🚀 Loading data from {INPUT_CSV}...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"❌ Error: File not found at {INPUT_CSV}")
        print("Please run 'python run_optimization.py' first to generate the data.")
        return

    print("📊 Preparing data for heatmap...")
    # 1. 数据透视 (Pivot)
    # 将长格式的 DataFrame 转换为矩阵格式，行是 Quality_Weight，列是 RRF_Weight
    heatmap_data = df.pivot(index="Quality_Weight", columns="RRF_Weight", values="Accuracy")
    
    # 为了让 Y 轴从下到上递增（符合直觉），我们需要反转索引
    heatmap_data = heatmap_data.sort_index(ascending=False)

    # 2. 绘图设置
    plt.figure(figsize=(10, 8), dpi=300) # 高分辨率学术图
    sns.set_theme(style="white") # 设置 seaborn 风格

    # 3. 绘制热力图
    # 使用 'YlOrRd' (Yellow-Orange-Red) 颜色映射，暖色代表高准确率
    # annot=True 会在每个格子里显示数值
    # fmt=".1%" 将数值格式化为百分比 (如 86.8%)
    ax = sns.heatmap(
        heatmap_data, 
        annot=True, 
        fmt=".1%", 
        cmap="YlOrRd", 
        cbar_kws={'label': 'Accuracy (%)'},
        linewidths=.5, # 添加格子边框
        square=True    #让格子呈正方形
    )

    # 4. 添加标题和轴标签
    plt.title("Figure 6: Hyperparameter Sensitivity Heatmap\n(RRF Weight vs. Quality Weight)", fontsize=14, fontweight='bold', pad=20)
    plt.xlabel("$W_{RRF}$ (RRF Rank Weight)", fontsize=12, labelpad=10)
    plt.ylabel("$W_{Quality}$ (RQP Quality Weight)", fontsize=12, labelpad=10)

    # 5. 寻找并标注最高点
    # 找到 Accuracy 最大的行和列索引
    # stack() 将 DataFrame 展平为 Series，idxmax() 找到最大值的索引 (row_label, col_label)
    best_coords = heatmap_data.stack().idxmax()
    best_quality_w, best_rrf_w = best_coords
    best_accuracy = heatmap_data.loc[best_quality_w, best_rrf_w]
    
    print(f"🏆 Peak Accuracy found: {best_accuracy:.2%} at RRF={best_rrf_w}, Quality={best_quality_w}")

    # 在图中添加标注框
    # 获取最高点在图中的坐标 (列索引, 行索引)
    # get_loc 获取标签在索引中的整数位置
    col_idx = heatmap_data.columns.get_loc(best_rrf_w)
    row_idx = heatmap_data.index.get_loc(best_quality_w)
    
    # 在对应格子的中心添加文本 annotation
    # xy 是标注点的坐标 (x+0.5, y+0.5 是格子中心)
    # xytext 是文本框的偏移位置
    ax.annotate(
        f'Peak:\n{best_accuracy:.1%}', 
        xy=(col_idx + 0.5, row_idx + 0.5),
        xytext=(0, 40), # 向上偏移
        textcoords='offset points',
        ha='center', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='red', lw=2),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='red', lw=2),
        fontsize=11, fontweight='bold', color='red'
    )

    # 6. 保存图片
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, bbox_inches='tight')
    print(f"\n✅ Heatmap saved to {OUTPUT_IMG}")
    plt.show()
