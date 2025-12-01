# plot_correlation.py
"""
Script to generate Figure 5: Confidence Gate Mechanism & Correlation Analysis.
读取 correlation_data_extracted.json 文件，计算相关性并绘制散点图。
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from scipy.stats import pearsonr

# --- 配置 ---
# 输入文件路径 (确保你已经运行过 2.py)
INPUT_JSON = "correlation_data_extracted.json"
# 输出图片路径
OUTPUT_IMG = os.path.join("results", "figure5_correlation_analysis.png") # 假设 results 目录存在

def plot_correlation_heatmap():
    print(f"🚀 Loading data from {INPUT_JSON}...")
    try:
        with open(INPUT_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: File not found at {INPUT_JSON}")
        print("Please run 'python 2.py' first to generate the correlation data.")
        return
    except json.JSONDecodeError:
        print(f"❌ Error: Could not decode JSON from {INPUT_JSON}")
        return

    rule_scores = np.array(data["rule_scores"])
    llm_scores = np.array(data["llm_scores"])

    # 1. 数据标准化 (Normalization)
    # 将分数标准化到 0-1 范围，以便更好地可视化和比较
    def normalize_scores(scores):
        if len(scores) == 0:
            return np.array([])
        min_val = np.min(scores)
        max_val = np.max(scores)
        if max_val == min_val: # 避免除以零
            return np.zeros_like(scores)
        return (scores - min_val) / (max_val - min_val)

    normalized_rule_scores = normalize_scores(rule_scores)
    normalized_llm_scores = normalize_scores(llm_scores)
    
    # 2. 计算皮尔逊相关系数
    if len(normalized_rule_scores) > 1 and len(normalized_llm_scores) > 1:
        correlation, _ = pearsonr(normalized_rule_scores, normalized_llm_scores)
    else:
        correlation = 0.0 # 数据不足时，相关系数为0
    print(f"📊 Calculated Pearson Correlation (ρ): {correlation:.3f}")

    # 3. 绘图设置
    plt.figure(figsize=(10, 8), dpi=300)
    sns.set_theme(style="whitegrid") # 设置 seaborn 风格，带网格

    # 4. 绘制散点图
    sns.scatterplot(
        x=normalized_rule_scores,
        y=normalized_llm_scores,
        alpha=0.6, # 透明度
        s=50,      # 点的大小
        color='steelblue' # 点的颜色
    )

    # 5. 添加标题和轴标签
    plt.title(
        "Figure 5: Visualization of the correlation ($\\rho$) between Rule-RQP and LLM-RQP scores.\n"
        "The lack of significant correlation ($\\rho < 0.1$) indicates that structural heuristics are insufficient "
        "for the MATH dataset, triggering the Gate to rely on the robust LLM judge.",
        fontsize=12, pad=20
    )
    plt.xlabel("Rule-RQP Score (Normalized)", fontsize=12)
    plt.ylabel("LLM-RQP Score (Normalized)", fontsize=12)
    
    # 6. 标注相关系数
    plt.text(
        0.95, 0.95, 
        f"Pearson Correlation ($\\rho$) = {correlation:.3f}", 
        transform=plt.gca().transAxes, # 相对坐标
        horizontalalignment='right', verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8, ec='gray'),
        fontsize=11
    )

    # 7. 添加注释框
    plt.annotate(
        "Low Correlation Detected: Rules fail to capture complex logic.\n"
        "Gate triggers LLM-Fallback.",
        xy=(np.mean(normalized_rule_scores) + 0.1, np.mean(normalized_llm_scores) + 0.1), # 注释指向大概的中心偏右上
        xytext=(0.6, 0.7), # 文本框位置
        textcoords='axes fraction', # 相对坐标
        arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8, headlength=8),
        bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='red', lw=2, linestyle='dashed'),
        ha='center', va='center',
        fontsize=12, color='red'
    )

    # 8. 设置轴范围和网格
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 9. 保存图片
    os.makedirs(os.path.dirname(OUTPUT_IMG), exist_ok=True) # 确保输出目录存在
    plt.tight_layout() # 自动调整布局，防止标签重叠
    plt.savefig(OUTPUT_IMG, bbox_inches='tight')
    print(f"\n✅ Heatmap saved to {OUTPUT_IMG}")
    plt.show()

if __name__ == "__main__":
    # 检查是否需要安装 seaborn
    try:
        import seaborn
    except ImportError:
        print("⚠️ Seaborn library not found. Please install it using: pip install seaborn")
        # 退出或提示用户安装，避免后续报错
        # os.system(f"{sys.executable} -m pip install seaborn") # 如果需要自动安装
        # print("✅ Seaborn installed. Please re-run the script.")
        exit() # 暂时退出，让用户手动安装

    plot_correlation_heatmap()