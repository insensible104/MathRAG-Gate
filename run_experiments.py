# run_experiments.py
"""
Experiment Orchestration Script for MathRAG-Gate.
MathRAG-Gate 实验编排脚本

VERSION: TIMESTAMPED & LOGGING
This script automates the stability analysis (5 Runs) and saves results
into a UNIQUE timestamped directory to prevent overwriting previous experiments.
此版本会自动创建一个带时间戳的独立文件夹（如 results/exp_2023...），防止覆盖旧数据。
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import logging
import time
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# [OPTIMIZATION] Suppress verbose logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Import project modules
from src.config import settings  # Import global settings
from src.utils.data_loader import load_math_data
from src.retriever.main_retriever import MainRQARRetriever
from src.retriever.dense_retriever import DenseRetriever
from src.retriever.sparse_retriever import SparseRetriever
from src.retriever.hybrid_retriever import HybridRetriever
from src.eval.evaluate import Evaluator

def setup_experiment_dir():
    """
    Creates a unique directory for this experiment run based on timestamp.
    基于当前时间戳为本次实验创建一个唯一的目录。
    
    Returns:
        str: The path to the new directory.
    """
    # Get current time string (e.g., 20251124_153000)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join("results", f"exp_{timestamp}")
    
    # Create the directory
    os.makedirs(exp_dir, exist_ok=True)
    
    print(f"\n📂 [System] Creating new experiment directory: {exp_dir}")
    print(f"📂 [系统] 已创建新的实验目录: {exp_dir}")
    
    # [CRITICAL] Update the global settings path!
    # This ensures that Evaluator, MetricsLogger, and save_results 
    # ALL point to this new folder automatically.
    settings.RESULTS_DIR = exp_dir
    
    return exp_dir

def plot_results(aggregated_df, output_dir):
    """Generates and saves the benchmark plot."""
    print(f"\n📊 Generating visualization plot... / 正在生成可视化图表...")
    
    # [FIX] Chinese Font Support
    import matplotlib
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    methods = aggregated_df['Method']
    acc_mean = aggregated_df['Accuracy_Mean'] * 100
    acc_std = aggregated_df['Accuracy_Std'] * 100
    rec_mean = aggregated_df['Recall_Mean'] * 100
    
    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, acc_mean, width, yerr=acc_std, label='Accuracy', capsize=5, color='#4CAF50', alpha=0.8)
    rects2 = ax.bar(x + width/2, rec_mean, width, label='Recall@5', capsize=5, color='#2196F3', alpha=0.8)

    ax.set_ylabel('Percentage (%)')
    ax.set_title('MathRAG-Gate Performance Benchmark (5 Runs Average)\nMathRAG-Gate 性能基准测试 (5轮平均)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.legend(loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    ax.bar_label(rects1, padding=3, fmt='%.1f')
    ax.bar_label(rects2, padding=3, fmt='%.1f')

    save_path = os.path.join(output_dir, "benchmark_results.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ Plot saved to {save_path}")

def show_samples(run_id, output_dir):
    """Extracts qualitative samples."""
    file_path = f"{output_dir}/run_{run_id}_MathRAG-Gate_Ours_detailed.csv"
    if not os.path.exists(file_path):
        return

    df = pd.read_csv(file_path)
    print("\n" + "="*70)
    print(f"🔍 QUALITATIVE ANALYSIS (Samples from Run {run_id}) / 定性分析")
    print("="*70)

    success_df = df[df['is_correct'] == True]
    if not success_df.empty:
        sample = success_df.iloc[0]
        print(f"\n✅ [SUCCESS CASE]")
        print(f"Q: {sample['query']}")
        print(f"Gold: {sample['true_answer']}")
        print(f"Pred: {sample['pred_answer']}")
        print(f"Reasoning: {sample['response_text'][:200].replace(chr(10), ' ')}...")
        print("-" * 60)

    fail_df = df[df['is_correct'] == False]
    if not fail_df.empty:
        sample = fail_df.iloc[0]
        print(f"\n❌ [FAILURE CASE]")
        print(f"Q: {sample['query']}")
        print(f"Gold: {sample['true_answer']}")
        print(f"Pred: {sample['pred_answer']}")
        print(f"Reasoning: {sample['response_text'][:200].replace(chr(10), ' ')}...")
        print("-" * 60)

def main():
    # 1. Setup Directory FIRST
    exp_dir = setup_experiment_dir()
    
    print("🚀 Starting Repeated Experiments... / 启动重复实验...")
    
    # 2. Load Data
    print("\n📚 Loading MATH Dataset...")
    kb_docs, test_questions, test_answers = load_math_data()
    
    # 3. Build Base Retrievers
    print("\n🔧 Building Base Retrievers...")
    dense_retriever = DenseRetriever(kb_docs, model_name=settings.EMBEDDING_MODEL_NAME)
    sparse_retriever_obj = SparseRetriever(kb_docs)
    hybrid_retriever = HybridRetriever(
        documents=kb_docs, 
        dense_retriever=dense_retriever.retriever,
        sparse_retriever=sparse_retriever_obj.retriever
    )
    
    all_runs_summary = []
    NUM_RUNS = 5
    SAMPLES_PER_RUN = 100 # Or 500 if you have time
    
    print(f"\n⚡ Starting Loop: {NUM_RUNS} Runs x {SAMPLES_PER_RUN} Samples")

    # 4. Experiment Loop
    for i in range(1, NUM_RUNS + 1):
        print(f"\n{'='*20} RUN {i}/{NUM_RUNS} {'='*20}")
        
        print("🔄 Re-calibrating Confidence Gate...")
        main_retriever = MainRQARRetriever(
            documents=kb_docs, 
            hybrid_retriever=hybrid_retriever
        )
        
        # Initialize Evaluator (It will read settings.RESULTS_DIR which we updated)
        evaluator = Evaluator(test_questions, test_answers)
        
        print(f"📊 Running evaluation for Run {i}...")
        results, summary_df = evaluator.run_ablation_study(
            dense_retriever=dense_retriever.retriever,
            sparse_retriever=sparse_retriever_obj.retriever,
            hybrid_retriever=hybrid_retriever,
            main_retriever=main_retriever,
            num_samples=SAMPLES_PER_RUN 
        )
        
        # Save results into the specific exp_dir
        evaluator.save_results(results, summary_df, run_id=f"run_{i}")
        
        summary_df['Run'] = i
        all_runs_summary.append(summary_df)
        print(f"✅ Run {i} complete.")

    # 5. Aggregation
    print("\n📈 Calculating Statistics...")
    final_df = pd.concat(all_runs_summary)
    
    agg_df = final_df.groupby('Method').agg({
        'Accuracy (%)': ['mean', 'std'],
        'Recall@5 (%)': ['mean', 'std'],
        'Latency (s)': ['mean']
    }).reset_index()
    
    agg_df.columns = ['Method', 'Accuracy_Mean', 'Accuracy_Std', 'Recall_Mean', 'Recall_Std', 'Latency_Mean']
    
    print("\n📊 AGGREGATED RESULTS (Mean ± Std)")
    print("="*60)
    print(agg_df.to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))
    
    agg_path = os.path.join(exp_dir, "final_aggregated_report.csv")
    agg_df.to_csv(agg_path, index=False)
    print(f"\n💾 Saved to {agg_path}")

    # 6. Visualization & Sampling
    try:
        plot_results(agg_df, output_dir=exp_dir)
        show_samples(run_id=NUM_RUNS, output_dir=exp_dir)
    except Exception as e:
        print(f"⚠️ Visualization failed: {e}")

    print(f"\n🎉 All Done! Check folder: {exp_dir}")

if __name__ == "__main__":
    main()