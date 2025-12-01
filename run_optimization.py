# run_optimization.py
"""
Hyperparameter Optimization Script for MathRAG-Gate.
MathRAG-Gate 超参数自动调优脚本

This script performs a Grid Search to find the optimal weights for 
RRF (Rank Score) vs. Quality (LLM/Rule Score).

It automates the following loop:
1. Define a range of parameters (e.g., Quality Weight from 0.1 to 0.9).
2. Run evaluation on a small subset (e.g., 50 samples) for speed.
3. Log Accuracy and Recall for each setting.
4. Select the best configuration.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import logging
from dotenv import load_dotenv

# Load env
load_dotenv()
# Suppress logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from src.config import settings
from src.utils.data_loader import load_math_data
from src.retriever.main_retriever import MainRQARRetriever
from src.retriever.dense_retriever import DenseRetriever
from src.retriever.sparse_retriever import SparseRetriever
from src.retriever.hybrid_retriever import HybridRetriever
from src.eval.evaluate import Evaluator

def main():
    print("🚀 Starting Hyperparameter Optimization... / 启动超参数自动调优...")
    
    # 1. Setup
    # Use a dedicated folder for optimization logs
    opt_dir = os.path.join("results", "optimization_logs")
    os.makedirs(opt_dir, exist_ok=True)
    settings.RESULTS_DIR = opt_dir # Redirect logs here
    
    # 2. Load Data & Retrievers ONCE (Efficiency)
    print("\n📚 Loading Data & Building Base Retrievers...")
    kb_docs, test_questions, test_answers = load_math_data()
    
    dense_retriever = DenseRetriever(kb_docs, model_name=settings.EMBEDDING_MODEL_NAME)
    sparse_retriever_obj = SparseRetriever(kb_docs)
    hybrid_retriever = HybridRetriever(
        documents=kb_docs, 
        dense_retriever=dense_retriever.retriever,
        sparse_retriever=sparse_retriever_obj.retriever
    )

    # 3. Define Search Space (Grid Search)
    # We explore different balances between RRF (Rank) and Quality.
    # Strategy: RRF_WEIGHT + QUALITY_WEIGHT = 1.0
    # 我们定义不同的权重组合，探索 排名分 vs 质量分 的最佳平衡。
    search_space = [
        {"rrf": 0.9, "quality": 0.1}, # Mostly trust Hybrid Rank
        {"rrf": 0.7, "quality": 0.3}, # Traditional setting
        {"rrf": 0.5, "quality": 0.5}, # Balanced
        {"rrf": 0.3, "quality": 0.7}, # MathRAG-Gate Preference (Current)
        {"rrf": 0.1, "quality": 0.9}, # Mostly trust Judge
    ]
    
    # Optimization Config
    # Use a smaller sample size for speed (e.g., 50 or 100)
    # 使用较小的样本量（如 50）来快速筛选，确定参数后再跑全量
    SAMPLE_SIZE = 50 
    
    results_log = []

    print(f"\n⚡ Grid Search: {len(search_space)} combinations x {SAMPLE_SIZE} samples")

    # 4. Optimization Loop
    for idx, params in enumerate(search_space):
        rrf_w = params["rrf"]
        qual_w = params["quality"]
        
        print(f"\n🔄 [Config {idx+1}/{len(search_space)}] Testing weights: RRF={rrf_w}, Quality={qual_w}")
        
        # --- Apply Settings Dynamically ---
        settings.RRF_WEIGHT = rrf_w
        settings.QUALITY_WEIGHT = qual_w
        
        # Re-initialize Main Retriever (to apply new settings logic if needed)
        # Although weights are read from settings at runtime, re-init implies fresh gate check
        main_retriever = MainRQARRetriever(
            documents=kb_docs, 
            hybrid_retriever=hybrid_retriever
        )
        
        # Init Evaluator
        evaluator = Evaluator(test_questions, test_answers)
        
        # Run ONLY MathRAG-Gate (We don't need to re-test Dense/Sparse every time)
        # We assume Dense/Sparse baselines are constant.
        res, acc, rec, lat = evaluator._evaluate_single_method(
            retriever=main_retriever,
            method_name=f"Gate_R{rrf_w}_Q{qual_w}",
            num_samples=SAMPLE_SIZE
        )
        
        # Record Result
        results_log.append({
            "RRF_Weight": rrf_w,
            "Quality_Weight": qual_w,
            "Accuracy": acc,
            "Recall": rec,
            "Latency": lat
        })
        
        print(f"   -> Result: Accuracy={acc:.2%}, Recall={rec:.2%}")

    # 5. Analysis & Export
    print("\n" + "="*60)
    print("🏆 OPTIMIZATION RESULTS / 调优结果")
    print("="*60)
    
    df = pd.DataFrame(results_log)
    # Sort by Accuracy descending
    df = df.sort_values(by="Accuracy", ascending=False)
    
    print(df.to_string(index=False))
    
    # Save to CSV
    df.to_csv(os.path.join(opt_dir, "optimization_report.csv"), index=False)
    
    # 6. Visualization (Weight vs Accuracy)
    try:
        plt.figure(figsize=(10, 6))
        # Fix Chinese fonts
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        plt.plot(df["Quality_Weight"], df["Accuracy"], marker='o', linestyle='-', color='b', label='Accuracy')
        plt.xlabel("Quality Weight (Trust in Judge) / 质量权重")
        plt.ylabel("Accuracy / 准确率")
        plt.title("Hyperparameter Tuning: Impact of Quality Weight\n超参数调优：质量权重对性能的影响")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(opt_dir, "tuning_curve.png"))
        print(f"\n📊 Tuning curve saved to {opt_dir}/tuning_curve.png")
    except Exception as e:
        print(f"Plotting error: {e}")

    # 7. Suggestion
    best_config = df.iloc[0]
    print(f"\n✅ Best Configuration Found: RRF={best_config['RRF_Weight']}, Quality={best_config['Quality_Weight']}")
    print(f"👉 Update your src/config.py with these values!")

if __name__ == "__main__":
    main()