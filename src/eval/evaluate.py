# src/eval/evaluate.py
"""
Evaluation script for the MathRAG-Gate project.

VERSION: STAGED BATCHING + CoT + TIMESTAMP COMPATIBILITY
"""

import os
import time
import re
import pandas as pd
import sys
from typing import List, Dict, Any
from scipy.stats import wilcoxon
import logging
from tqdm import tqdm

from src.config import settings
from src.monitoring.metrics import SystemMetrics
from llama_index.llms.ollama import Ollama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, test_questions: List[str], test_answers: List[str]):
        self.test_questions = test_questions
        self.test_answers = test_answers
        self.metrics_logger = SystemMetrics()
        self.answer_llm = None

    def _load_generator_model(self):
        if self.answer_llm is not None: return
        print(f"\n⏳ [Phase Switch] Loading Generator Model ({settings.LLM_MODEL_NAME})...")
        try:
            self.answer_llm = Ollama(
                model=settings.LLM_MODEL_NAME,
                request_timeout=120.0,
                temperature=0.0
            )
            self.answer_llm.complete("1+1=")
            print(f"✅ Generator Model loaded!")
        except Exception as e:
            logger.warning(f"Warm-up failed: {e}")

    def _extract_final_answer(self, text: str) -> str:
        """Extracts answer, prioritizing \\boxed{} for CoT."""
        if not text: return ""
        # Priority 1: Boxed LaTeX answer
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
        if boxed_match:
            content = boxed_match.group(1)
            num_match = re.findall(r'-?\d+\.?\d*', content)
            if num_match: return num_match[-1]
        
        # Priority 2: Last number
        matches = re.findall(r'-?\d+\.?\d*', text)
        return matches[-1] if matches else ""

    def _calculate_recall_at_k(self, retrieved_docs: List[Any], gold_answer: str, k: int = 5) -> bool:
        gold_nums = re.findall(r'-?\d+\.?\d*', gold_answer)
        if not gold_nums: return False
        for i, doc in enumerate(retrieved_docs[:k]):
            doc_text = doc.node.text if hasattr(doc, 'node') else str(doc)
            for num in gold_nums:
                if num in doc_text: return True
        return False

    def _generate_answer_with_llm(self, question: str, retrieved_nodes: List[Any]) -> str:
        if not retrieved_nodes or self.answer_llm is None: return ""
        
        context = "\n\n".join([f"Reference:\n{node.node.text}" for node in retrieved_nodes[:3]])
        
        # [OPTIMIZATION] Chain-of-Thought Prompt
        prompt = f"""
You are a math expert. Solve the problem below using the provided Reference Context.

### Reference Context:
{context}

### Question:
{question}

### Instructions:
1. **Think Step-by-Step**: Analyze the problem logic carefully. Write out your reasoning process.
2. **Use References**: The references may contain similar problems. Borrow their logic but apply them to the current numbers.
3. **Format**: At the very end, put the final numerical answer inside \\boxed{{}}. Example: \\boxed{{42}}.

### Solution:
Let's think step by step.
"""
        try:
            response = self.answer_llm.complete(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return ""

    def _evaluate_single_method(self, retriever, method_name: str, num_samples: int):
        logger.info(f"🚀 Starting Staged Evaluation for {method_name}...")
        num_to_process = min(num_samples, len(self.test_questions))
        
        # PHASE 1: RETRIEVAL
        print(f"📥 [Phase 1/2] Batch Retrieval ({num_to_process} samples)...")
        retrieval_cache = []
        for i in tqdm(range(num_to_process), desc="Phase 1", unit="sample", file=sys.stdout):
            q = self.test_questions[i]
            try:
                retrieved_nodes = retriever.retrieve(q)
            except Exception:
                retrieved_nodes = []
            retrieval_cache.append({
                "index": i, "query": q, "true_answer_raw": self.test_answers[i],
                "retrieved_nodes": retrieved_nodes
            })

        # PHASE 2: GENERATION
        print(f"🧠 [Phase 2/2] Batch Generation (CoT Reasoning)...")
        self._load_generator_model()
        
        final_results = []
        correct_count = 0
        recall_count = 0
        total_time = 0.0 # We simplify latency tracking for CoT

        for item in tqdm(retrieval_cache, desc="Phase 2", unit="sample", file=sys.stdout):
            start = time.time()
            resp = self._generate_answer_with_llm(item['query'], item['retrieved_nodes'])
            latency = time.time() - start
            total_time += latency

            pred = self._extract_final_answer(resp)
            true = self._extract_final_answer(item['true_answer_raw'])
            
            is_correct = (pred == true)
            if is_correct: correct_count += 1
            is_retrieved = self._calculate_recall_at_k(item['retrieved_nodes'], item['true_answer_raw'])
            if is_retrieved: recall_count += 1

            # Log
            use_rule = retriever.gate.use_rule if hasattr(retriever, 'gate') else False
            quality = item['retrieved_nodes'][0].score if item['retrieved_nodes'] else 0.0
            self.metrics_logger.log_query(item['query'], method_name, latency, use_rule, quality, resp)
            
            final_results.append({
                "query": item['query'], "true_answer": true, "pred_answer": pred,
                "is_correct": is_correct, "latency": latency, "response_text": resp,
                "recall_at_5": is_retrieved
            })

        accuracy = correct_count / len(final_results) if final_results else 0
        recall = recall_count / len(final_results) if final_results else 0
        avg_lat = total_time / len(final_results) if final_results else 0
        
        logger.info(f"{method_name} - Accuracy: {accuracy:.2%}, Recall: {recall:.2%}")
        return final_results, accuracy, recall, avg_lat

    def run_ablation_study(self, dense_retriever, sparse_retriever, hybrid_retriever, main_retriever, num_samples: int = 50):
        methods = {
            "Dense Retrieval": dense_retriever,
            "Sparse Retrieval (BM25)": sparse_retriever,
            "Hybrid Retrieval (RRF)": hybrid_retriever,
            "MathRAG-Gate (Ours)": main_retriever
        }
        all_results = {}
        summary_data = []

        for name, retriever in methods.items():
            res, acc, rec, lat = self._evaluate_single_method(retriever, name, num_samples)
            all_results[name] = res
            summary_data.append({"Method": name, "Accuracy (%)": acc, "Recall@5 (%)": rec, "Latency (s)": lat})

        return all_results, pd.DataFrame(summary_data)

    def save_results(self, all_results, summary_df, run_id=""):
        # Use settings.RESULTS_DIR which is dynamically updated in run_experiments.py
        output_dir = settings.RESULTS_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        for name, res in all_results.items():
            safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
            pd.DataFrame(res).to_csv(f"{output_dir}/{run_id}_{safe_name}_detailed.csv", index=False)
        
        summary_df.to_csv(f"{output_dir}/{run_id}_summary_table.csv", index=False)