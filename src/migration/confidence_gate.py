# src/migration/confidence_gate.py
"""
Confidence-Gated Rule Transfer Mechanism.
This module is the core innovation of MathRAG-Gate, responsible for
deciding whether to trust the rule-based RQP or fall back to the LLM-based RQP.

VERSION: SERIAL (Single-threaded) with Progress Bars
"""

import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm
from src.retriever.rqar_rule import score_reasoning_quality
from src.retriever.rqar_llm import score_reasoning_quality_with_llm
from src.config import settings

class ConfidenceGate:
    """
    A gate that decides which RQP to use based on their consistency.
    
    Attributes:
        use_rule (bool): Flag indicating if the rule-based RQP should be used.
        threshold (float): The correlation threshold for using the rule.
        sample_size (int): Number of samples to use for the consistency check.
    """
    
    def __init__(self):
        self.use_rule = True  # Default to using the rule
        self.threshold = settings.GATE_THRESHOLD  # e.g., 0.45
        self.sample_size = settings.GATE_SAMPLE_SIZE  # e.g., 200

    def check_consistency(self, documents):
        """
        Checks the consistency between the rule-based and LLM-based scorers.
        
        This method samples a subset of documents, scores them with both methods,
        and calculates the Pearson correlation coefficient (rho).
        
        Args:
            documents (List[Document]): A list of LlamaIndex Document objects.
                                       
        Returns:
            tuple: (rho: float, use_rule: bool)
        """
        print(f"🔍 Checking consistency between Rule and LLM scorers on {self.sample_size} samples...")
        
        # Sample documents
        sample_size = min(self.sample_size, len(documents))
        sample_indices = np.random.choice(len(documents), size=sample_size, replace=False)
        sample_docs = [documents[i] for i in sample_indices]

        rule_scores = []
        llm_scores = []

        # 1. Rule Scoring (Fast)
        # Serial execution is fine here as it's CPU-bound and very fast
        for doc in tqdm(sample_docs, desc="[1/2] Rule Scoring", leave=False):
            doc_text = doc.text
            rule_score = score_reasoning_quality(doc_text)
            rule_scores.append(rule_score)

        # 2. LLM Scoring (Slow)
        # Serial execution to prevent GPU thrashing / OOM
        for doc in tqdm(sample_docs, desc="[2/2] LLM Scoring (Serial)"):
            doc_text = doc.text
            llm_score = score_reasoning_quality_with_llm(doc_text)
            llm_scores.append(llm_score)

        # Calculate Pearson correlation
        # Handle edge cases where all scores are the same
        if len(set(rule_scores)) <= 1 or len(set(llm_scores)) <= 1:
            print("⚠️ Warning: Low variance in scores, defaulting correlation to 0.0")
            rho = 0.0
        else:
            rho, _ = pearsonr(rule_scores, llm_scores)

        print(f"✅ Consistency Coefficient (ρ): {rho:.3f}")

        # Make the decision
        self.use_rule = rho >= self.threshold
        decision_str = "USE RULE-BASED RQP" if self.use_rule else "FALL BACK TO LLM RQP"
        print(f"🧠 Gate Decision: {decision_str}")

        return rho, self.use_rule