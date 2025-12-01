# src/retriever/rqar_explainer.py
"""
Explainable Reasoning Quality Assessment.
This module generates human-readable explanations for why a particular
explanation was scored highly or lowly by the rule-based RQP.
"""

import re
from .rqar_rule import score_reasoning_quality

def generate_explanation(text: str) -> dict:
    """
    Generates a structured explanation for the reasoning quality score.
    
    Returns a dictionary containing:
    - "score": The numerical quality score (0.0-1.0)
    - "explanation": A human-readable string explaining the score
    - "features": A dictionary of the raw features used in the scoring
    
    Args:
        text (str): The text of the math explanation/solution.
        
    Returns:
        dict: A dictionary with 'score', 'explanation', and 'features'.
    """
    # Reuse the existing scoring function to get the score
    score = score_reasoning_quality(text)
    text_lower = text.lower()
    
    # Extract the same features used in the scoring function
    causal_words = ["because", "since", "as a result", "therefore", "thus", "hence", "so", "consequently"]
    causal_count = sum(text_lower.count(word) for word in causal_words)
    
    sequential_words = ["first", "secondly", "next", "then", "after that", "subsequently", "finally", "lastly", "in conclusion"]
    sequential_count = sum(text_lower.count(word) for word in sequential_words)
    
    calc_pattern = r'\d+\s*[\+\-\*\/]\s*\d+\s*=\s*\d+'
    calc_matches = len(re.findall(calc_pattern, text))
    
    formula_pattern = r'[a-zA-Z]\s*(?:\^|\*\*)\s*\d+\s*[\+\-\*\/]\s*[a-zA-Z]\s*(?:\^|\*\*)\s*\d+\s*=\s*[a-zA-Z]'
    formula_matches = len(re.findall(formula_pattern, text))
    
    word_count = len(text.split())
    vague_phrases = ["obviously", "clearly", "it's easy to see", "we can see", "trivially", "it follows that"]
    has_vague_phrases = any(phrase in text_lower for phrase in vague_phrases)
    
    # Build the human-readable explanation
    explanation_parts = []
    if causal_count > 0:
        explanation_parts.append(f"Contains {causal_count} causal reasoning marker(s) (e.g., 'because', 'therefore').")
    if sequential_count > 0:
        explanation_parts.append(f"Uses {sequential_count} sequential markers (e.g., 'first', 'finally').")
    if calc_matches > 0:
        explanation_parts.append(f"Contains {calc_matches} explicit calculation step(s) (e.g., '3\\times2=6').")
    if formula_matches > 0:
        explanation_parts.append(f"Contains {formula_matches} symbolic formula(s).")
    if word_count < 40:
        explanation_parts.append("Answer is very short and likely skips intermediate steps.")
    if has_vague_phrases:
        explanation_parts.append("Contains vague phrases like 'clearly', which may hide gaps in the logic.")
    
    if not explanation_parts:
        explanation = "No clear reasoning patterns detected in the explanation."
    else:
        explanation = " ".join(explanation_parts)
    
    # Package the features for potential further analysis
    features = {
        "causal_count": causal_count,
        "sequential_count": sequential_count,
        "calculation_steps": calc_matches,
        "symbolic_formulas": formula_matches,
        "word_count": word_count,
        "has_vague_phrases": has_vague_phrases
    }
    
    return {
        "score": score,
        "explanation": explanation,
        "features": features
    }