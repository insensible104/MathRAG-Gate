# src/retriever/rqar_rule.py
"""
Rule-based Reasoning Quality Predictor (RQP) for English Math Explanations.

UPGRADE: STRUCTURE AWARENESS
Added regex patterns to detect structural markers (Steps, Variable Definitions, LaTeX environments)
that strongly correlate with high-quality reasoning in the MATH dataset.
"""

import re

def score_reasoning_quality(text: str) -> float:
    """
    Scores the reasoning quality of a math explanation on a scale of 0.0 to 1.0.
    """
    if not text or len(text.strip()) == 0:
        return 0.0

    score = 0.0
    text_lower = text.lower()

    # --- 1. Logic & Connectors (Weight: 0.3) ---
    causal_words = ["because", "since", "therefore", "thus", "hence", "implies", "consequently"]
    causal_count = sum(text_lower.count(w) for w in causal_words)
    # LaTeX logic symbols: \implies, \because, \therefore
    latex_logic_count = len(re.findall(r'\\(implies|because|therefore|Rightarrow)', text))
    score += min((causal_count + latex_logic_count) * 0.05, 0.3)

    # --- 2. Structural Rigor (Weight: 0.3) ---
    # [UPGRADE] Detect specific structural markers
    
    # "Step 1", "Case 1"
    structure_matches = len(re.findall(r'(?:step|case|method)\s*\d+', text_lower))
    
    # Variable definitions: "Let x be", "Define y"
    definition_matches = len(re.findall(r'(?:let|define|assume|suppose)\s+[a-z]', text_lower))
    
    # LaTeX alignment environments (sign of structured math)
    env_matches = len(re.findall(r'\\begin\{(align|equation|split)\}', text))
    
    score += min((structure_matches + definition_matches + env_matches) * 0.06, 0.3)

    # --- 3. Mathematical Density (Weight: 0.25) ---
    # Basic equality "x = y"
    equality_matches = len(re.findall(r'[^\=\n]+\=[^\=\n]+', text))
    # Complex LaTeX: \frac, \sqrt, \int
    latex_matches = len(re.findall(r'\\(frac|sqrt|int|sum|prod|cdot)', text))
    
    score += min((equality_matches + latex_matches) * 0.03, 0.25)

    # --- 4. The "Holy Grail" (Weight: 0.15) ---
    # Detect boxed final answer, specific to MATH dataset
    if "\\boxed{" in text:
        score += 0.15

    # --- 5. Penalties ---
    word_count = len(text.split())
    if word_count < 30:
        score *= 0.5  # Too short to be a good explanation
    
    # Penalize vagueness only if no math is present
    if equality_matches == 0 and "obviously" in text_lower:
        score -= 0.1

    return max(0.0, min(score, 1.0))