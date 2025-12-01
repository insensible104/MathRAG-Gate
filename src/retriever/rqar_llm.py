# src/retriever/rqar_llm.py
"""
LLM-based Reasoning Quality Predictor (RQP).
This module uses a Local Ollama model to perform evaluation of mathematical reasoning quality.

UPGRADE: FEW-SHOT PROMPTING
To stabilize the output of the small 0.5B model, we provide concrete examples (Few-Shot)
of high and low-quality reasoning within the prompt.
"""

import re
import logging
from typing import Optional

# Import ONLY Ollama
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import LLM

from src.config import settings

logger = logging.getLogger(__name__)

# --- GLOBAL SINGLETON INSTANCE ---
_JUDGE_LLM: Optional[LLM] = None

def _get_judge_llm() -> LLM:
    """
    Retrieves the global LLM instance. Initializes it if it doesn't exist.
    """
    global _JUDGE_LLM
    if _JUDGE_LLM is not None:
        return _JUDGE_LLM

    logger.info(f"🦙 Initializing QJudge using Local Ollama (Model: {settings.QJUDGE_MODEL_NAME})...")
    
    _JUDGE_LLM = Ollama(
        model=settings.QJUDGE_MODEL_NAME,
        request_timeout=120.0,
        temperature=0.1, # Low temp for consistency
        additional_kwargs={"num_predict": 10} # Limit output tokens
    )
    
    logger.info("✅ QJudge LLM (Ollama) initialized successfully.")
    return _JUDGE_LLM

def score_reasoning_quality_with_llm(text: str) -> float:
    """
    Uses the singleton LLM instance to perform a Few-Shot evaluation.
    
    Args:
        text (str): The text of the math explanation/solution.
        
    Returns:
        float: A normalized quality score between 0.0 and 1.0.
    """
    if not text:
        return 0.0

    truncated_text = text[:2000]

    # [UPGRADE] Few-Shot Prompt
    # We explicitly show the model what a "1" and a "5" look like.
    prompt = f"""
You are a math grader. Rate the reasoning quality of the Target Answer from 1 to 5.

### Example 1 (Low Quality - Score 1)
**Answer**: "The answer is 5."
**Reason**: No steps, no logic, just a number.
**Rating**: [[1]]

### Example 2 (High Quality - Score 5)
**Answer**: "First, let x be the width. Since the area is 20, we have x * (x+1) = 20. Solving for x, we get x=4. Therefore..."
**Reason**: Clear variables, logical steps, and derivation.
**Rating**: [[5]]

### Target Answer to Grade
{truncated_text}

### Task
Rate from 1 to 5 based on logic and clarity.
Output ONLY the score in brackets, e.g., [[3]].

**Rating**:
"""
    
    try:
        llm = _get_judge_llm()
        response = llm.complete(prompt)
        response_text = response.text.strip()
        
        # --- Anchor Parsing ---
        match = re.search(r'\[\[(\d)\]\]', response_text)
        
        if match:
            raw_score = int(match.group(1))
        else:
            # Fallback for loose digits
            match_loose = re.search(r'\b[1-5]\b', response_text)
            if match_loose:
                raw_score = int(match_loose.group())
            else:
                return 0.0

        # Normalize 1-5 to 0.0-1.0
        return min(max(raw_score / 5.0, 0.0), 1.0)
            
    except Exception as e:
        logger.error(f"Error in LLM scoring: {e}")
        return 0.0