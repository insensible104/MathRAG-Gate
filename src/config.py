# src/config.py
"""
Configuration management for the MathRAG-Gate project.

SYSTEM TUNING:
Adjusted weights to prioritize Quality Scoring over RRF ranking, 
as experiments showed RRF introduces noise in the MATH dataset.
"""

from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import ConfigDict

class Settings(BaseSettings):
    """
    A Pydantic model to hold all project settings.
    """
    
    # --- LLM Configuration ---
    LLM_MODEL_NAME: str = "qwen2.5:7b"
    
    # The smaller LLM for judging (0.5B)
    QJUDGE_MODEL_NAME: str = "qwen2.5:0.5b"

    # Optional API Key
    DASHSCOPE_API_KEY: Optional[str] = None

    # --- Retrieval Configuration ---
    # Use English embedding model for MATH dataset
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"

    DENSE_TOP_K: int = 5
    SPARSE_TOP_K: int = 5
    
    # We retrieve top-10 candidates for re-ranking
    HYBRID_TOP_K: int = 10

    # --- RQAR (Re-ranking) Tuning ---
    # [TUNING] Lower RRF weight because Hybrid baseline was poor (31%)
    # [TUNING] Higher Quality weight to trust our improved Judge (0.9)
    RRF_WEIGHT: float = 0.1
    QUALITY_WEIGHT: float = 0.9

    # --- Confidence Gate Configuration ---
    GATE_THRESHOLD: float = 0.45
    GATE_SAMPLE_SIZE: int = 200

    # --- Path Configuration ---
    RESULTS_DIR: str = "results"
    DATA_DIR: str = "data"

    # --- Pydantic Configuration ---
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()