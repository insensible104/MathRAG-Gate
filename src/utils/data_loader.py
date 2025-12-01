"""
Data loading and preprocessing for the MathRAG-Gate project.

This module loads the MATH dataset, forces the cache to be stored in the local
'data/' folder, handles different data formats, and returns LlamaIndex-compatible
documents and test data.
"""

import os
import json
import logging
import sys
from datasets import load_dataset
from llama_index.core.schema import Document
from src.config import settings # ADDED: Use settings for DATA_DIR

# --- Setup Environment Variables (MUST be done before importing datasets) ---

# 1. HF Mirror setup for China users
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "0" 

# 2. [NEW] Set custom cache directory to the local 'data/' folder
DATA_DIR = settings.DATA_DIR
# Ensure the directory exists
os.makedirs(DATA_DIR, exist_ok=True)
# Set the environment variable for Hugging Face to use this path
os.environ["HF_DATASETS_CACHE"] = os.path.abspath(DATA_DIR) 

# --- Logging Setup ---
# Set up simple logging (needs to be done here or in main.py)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Heuristic Check for Local Existence ---
# Hugging Face caches datasets in a specific subfolder structure (e.g., 'data/DigitalLearningGmbH___MATH-lighteval')
# We use this to provide clear feedback to the user.
DATASET_NAME_PART = "DigitalLearningGmbH___MATH-lighteval"
cache_path_guess = os.path.join(os.environ["HF_DATASETS_CACHE"], DATASET_NAME_PART)
# Check if the expected dataset folder structure exists
IS_DOWNLOADED_LOCALLY = os.path.exists(cache_path_guess)


def load_math_data():
    """
    Loads and preprocesses the MATH dataset.

    - Forces download/cache location to the local 'data/' directory.
    - Returns LlamaIndex Documents for KB + raw strings for test.
    """
    
    # 根据本地检查结果，决定加载策略
    if IS_DOWNLOADED_LOCALLY:
        logger.info("✅ 检测到本地数据集。正在从 'data/' 文件夹加载 (强制离线模式)...")
        # 优化点：使用 local_files_only=True 强制只读取本地文件，完全跳过网络检查。
        load_mode_args = {"local_files_only": True}
    else:
        logger.info("📚 正在从 Hugging Face 下载数据集到 'data/' 文件夹 (首次联网，后续自动离线)...")
        # 首次下载：允许联网，并要求如果缓存已存在则重用
        load_mode_args = {"download_mode": "reuse_cache_if_exists"}

    try:
        # load_dataset 依赖于 HF_DATASETS_CACHE env variable
        dataset = load_dataset(
            "DigitalLearningGmbH/MATH-lighteval",
            "default",
            **load_mode_args # 动态传入加载参数
        )
    except Exception as e:
        logger.error(f"❌ 无法加载数据集: {e}")
        if IS_DOWNLOADED_LOCALLY:
            logger.error("请检查本地缓存文件是否完整或损坏。")
        else:
            logger.error("请检查网络连接或 VPN/代理设置是否稳定。")
        raise

    all_items = dataset["train"]
    total = len(all_items)
    logger.info(f"✅ 成功加载 {total} 条数学题目")

    # === 智能解析：自动检测数据格式（dict 或 JSON 字符串）===
    # ... (后续数据处理逻辑保持不变)
    parsed_items = []
    for item in all_items:
        if isinstance(item, dict):
            # 已是字典格式
            parsed_items.append(item)
        elif isinstance(item, str):
            # 是 JSON 字符串，尝试解析
            try:
                # Assuming the item is a valid JSON string containing the necessary fields
                parsed_items.append(json.loads(item))
            except (json.JSONDecodeError, TypeError) as parse_err:
                logger.warning(f"⚠️ 跳过无法解析的条目: {str(item)[:100]}...")
                continue
        else:
            logger.warning(f"⚠️ 跳过非字符串/非字典条目: {type(item)}")
            continue

    if not parsed_items:
        raise ValueError("❌ 数据集中没有有效条目！")

    # === 划分知识库（80%）和测试集（20%）===
    split_idx = int(len(parsed_items) * 0.8)
    kb_items = parsed_items[:split_idx]
    test_items = parsed_items[split_idx:]

    # === 构建知识库：转为 LlamaIndex Document ===
    kb_docs = [
        # Combining question and solution provides the best context for retrieval
        Document(text=f"Question: {item['problem']}\nAnswer: {item['solution']}")
        for item in kb_items
        if "problem" in item and "solution" in item
    ]

    # === 构建测试集 ===
    test_questions = [
        item["problem"] for item in test_items
        if "problem" in item and "solution" in item
    ]
    test_answers = [
        item["solution"] for item in test_items
        if "problem" in item and "solution" in item
    ]

    # 确保测试集长度一致
    min_len = min(len(test_questions), len(test_answers))
    test_questions = test_questions[:min_len]
    test_answers = test_answers[:min_len]

    logger.info(f"✅ 知识库文档数: {len(kb_docs)}")
    logger.info(f"✅ 测试集问题数: {len(test_questions)}")

    return kb_docs, test_questions, test_answers