"""
Configuration module for MRCD Framework.
Centralizes all constants, hyperparameters, and path configurations.
Loads environment variables from .env file via python-dotenv.
"""

import os
from pathlib import Path

# Load .env from project root
try:
    from dotenv import load_dotenv
    _project_root = Path(__file__).resolve().parent.parent
    load_dotenv(_project_root / ".env")
except ImportError:
    pass  # dotenv not installed, rely on system env vars

# ============================================================
# Data Paths
# ============================================================
DATA_DIR = os.environ.get(
    "MRCD_DATA_DIR",
    r"/kaggle/input/datasets/chinhde/twitter15-16",
)
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
VAL_CSV = os.path.join(DATA_DIR, "val.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")

# SLM model path (pre-trained checkpoint)
MODEL_PATH = os.environ.get(
    "MRCD_MODEL_PATH",
    "bert-base-uncased",  # Changed from RoBERTa to BERT for EANN
)

# ============================================================
# LLM Configuration
# ============================================================
LLM_MODEL_NAME = os.environ.get(
    "LLM_MODEL_NAME",
    "meta-llama/Meta-Llama-3-8B-Instruct",
)

LLM_MAX_NEW_TOKENS = 128
LLM_MAX_OUTPUT_TOKENS_EXTRACTION = 96
# LLM chỉ trả về "Real" hoặc "Fake" 
LLM_MAX_OUTPUT_TOKENS_CLASSIFICATION = 1
LLM_TEMPERATURE = 0.0
LLM_TOP_P = 1.0
# Backend for LLM: "hf" (Hugging Face) or "vllm"
LLM_BACKEND = os.environ.get("LLM_BACKEND", "hf")
# ============================================================
# SLM Backend Configuration
# ============================================================
# "hf" for HuggingFace Transformers, "vllm" for vLLM
SLM_BACKEND = os.environ.get("SLM_BACKEND", "hf")

# ============================================================
# Pipeline Hyperparameters
# ============================================================
CONFIDENCE_THRESHOLD = 0.8
NUM_LOOP = 3
TOP_K_DEMOS = 4
FACT_TOP_K = 3

# Bootstrap settings
BOOTSTRAP_ENABLE_PARALLEL = True
BOOTSTRAP_MAX_WORKERS = min(16, max(4, (os.cpu_count() or 4) * 2))

# Parallel crawling settings
CRAWL_MAX_WORKERS = min(8, max(2, (os.cpu_count() or 4)))

# ============================================================
# SLM Fine-tune Configuration
# ============================================================
ENABLE_SLM_FINETUNE = True
SLM_FINETUNE_EPOCHS = 1
SLM_FINETUNE_BATCH_SIZE = 32
SLM_FINETUNE_LR = 2e-5
SLM_FINETUNE_WEIGHT_DECAY = 0.01
SLM_FINETUNE_MIN_SAMPLES = 16

# ============================================================
# Knowledge Retrieval Mode
# ============================================================
# "wiki_only" = chỉ lấy Wikipedia
# "full"      = Wikipedia + fact-check crawl + rerank
KNOWLEDGE_MODE = os.environ.get("MRCD_KNOWLEDGE_MODE", "full")

# Bật cờ này để tải toàn bộ nội dung trang Wikipedia thay vì chỉ summary.
# Có thể tiêu thụ nhiều token của LLM hơn nếu bật.
WIKI_FETCH_FULL = os.environ.get("MRCD_WIKI_FETCH_FULL", "false").lower() == "true"

# ============================================================
# Retrieval Configuration
# ============================================================
AG_NEWS_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"

TRUST_DOMAINS = [
    "apnews.com",
    "reuters.com",
    "afp.com",
    "factcheck.org",
    "snopes.com",
    "fullfact.org",
    "who.int",
    "cdc.gov",
    "npr.org",
    "pbs.org",
    "theguardian.com",
    "theconversation.com",
]

# ============================================================
# Debug / Logging
# ============================================================
RETRIEVAL_DEBUG_CSV = os.environ.get("MRCD_RETRIEVAL_DEBUG_CSV", None)
RESULTS_CSV = os.environ.get("MRCD_RESULTS_CSV", "results.csv")
TRACE_CSV = os.environ.get("MRCD_TRACE_CSV", "trace.csv")
