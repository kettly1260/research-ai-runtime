import os
import yaml
from typing import Any, Dict, Optional

OVMS_BASE = os.getenv("OVMS_BASE", "http://ovms:8001")
OVMS_TIMEOUT = int(os.getenv("OVMS_TIMEOUT", "60"))
MODEL_IDLE_UNLOAD_SECONDS = int(os.getenv("MODEL_IDLE_UNLOAD_SECONDS", "1200"))
MAX_LOADED_MODELS = int(os.getenv("MAX_LOADED_MODELS", "2"))
MODEL_IDLE_SWEEP_SECONDS = int(os.getenv("MODEL_IDLE_SWEEP_SECONDS", "60"))

OVMS_CONFIG_PATH = os.getenv("OVMS_CONFIG_PATH", "/ovms-config/config.json")
OVMS_MODEL_CATALOG_PATH = os.getenv("OVMS_MODEL_CATALOG_PATH", "/ovms-config/model_catalog.json")
MODEL_REGISTRY_PATH = os.getenv("MODEL_REGISTRY_PATH", "/ovms-config/model_registry.json")
MODELS_CONFIG_PATH = os.getenv("MODELS_CONFIG_PATH", "/config/models.yaml")
OVMS_CONFIG_RELOAD_URL = os.getenv("OVMS_CONFIG_RELOAD_URL", "{ovms_base}/v1/config/reload")
# GenAI v3 (graph-backed) OpenAI-compatible embeddings endpoint.
OVMS_GENAI_EMBEDDINGS_URL = os.getenv(
    "OVMS_GENAI_EMBEDDINGS_URL", "{ovms_base}/v3/embeddings"
)
OVMS_CONFIG_UPDATE_MODE = os.getenv("OVMS_CONFIG_UPDATE_MODE", "poll").strip().lower()
if OVMS_CONFIG_UPDATE_MODE not in {"poll", "api"}:
    raise ValueError("OVMS_CONFIG_UPDATE_MODE must be 'poll' or 'api'")

PINNED_MODELS = {
    item.strip()
    for item in os.getenv("PINNED_MODELS", "").split(",")
    if item.strip()
}

# Rerank settings
RERANK_MAX_LENGTH = int(os.getenv("RERANK_MAX_LENGTH", "256"))
RERANK_MAX_DOCS = int(os.getenv("RERANK_MAX_DOCS", "8"))
RERANK_MAX_CHARS = int(os.getenv("RERANK_MAX_CHARS", "1000"))
RERANK_BATCH_SIZE = int(os.getenv("RERANK_BATCH_SIZE", "2"))
RERANK_FALLBACK_MAX_LENGTH = int(os.getenv("RERANK_FALLBACK_MAX_LENGTH", "96"))
RERANK_FALLBACK_TIMEOUT = int(os.getenv("RERANK_FALLBACK_TIMEOUT", "180"))
RERANK_STRICT_RECALL = os.getenv("RERANK_STRICT_RECALL", "1") == "1"
RERANK_TOKENIZER_PATH = os.getenv("RERANK_TOKENIZER_PATH", "/tokenizers/qwen-reranker")
RERANK_FALLBACK_TOKENIZER = os.getenv("RERANK_FALLBACK_TOKENIZER", "Qwen/Qwen3-Reranker-0.6B")

# BGE settings
BGE_TOKENIZER_PATH = os.getenv("BGE_TOKENIZER_PATH", "/tokenizers/bge-m3")
BGE_FALLBACK_TOKENIZER = os.getenv("BGE_FALLBACK_TOKENIZER", "BAAI/bge-m3")
BGE_MAX_TOTAL_TOKENS = int(os.getenv("BGE_MAX_TOTAL_TOKENS", "8192"))
BGE_CHUNK_TOKENS = int(os.getenv("BGE_CHUNK_TOKENS", "1024"))
BGE_CHUNK_OVERLAP = int(os.getenv("BGE_CHUNK_OVERLAP", "128"))
BGE_BATCH_SIZE = int(os.getenv("BGE_BATCH_SIZE", "4"))

# Qwen pooled-IR long-text settings. Logical requests may be much longer than
# a single GPU inference window; they are chunked internally and merged back to
# one embedding so callers keep the normal OpenAI-compatible API contract.
POOLED_MAX_TOTAL_TOKENS = int(os.getenv("POOLED_MAX_TOTAL_TOKENS", "32768"))
POOLED_CHUNK_TOKENS = int(os.getenv("POOLED_CHUNK_TOKENS", "1024"))
POOLED_CHUNK_OVERLAP = int(os.getenv("POOLED_CHUNK_OVERLAP", "128"))
POOLED_CHUNK_BATCH_SIZE = max(1, int(os.getenv("POOLED_CHUNK_BATCH_SIZE", "1")))

# Adaptive batching settings
ADAPTIVE_EMBEDDING_BATCHING = os.getenv("ADAPTIVE_EMBEDDING_BATCHING", "1") == "1"
ADAPTIVE_BATCH_WAIT_MS = float(os.getenv("ADAPTIVE_BATCH_WAIT_MS", "8"))
ADAPTIVE_BATCH_SUBMIT_TIMEOUT = float(os.getenv("ADAPTIVE_BATCH_SUBMIT_TIMEOUT", str(max(OVMS_TIMEOUT + 30, 90))))
QWEN_TOKENIZER_WORKERS = max(1, int(os.getenv("QWEN_TOKENIZER_WORKERS", "4")))
QWEN_SHORT_PRIORITY_MS = max(0.0, float(os.getenv("QWEN_SHORT_PRIORITY_MS", "25")))
POOLED_LIST_SUBMIT_TIMEOUT = max(ADAPTIVE_BATCH_SUBMIT_TIMEOUT, float(os.getenv("POOLED_LIST_SUBMIT_TIMEOUT", "600")))

# Device Broker resource budgets
CPU_SEMAPHORE_LIMIT = int(os.getenv("CPU_SEMAPHORE_LIMIT", "4"))
CPU_THREAD_BUDGET = int(os.getenv("CPU_THREAD_BUDGET", "4"))
MAX_QUEUE_DEPTH = int(os.getenv("MAX_QUEUE_DEPTH", "64"))


def load_yaml_models_config(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict):
                return data
    except Exception as exc:
        print(f"Warning: Failed to load models config from {path}: {exc}", flush=True)
    return None
