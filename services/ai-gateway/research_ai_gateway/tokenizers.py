import os
import gc
import threading
from concurrent.futures import ThreadPoolExecutor
from fastapi import HTTPException
from transformers import AutoTokenizer
from .config import (
    BGE_TOKENIZER_PATH,
    BGE_FALLBACK_TOKENIZER,
    RERANK_TOKENIZER_PATH,
    RERANK_FALLBACK_TOKENIZER,
    QWEN_TOKENIZER_WORKERS,
)

_embedding_tokenizers = {}
_embedding_tokenizer_lock = threading.Lock()

_rerank_tokenizer = None
_rerank_tokenizer_error = None
_rerank_tokenizer_lock = threading.Lock()

_bge_tokenizer = None
_bge_tokenizer_lock = threading.Lock()

qwen_tokenizer_executor = ThreadPoolExecutor(
    max_workers=QWEN_TOKENIZER_WORKERS,
    thread_name_prefix="qwen-tokenize",
)


def _configure_rerank_tokenizer(tokenizer):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_bge_tokenizer():
    global _bge_tokenizer
    if _bge_tokenizer is not None:
        return _bge_tokenizer
    with _bge_tokenizer_lock:
        if _bge_tokenizer is not None:
            return _bge_tokenizer
        if os.path.isdir(BGE_TOKENIZER_PATH):
            _bge_tokenizer = AutoTokenizer.from_pretrained(BGE_TOKENIZER_PATH, local_files_only=True)
        else:
            _bge_tokenizer = AutoTokenizer.from_pretrained(BGE_FALLBACK_TOKENIZER)
        return _bge_tokenizer


def get_embedding_tokenizer(path: str):
    if not path:
        raise HTTPException(status_code=500, detail="embedding tokenizer path is not configured")
    with _embedding_tokenizer_lock:
        tokenizer = _embedding_tokenizers.get(path)
        if tokenizer is not None:
            return tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        except Exception as exc:
            try:
                tokenizer = AutoTokenizer.from_pretrained(path)
            except Exception as e2:
                raise HTTPException(
                    status_code=500,
                    detail=f"failed to load embedding tokenizer from {path}: {exc} / {e2}",
                ) from e2
        _embedding_tokenizers[path] = tokenizer
        return tokenizer


def get_rerank_tokenizer():
    global _rerank_tokenizer, _rerank_tokenizer_error
    if _rerank_tokenizer is not None:
        return _rerank_tokenizer

    with _rerank_tokenizer_lock:
        if _rerank_tokenizer is not None:
            return _rerank_tokenizer

        load_errors = []
        if os.path.isdir(RERANK_TOKENIZER_PATH):
            try:
                _rerank_tokenizer = AutoTokenizer.from_pretrained(
                    RERANK_TOKENIZER_PATH,
                    local_files_only=True,
                )
                _rerank_tokenizer = _configure_rerank_tokenizer(_rerank_tokenizer)
                _rerank_tokenizer_error = None
                return _rerank_tokenizer
            except Exception as exc:
                load_errors.append(f"local tokenizer load failed: {exc}")

        try:
            _rerank_tokenizer = AutoTokenizer.from_pretrained(RERANK_FALLBACK_TOKENIZER)
            _rerank_tokenizer = _configure_rerank_tokenizer(_rerank_tokenizer)
            _rerank_tokenizer_error = None
            return _rerank_tokenizer
        except Exception as exc:
            load_errors.append(f"remote tokenizer load failed: {exc}")

        _rerank_tokenizer_error = " | ".join(load_errors) if load_errors else "unknown error"
        raise HTTPException(
            status_code=503,
            detail=(
                "rerank tokenizer unavailable; ensure protobuf is installed and "
                f"tokenizer files exist at {RERANK_TOKENIZER_PATH}. "
                f"last_error={_rerank_tokenizer_error}"
            ),
        )


def release_tokenizers_for_model(model_name: str, tokenizer_paths: set):
    global _bge_tokenizer, _rerank_tokenizer
    released = False
    if model_name in ("bge-m3-i8", "bge-m3"):
        with _bge_tokenizer_lock:
            if _bge_tokenizer is not None:
                _bge_tokenizer = None
                released = True
    if model_name == "qwen-reranker" or RERANK_TOKENIZER_PATH in tokenizer_paths:
        with _rerank_tokenizer_lock:
            if _rerank_tokenizer is not None:
                _rerank_tokenizer = None
                released = True
    if tokenizer_paths:
        with _embedding_tokenizer_lock:
            for p in tokenizer_paths:
                if _embedding_tokenizers.pop(p, None) is not None:
                    released = True
    if released:
        gc.collect()
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
