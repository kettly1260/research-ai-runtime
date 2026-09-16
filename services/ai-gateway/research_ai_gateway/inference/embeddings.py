import time
import threading
from typing import Any, List, Tuple
import numpy as np
from fastapi import HTTPException
from ..config import (
    BGE_MAX_TOTAL_TOKENS,
    BGE_CHUNK_TOKENS,
    BGE_CHUNK_OVERLAP,
    BGE_BATCH_SIZE,
    OVMS_TIMEOUT,
    ADAPTIVE_EMBEDDING_BATCHING,
    ADAPTIVE_BATCH_WAIT_MS,
    ADAPTIVE_BATCH_SUBMIT_TIMEOUT,
    QWEN_TOKENIZER_WORKERS,
    QWEN_SHORT_PRIORITY_MS,
    POOLED_LIST_SUBMIT_TIMEOUT,
)
from ..tokenizers import (
    get_bge_tokenizer,
    get_embedding_tokenizer,
    qwen_tokenizer_executor,
)
from ..ovms_client import ovms_predict, ovms_genai_embeddings


def normalize_embedding_input(raw_input: Any) -> List[str]:
    if isinstance(raw_input, str):
        return [raw_input]
    if isinstance(raw_input, list):
        normalized = []
        for item in raw_input:
            if isinstance(item, str):
                normalized.append(item)
            else:
                normalized.append(str(item))
        return normalized
    raise HTTPException(status_code=400, detail="input must be a string or list of strings")


# ---------- BGE-M3 ----------

def split_ids_for_bge(token_ids: List[int]) -> List[List[int]]:
    if not token_ids:
        return [[]]
    max_total = max(1, BGE_MAX_TOTAL_TOKENS)
    chunk_tokens = max(8, BGE_CHUNK_TOKENS)
    overlap = max(0, min(BGE_CHUNK_OVERLAP, chunk_tokens - 4))
    step = max(1, chunk_tokens - overlap)

    clipped = token_ids[:max_total]
    chunks = []
    for start in range(0, len(clipped), step):
        end = min(start + chunk_tokens, len(clipped))
        chunks.append(clipped[start:end])
        if end >= len(clipped):
            break
    return chunks


def build_bge_instances(texts: List[str]) -> Tuple[List[dict], List[List[int]], int]:
    instances = []
    mapping = []
    prompt_tokens = 0
    max_len = max(8, BGE_CHUNK_TOKENS)
    max_content_tokens = max(1, max_len - 2)

    tokenizer = get_bge_tokenizer()
    normalized_texts = [text or "" for text in texts]
    encoded_batch = tokenizer(
        normalized_texts,
        add_special_tokens=False,
        return_attention_mask=False,
    )
    token_id_batch = encoded_batch.get("input_ids", [])

    for text_index, token_ids in enumerate(token_id_batch):
        chunks = split_ids_for_bge(token_ids)
        for chunk_ids in chunks:
            chunk_ids = chunk_ids[:max_content_tokens]
            if hasattr(tokenizer, "build_inputs_with_special_tokens"):
                input_ids_list = tokenizer.build_inputs_with_special_tokens(chunk_ids)
            else:
                input_ids_list = [tokenizer.bos_token_id, *chunk_ids, tokenizer.eos_token_id]

            input_ids = np.asarray(input_ids_list, dtype=np.int64)
            attention_mask = np.ones_like(input_ids, dtype=np.int64)

            prompt_tokens += int(attention_mask.sum())
            instances.append({
                "input_ids": input_ids.tolist(),
                "attention_mask": attention_mask.tolist(),
            })
            mapping.append([text_index, int(attention_mask.sum())])

    return instances, mapping, prompt_tokens


def mean_pool(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask = attention_mask.astype(np.float32)[:, :, None]
    weighted = last_hidden_state * mask
    summed = weighted.sum(axis=1)
    counts = np.maximum(mask.sum(axis=1), 1e-9)
    return summed / counts


def pad_instances_for_ovms(instances: List[dict]) -> List[dict]:
    if not instances:
        return []
    tokenizer = get_bge_tokenizer()
    pad_id = tokenizer.pad_token_id or 0
    max_len = max(len(item["input_ids"]) for item in instances)
    padded = []
    for item in instances:
        ids = list(item["input_ids"])
        mask = list(item["attention_mask"])
        pad_count = max_len - len(ids)
        if pad_count > 0:
            ids.extend([int(pad_id)] * pad_count)
            mask.extend([0] * pad_count)
        padded.append({"input_ids": ids, "attention_mask": mask})
    return padded


def extract_batch_vectors(response: dict, batch_masks: np.ndarray) -> np.ndarray:
    predictions = response.get("predictions", [])
    if predictions and not isinstance(predictions[0], dict):
        tensor = np.asarray(predictions, dtype=np.float32)
        if tensor.ndim == 2:
            return tensor
        if tensor.ndim == 3:
            return mean_pool(tensor, batch_masks)
        raise HTTPException(status_code=502, detail="unexpected OVMS prediction shape")

    if predictions and isinstance(predictions[0], dict):
        first = predictions[0]
        if "sentence_embedding" in first:
            return np.asarray([item.get("sentence_embedding", []) for item in predictions], dtype=np.float32)
        token_key = None
        for cand in ("token_embeddings", "last_hidden_state", "output"):
            if cand in first:
                token_key = cand
                break
        if token_key is not None:
            token_tensor = np.asarray([item.get(token_key, []) for item in predictions], dtype=np.float32)
            if token_tensor.ndim != 3:
                raise HTTPException(status_code=502, detail=f"unexpected token output shape for key: {token_key}")
            return mean_pool(token_tensor, batch_masks)
        raise HTTPException(status_code=502, detail=f"unsupported OVMS prediction keys: {sorted(first.keys())}")
    raise HTTPException(status_code=502, detail="empty OVMS predictions")


def run_bge_embedding(model_name: str, texts: List[str]) -> Tuple[List[List[float]], int]:
    instances, mapping, prompt_tokens = build_bge_instances(texts)
    if not instances:
        return [[] for _ in texts], 0

    chunk_vectors = []
    for start in range(0, len(instances), max(1, BGE_BATCH_SIZE)):
        raw_batch = instances[start:start + max(1, BGE_BATCH_SIZE)]
        batch = pad_instances_for_ovms(raw_batch)
        response = ovms_predict(model_name, {"instances": batch}, timeout=OVMS_TIMEOUT)
        batch_masks = np.asarray([item["attention_mask"] for item in batch], dtype=np.int64)
        vectors = extract_batch_vectors(response, batch_masks)
        if vectors.ndim != 2:
            raise HTTPException(status_code=502, detail="unexpected OVMS embedding shape")
        chunk_vectors.extend(vectors)

    merged = [np.zeros_like(chunk_vectors[0], dtype=np.float32) for _ in texts]
    weights = [0.0 for _ in texts]
    for vec, (text_index, token_count) in zip(chunk_vectors, mapping):
        w = float(max(token_count, 1))
        merged[text_index] += vec * w
        weights[text_index] += w

    embeddings = []
    for idx in range(len(texts)):
        out = merged[idx] if weights[idx] <= 0 else (merged[idx] / weights[idx])
        norm = float(np.linalg.norm(out))
        if norm > 0:
            out = out / norm
        embeddings.append(out.astype(np.float32).tolist())
    return embeddings, prompt_tokens


def call_embedding(model_name: str, raw_input: Any, response_model_name: str) -> dict:
    texts = normalize_embedding_input(raw_input)
    embeddings, prompt_tokens = run_bge_embedding(model_name, texts)
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": idx, "embedding": emb}
            for idx, emb in enumerate(embeddings)
        ],
        "model": response_model_name,
        "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
    }


# ---------- Sentence Transformer (e.g. Arctic) ----------

def call_sentence_transformer_embedding(
    model_name: str,
    raw_input: Any,
    response_model_name: str,
    cfg: dict,
    input_type: str = None,
) -> dict:
    texts = normalize_embedding_input(raw_input)
    tokenizer = get_embedding_tokenizer(cfg.get("tokenizer_path"))
    max_length = max(8, int(cfg.get("max_length", 8192)))
    batch_size = max(1, int(cfg.get("batch_size", 2)))
    if input_type == "query" and cfg.get("query_prefix"):
        prefix = str(cfg.get("query_prefix"))
        texts = [prefix + text for text in texts]

    embeddings = []
    prompt_tokens = 0
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )
        input_ids = encoded["input_ids"].astype(np.int64)
        attention_mask = encoded["attention_mask"].astype(np.int64)
        prompt_tokens += int(attention_mask.sum())
        payload = {
            "instances": [
                {
                    "input_ids": input_ids[idx].tolist(),
                    "attention_mask": attention_mask[idx].tolist(),
                }
                for idx in range(input_ids.shape[0])
            ]
        }
        response = ovms_predict(model_name, payload, timeout=OVMS_TIMEOUT)
        vectors = extract_batch_vectors(response, attention_mask)
        if vectors.ndim != 2:
            raise HTTPException(status_code=502, detail="unexpected sentence-transformer embedding shape")
        for vector in vectors:
            vector = np.asarray(vector, dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            if norm > 0:
                vector = vector / norm
            embeddings.append(vector.tolist())

    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": idx, "embedding": emb}
            for idx, emb in enumerate(embeddings)
        ],
        "model": response_model_name,
        "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
    }


# ---------- GenAI v3 (graph-backed) embeddings ----------

def normalize_genai_embeddings_response(result: Any, response_model_name: str) -> dict:
    """Normalises an OVMS GenAI embeddings payload to the gateway's response shape.

    The legacy gateway passed the backend payload through with only ``model``
    replaced. We keep the observable shape identical (``object``/``data``/
    ``model``/``usage``) while validating the vectors so a malformed backend
    response surfaces as a 502 instead of a silent bad vector.
    """
    if not isinstance(result, dict):
        raise HTTPException(status_code=502, detail="embedding backend returned non-object JSON")

    raw_data = result.get("data")
    if not isinstance(raw_data, list) or not raw_data:
        raise HTTPException(status_code=502, detail="empty genai embedding response")

    normalized: List[dict] = []
    for position, item in enumerate(raw_data):
        if not isinstance(item, dict):
            raise HTTPException(
                status_code=502,
                detail=f"unexpected genai embedding item type: {type(item).__name__}",
            )
        embedding = item.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise HTTPException(
                status_code=502,
                detail="genai embedding item is missing a usable embedding vector",
            )
        try:
            vector = [float(value) for value in embedding]
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=502,
                detail="genai embedding vector contains non-numeric values",
            ) from exc
        try:
            index = int(item.get("index", position))
        except (TypeError, ValueError):
            index = position
        normalized.append({"object": "embedding", "index": index, "embedding": vector})

    normalized.sort(key=lambda entry: entry["index"])

    usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
    try:
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
    except (TypeError, ValueError):
        prompt_tokens = 0
    try:
        total_tokens = int(usage.get("total_tokens") or prompt_tokens)
    except (TypeError, ValueError):
        total_tokens = prompt_tokens

    return {
        "object": "list",
        "data": normalized,
        "model": response_model_name,
        "usage": {"prompt_tokens": prompt_tokens, "total_tokens": total_tokens},
    }


def call_genai_embedding(model_name: str, raw_input: Any, response_model_name: str) -> dict:
    """Graph-backed (GenAI v3) embeddings via OVMS ``/v3/embeddings``."""
    texts = normalize_embedding_input(raw_input)
    result = ovms_genai_embeddings(model_name, texts)
    return normalize_genai_embeddings_response(result, response_model_name)


# ---------- Qwen Pooled IR ----------

def _tokenize_qwen_text(text: str, tokenizer_path: str):
    tokenizer = get_embedding_tokenizer(tokenizer_path)
    encoded = tokenizer(
        text or "",
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=True,
    )
    input_ids = np.asarray(encoded.get("input_ids", []), dtype=np.int64)
    attention_mask = np.asarray(encoded.get("attention_mask", []), dtype=np.int64)
    if input_ids.size == 0:
        fallback = tokenizer("", add_special_tokens=True, truncation=False, return_attention_mask=True)
        input_ids = np.asarray(fallback.get("input_ids", []), dtype=np.int64)
        attention_mask = np.asarray(fallback.get("attention_mask", []), dtype=np.int64)
    if attention_mask.size == 0:
        attention_mask = np.ones(input_ids.shape, dtype=np.int64)
    return input_ids, attention_mask


def tokenize_qwen_texts(texts: List[str], tokenizer_path: str):
    futures = [
        qwen_tokenizer_executor.submit(_tokenize_qwen_text, text, tokenizer_path)
        for text in texts
    ]
    return [future.result() for future in futures]


def _pad_qwen_prepared(prepared, tokenizer_path: str):
    if not prepared:
        return [], 0
    tokenizer = get_embedding_tokenizer(tokenizer_path)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    max_len = max(len(input_ids) for input_ids, _ in prepared)
    instances = []
    prompt_tokens = 0
    for input_ids, attention_mask in prepared:
        ids = np.asarray(input_ids, dtype=np.int64).tolist()
        mask = np.asarray(attention_mask, dtype=np.int64).tolist()
        prompt_tokens += int(sum(mask))
        pad_count = max_len - len(ids)
        if pad_count > 0:
            ids.extend([int(pad_id)] * pad_count)
            mask.extend([0] * pad_count)
        instances.append({"input_ids": ids, "attention_mask": mask})
    return instances, prompt_tokens


def _extract_pooled_vectors(response: dict) -> np.ndarray:
    predictions = response.get("predictions", [])
    if not predictions:
        raise HTTPException(status_code=502, detail="empty pooled-IR prediction")
    if isinstance(predictions[0], dict):
        keys = list(predictions[0].keys())
        if len(keys) != 1:
            raise HTTPException(status_code=502, detail=f"unexpected pooled-IR prediction keys: {sorted(keys)}")
        predictions = [row.get(keys[0], []) for row in predictions]
    vectors = np.asarray(predictions, dtype=np.float32)
    if vectors.ndim != 2:
        raise HTTPException(status_code=502, detail=f"unexpected pooled-IR prediction shape: {list(vectors.shape)}")
    return vectors


def predict_pooled_ir(model_name: str, prepared, tokenizer_path: str):
    instances, prompt_tokens = _pad_qwen_prepared(prepared, tokenizer_path)
    response = ovms_predict(model_name, {"instances": instances}, timeout=OVMS_TIMEOUT)
    return _extract_pooled_vectors(response), prompt_tokens


def pooled_ir_response(model_name: str, texts: List[str], response_model_name: str, tokenizer_path: str, broker=None):
    prepared = tokenize_qwen_texts(texts, tokenizer_path)
    if broker:
        with broker.embedding_inference_lock:
            # Hold the lease for the entire backend inference and use the
            # concrete runtime alias selected by DeviceBroker.  Merely calling
            # ensure_model_available() releases the lease before inference and
            # then incorrectly addresses the unsuffixed logical model name,
            # which breaks zero-resident and CPU-spillover operation.
            with broker.lease(model_name) as active_model:
                vectors, prompt_tokens = predict_pooled_ir(active_model, prepared, tokenizer_path)
    else:
        vectors, prompt_tokens = predict_pooled_ir(model_name, prepared, tokenizer_path)
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": idx, "embedding": vector.tolist()}
            for idx, vector in enumerate(vectors)
        ],
        "model": response_model_name,
        "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
    }


# ---------- Adaptive Batchers ----------

class _AdaptiveEmbeddingItem:
    def __init__(self, text: str, token_count: int, response_model_name: str):
        self.text = text
        self.token_count = token_count
        self.response_model_name = response_model_name
        self.created = time.monotonic()
        self.event = threading.Event()
        self.result = None
        self.error = None
        self.model_name = None
        self.bucket = None
        self.tokenizer_path = None
        self.input_ids = None
        self.attention_mask = None


class AdaptiveGenAIBatcher:
    """Length-aware cross-request batching for single-text GenAI (graph) embeddings.

    Ported from the legacy gateway so the graph-backed
    ``qwen3-embedding-0.6b-int8`` model keeps identical batching behaviour.
    The thresholds are based on measured UHD 730 throughput for Qwen3
    Embedding 0.6B INT4. Long inputs bypass active waiting because batching
    brings little or no throughput gain there.
    """

    backend_name = "genai_v3"
    timeout_detail = "adaptive embedding batch timeout"
    failure_detail = "adaptive embedding batch failed"

    def __init__(self, broker=None):
        self.broker = broker
        self._condition = threading.Condition()
        self._queues = {"le256": [], "257_384": [], "385_512": [], "gt512": []}
        self._stats_lock = threading.Lock()
        self._stats = {
            "submitted_items": 0,
            "backend_batches": 0,
            "backend_items": 0,
            "batched_items": 0,
            "backend_seconds": 0.0,
            "queue_wait_seconds": 0.0,
            "batch_size_counts": {"1": 0, "2": 0, "3": 0, "4": 0},
            "bucket_items": {"le256": 0, "257_384": 0, "385_512": 0, "gt512": 0},
            "errors": 0,
        }
        self._worker = threading.Thread(
            target=self._run,
            name="adaptive-genai-embedding-batcher",
            daemon=True,
        )
        self._worker.start()

    @staticmethod
    def _policy(token_count: int):
        if token_count <= 256:
            return "le256", 4, max(0.0, ADAPTIVE_BATCH_WAIT_MS) / 1000.0
        if token_count <= 384:
            return "257_384", 2, max(0.0, ADAPTIVE_BATCH_WAIT_MS) / 1000.0
        if token_count <= 512:
            # Only combine requests already waiting; do not add latency just
            # to chase the small ~5-7% throughput gain seen in this range.
            return "385_512", 2, 0.0
        return "gt512", 1, 0.0

    def _token_count(self, text: str, tokenizer_path: str) -> int:
        tokenizer = get_embedding_tokenizer(tokenizer_path)
        encoded = tokenizer(
            text or "",
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
        )
        ids = encoded.get("input_ids", [])
        return max(1, len(ids))

    def submit(self, model_name: str, response_model_name: str, text: str, tokenizer_path: str):
        token_count = self._token_count(text, tokenizer_path)
        bucket, _, _ = self._policy(token_count)
        item = _AdaptiveEmbeddingItem(text, token_count, response_model_name)
        item.model_name = model_name
        item.bucket = bucket
        item.tokenizer_path = tokenizer_path

        with self._stats_lock:
            self._stats["submitted_items"] += 1
            self._stats["bucket_items"][bucket] += 1

        with self._condition:
            self._queues[bucket].append(item)
            self._condition.notify_all()

        if not item.event.wait(timeout=ADAPTIVE_BATCH_SUBMIT_TIMEOUT):
            raise HTTPException(status_code=504, detail=self.timeout_detail)
        if item.error is not None:
            if isinstance(item.error, HTTPException):
                raise item.error
            raise HTTPException(
                status_code=502, detail=f"{self.failure_detail}: {item.error}"
            )
        return item.result

    def _select_bucket_locked(self):
        candidates = []
        for name, queue in self._queues.items():
            if queue:
                candidates.append((queue[0].created, name))
        if not candidates:
            return None
        candidates.sort(key=lambda entry: entry[0])
        return candidates[0][1]

    def _take_batch(self):
        with self._condition:
            while True:
                bucket = self._select_bucket_locked()
                if bucket is None:
                    self._condition.wait()
                    continue

                queue = self._queues[bucket]
                head = queue[0]
                _, max_batch, wait_seconds = self._policy(head.token_count)
                deadline = head.created + wait_seconds

                while len(queue) < max_batch and wait_seconds > 0:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._condition.wait(timeout=remaining)

                take = min(max_batch, len(queue))
                batch = queue[:take]
                del queue[:take]
                return batch

    def _predict_batch(self, model_name: str, texts: List[str], response_model_name: str) -> dict:
        if self.broker:
            # The model switch and inference stay in one critical section so
            # another embedding family cannot evict this model mid-batch.
            with self.broker.embedding_inference_lock:
                with self.broker.lease(model_name) as active_model:
                    return call_genai_embedding(active_model, texts, response_model_name)
        return call_genai_embedding(model_name, texts, response_model_name)

    def _finish_error(self, batch, exc):
        with self._stats_lock:
            self._stats["errors"] += len(batch)
        for item in batch:
            item.error = exc
            item.event.set()

    def _run(self):
        while True:
            batch = self._take_batch()
            if not batch:
                continue

            model_name = batch[0].model_name
            texts = [item.text for item in batch]
            started = time.monotonic()
            try:
                result = self._predict_batch(model_name, texts, batch[0].response_model_name)

                data = sorted(result.get("data", []), key=lambda entry: entry.get("index", 0))
                if len(data) != len(batch):
                    raise RuntimeError(
                        f"embedding backend returned {len(data)} vectors for {len(batch)} inputs"
                    )

                finished = time.monotonic()
                backend_seconds = finished - started
                wait_seconds = sum(max(0.0, started - item.created) for item in batch)
                with self._stats_lock:
                    size_key = str(len(batch))
                    self._stats["backend_batches"] += 1
                    self._stats["backend_items"] += len(batch)
                    if len(batch) > 1:
                        self._stats["batched_items"] += len(batch)
                    self._stats["backend_seconds"] += backend_seconds
                    self._stats["queue_wait_seconds"] += wait_seconds
                    self._stats["batch_size_counts"].setdefault(size_key, 0)
                    self._stats["batch_size_counts"][size_key] += 1

                for idx, item in enumerate(batch):
                    entry = dict(data[idx])
                    entry["index"] = 0
                    item.result = {
                        "object": "list",
                        "data": [entry],
                        "model": item.response_model_name,
                        "usage": {
                            "prompt_tokens": item.token_count,
                            "total_tokens": item.token_count,
                        },
                    }
                    item.event.set()
            except Exception as exc:
                self._finish_error(batch, exc)

    def stats(self) -> dict:
        with self._condition:
            queue_depths = {name: len(queue) for name, queue in self._queues.items()}
        with self._stats_lock:
            snap = dict(self._stats)
            snap["batch_size_counts"] = dict(self._stats["batch_size_counts"])
            snap["bucket_items"] = dict(self._stats["bucket_items"])
        backend_items = max(1, snap["backend_items"])
        submitted = max(1, snap["submitted_items"])
        snap.update(
            {
                "backend": self.backend_name,
                "enabled": ADAPTIVE_EMBEDDING_BATCHING,
                "wait_ms": ADAPTIVE_BATCH_WAIT_MS,
                "queue_depths": queue_depths,
                "mean_backend_ms_per_item": 1000.0 * snap["backend_seconds"] / backend_items,
                "mean_queue_wait_ms": 1000.0 * snap["queue_wait_seconds"] / submitted,
                "saved_backend_calls": max(0, snap["backend_items"] - snap["backend_batches"]),
                "policy": {
                    "<=256": {"max_batch": 4, "wait_ms": ADAPTIVE_BATCH_WAIT_MS},
                    "257-384": {"max_batch": 2, "wait_ms": ADAPTIVE_BATCH_WAIT_MS},
                    "385-512": {"max_batch": 2, "wait_ms": 0},
                    ">512": {"max_batch": 1, "wait_ms": 0},
                },
            }
        )
        return snap


class AdaptivePooledIRBatcher:
    """Length-aware cross-request batching for Pooled IR."""

    def __init__(self, broker=None):
        self.broker = broker
        self._condition = threading.Condition()
        self._queues = {"le256": [], "257_384": [], "385_512": [], "gt512": []}
        self._stats_lock = threading.Lock()
        self._stats = {
            "submitted_items": 0,
            "backend_batches": 0,
            "backend_items": 0,
            "batched_items": 0,
            "backend_seconds": 0.0,
            "queue_wait_seconds": 0.0,
            "batch_size_counts": {"1": 0, "2": 0, "3": 0, "4": 0},
            "bucket_items": {"le256": 0, "257_384": 0, "385_512": 0, "gt512": 0},
            "errors": 0,
            "token_sum": 0,
            "max_tokens_seen": 0,
            "request_count": 0,
            "list_request_count": 0,
            "max_request_items": 0,
        }
        self._worker = threading.Thread(
            target=self._run,
            name="adaptive-pooled-ir-batcher",
            daemon=True,
        )
        self._worker.start()

    @staticmethod
    def _policy(token_count: int):
        if token_count <= 256:
            return "le256", 4, max(0.0, ADAPTIVE_BATCH_WAIT_MS) / 1000.0
        if token_count <= 384:
            return "257_384", 2, max(0.0, ADAPTIVE_BATCH_WAIT_MS) / 1000.0
        if token_count <= 512:
            return "385_512", 2, 0.0
        return "gt512", 1, 0.0

    def submit(self, model_name: str, response_model_name: str, text: str, tokenizer_path: str):
        prepared = qwen_tokenizer_executor.submit(_tokenize_qwen_text, text, tokenizer_path).result()
        input_ids, attention_mask = prepared
        token_count = max(1, int(sum(attention_mask)))
        bucket, _, _ = self._policy(token_count)
        item = _AdaptiveEmbeddingItem(text, token_count, response_model_name)
        item.model_name = model_name
        item.bucket = bucket
        item.tokenizer_path = tokenizer_path
        item.input_ids = input_ids
        item.attention_mask = attention_mask

        with self._stats_lock:
            self._stats["request_count"] += 1
            self._stats["max_request_items"] = max(self._stats["max_request_items"], 1)
            self._stats["submitted_items"] += 1
            self._stats["bucket_items"][bucket] += 1
            self._stats["token_sum"] += token_count
            self._stats["max_tokens_seen"] = max(self._stats["max_tokens_seen"], token_count)

        with self._condition:
            self._queues[bucket].append(item)
            self._condition.notify_all()

        if not item.event.wait(timeout=ADAPTIVE_BATCH_SUBMIT_TIMEOUT):
            raise HTTPException(status_code=504, detail="adaptive pooled-IR batch timeout")
        if item.error is not None:
            if isinstance(item.error, HTTPException):
                raise item.error
            raise HTTPException(status_code=502, detail=f"adaptive pooled-IR batch failed: {item.error}")
        return item.result

    def submit_many(self, model_name: str, response_model_name: str, texts: List[str], tokenizer_path: str):
        prepared_items = tokenize_qwen_texts(texts, tokenizer_path)
        items = []
        token_sum = 0
        max_tokens = 0
        for text, prepared in zip(texts, prepared_items):
            input_ids, attention_mask = prepared
            token_count = max(1, int(np.asarray(attention_mask, dtype=np.int64).sum()))
            bucket, _, _ = self._policy(token_count)
            item = _AdaptiveEmbeddingItem(text, token_count, response_model_name)
            item.model_name = model_name
            item.bucket = bucket
            item.tokenizer_path = tokenizer_path
            item.input_ids = input_ids
            item.attention_mask = attention_mask
            items.append(item)
            token_sum += token_count
            max_tokens = max(max_tokens, token_count)

        with self._stats_lock:
            self._stats["request_count"] += 1
            self._stats["list_request_count"] += 1
            self._stats["max_request_items"] = max(self._stats["max_request_items"], len(items))
            self._stats["submitted_items"] += len(items)
            for it in items:
                self._stats["bucket_items"][it.bucket] += 1
            self._stats["token_sum"] += token_sum
            self._stats["max_tokens_seen"] = max(self._stats["max_tokens_seen"], max_tokens)

        with self._condition:
            for it in items:
                self._queues[it.bucket].append(it)
            self._condition.notify_all()

        deadline = time.monotonic() + POOLED_LIST_SUBMIT_TIMEOUT
        for it in items:
            rem = deadline - time.monotonic()
            if rem <= 0 or not it.event.wait(timeout=rem):
                raise HTTPException(status_code=504, detail="adaptive pooled-IR list batch timeout")
            if it.error is not None:
                if isinstance(it.error, HTTPException):
                    raise it.error
                raise HTTPException(status_code=502, detail=f"adaptive pooled-IR list batch failed: {it.error}")

        return {
            "object": "list",
            "data": [
                {"object": "embedding", "index": idx, "embedding": it.result["data"][0]["embedding"]}
                for idx, it in enumerate(items)
            ],
            "model": response_model_name,
            "usage": {"prompt_tokens": token_sum, "total_tokens": token_sum},
        }

    def _select_bucket_locked(self):
        now = time.monotonic()
        priority_after = QWEN_SHORT_PRIORITY_MS / 1000.0
        for name in ("le256", "257_384", "385_512"):
            q = self._queues[name]
            if q and (now - q[0].created) >= priority_after:
                return name
        for name in ("gt512", "385_512", "257_384", "le256"):
            if self._queues[name]:
                return name
        return None

    def _take_batch(self):
        with self._condition:
            while True:
                bucket = self._select_bucket_locked()
                if bucket is None:
                    self._condition.wait()
                    continue
                q = self._queues[bucket]
                first = q[0]
                _, max_batch, max_wait = self._policy(first.token_count)
                age = time.monotonic() - first.created
                if len(q) < max_batch and age < max_wait:
                    self._condition.wait(timeout=max(0.001, max_wait - age))
                    continue
                batch = []
                target_model = first.model_name
                i = 0
                while i < len(q) and len(batch) < max_batch:
                    if q[i].model_name == target_model:
                        batch.append(q.pop(i))
                    else:
                        i += 1
                return batch

    def _predict_batch(self, model_name: str, prepared, tokenizer_path: str):
        if self.broker:
            with self.broker.embedding_inference_lock:
                with self.broker.lease(model_name) as active_model:
                    return predict_pooled_ir(active_model, prepared, tokenizer_path)
        return predict_pooled_ir(model_name, prepared, tokenizer_path)

    def _run(self):
        while True:
            batch = self._take_batch()
            if not batch:
                continue
            model_name = batch[0].model_name
            tokenizer_path = batch[0].tokenizer_path
            prepared = [(item.input_ids, item.attention_mask) for item in batch]
            now = time.monotonic()
            wait_s = sum(now - item.created for item in batch)
            try:
                t0 = time.monotonic()
                vectors, _ = self._predict_batch(model_name, prepared, tokenizer_path)
                backend_s = time.monotonic() - t0

                for idx, item in enumerate(batch):
                    item.result = {
                        "object": "list",
                        "data": [{
                            "object": "embedding",
                            "index": 0,
                            "embedding": vectors[idx].tolist(),
                        }],
                        "model": item.response_model_name,
                        "usage": {
                            "prompt_tokens": item.token_count,
                            "total_tokens": item.token_count,
                        },
                    }
                    item.event.set()

                with self._stats_lock:
                    self._stats["backend_batches"] += 1
                    self._stats["backend_items"] += len(batch)
                    if len(batch) > 1:
                        self._stats["batched_items"] += len(batch)
                    self._stats["backend_seconds"] += backend_s
                    self._stats["queue_wait_seconds"] += wait_s
                    b_key = str(min(len(batch), 4))
                    self._stats["batch_size_counts"][b_key] += 1
            except Exception as exc:
                with self._stats_lock:
                    self._stats["errors"] += 1
                for item in batch:
                    item.error = exc
                    item.event.set()

    def stats(self) -> dict:
        with self._stats_lock:
            snap = dict(self._stats)
            snap["batch_size_counts"] = dict(self._stats["batch_size_counts"])
            snap["bucket_items"] = dict(self._stats["bucket_items"])
        submitted = max(1, snap["submitted_items"])
        snap.update({
            "backend": "pooled_ir",
            "tokenizer_workers": QWEN_TOKENIZER_WORKERS,
            "short_priority_ms": QWEN_SHORT_PRIORITY_MS,
            "mean_tokens": float(snap.get("token_sum", 0)) / submitted,
            "max_tokens_seen": int(snap.get("max_tokens_seen", 0)),
        })
        return snap
