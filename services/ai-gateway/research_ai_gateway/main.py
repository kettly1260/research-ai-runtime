import time
from typing import Any, Dict
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from .config import ADAPTIVE_EMBEDDING_BATCHING
from .registry import MODEL_REGISTRY
from .broker import DEVICE_BROKER
from . import ovms_protocol
from .inference import (
    call_rerank,
    normalize_embedding_input,
    call_embedding,
    call_genai_embedding,
    call_sentence_transformer_embedding,
    pooled_ir_response,
    AdaptiveGenAIBatcher,
    AdaptivePooledIRBatcher,
    run_multimodal_image_embedding,
    run_multimodal_text_embedding,
    run_dino_embedding,
)

adaptive_genai_batcher = AdaptiveGenAIBatcher(broker=DEVICE_BROKER)
adaptive_pooled_batcher = AdaptivePooledIRBatcher(broker=DEVICE_BROKER)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start Device Broker idle sweeper for Zero-Resident model management
    DEVICE_BROKER.start_sweeper()
    yield
    # Shutdown


app = FastAPI(
    title="Research AI Gateway",
    version="0.2.0",
    description="Universal AI Inference Gateway with GPU-first and CPU spillover broker",
    lifespan=lifespan,
)


# ---------- Model Management APIs ----------

@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": cfg.get("owned_by", "openvino"),
                "permission": [],
            }
            for model_id, cfg in MODEL_REGISTRY.items()
        ]
    }


@app.get("/models")
def models_compat():
    return models()


@app.get("/v1/models/{model_id}")
def model_detail(model_id: str):
    cfg = MODEL_REGISTRY.get(model_id)
    if not cfg:
        raise HTTPException(status_code=404, detail="model not found")
    return {
        "id": model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": cfg.get("owned_by", "openvino"),
        "permission": [],
    }


# ---------- Rerank API ----------

@app.post("/v1/rerank")
def rerank(req: Dict[str, Any]):
    model = req.get("model", "qwen-reranker")
    top_n = req.get("top_n")
    if top_n is None:
        top_n = 0
    else:
        try:
            top_n = int(top_n)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="top_n must be an integer") from exc
        if top_n < 0:
            raise HTTPException(status_code=400, detail="top_n must be >= 0")

    cfg = MODEL_REGISTRY.get(model)
    if not cfg:
        raise HTTPException(status_code=404, detail="model not found")

    ovms_model = cfg.get("ovms_model", model)
    preferred_device = cfg.get("preferred_device", "GPU")

    query = req.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    documents = req.get("documents", [])

    with DEVICE_BROKER.lease(ovms_model, preferred_device=preferred_device) as active_model:
        ranked, rerank_stats = call_rerank(active_model, query, documents)
    if top_n > 0:
        ranked = ranked[:top_n]

    from .config import RERANK_STRICT_RECALL
    return {
        "results": [
            {
                "text": item["text"],
                "score": item["score"],
                "document": item["document"],
            }
            for item in ranked
        ],
        "meta": {
            "strict_recall": RERANK_STRICT_RECALL,
            "total_documents": rerank_stats["total_documents"],
            "scored_documents": rerank_stats["scored_documents"],
            "discarded_documents": rerank_stats["discarded_documents"],
            "text_truncated_documents": rerank_stats["text_truncated_documents"],
            "applied_top_n": top_n if top_n > 0 else None,
        },
    }


# ---------- Embeddings API (Text, Multimodal, DINO) ----------

@app.post("/v1/embeddings")
def embeddings(req: Dict[str, Any]):
    model = req.get("model", "bge-m3")
    cfg = MODEL_REGISTRY.get(model)
    if not cfg:
        raise HTTPException(status_code=404, detail="model not found")

    ovms_model = cfg.get("ovms_model", model)
    preferred_device = cfg.get("preferred_device", "GPU")
    model_type = cfg.get("type", "embedding")
    modality = req.get("modality", "text")
    raw_input = req.get("input")

    # 1. DINO Embedding (Image -> Image only)
    if model_type == "dino_embedding" or model.lower().startswith("dino"):
        if modality != "image":
            raise HTTPException(status_code=400, detail="DINO only supports image-to-image embeddings (modality='image')")
        items = raw_input if isinstance(raw_input, list) else [raw_input]
        with DEVICE_BROKER.lease(ovms_model, preferred_device=preferred_device) as active_model:
            vectors = run_dino_embedding(active_model, items, model_cfg=cfg)
        return {
            "object": "list",
            "data": [
                {"object": "embedding", "index": idx, "embedding": vec}
                for idx, vec in enumerate(vectors)
            ],
            "model": model,
            "usage": {"prompt_tokens": len(items), "total_tokens": len(items)},
        }

    # 2. Multimodal Embedding (Text or Image)
    if model_type == "multimodal_embedding" or modality == "image":
        if modality == "image":
            items = raw_input if isinstance(raw_input, list) else [raw_input]
            vision_ovms = cfg.get("ovms_vision_model") or (f"{ovms_model}-vision" if "text" not in ovms_model else ovms_model.replace("text", "vision"))
            with DEVICE_BROKER.lease(vision_ovms, preferred_device=preferred_device) as active_model:
                vectors = run_multimodal_image_embedding(active_model, items, model_cfg=cfg)
            return {
                "object": "list",
                "data": [
                    {"object": "embedding", "index": idx, "embedding": vec}
                    for idx, vec in enumerate(vectors)
                ],
                "model": model,
                "usage": {"prompt_tokens": len(items), "total_tokens": len(items)},
            }
        else:
            texts = normalize_embedding_input(raw_input)
            text_ovms = cfg.get("ovms_text_model") or (f"{ovms_model}-text" if "vision" not in ovms_model else ovms_model.replace("vision", "text"))
            with DEVICE_BROKER.lease(text_ovms, preferred_device=preferred_device) as active_model:
                vectors, prompt_tokens = run_multimodal_text_embedding(
                    active_model, texts, tokenizer_path=cfg.get("tokenizer_path"), model_cfg=cfg
                )
            return {
                "object": "list",
                "data": [
                    {"object": "embedding", "index": idx, "embedding": vec}
                    for idx, vec in enumerate(vectors)
                ],
                "model": model,
                "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
            }

    # 3. Standard Text Embedding (pooled_ir, genai_v3, sentence_transformer, bge_m3)
    backend = cfg.get("embedding_backend")

    if backend == "genai_v3":
        # Graph-backed OVMS model served through the OpenAI-compatible
        # /v3/embeddings endpoint. Parity with the legacy gateway: the adaptive
        # batcher is only engaged when the registry entry opts in AND provides a
        # tokenizer path, because batching needs a local tokenizer for the
        # length-aware policy.
        texts = normalize_embedding_input(raw_input)
        if (
            ADAPTIVE_EMBEDDING_BATCHING
            and cfg.get("adaptive_batching") is True
            and len(texts) == 1
            and cfg.get("tokenizer_path")
        ):
            return adaptive_genai_batcher.submit(
                ovms_model,
                model,
                texts[0],
                cfg["tokenizer_path"],
            )
        with DEVICE_BROKER.lease(ovms_model, preferred_device=preferred_device) as active_model:
            return call_genai_embedding(active_model, texts, model)

    if backend == "pooled_ir":
        texts = normalize_embedding_input(raw_input)
        if (
            ADAPTIVE_EMBEDDING_BATCHING
            and cfg.get("adaptive_batching") is True
            and cfg.get("tokenizer_path")
        ):
            if len(texts) == 1:
                return adaptive_pooled_batcher.submit(
                    ovms_model,
                    model,
                    texts[0],
                    cfg["tokenizer_path"],
                )
            return adaptive_pooled_batcher.submit_many(
                ovms_model,
                model,
                texts,
                cfg["tokenizer_path"],
            )
        return pooled_ir_response(
            ovms_model,
            texts,
            model,
            cfg.get("tokenizer_path"),
            broker=DEVICE_BROKER,
        )

    if backend == "sentence_transformer":
        with DEVICE_BROKER.lease(ovms_model, preferred_device=preferred_device) as active_model:
            return call_sentence_transformer_embedding(
                active_model,
                raw_input,
                model,
                cfg,
                input_type=req.get("input_type"),
            )

    # Default: BGE-M3 / generic embedding
    with DEVICE_BROKER.lease(ovms_model, preferred_device=preferred_device) as active_model:
        return call_embedding(active_model, raw_input, model)


# ---------- Stats & Health APIs ----------

@app.get("/v1/embedding-batch-stats")
def embedding_batch_stats():
    # Parity with the legacy gateway: the reported batcher follows the backend
    # configured for the primary Qwen embedding entry, so operators see the
    # stats of the batching policy that is actually in use.
    cfg = MODEL_REGISTRY.get("qwen3-embedding-0.6b-int4") or {}
    if cfg.get("embedding_backend") == "pooled_ir":
        return adaptive_pooled_batcher.stats()
    return adaptive_genai_batcher.stats()


@app.get("/v1/broker/metrics")
def broker_metrics():
    return {
        "metrics": DEVICE_BROKER.metrics,
        "loaded_models": {
            m: {"loaded": is_loaded, "device": DEVICE_BROKER.model_device.get(m, "GPU")}
            for m, is_loaded in DEVICE_BROKER.model_loaded.items()
        },
    }


@app.get("/health")
def health():
    # Protocol diagnostics are part of the health payload so a protocol mismatch
    # can be identified from the outside without reading container logs.
    return {
        "status": "ok",
        "service": "ai-gateway",
        **ovms_protocol.protocol_diagnostics(),
    }
