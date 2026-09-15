from typing import Any, Dict, List, Tuple
import numpy as np
import requests
from fastapi import HTTPException
from ..config import (
    RERANK_MAX_LENGTH,
    RERANK_MAX_DOCS,
    RERANK_MAX_CHARS,
    RERANK_BATCH_SIZE,
    RERANK_FALLBACK_MAX_LENGTH,
    RERANK_FALLBACK_TIMEOUT,
    RERANK_STRICT_RECALL,
    OVMS_TIMEOUT,
)
from ..tokenizers import get_rerank_tokenizer
from .. import ovms_client


def extract_document_text(document: Any) -> str:
    if document is None:
        return ""
    if isinstance(document, str):
        return document.strip()
    if isinstance(document, (list, tuple)):
        parts = [extract_document_text(item) for item in document]
        return " ".join(p for p in parts if p)
    if isinstance(document, dict):
        preferred_keys = [
            "title", "name", "abstract", "summary", "snippet", "description",
            "text", "content", "route", "reaction", "conditions", "reagents",
            "reactants", "products", "source", "institution", "degree", "year"
        ]
        parts = []
        for k in preferred_keys:
            val = document.get(k)
            if isinstance(val, str) and val.strip():
                parts.append(val.strip())
            elif isinstance(val, (int, float)):
                parts.append(str(val))
        for k in ("steps", "step_list", "items", "routes", "documents"):
            val = document.get(k)
            if isinstance(val, list):
                for item in val:
                    t = extract_document_text(item)
                    if t:
                        parts.append(t)
        if not parts:
            for val in document.values():
                if isinstance(val, str) and val.strip():
                    parts.append(val.strip())
        return " ".join(parts).strip()
    return str(document).strip()


def build_rerank_payload_documents(docs: List[Any]) -> Tuple[List[Tuple[Any, str]], Dict[str, int]]:
    total_documents = len(docs)
    discarded_documents = 0
    docs_to_use = docs
    if total_documents > RERANK_MAX_DOCS:
        if RERANK_STRICT_RECALL:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"documents count ({total_documents}) exceeds RERANK_MAX_DOCS ({RERANK_MAX_DOCS}) "
                    "under strict recall mode; increase RERANK_MAX_DOCS or reduce candidate set"
                ),
            )
        docs_to_use = docs[:RERANK_MAX_DOCS]
        discarded_documents = total_documents - RERANK_MAX_DOCS

    text_truncated_documents = 0
    normalized = []
    for doc in docs_to_use:
        text = extract_document_text(doc)
        if not text:
            text = str(doc)
        if len(text) > RERANK_MAX_CHARS:
            text_truncated_documents += 1
        normalized.append((doc, text[:RERANK_MAX_CHARS]))

    return normalized, {
        "total_documents": total_documents,
        "scored_documents": len(normalized),
        "discarded_documents": discarded_documents,
        "text_truncated_documents": text_truncated_documents,
    }


def run_ovms_rerank_batch(
    model_name: str,
    query: str,
    batch_documents: List[Tuple[Any, str]],
    max_length: int = RERANK_MAX_LENGTH,
    timeout: int = OVMS_TIMEOUT,
) -> List[float]:
    pairs = [[query, text] for _, text in batch_documents]
    rerank_tokenizer = get_rerank_tokenizer()
    inputs = rerank_tokenizer(
        pairs,
        padding=True,
        truncation=True,
        max_length=max(8, max_length),
        return_tensors="np",
    )

    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask = inputs["attention_mask"].astype(np.int64)
    seq_len = input_ids.shape[1]
    batch = input_ids.shape[0]

    position_ids = np.tile(np.arange(seq_len), (batch, 1)).astype(np.int64)

    payload = {"instances": []}
    for i in range(batch):
        payload["instances"].append({
            "input_ids": input_ids[i].tolist(),
            "attention_mask": attention_mask[i].tolist(),
            "position_ids": position_ids[i].tolist(),
        })

    resp = ovms_client.ovms_predict(model_name, payload, timeout=timeout)
    logits = np.array(resp["predictions"])
    scores = logits[:, -1, :].max(axis=1).tolist()
    return scores


def call_rerank(model_name: str, query: str, docs: List[Any]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    if not docs:
        return [], {
            "total_documents": 0,
            "scored_documents": 0,
            "discarded_documents": 0,
            "text_truncated_documents": 0,
        }

    normalized_docs, rerank_stats = build_rerank_payload_documents(docs)
    scores = []
    try:
        for start in range(0, len(normalized_docs), RERANK_BATCH_SIZE):
            batch_docs = normalized_docs[start:start + RERANK_BATCH_SIZE]
            scores.extend(run_ovms_rerank_batch(model_name, query, batch_docs))
    except HTTPException as exc:
        if exc.status_code != 504:
            raise
        scores = []
        for item in normalized_docs:
            scores.extend(
                run_ovms_rerank_batch(
                    model_name,
                    query,
                    [item],
                    max_length=min(RERANK_MAX_LENGTH, RERANK_FALLBACK_MAX_LENGTH),
                    timeout=max(OVMS_TIMEOUT, RERANK_FALLBACK_TIMEOUT),
                )
            )

    ranked = sorted(zip(normalized_docs, scores), key=lambda x: x[1], reverse=True)
    return [
        {
            "document": original,
            "text": text,
            "score": score,
        }
        for (original, text), score in ranked
    ], rerank_stats
