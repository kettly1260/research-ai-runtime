"""Frozen contracts for Research AI Runtime.

Per specification Section 35:
T0 — Contracts (FROZEN)
- ModelRequest
- ModelResponse
- ParseRequest
- ParsedDocument
- ProviderConfig
- ProviderStatus
- MediaRecord
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field


# ==============================================================================
# 1. Model Requests & Responses (/v1/embeddings, /v1/rerank, multimodal, DINO)
# ==============================================================================

class EmbeddingItem(BaseModel):
    object: str = "embedding"
    index: int
    embedding: List[float]


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str], List[Any]]
    modality: Optional[Literal["text", "image"]] = "text"
    encoding_format: Optional[str] = "float"
    user: Optional[str] = None

    model_config = ConfigDict(extra="ignore")


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingItem]
    model: str
    usage: UsageInfo = Field(default_factory=UsageInfo)


class RerankRequest(BaseModel):
    model: str = "qwen-reranker"
    query: str
    documents: List[str]
    top_n: Optional[int] = None
    return_documents: bool = True

    model_config = ConfigDict(extra="ignore")


class RerankResultItem(BaseModel):
    index: Optional[int] = None
    document: Optional[Union[str, Dict[str, Any]]] = None
    text: Optional[str] = None
    relevance_score: Optional[float] = None
    score: Optional[float] = None


class RerankMeta(BaseModel):
    strict_recall: bool = True
    total_documents: int
    scored_documents: int
    discarded_documents: int = 0
    text_truncated_documents: int = 0
    applied_top_n: Optional[int] = None


class RerankResponse(BaseModel):
    results: List[RerankResultItem]
    meta: RerankMeta


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "openvino"
    permission: List[Any] = Field(default_factory=list)


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelCard]


# Generic umbrella request/response for model inference
class ModelRequest(BaseModel):
    type: Literal["embedding", "rerank", "multimodal_embedding", "dino_embedding"]
    payload: Union[EmbeddingRequest, RerankRequest, Dict[str, Any]]


class ModelResponse(BaseModel):
    type: Literal["embedding", "rerank", "multimodal_embedding", "dino_embedding"]
    payload: Union[EmbeddingResponse, RerankResponse, Dict[str, Any]]


# ==============================================================================
# 2. Document Parser Contracts (POST /v1/document/parse)
# ==============================================================================

class ParseRequest(BaseModel):
    file_path: Optional[str] = None
    file_url: Optional[str] = None
    file_content_base64: Optional[str] = None
    mime_type: str = "application/pdf"
    privacy: Literal["public", "private"] = "private"
    needs: List[str] = Field(
        default_factory=lambda: ["pdf", "layout", "figures", "formula", "ocr"]
    )
    priority: int = 100
    preferred_provider: Optional[str] = None
    extract_images: bool = True

    model_config = ConfigDict(extra="ignore")


class DocumentBlock(BaseModel):
    block_id: Optional[str] = None
    type: Literal["text", "heading", "formula", "table", "figure", "list", "code", "header", "footer"]
    content: str
    page: Optional[int] = None
    bbox: Optional[List[float]] = None  # [x0, y0, x1, y1]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentFigure(BaseModel):
    figure_id: str
    page: int
    bbox: Optional[List[float]] = None  # [x0, y0, x1, y1]
    image: Optional[str] = None  # base64 encoded image or path/URL
    caption: Optional[str] = None
    source_ref: Optional[str] = None
    ocr_text: Optional[str] = None
    figure_type: Optional[str] = None  # e.g., "chart", "diagram", "photo", "spectrum"


class DocumentTable(BaseModel):
    table_id: str
    page: int
    bbox: Optional[List[float]] = None
    caption: Optional[str] = None
    html: Optional[str] = None
    markdown: Optional[str] = None
    raw_data: Optional[Any] = None


class DocumentPage(BaseModel):
    page_number: int
    width: Optional[float] = None
    height: Optional[float] = None
    text: Optional[str] = None


class ProviderExecutionInfo(BaseModel):
    provider_name: str
    latency_ms: float
    status: Literal["success", "fallback", "error"]
    fallback_reason: Optional[str] = None
    attempted_providers: List[str] = Field(default_factory=list)


class ParsedDocument(BaseModel):
    document_id: str
    markdown: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    pages: List[DocumentPage] = Field(default_factory=list)
    blocks: List[DocumentBlock] = Field(default_factory=list)
    figures: List[DocumentFigure] = Field(default_factory=list)
    tables: List[DocumentTable] = Field(default_factory=list)
    raw_provider_result: Optional[Dict[str, Any]] = None
    provider_info: Optional[ProviderExecutionInfo] = None


# ==============================================================================
# 3. Provider Configuration & Status Contracts
# ==============================================================================

class ProviderLifecycle(BaseModel):
    mode: Literal["external", "persistent", "on_demand", "model_on_demand"] = "external"
    startup_command: Optional[str] = None
    stop_command: Optional[str] = None
    container_name: Optional[str] = None


class ProviderConfig(BaseModel):
    name: Optional[str] = None
    driver: str  # mineru, paddleocr, generic_ocr, etc.
    transport: str = "http"  # http, local_cli
    location: Literal["remote", "local"]
    endpoint_env: Optional[str] = None
    api_key_env: Optional[str] = None
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    enabled: bool = True
    capabilities: Dict[str, bool] = Field(default_factory=dict)
    priority: int = 100
    timeout: float = 300.0
    lifecycle: ProviderLifecycle = Field(default_factory=ProviderLifecycle)
    model: Optional[str] = None

    model_config = ConfigDict(extra="ignore")


class ProviderStatus(BaseModel):
    name: str
    available: bool = True
    last_success: Optional[float] = None
    last_failure: Optional[float] = None
    latency: Optional[float] = None
    failure_count: int = 0
    quota_state: Literal["normal", "exhausted", "unknown"] = "normal"
    cooldown_until: Optional[float] = None


# ==============================================================================
# 4. Media Record Contract (LanceDB Media Layer)
# ==============================================================================

class MediaRecord(BaseModel):
    media_id: str
    source_type: str  # "figure", "table", "chart", "page_screenshot", "image"
    source_uri: str
    source_native_id: Optional[str] = None
    parent_uri: Optional[str] = None

    mime_type: str = "image/png"
    page: Optional[int] = None
    figure_number: Optional[str] = None

    caption: Optional[str] = None
    ocr_text: Optional[str] = None
    vlm_summary: Optional[str] = None

    hash: str
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    multimodal_vector: Optional[List[float]] = None
    dino_vector: Optional[List[float]] = None

    parser_provider: Optional[str] = None
    parser_version: Optional[str] = None
    embedding_model: Optional[str] = None
    embedding_version: Optional[str] = None

    model_config = ConfigDict(extra="ignore")
