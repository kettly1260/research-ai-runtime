from typing import Any, Dict, List, Optional, Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from contracts import ParseRequest, ParsedDocument
from .parser import PARSER_MANAGER
from .media import MEDIA_INGESTOR, MEDIA_SEARCH, MEDIA_STORE

app = FastAPI(
    title="Research Media Service",
    version="0.2.0",
    description="Pluggable Document Parser, Scientific Media Ingestion, and LanceDB Search",
)


# ---------- Document Parsing API ----------

@app.post("/v1/document/parse", response_model=ParsedDocument)
async def parse_document(request: ParseRequest):
    return await PARSER_MANAGER.parse(request)


# ---------- Media Ingestion API ----------

class IngestRequest(BaseModel):
    document: ParsedDocument
    source_uri: Optional[str] = None
    embedding_model: str = "jina-clip-v2"
    dino_model: str = "dinov2-small"


@app.post("/v1/media/ingest")
async def ingest_media(req: IngestRequest):
    return await MEDIA_INGESTOR.ingest_document(
        doc=req.document,
        source_uri=req.source_uri or req.document.document_id,
        embedding_model=req.embedding_model,
        dino_model=req.dino_model,
    )


# ---------- Media Search API ----------

class MediaSearchRequest(BaseModel):
    mode: Literal["text_to_image", "image_to_image"] = "text_to_image"
    query_text: Optional[str] = None
    query_image: Optional[str] = None
    top_k: int = 10
    filter_expr: Optional[str] = None
    model: Optional[str] = None


@app.post("/v1/media/search")
async def search_media(req: MediaSearchRequest):
    if req.mode == "text_to_image":
        if not req.query_text:
            raise HTTPException(status_code=400, detail="query_text is required for text_to_image search")
        model = req.model or "jina-clip-v2"
        results = await MEDIA_SEARCH.search_by_text(
            query=req.query_text,
            top_k=req.top_k,
            filter_expr=req.filter_expr,
            model=model,
        )
        return {"results": results, "mode": "text_to_image", "count": len(results)}

    elif req.mode == "image_to_image":
        if not req.query_image:
            raise HTTPException(status_code=400, detail="query_image is required for image_to_image search")
        model = req.model or "dinov2-small"
        results = await MEDIA_SEARCH.search_by_image(
            image_input=req.query_image,
            top_k=req.top_k,
            filter_expr=req.filter_expr,
            model=model,
        )
        return {"results": results, "mode": "image_to_image", "count": len(results)}

    raise HTTPException(status_code=400, detail=f"Unsupported search mode: {req.mode}")


# ---------- Provider Status & Management ----------

@app.get("/v1/providers")
def list_providers():
    drivers = PARSER_MANAGER.registry.list_drivers()
    return {
        "config_path": PARSER_MANAGER.registry.resolved_config_path,
        "config_reload_error": PARSER_MANAGER.registry.last_reload_error,
        "providers": [
            {
                "name": p.name,
                "driver": p.definition.driver,
                "location": p.definition.location,
                "enabled": p.definition.enabled,
                "priority": p.definition.priority,
                "model": p.definition.model,
                "api_mode": p.definition.api_mode,
                "capabilities": p.definition.capabilities,
                "status": p.status.model_dump(),
            }
            for p in drivers
        ]
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "research-media",
        "media_records_count": MEDIA_STORE.count(),
    }
