import uuid
import time
import httpx
from typing import Any, Dict, List
from ..base import BaseParserProvider
from contracts import (
    ParseRequest,
    ParsedDocument,
    DocumentBlock,
    DocumentFigure,
    DocumentTable,
    DocumentPage,
    ProviderConfig,
)


class PaddleOCRProvider(BaseParserProvider):
    """Driver for PaddleOCR Cloud & Local HTTP API."""

    def __init__(self, name: str, config: ProviderConfig):
        super().__init__(name, config)
        self.endpoint = config.endpoint or ""
        self.api_key = config.api_key or ""

    async def parse(self, request: ParseRequest) -> ParsedDocument:
        t0 = time.monotonic()
        doc_id = str(uuid.uuid4())

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "use_gpu": False,
            "table": True,
            "layout": True,
        }
        if request.file_url:
            payload["url"] = request.file_url
        elif request.file_content_base64:
            payload["file_base64"] = request.file_content_base64
        elif request.file_path:
            payload["file_path"] = request.file_path

        timeout = min(self.config.timeout, 300.0)

        async with httpx.AsyncClient(timeout=timeout) as client:
            url = f"{self.endpoint.rstrip('/')}/layout_parse"
            resp = await client.post(url, json=payload, headers=headers)

            if resp.status_code == 429:
                self.record_failure(Exception("PaddleOCR quota exhausted"), quota_exhausted=True)
                raise RuntimeError("PaddleOCR quota exhausted (HTTP 429)")

            if resp.status_code != 200:
                self.record_failure(Exception(f"HTTP {resp.status_code}: {resp.text}"))
                raise RuntimeError(f"PaddleOCR request failed: {resp.status_code} - {resp.text}")

            raw_result = resp.json()

        latency_ms = (time.monotonic() - t0) * 1000.0
        self.record_success(latency_ms)

        return self._normalize_paddle_result(doc_id, raw_result)

    def _normalize_paddle_result(self, doc_id: str, raw: Dict[str, Any]) -> ParsedDocument:
        results = raw.get("result", raw)
        lines = []
        blocks: List[DocumentBlock] = []
        figures: List[DocumentFigure] = []
        tables: List[DocumentTable] = []
        pages: List[DocumentPage] = []

        for idx, item in enumerate(results if isinstance(results, list) else []):
            label = item.get("type", "text").lower()
            content = item.get("res", "")
            if isinstance(content, list):
                # Text detection box format
                text_parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in content]
                content = "\n".join(text_parts)
            lines.append(content)

            b_type = "text"
            if "title" in label or "header" in label:
                b_type = "heading"
            elif "table" in label:
                b_type = "table"
                tables.append(DocumentTable(
                    table_id=f"{doc_id}-tbl{len(tables)}",
                    page=item.get("page", 1),
                    bbox=item.get("bbox"),
                    html=item.get("html"),
                    markdown=content,
                ))
            elif "figure" in label or "image" in label:
                b_type = "figure"
                figures.append(DocumentFigure(
                    figure_id=f"{doc_id}-fig{len(figures)}",
                    page=item.get("page", 1),
                    bbox=item.get("bbox"),
                    image=item.get("img"),
                    caption=item.get("caption"),
                ))

            blocks.append(DocumentBlock(
                block_id=f"{doc_id}-b{idx}",
                type=b_type,
                content=content,
                page=item.get("page", 1),
                bbox=item.get("bbox"),
            ))

        markdown_text = "\n\n".join(lines)
        return ParsedDocument(
            document_id=doc_id,
            markdown=markdown_text,
            metadata=raw.get("meta", {}),
            pages=pages,
            blocks=blocks,
            figures=figures,
            tables=tables,
            raw_provider_result=raw,
        )
