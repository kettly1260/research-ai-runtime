import uuid
import time
import httpx
from typing import Any, Dict, List, Optional
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


class MinerUProvider(BaseParserProvider):
    """Driver for MinerU Cloud & Local Parser API."""

    def __init__(self, name: str, config: ProviderConfig):
        super().__init__(name, config)
        self.endpoint = config.endpoint or "https://mineru.net/api/v4"
        self.api_key = config.api_key or ""

    async def parse(self, request: ParseRequest) -> ParsedDocument:
        t0 = time.monotonic()
        doc_id = str(uuid.uuid4())

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Prepare payload
        payload = {
            "is_ocr": True,
            "enable_formula": True,
            "enable_table": True,
        }
        if request.file_url:
            payload["url"] = request.file_url
        elif request.file_content_base64:
            payload["file_base64"] = request.file_content_base64
        elif request.file_path:
            payload["file_path"] = request.file_path

        timeout = min(self.config.timeout, 300.0)

        async with httpx.AsyncClient(timeout=timeout) as client:
            # MinerU standard extraction endpoint
            url = f"{self.endpoint.rstrip('/')}/extract"
            resp = await client.post(url, json=payload, headers=headers)

            if resp.status_code == 429:
                self.record_failure(Exception("Quota exhausted"), quota_exhausted=True)
                raise RuntimeError("MinerU quota exhausted (HTTP 429)")

            if resp.status_code != 200:
                self.record_failure(Exception(f"HTTP {resp.status_code}: {resp.text}"))
                raise RuntimeError(f"MinerU extraction failed: {resp.status_code} - {resp.text}")

            raw_result = resp.json()

        latency_ms = (time.monotonic() - t0) * 1000.0
        self.record_success(latency_ms)

        # Normalize MinerU result into ParsedDocument
        return self._normalize_mineru_result(doc_id, raw_result)

    def _normalize_mineru_result(self, doc_id: str, raw: Dict[str, Any]) -> ParsedDocument:
        data = raw.get("data", raw)
        markdown_text = data.get("markdown", "") or data.get("text", "")

        blocks: List[DocumentBlock] = []
        figures: List[DocumentFigure] = []
        tables: List[DocumentTable] = []
        pages: List[DocumentPage] = []

        # Parse layout blocks if present
        for idx, block in enumerate(data.get("layout_blocks", [])):
            b_type = block.get("type", "text")
            b_content = block.get("content", "")
            page_num = block.get("page", 1)
            bbox = block.get("bbox")
            blocks.append(DocumentBlock(
                block_id=f"{doc_id}-b{idx}",
                type=b_type if b_type in ("text", "heading", "formula", "table", "figure") else "text",
                content=b_content,
                page=page_num,
                bbox=bbox,
                metadata=block.get("metadata", {}),
            ))

        # Parse figures
        for idx, fig in enumerate(data.get("figures", [])):
            figures.append(DocumentFigure(
                figure_id=fig.get("id", f"{doc_id}-fig{idx}"),
                page=fig.get("page", 1),
                bbox=fig.get("bbox"),
                image=fig.get("image") or fig.get("image_base64"),
                caption=fig.get("caption"),
                source_ref=fig.get("source_ref"),
                ocr_text=fig.get("ocr_text"),
                figure_type=fig.get("type", "chart"),
            ))

        # Parse tables
        for idx, tbl in enumerate(data.get("tables", [])):
            tables.append(DocumentTable(
                table_id=tbl.get("id", f"{doc_id}-tbl{idx}"),
                page=tbl.get("page", 1),
                bbox=tbl.get("bbox"),
                caption=tbl.get("caption"),
                html=tbl.get("html"),
                markdown=tbl.get("markdown"),
            ))

        # Pages
        for idx, page in enumerate(data.get("pages", [])):
            pages.append(DocumentPage(
                page_number=page.get("page_number", idx + 1),
                width=page.get("width"),
                height=page.get("height"),
                text=page.get("text"),
            ))

        return ParsedDocument(
            document_id=doc_id,
            markdown=markdown_text,
            metadata=raw.get("metadata", {}),
            pages=pages,
            blocks=blocks,
            figures=figures,
            tables=tables,
            raw_provider_result=raw,
        )
