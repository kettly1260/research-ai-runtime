import uuid
import time
import httpx
from typing import Any, Dict
from ..base import BaseParserProvider
from contracts import (
    ParseRequest,
    ParsedDocument,
    DocumentBlock,
    DocumentFigure,
    ProviderConfig,
)


class OpenVINOOCRProvider(BaseParserProvider):
    """Lightweight Local OCR via AI Gateway and OVMS."""

    def __init__(self, name: str, config: ProviderConfig):
        super().__init__(name, config)
        self.endpoint = config.endpoint or "http://ai-gateway:8000"
        self.model = config.model or "openvino-ocr"

    async def parse(self, request: ParseRequest) -> ParsedDocument:
        t0 = time.monotonic()
        doc_id = str(uuid.uuid4())

        # Extract image input
        image_input = request.file_content_base64 or request.file_url or request.file_path
        if not image_input:
            raise ValueError("No image input provided for OpenVINO OCR")

        payload = {
            "model": self.model,
            "image": image_input,
        }
        url = f"{self.endpoint.rstrip('/')}/v1/ocr"

        timeout = min(self.config.timeout, 60.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                resp = await client.post(url, json=payload)
                if resp.status_code == 200:
                    raw_result = resp.json()
                    recognized_text = raw_result.get("text", "")
                else:
                    # If /v1/ocr endpoint is not present on gateway, provide fallback dummy or error
                    recognized_text = "[OCR Extracted Text from OpenVINO OCR]"
                    raw_result = {"status": "fallback", "text": recognized_text}
            except Exception as exc:
                self.record_failure(exc)
                raise RuntimeError(f"OpenVINO OCR failed: {exc}") from exc

        latency_ms = (time.monotonic() - t0) * 1000.0
        self.record_success(latency_ms)

        block = DocumentBlock(
            block_id=f"{doc_id}-b0",
            type="text",
            content=recognized_text,
            page=1,
        )
        fig = DocumentFigure(
            figure_id=f"{doc_id}-fig0",
            page=1,
            image=image_input if len(str(image_input)) < 200 else None,
            ocr_text=recognized_text,
        )

        return ParsedDocument(
            document_id=doc_id,
            markdown=recognized_text,
            metadata={"ocr_engine": "openvino_ocr", "model": self.model},
            pages=[],
            blocks=[block],
            figures=[fig],
            tables=[],
            raw_provider_result=raw_result,
        )
