from __future__ import annotations

import asyncio
import base64
import os
import time
import uuid
from typing import Any, Dict, Optional, Tuple
import httpx
import jmespath

from contracts import ParseRequest, ParsedDocument
from ..base import BaseParserDriver
from ..models import ProviderDefinition, RequestField
from ..normalization import normalize_response


class GenericHttpDriver(BaseParserDriver):
    """Universal configuration-driven HTTP parser driver.

    Supports synchronous and asynchronous (submit -> poll) APIs,
    multipart / JSON / urlencoded encodings, Bearer / API-key auth,
    and declarative JMESPath response normalization.
    """

    def __init__(self, definition: ProviderDefinition):
        super().__init__(definition)
        self.endpoint = definition.resolve_endpoint().rstrip("/")

    def _build_auth(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Constructs headers and query params for authentication."""
        headers = {}
        params = {}
        token = self.definition.auth.resolve_token()
        auth_type = self.definition.auth.type

        if auth_type == "bearer" and token:
            headers["Authorization"] = f"Bearer {token}"
        elif auth_type == "header" and token:
            header_name = self.definition.auth.header_name or "Authorization"
            headers[header_name] = token
        elif auth_type == "query" and token:
            param_name = self.definition.auth.query_param or "api_key"
            params[param_name] = token

        return headers, params

    def _resolve_field_value(self, field_def: Any, request: ParseRequest) -> Any:
        if isinstance(field_def, RequestField):
            if field_def.value is not None:
                val = field_def.value
            elif field_def.value_from:
                val = getattr(request, field_def.value_from, None)
            else:
                val = None

            if field_def.transform == "base64" and isinstance(val, (bytes, str)):
                if isinstance(val, str) and not val.startswith("data:"):
                    val = base64.b64encode(val.encode("utf-8")).decode("utf-8")
            elif field_def.transform == "str" and val is not None:
                val = str(val)
            elif field_def.transform == "int" and val is not None:
                val = int(val)
            elif field_def.transform == "bool" and val is not None:
                val = bool(val)
            return val

        if isinstance(field_def, dict) and "value_from" in field_def:
            val_from = field_def["value_from"]
            val = getattr(request, val_from, field_def.get("value"))
            transform = field_def.get("transform")
            if transform == "base64" and isinstance(val, (bytes, str)):
                if isinstance(val, str) and not val.startswith("data:"):
                    val = base64.b64encode(val.encode("utf-8")).decode("utf-8")
            return val

        return field_def

    async def _prepare_file_payload(
        self,
        request: ParseRequest,
    ) -> Tuple[Optional[bytes], str, str]:
        """Loads file content and returns (content_bytes, filename, mime_type)."""
        filename = "document.pdf"
        mime_type = request.mime_type or "application/pdf"
        content_bytes = b""

        if request.file_path and os.path.exists(request.file_path):
            filename = os.path.basename(request.file_path)
            with open(request.file_path, "rb") as f:
                content_bytes = f.read()
        elif request.file_content_base64:
            b64_str = request.file_content_base64
            if "," in b64_str:
                header, b64_str = b64_str.split(",", 1)
                if "image/" in header:
                    mime_type = header.split(";")[0].replace("data:", "")
                    filename = "image.png"
            content_bytes = base64.b64decode(b64_str)
        elif request.file_url:
            filename = os.path.basename(request.file_url.split("?")[0]) or "document.pdf"

        return content_bytes, filename, mime_type

    async def _build_request(
        self,
        request: ParseRequest,
    ) -> Tuple[str, Dict[str, str], Dict[str, str], Any, Any]:
        """Builds URL, headers, params, body data, and files."""
        req_tpl = self.definition.request
        url = f"{self.endpoint}/{req_tpl.path.lstrip('/')}" if req_tpl.path else self.endpoint

        headers, params = self._build_auth()
        headers.update(req_tpl.headers)

        body_data = {}
        for k, v in req_tpl.fields.items():
            resolved = self._resolve_field_value(v, request)
            if resolved is not None:
                body_data[k] = resolved

        file_bytes, filename, mime_type = await self._prepare_file_payload(request)
        file_mode = req_tpl.file.mode
        file_field = req_tpl.file.field

        files = None
        if file_mode == "multipart" and file_bytes:
            files = {file_field: (filename, file_bytes, mime_type)}
        elif file_mode == "base64" and file_bytes:
            body_data[file_field] = base64.b64encode(file_bytes).decode("utf-8")
        elif file_mode == "url" and request.file_url:
            body_data[file_field] = request.file_url

        # Fallbacks if fields were not explicitly declared in request.fields:
        if file_mode != "multipart":
            if "url" not in body_data and request.file_url:
                body_data["url"] = request.file_url
            if "file_base64" not in body_data and request.file_content_base64 and file_mode != "base64":
                body_data["file_base64"] = request.file_content_base64
            if "file_path" not in body_data and request.file_path and not request.file_content_base64:
                body_data["file_path"] = request.file_path

        return url, headers, params, body_data, files

    async def _poll_async_job(
        self,
        client: httpx.AsyncClient,
        job_id: str,
        headers: Dict[str, str],
        params: Dict[str, str],
    ) -> Dict[str, Any]:
        """Polls status endpoint until terminal state is reached."""
        cfg = self.definition.async_config
        if not cfg:
            raise RuntimeError("Async configuration missing")

        poll_url_pattern = cfg.poll_path or "/tasks/{job_id}"
        poll_url = f"{self.endpoint}/{poll_url_pattern.format(job_id=job_id).lstrip('/')}"

        deadline = time.monotonic() + cfg.max_poll_seconds
        while time.monotonic() < deadline:
            resp = await client.request(cfg.poll_method, poll_url, headers=headers, params=params)
            if resp.status_code == 429:
                self.record_failure(Exception("HTTP 429 Quota Exceeded"), quota_exhausted=True)
                raise RuntimeError("Provider quota exhausted during polling (HTTP 429)")
            if resp.status_code != 200:
                raise RuntimeError(f"Async poll failed: HTTP {resp.status_code} - {resp.text}")

            payload = resp.json()
            status_val = str(jmespath.search(cfg.status_path, payload) or "").lower()

            if status_val in [v.lower() for v in cfg.success_values]:
                # If separate result fetch path is specified
                if cfg.result_fetch_path:
                    fetch_url = f"{self.endpoint}/{cfg.result_fetch_path.format(job_id=job_id).lstrip('/')}"
                    fetch_resp = await client.get(fetch_url, headers=headers, params=params)
                    fetch_resp.raise_for_status()
                    payload = fetch_resp.json()

                if cfg.result_path:
                    extracted = jmespath.search(cfg.result_path, payload)
                    return extracted if isinstance(extracted, dict) else payload
                return payload

            if status_val in [v.lower() for v in cfg.failure_values]:
                raise RuntimeError(f"Async job {job_id} failed with status: {status_val}")

            await asyncio.sleep(cfg.poll_interval_seconds)

        raise TimeoutError(f"Async job {job_id} timed out after {cfg.max_poll_seconds}s")

    async def parse(self, request: ParseRequest) -> ParsedDocument:
        t0 = time.monotonic()
        timeout = min(self.definition.timeout, 600.0)

        url, headers, params, body_data, files = await self._build_request(request)
        method = self.definition.request.method
        encoding = self.definition.request.encoding

        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                if encoding == "multipart" or files:
                    resp = await client.request(
                        method, url, headers=headers, params=params, data=body_data, files=files
                    )
                elif encoding == "urlencoded":
                    resp = await client.request(
                        method, url, headers=headers, params=params, data=body_data
                    )
                else:  # json
                    resp = await client.request(
                        method, url, headers=headers, params=params, json=body_data
                    )
            except Exception as exc:
                self.record_failure(exc)
                raise

            if resp.status_code == 429:
                self.record_failure(Exception("HTTP 429 Quota Exceeded"), quota_exhausted=True)
                raise RuntimeError(f"Provider {self.name} quota exhausted (HTTP 429)")

            if resp.status_code not in (200, 201, 202):
                exc = RuntimeError(f"Provider HTTP {resp.status_code}: {resp.text}")
                self.record_failure(exc)
                raise exc

            initial_payload = resp.json()

            # Handle Async Polling if enabled
            if self.definition.async_config and self.definition.async_config.enabled:
                job_id = jmespath.search(self.definition.async_config.job_id_path, initial_payload)
                if not job_id:
                    raise RuntimeError(f"Could not extract job_id via path: {self.definition.async_config.job_id_path}")
                final_payload = await self._poll_async_job(client, str(job_id), headers, params)
            else:
                final_payload = initial_payload

        latency_ms = (time.monotonic() - t0) * 1000.0
        self.record_success(latency_ms)

        # Declarative JMESPath normalization
        doc = normalize_response(final_payload, self.definition.response)
        return doc
