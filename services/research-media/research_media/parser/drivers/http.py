from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
import httpx
import jmespath

from contracts import ParseRequest, ParsedDocument
from ..base import BaseParserDriver
from ..models import ProviderDefinition, RequestField, WorkflowStep
from ..normalization import normalize_response


_OUTPUT_PRIVATE_KEYS = {
    "token",
    "api_key",
    "access_token",
    "refresh_token",
    "authorization",
    "password",
    "secret",
    "file_content_base64",
    "upload_url",
    "agent_upload_url",
    "full_zip_url",
    "markdown_url",
    "jsonl_url",
    "resulturl",
}


def _sanitize_output_payload(value: Any) -> Any:
    """Remove credentials and temporary signed-resource URLs from API output."""
    if isinstance(value, dict):
        cleaned: Dict[str, Any] = {}
        for key, item in value.items():
            if str(key).lower() in _OUTPUT_PRIVATE_KEYS:
                continue
            cleaned[key] = _sanitize_output_payload(item)
        return cleaned
    if isinstance(value, list):
        return [_sanitize_output_payload(item) for item in value]
    return value


def _interpolate_value(val: Any, context: Dict[str, Any]) -> Any:
    """Recursively interpolates ${var} placeholders from context into values."""
    if isinstance(val, str):
        if val.startswith("${") and val.endswith("}"):
            var_name = val[2:-1].strip()
            if var_name in context:
                return context[var_name]
        res = val
        for k, v in context.items():
            if isinstance(v, (str, int, float, bool)):
                res = res.replace(f"${{{k}}}", str(v))
        return res
    elif isinstance(val, dict):
        return {k: _interpolate_value(v, context) for k, v in val.items()}
    elif isinstance(val, list):
        return [_interpolate_value(item, context) for item in val]
    return val


def _eval_condition(condition: str, context: Dict[str, Any]) -> bool:
    """Evaluates declarative conditions like 'file_path != null'.

    Fails fast with ValueError on invalid syntax or undefined variables rather than silently failing open.
    """
    if not condition or not condition.strip():
        return True
    env = {k: v for k, v in context.items() if isinstance(k, str) and k.isidentifier()}
    env["null"] = None
    env["true"] = True
    env["false"] = False
    try:
        py_cond = (
            condition.replace("&&", " and ")
            .replace("||", " or ")
            .replace("!=", " is not ")
            .replace("==", " == ")
        )
        # Workflow variables are created by earlier steps. Missing variables
        # are intentionally treated as null so a skipped branch does not turn
        # a later condition into a NameError.
        reserved = {"and", "or", "not", "is", "null", "true", "false"}
        for name in re.findall(r"\b[A-Za-z_]\w*\b", py_cond):
            if name not in reserved and name not in env:
                env[name] = None
        return bool(eval(py_cond, {"__builtins__": {}}, env))
    except Exception as exc:
        raise ValueError(f"Invalid workflow condition expression '{condition}': {exc}") from exc


class GenericHttpDriver(BaseParserDriver):
    """Universal configuration-driven HTTP parser driver.

    Supports:
    1. Multi-step declarative workflows (variable extraction, presigned upload, polling, download).
    2. Legacy synchronous and asynchronous (submit -> poll) APIs.
    3. Multipart / JSON / urlencoded / raw binary encodings.
    4. Bearer / API-key / Custom header authentication.
    5. Declarative JMESPath response normalization.
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
        def _lookup(value_from: str) -> Any:
            if value_from == "provider.model":
                return self.definition.model
            if value_from == "provider.api_mode":
                return self.definition.api_mode
            if value_from == "provider.options":
                return self.definition.options
            if value_from.startswith("provider.options."):
                return self.definition.options.get(value_from.split(".", 2)[2])
            return getattr(request, value_from, None)

        if isinstance(field_def, RequestField):
            if field_def.value is not None:
                val = field_def.value
            elif field_def.value_from:
                val = _lookup(field_def.value_from)
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
            val = _lookup(val_from)
            if val is None:
                val = field_def.get("value")
            transform = field_def.get("transform")
            if transform == "base64" and isinstance(val, (bytes, str)):
                if isinstance(val, str) and not val.startswith("data:"):
                    val = base64.b64encode(val.encode("utf-8")).decode("utf-8")
            return val

        return field_def

    @staticmethod
    def _decode_workflow_response(resp: httpx.Response, response_format: str) -> Dict[str, Any]:
        if response_format == "json":
            try:
                payload = resp.json()
                return payload if isinstance(payload, dict) else {"result": payload}
            except Exception:
                # Presigned PUT endpoints commonly return an empty or plain
                # text 200/204 body. Preserve tolerant legacy behavior.
                return {"text": resp.text}
        if response_format == "jsonl":
            items: List[Any] = []
            for raw_line in resp.text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
            return {"jsonl": items}
        if response_format == "binary":
            return {"content_base64": base64.b64encode(resp.content).decode("ascii")}
        return {"text": resp.text}

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
        """Builds URL, headers, params, body data, and files for legacy single-step request."""
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

    async def _execute_workflow(
        self,
        client: httpx.AsyncClient,
        request: ParseRequest,
    ) -> Dict[str, Any]:
        """Executes a multi-step declarative HTTP workflow with JMESPath variable extraction."""
        file_bytes, filename, mime_type = await self._prepare_file_payload(request)
        auth_headers, auth_params = self._build_auth()
        token = self.definition.auth.resolve_token() or ""

        context: Dict[str, Any] = {
            "endpoint": self.endpoint,
            "api_key": token,
            "token": token,
            "file_path": request.file_path,
            "file_url": request.file_url,
            "file_content_base64": request.file_content_base64,
            "filename": filename,
            "mime_type": mime_type,
            "document_id": str(uuid.uuid4()),
            "model": self.definition.model,
            "api_mode": self.definition.api_mode,
            "options": dict(self.definition.options),
            "options_json": json.dumps(self.definition.options, ensure_ascii=False, separators=(",", ":")),
        }
        for option_name, option_value in self.definition.options.items():
            context[f"options.{option_name}"] = option_value

        last_response_data: Any = {}

        for step in (self.definition.workflow or []):
            # 1. Condition check
            if step.condition and not _eval_condition(step.condition, context):
                continue

            # 2. Resolve URL
            raw_url = step.url or step.path or ""
            step_url = str(_interpolate_value(raw_url, context))
            if not step_url.startswith("http://") and not step_url.startswith("https://"):
                step_url = f"{self.endpoint}/{step_url.lstrip('/')}"

            # 3. Resolve Headers and Query Parameters
            step_headers = dict(auth_headers) if step.use_auth else {}
            if step.headers:
                resolved_headers = _interpolate_value(step.headers, context)
                if isinstance(resolved_headers, dict):
                    step_headers.update({str(k): str(v) for k, v in resolved_headers.items()})

            step_params = dict(auth_params) if step.use_auth else {}
            if step.params:
                resolved_params = _interpolate_value(step.params, context)
                if isinstance(resolved_params, dict):
                    step_params.update({str(k): str(v) for k, v in resolved_params.items()})

            # 4. Execute Step
            if step.type == "poll":
                deadline = time.monotonic() + step.max_poll_seconds
                status_path = step.status_path or "status"
                success_vals = [s.lower() for s in step.success_values]
                failure_vals = [f.lower() for f in step.failure_values]

                poll_resp_data = None
                while time.monotonic() < deadline:
                    r = await client.request(
                        step.method, step_url, headers=step_headers, params=step_params
                    )
                    if r.status_code == 429:
                        self.record_failure(Exception("HTTP 429 Quota Exceeded"), quota_exhausted=True)
                        raise RuntimeError(f"Provider {self.name} quota exhausted (HTTP 429)")
                    if r.status_code not in (200, 201, 202):
                        raise RuntimeError(f"Workflow poll failed: HTTP {r.status_code} - {r.text}")

                    payload = r.json()
                    poll_resp_data = payload
                    cur_status = str(jmespath.search(status_path, payload) or "").lower()

                    if cur_status in success_vals:
                        break
                    if cur_status in failure_vals:
                        raise RuntimeError(f"Workflow poll step '{step.name}' failed with status: {cur_status}")

                    await asyncio.sleep(step.poll_interval_seconds)
                else:
                    raise TimeoutError(f"Workflow poll step '{step.name}' timed out after {step.max_poll_seconds}s")

                last_response_data = poll_resp_data

            else:
                encoding = step.encoding
                retry_attempts = max(1, int(step.retry_attempts))
                retry_statuses = set(step.retry_statuses)
                resp = None

                for attempt in range(retry_attempts):
                    if encoding in ("binary_file", "raw_file"):
                        resp = await client.request(
                            step.method, step_url, headers=step_headers, params=step_params, content=file_bytes
                        )
                    elif encoding == "multipart":
                        file_field = step.file_field or "file"
                        files = {file_field: (filename, file_bytes, mime_type)}
                        data = _interpolate_value(step.body, context) if step.body else None
                        resp = await client.request(
                            step.method, step_url, headers=step_headers, params=step_params, data=data, files=files
                        )
                    elif encoding == "urlencoded":
                        data = _interpolate_value(step.body, context)
                        resp = await client.request(
                            step.method, step_url, headers=step_headers, params=step_params, data=data
                        )
                    else:  # json
                        json_data = _interpolate_value(step.body, context) if step.body is not None else None
                        resp = await client.request(
                            step.method, step_url, headers=step_headers, params=step_params, json=json_data
                        )

                    if (
                        resp.status_code in retry_statuses
                        and attempt + 1 < retry_attempts
                    ):
                        await asyncio.sleep(max(0.0, float(step.retry_delay_seconds)))
                        continue
                    break

                if resp is None:
                    raise RuntimeError(f"Workflow step '{step.name}' produced no HTTP response")

                if resp.status_code == 429:
                    self.record_failure(Exception("HTTP 429 Quota Exceeded"), quota_exhausted=True)
                    raise RuntimeError(f"Provider {self.name} quota exhausted (HTTP 429)")
                if resp.status_code not in (200, 201, 202, 204):
                    exc = RuntimeError(f"Workflow step '{step.name}' HTTP {resp.status_code}: {resp.text}")
                    self.record_failure(exc)
                    raise exc

                try:
                    last_response_data = self._decode_workflow_response(resp, step.response_format)
                except Exception as exc:
                    raise RuntimeError(
                        f"Workflow step '{step.name}' could not decode {step.response_format} response: {exc}"
                    ) from exc

            # 5. Extract JMESPath variables into context
            if step.exports and isinstance(last_response_data, dict):
                for var_name, jpath in step.exports.items():
                    val = jmespath.search(jpath, last_response_data)
                    if val is not None:
                        context[var_name] = val

            # 6. Handle action: download_zip or explicit zip download
            if getattr(step, "action", None) in ("download_zip", "download_and_extract_zip"):
                zip_url_val = _interpolate_value(step.url or context.get("full_zip_url"), context)
                if zip_url_val:
                    zip_data = await self._unpack_zip_archive(client, zip_url_val)
                    context.update(zip_data)
                    if isinstance(last_response_data, dict):
                        last_response_data.update(zip_data)

        final_payload = dict(last_response_data) if isinstance(last_response_data, dict) else {}
        for k, v in context.items():
            if k not in final_payload:
                final_payload[k] = v

        # If a full_zip_url was discovered in the workflow and not yet unpacked, unpack it
        zip_url = context.get("full_zip_url") or final_payload.get("full_zip_url")
        has_md = bool(
            final_payload.get("markdown")
            or (isinstance(final_payload.get("extract_result"), dict) and final_payload["extract_result"].get("markdown"))
        )
        if zip_url and not has_md:
            try:
                zip_data = await self._unpack_zip_archive(client, zip_url)
                final_payload.update(zip_data)
            except Exception as exc:
                self.record_failure(exc)
                raise RuntimeError(f"Failed to unpack MinerU ZIP from {zip_url}: {exc}") from exc

        return _sanitize_output_payload(final_payload)

    async def _unpack_zip_archive(self, client: httpx.AsyncClient, zip_url: str) -> Dict[str, Any]:
        """Downloads a MinerU/standard parser result ZIP and extracts markdown, content list, figures, and tables."""
        import zipfile
        import io
        import json

        def _clean_caption(val: Any) -> str:
            if val is None:
                return ""
            if isinstance(val, (list, tuple)):
                return " ".join(str(c).strip() for c in val if c is not None and str(c).strip()).strip()
            return str(val).strip()

        # Streaming download into memory buffer
        bio = io.BytesIO()
        async with client.stream("GET", zip_url, timeout=120.0) as resp:
            if resp.status_code != 200:
                raise RuntimeError(f"Failed to download parser ZIP from {zip_url}: HTTP {resp.status_code}")
            async for chunk in resp.aiter_bytes(chunk_size=65536):
                bio.write(chunk)
        zip_bytes = bio.getvalue()

        extracted: Dict[str, Any] = {
            "markdown": "",
            "pages": [],
            "figures": [],
            "tables": [],
        }

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            file_names = zf.namelist()
            # 1. Locate full.md or primary markdown
            md_candidates = [f for f in file_names if f.endswith(".md")]
            if "full.md" in md_candidates:
                extracted["markdown"] = zf.read("full.md").decode("utf-8", errors="replace")
            elif md_candidates:
                extracted["markdown"] = zf.read(md_candidates[0]).decode("utf-8", errors="replace")

            # 2. Locate content_list.json (official MinerU schema)
            content_list_file = next((f for f in file_names if f.endswith("content_list.json")), None)
            if content_list_file:
                try:
                    c_data = json.loads(zf.read(content_list_file).decode("utf-8"))
                except Exception as exc:
                    raise ValueError(f"Malformed {content_list_file} in ZIP archive: {exc}") from exc
                if isinstance(c_data, list):
                    pages_dict = {}
                    for item in c_data:
                        item_type = item.get("type", "")
                        page_idx = item.get("page_idx", 1)
                        if page_idx not in pages_dict:
                            pages_dict[page_idx] = {"page_number": page_idx, "text": ""}
                        if item_type in ("text", "title"):
                            pages_dict[page_idx]["text"] += item.get("text", "") + "\n"
                        elif item_type == "image":
                            img_path = item.get("img_path", "")
                            b64_img = None
                            if img_path and img_path in file_names:
                                b64_img = base64.b64encode(zf.read(img_path)).decode("utf-8")
                            # MinerU official outputs 'image_caption': ["Fig 1..."] as list[str]
                            raw_cap = item.get("image_caption") or item.get("img_caption") or item.get("caption")
                            caption_str = _clean_caption(raw_cap)
                            extracted["figures"].append({
                                "figure_number": str(len(extracted["figures"]) + 1),
                                "caption": caption_str,
                                "page": page_idx,
                                "image_base64": b64_img,
                            })
                        elif item_type == "table":
                            # MinerU official outputs 'table_caption': ["Table 1..."] and 'table_body': "<html>...</html>"
                            raw_cap = item.get("table_caption") or item.get("caption")
                            caption_str = _clean_caption(raw_cap)
                            body_content = item.get("table_body") or item.get("text") or item.get("markdown") or ""
                            extracted["tables"].append({
                                "table_number": str(len(extracted["tables"]) + 1),
                                "caption": caption_str,
                                "page": page_idx,
                                "markdown": body_content,
                            })
                    extracted["pages"] = list(pages_dict.values())

        return extracted

    async def parse(self, request: ParseRequest) -> ParsedDocument:
        t0 = time.monotonic()
        configured_timeout = (
            self.definition.timeout_seconds
            if self.definition.timeout_seconds is not None
            else self.definition.timeout
        )
        timeout = min(configured_timeout, 600.0)

        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                if self.definition.workflow:
                    final_payload = await self._execute_workflow(client, request)
                else:
                    url, headers, params, body_data, files = await self._build_request(request)
                    method = self.definition.request.method
                    encoding = self.definition.request.encoding

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

                    if resp.status_code == 429:
                        self.record_failure(Exception("HTTP 429 Quota Exceeded"), quota_exhausted=True)
                        raise RuntimeError(f"Provider {self.name} quota exhausted (HTTP 429)")

                    if resp.status_code not in (200, 201, 202):
                        exc = RuntimeError(f"Provider HTTP {resp.status_code}: {resp.text}")
                        self.record_failure(exc)
                        raise exc

                    initial_payload = resp.json()

                    if self.definition.async_config and self.definition.async_config.enabled:
                        job_id = jmespath.search(self.definition.async_config.job_id_path, initial_payload)
                        if not job_id:
                            raise RuntimeError(
                                f"Could not extract job_id via path: {self.definition.async_config.job_id_path}"
                            )
                        final_payload = await self._poll_async_job(client, str(job_id), headers, params)
                    else:
                        final_payload = initial_payload
            except Exception as exc:
                self.record_failure(exc)
                raise

        latency_ms = (time.monotonic() - t0) * 1000.0
        self.record_success(latency_ms)

        # Declarative JMESPath normalization
        doc = normalize_response(final_payload, self.definition.response)
        return doc
