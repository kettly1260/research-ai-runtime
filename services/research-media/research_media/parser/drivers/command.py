from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional
from contracts import ParseRequest, ParsedDocument
from ..base import BaseParserDriver
from ..models import ProviderDefinition
from ..normalization import normalize_response


def _interpolate_str(template: str, context: Dict[str, Any]) -> str:
    res = template
    for k, v in context.items():
        if v is not None:
            res = res.replace(f"${{{k}}}", str(v))
    return res


class GenericCommandDriver(BaseParserDriver):
    """Protocol-based command / CLI parser driver.

    Executes arbitrary system commands/CLIs declaratively specified in YAML
    (e.g., local OCR tools, custom conversion scripts) without hardcoded vendor Python logic.
    """

    def __init__(self, definition: ProviderDefinition):
        super().__init__(definition)
        self.command = definition.command or ""
        self.args_template = definition.args or []
        self.output_format = definition.output_format or "json"

    def is_available(self) -> bool:
        if not super().is_available():
            return False
        # If command is specified, check if executable is in PATH
        if self.command:
            cmd_name = self.command.split()[0]
            if not shutil.which(cmd_name) and not os.path.exists(cmd_name):
                return False
        return True

    async def parse(self, request: ParseRequest) -> ParsedDocument:
        t0 = time.monotonic()
        temp_file_to_clean = None
        file_path = request.file_path

        try:
            if not file_path or not os.path.exists(file_path):
                if request.file_content_base64:
                    raw_b64 = request.file_content_base64
                    if "," in raw_b64:
                        _, raw_b64 = raw_b64.split(",", 1)
                    file_bytes = base64.b64decode(raw_b64)
                    suffix = ".pdf" if "pdf" in (request.mime_type or "") else ".png"
                    fd, temp_path = tempfile.mkstemp(suffix=suffix)
                    os.write(fd, file_bytes)
                    os.close(fd)
                    temp_file_to_clean = temp_path
                    file_path = temp_path
                else:
                    raise ValueError("No valid file_path or file_content_base64 provided for command driver")

            filename = os.path.basename(file_path)
            context = {
                "file_path": file_path,
                "filename": filename,
                "mime_type": request.mime_type or "application/octet-stream",
            }

            resolved_cmd = _interpolate_str(self.command, context)
            resolved_args = [_interpolate_str(arg, context) for arg in self.args_template]

            cmd_timeout = float(self.definition.timeout_seconds or self.definition.timeout or 60.0)

            def _run_sub():
                return subprocess.run(
                    [resolved_cmd] + resolved_args,
                    capture_output=True,
                    text=False,
                    timeout=cmd_timeout,
                )

            try:
                res = await asyncio.to_thread(_run_sub)
            except subprocess.TimeoutExpired as exc:
                t_err = TimeoutError(f"Command '{resolved_cmd}' timed out after {cmd_timeout}s")
                self.record_failure(t_err)
                raise t_err from exc
            stdout = res.stdout
            stderr = res.stderr

            if res.returncode != 0:
                err_msg = stderr.decode("utf-8", errors="replace").strip()
                exc = RuntimeError(f"Command '{resolved_cmd}' failed (exit {res.returncode}): {err_msg}")
                self.record_failure(exc)
                raise exc

            out_str = stdout.decode("utf-8", errors="replace").strip()
            if self.output_format == "json":
                try:
                    payload = json.loads(out_str)
                except Exception:
                    payload = {"text": out_str}
            elif self.output_format == "lines":
                lines = [l.strip() for l in out_str.splitlines() if l.strip()]
                payload = {
                    "text": out_str,
                    "blocks": [{"type": "text", "content": l} for l in lines],
                }
            else:
                payload = {
                    "text": out_str,
                    "markdown": out_str,
                }

            latency_ms = (time.monotonic() - t0) * 1000.0
            self.record_success(latency_ms)

            return normalize_response(payload, self.definition.response)

        finally:
            if temp_file_to_clean and os.path.exists(temp_file_to_clean):
                try:
                    os.unlink(temp_file_to_clean)
                except Exception:
                    pass
