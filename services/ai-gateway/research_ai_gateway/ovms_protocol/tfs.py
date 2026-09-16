"""OVMS 2026.1 TensorFlow-Serving (Classic Model) protocol adapter.

This is the protocol the production stack runs today.  It is deliberately a
pass-through: the canonical payload already *is* the TFS payload and the TFS
response already *is* the canonical response, so keeping this adapter free of
translation logic is what guarantees byte-identical 2026.1 behaviour.
"""

from __future__ import annotations

from typing import Any

from ..config import OVMS_TIMEOUT
from .base import BackendRequest, HttpClient, ProtocolAdapter, ProtocolPayloadError


class TfsAdapter(ProtocolAdapter):
    name = "tfs"

    def build_request(self, model_name: str, payload: dict) -> BackendRequest:
        if not isinstance(payload, dict) or "instances" not in payload:
            raise ProtocolPayloadError("tfs payload must contain an 'instances' list")
        return BackendRequest(path=f"/v1/models/{model_name}:predict", payload=payload)

    def normalize_response(self, model_name: str, body: Any, payload: dict) -> dict:
        if not isinstance(body, dict):
            raise ProtocolPayloadError(
                f"tfs backend returned {type(body).__name__} instead of a JSON object"
            )
        return body

    def model_available(self, model_name: str, http: HttpClient) -> bool:
        status, payload = http.get(f"/v1/models/{model_name}", float(min(OVMS_TIMEOUT, 15)))
        if status != 200 or not isinstance(payload, dict):
            return False

        statuses = payload.get("model_version_status", [])
        if not isinstance(statuses, list) or len(statuses) == 0:
            return True

        return any(str(entry.get("state", "")).upper() == "AVAILABLE" for entry in statuses)
