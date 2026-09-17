"""OVMS 2026.3+ KServe v2 protocol adapter.

OVMS 2026.3 repurposed ``POST /v1/models/{model}:predict`` for MediaPipe graphs,
so the Classic Model REST API is gone.  Classic IR models are now served through
the KServe v2 inference protocol::

    POST /v2/models/{model}/infer
    {"inputs": [{"name": ..., "shape": [...], "datatype": ..., "data": [...]}]}

Nothing about a model's tensor names, dtypes or output shapes is hardcoded here;
everything is derived from the payload the business layer already built and from
the shapes the backend reports back.
"""

from __future__ import annotations

from typing import Any

from ..config import OVMS_TIMEOUT
from .base import BackendRequest, HttpClient, ProtocolAdapter, ProtocolPayloadError
from .tensors import instances_to_inputs, outputs_to_predictions


class KserveAdapter(ProtocolAdapter):
    name = "kserve"

    def build_request(self, model_name: str, payload: dict) -> BackendRequest:
        if not isinstance(payload, dict):
            raise ProtocolPayloadError("kserve payload must be an object")
        inputs = instances_to_inputs(payload.get("instances"))

        body: dict = {"inputs": inputs}
        if payload.get("id") is not None:
            body["id"] = payload["id"]
        return BackendRequest(path=f"/v2/models/{model_name}/infer", payload=body)

    def normalize_response(self, model_name: str, body: Any, payload: dict) -> dict:
        if not isinstance(body, dict):
            raise ProtocolPayloadError(
                f"kserve backend returned {type(body).__name__} instead of a JSON object"
            )
        batch = len(payload.get("instances") or [])
        predictions = outputs_to_predictions(body.get("outputs"), batch)
        return {
            "predictions": predictions,
            "model_name": body.get("model_name", model_name),
        }

    def model_available(self, model_name: str, http: HttpClient) -> bool:
        """KServe v2 readiness probe.

        ``/v2/models/{name}/ready`` is the documented per-model readiness
        endpoint: 200 when the model can serve, 503 while it is loading or
        unloaded.  Older/partial implementations may not expose it, so a 404
        falls back to the model metadata endpoint.
        """
        timeout = float(min(OVMS_TIMEOUT, 15))

        status, _ = http.get(f"/v2/models/{model_name}/ready", timeout)
        if status == 200:
            return True
        if status == 503:
            return False

        metadata_status, _ = http.get(f"/v2/models/{model_name}", timeout)
        return metadata_status == 200
