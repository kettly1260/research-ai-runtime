"""OVMS backend client.

``ovms_predict()`` is the single entry point business code is allowed to use.
Everything protocol specific -- ``instances`` vs ``inputs``, ``predictions`` vs
``outputs``, ``/v1/models/{m}:predict`` vs ``/v2/models/{m}/infer`` -- is owned
by the adapter layer in :mod:`research_ai_gateway.ovms_protocol`.  This module
keeps the transport, the error mapping and the model-availability probe, and
delegates the rest.
"""

import json
import os
import requests
from typing import Any, List, Optional, Tuple
from fastapi import HTTPException
from .config import (
    OVMS_BASE,
    OVMS_TIMEOUT,
    OVMS_CONFIG_PATH,
    OVMS_CONFIG_RELOAD_URL,
    OVMS_MODEL_CATALOG_PATH,
    OVMS_GENAI_EMBEDDINGS_URL,
)
from . import ovms_protocol
from .ovms_protocol.base import ProtocolPayloadError

#: Backend error bodies are echoed into the gateway's 502 detail, but never in
#: full: an OVMS stack trace must not be able to flood a client response.
BACKEND_BODY_LIMIT = 400


def _truncate_body(body: Optional[str]) -> str:
    text = (body or "").strip()
    if len(text) > BACKEND_BODY_LIMIT:
        return text[:BACKEND_BODY_LIMIT] + "..."
    return text


class _RequestsHttpClient(ovms_protocol.HttpClient):
    """Transport handed to the protocol adapters (probing + availability)."""

    @staticmethod
    def _url(path: str) -> str:
        return f"{OVMS_BASE.rstrip('/')}{path}"

    def get(self, path: str, timeout: float) -> Tuple[Optional[int], Any]:
        try:
            response = requests.get(self._url(path), timeout=timeout)
        except requests.RequestException:
            return None, None
        try:
            return response.status_code, response.json()
        except ValueError:
            # Keep the raw text so marker-based probes can still inspect it.
            return response.status_code, response.text

    def post(self, path: str, payload: dict, timeout: float) -> Tuple[Optional[int], Any]:
        try:
            response = requests.post(self._url(path), json=payload, timeout=timeout)
        except requests.RequestException:
            return None, None
        try:
            return response.status_code, response.json()
        except ValueError:
            return response.status_code, response.text


HTTP_CLIENT = _RequestsHttpClient()


def ovms_url(model_name: str) -> str:
    """Classic TFS inference URL for ``model_name`` (2026.1 protocol)."""
    return f"{OVMS_BASE.rstrip('/')}/v1/models/{model_name}:predict"


def _post_inference(adapter, model_name: str, request, timeout: int) -> Any:
    """POST a prepared protocol request and map failures to unified semantics.

    backend HTTP error -> 502, timeout -> 504, connection error -> 502,
    malformed JSON -> 502.  The detail always names the protocol, the model and
    the backend status so a protocol mismatch can be diagnosed without guessing.
    """
    url = f"{OVMS_BASE.rstrip('/')}{request.path}"
    try:
        response = requests.post(url, json=request.payload, timeout=timeout)
    except requests.exceptions.Timeout as exc:
        raise HTTPException(
            status_code=504,
            detail=(
                f"backend timeout after {timeout}s "
                f"(protocol={adapter.name}, model={model_name}, path={request.path})"
            ),
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                f"backend request failed "
                f"(protocol={adapter.name}, model={model_name}, path={request.path}): {exc}"
            ),
        ) from exc

    if response.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=(
                f"backend returned HTTP {response.status_code} for model {model_name} "
                f"(protocol={adapter.name}, path={request.path}); "
                f"body={_truncate_body(response.text)}"
            ),
        )

    try:
        return response.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                f"backend returned invalid JSON for model {model_name} "
                f"(protocol={adapter.name}, path={request.path})"
            ),
        ) from exc


def ovms_predict(model_name: str, payload: dict, timeout: int = OVMS_TIMEOUT) -> dict:
    """Run one Classic Model inference and return the canonical response.

    ``payload`` is the gateway's canonical form, ``{"instances": [...]}``.  The
    adapter translates it to whichever wire protocol ``OVMS_PROTOCOL`` selects
    and normalises the response back to ``{"predictions": [...]}``.
    """
    adapter = ovms_protocol.resolve_adapter(HTTP_CLIENT)

    try:
        request = adapter.build_request(model_name, payload)
    except ProtocolPayloadError as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                f"cannot serialize inference request for model {model_name} "
                f"(protocol={adapter.name}): {exc}"
            ),
        ) from exc

    body = _post_inference(adapter, model_name, request, timeout)

    try:
        return adapter.normalize_response(model_name, body, payload)
    except ProtocolPayloadError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"unexpected {adapter.name} response for model {model_name}: {exc}",
        ) from exc


def genai_embeddings_url() -> str:
    return OVMS_GENAI_EMBEDDINGS_URL.format(ovms_base=OVMS_BASE.rstrip("/"))


def ovms_genai_embeddings(model_name: str, texts: List[str], timeout: int = OVMS_TIMEOUT) -> dict:
    """Calls the OVMS GenAI (graph-backed) OpenAI-compatible embeddings endpoint.

    Ported from the legacy gateway so graph models such as
    ``qwen3-embedding-0.6b`` keep byte-identical request/response semantics.
    This endpoint sits outside the Classic Model TFS/KServe split: it is the
    OpenAI-compatible GenAI v3 surface, served by both OVMS versions.
    """
    payload = {"model": model_name, "input": list(texts), "encoding_format": "float"}
    url = genai_embeddings_url()
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout as exc:
        raise HTTPException(
            status_code=504,
            detail=f"embedding backend timeout after {timeout}s",
        ) from exc
    except requests.exceptions.HTTPError as exc:
        resp = exc.response
        status = resp.status_code if resp is not None else "unknown"
        body = ""
        if resp is not None:
            body = (resp.text or "").strip()
            if len(body) > 400:
                body = body[:400] + "..."
        raise HTTPException(
            status_code=502,
            detail=f"embedding backend request failed: HTTP {status} {body}".rstrip(),
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"embedding backend request failed: {exc}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail="embedding backend returned invalid JSON",
        ) from exc


def read_model_config_file(path: str) -> dict:
    if not os.path.exists(path):
        return {"model_config_list": []}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid OVMS config payload at {path}")
    if "model_config_list" not in payload:
        payload["model_config_list"] = []
    return payload


def catalog_model_config(model_name: str) -> dict:
    payload = read_model_config_file(OVMS_MODEL_CATALOG_PATH)
    entries = payload.get("model_config_list", [])
    for item in entries:
        cfg = item.get("config", {}) if isinstance(item, dict) else {}
        if cfg.get("name") == model_name:
            return dict(cfg)

    # If model_name is an alias like <base>__gpu or <base>__cpu
    if "__" in model_name:
        base_name, device_suffix = model_name.rsplit("__", 1)
        for item in entries:
            cfg = item.get("config", {}) if isinstance(item, dict) else {}
            if cfg.get("name") == base_name:
                cloned = dict(cfg)
                cloned["name"] = model_name
                cloned["target_device"] = device_suffix.upper()
                return cloned

    raise KeyError(f"model not found in catalog: {model_name}")


def write_runtime_config(payload: dict) -> None:
    directory = os.path.dirname(OVMS_CONFIG_PATH) or "."
    os.makedirs(directory, exist_ok=True)
    tmp_path = f"{OVMS_CONFIG_PATH}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, OVMS_CONFIG_PATH)


def reload_ovms_config() -> bool:
    """Trigger a global OVMS config reload (TFS config API).

    Only used when ``OVMS_CONFIG_UPDATE_MODE=api``.  The default ``poll`` mode
    relies on OVMS's own filesystem polling (default 1s) instead, which is what
    the production stack runs.
    """
    url = OVMS_CONFIG_RELOAD_URL.format(ovms_base=OVMS_BASE.rstrip("/"))
    try:
        response = requests.post(url, timeout=max(OVMS_TIMEOUT, 30))
        return response.status_code in (200, 201, 202, 204)
    except requests.RequestException:
        return False


def is_model_available(model_name: str) -> bool:
    """Protocol-aware model readiness check used by the DeviceBroker.

    * ``tfs``    -- ``GET /v1/models/{model}`` plus ``model_version_status``
    * ``kserve`` -- ``GET /v2/models/{model}/ready``

    Must return ``False`` (not raise) for models that are not loaded, because
    the zero-resident broker polls it while cold-loading an alias.
    """
    try:
        adapter = ovms_protocol.resolve_adapter(HTTP_CLIENT)
    except HTTPException:
        return False
    return adapter.model_available(model_name, HTTP_CLIENT)
