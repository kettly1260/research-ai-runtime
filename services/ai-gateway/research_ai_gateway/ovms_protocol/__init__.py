"""OVMS wire-protocol adapters.

Public surface used by :mod:`ovms_client`:

* :func:`resolve_adapter` -- pick the adapter for the configured protocol
* :func:`protocol_diagnostics` -- report configured vs effective protocol
* :func:`reset_protocol_cache` -- drop the cached auto-detection result (tests)

``OVMS_PROTOCOL`` semantics:

``tfs``
    Force ``POST /v1/models/{model}:predict``.  Required for the current
    production OVMS 2026.1 stack, and the default so behaviour is unchanged for
    deployments that do not set the variable.
``kserve``
    Force ``POST /v2/models/{model}/infer``.  Required for OVMS 2026.3.1, where
    the Classic Model REST API no longer exists.
``auto``
    Probe the backend once and cache the result.  Intended for development,
    acceptance and CI only -- production must pin an explicit protocol.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

from fastapi import HTTPException

from ..config import OVMS_PROTOCOL
from .base import (
    BackendRequest,
    HttpClient,
    ProtocolAdapter,
    ProtocolDecision,
    ProtocolDetectionError,
    ProtocolPayloadError,
)
from .auto import probe_protocol
from .kserve import KserveAdapter
from .tfs import TfsAdapter

__all__ = [
    "BackendRequest",
    "HttpClient",
    "ProtocolAdapter",
    "ProtocolDecision",
    "ProtocolDetectionError",
    "ProtocolPayloadError",
    "configured_protocol",
    "get_adapter",
    "protocol_diagnostics",
    "reset_protocol_cache",
    "resolve_adapter",
    "resolve_decision",
]

_ADAPTERS: Dict[str, ProtocolAdapter] = {
    "tfs": TfsAdapter(),
    "kserve": KserveAdapter(),
}

_lock = threading.Lock()
_cached_decision: Optional[ProtocolDecision] = None


def configured_protocol() -> str:
    """The protocol requested via ``OVMS_PROTOCOL``."""
    return OVMS_PROTOCOL


def get_adapter(protocol: str) -> ProtocolAdapter:
    try:
        return _ADAPTERS[protocol]
    except KeyError as exc:  # pragma: no cover - config validates the value
        raise ValueError(f"unsupported OVMS protocol: {protocol!r}") from exc


def resolve_decision(http: HttpClient, force_refresh: bool = False) -> ProtocolDecision:
    """Resolve the effective protocol, caching ``auto`` detection.

    Detection runs once per process (startup / first use), never per inference.
    A failed detection is *not* cached, so the gateway recovers on its own once
    the backend becomes reachable.
    """
    if OVMS_PROTOCOL != "auto":
        return ProtocolDecision(OVMS_PROTOCOL, "configured", {})

    global _cached_decision
    with _lock:
        if _cached_decision is not None and not force_refresh:
            return _cached_decision
        decision = probe_protocol(http)
        _cached_decision = decision
        print(
            f"Detected OVMS protocol: {decision.protocol} "
            f"(source={decision.source}, evidence={decision.evidence})",
            flush=True,
        )
        return decision


def resolve_adapter(http: HttpClient) -> ProtocolAdapter:
    try:
        decision = resolve_decision(http)
    except ProtocolDetectionError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"OVMS protocol auto-detection failed: {exc}",
        ) from exc
    return get_adapter(decision.protocol)


def protocol_diagnostics() -> Dict[str, Any]:
    """Protocol state for ``/health`` and log-based troubleshooting."""
    configured = OVMS_PROTOCOL
    if configured != "auto":
        return {
            "ovms_protocol_configured": configured,
            "ovms_protocol_effective": configured,
            "ovms_protocol_source": "configured",
            "ovms_protocol_evidence": {},
        }

    with _lock:
        cached = _cached_decision
    return {
        "ovms_protocol_configured": configured,
        "ovms_protocol_effective": cached.protocol if cached else "unresolved",
        "ovms_protocol_source": cached.source if cached else "auto_pending",
        "ovms_protocol_evidence": dict(cached.evidence) if cached else {},
    }


def reset_protocol_cache() -> None:
    """Drop the cached auto-detection result (test helper)."""
    global _cached_decision
    with _lock:
        _cached_decision = None
