"""Shared contracts for the OVMS protocol adapters.

The gateway speaks exactly one internal request/response shape, inherited from
the 2026.1 TensorFlow-Serving (TFS) era:

* request  -> ``{"instances": [{"<tensor_name>": [...], ...}, ...]}``
* response -> ``{"predictions": [...]}``

Every business module (embeddings, rerank, DINO, multimodal) is written against
that shape and must stay unaware of which wire protocol is in use.  An adapter
is therefore responsible for two things only:

1. translating the canonical ``instances`` payload into a protocol request, and
2. normalising the protocol response back into ``predictions``.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


class ProtocolPayloadError(ValueError):
    """Raised when a canonical payload cannot be expressed in a wire protocol.

    Adapters translate this into a gateway ``HTTPException(502)`` so the failure
    mode stays identical to every other backend-contract violation.
    """


class ProtocolDetectionError(RuntimeError):
    """Raised when ``OVMS_PROTOCOL=auto`` cannot determine the backend protocol.

    Auto-detection is fail-closed: an undetectable backend must never silently
    fall back to a different protocol and then pretend the request succeeded.
    """


class HttpClient:
    """Transport injected by :mod:`ovms_client` into the protocol adapters.

    Keeping it as a duck-typed collaborator means adapters never import the HTTP
    layer, never open sockets at import time, and stay trivially testable.
    Both methods return ``(status_code, parsed_body)``; ``status_code`` is
    ``None`` when the call failed at transport level (timeout/refused).
    """

    def get(self, path: str, timeout: float) -> Tuple[Optional[int], Any]:
        raise NotImplementedError

    def post(self, path: str, payload: dict, timeout: float) -> Tuple[Optional[int], Any]:
        raise NotImplementedError


@dataclass(frozen=True)
class BackendRequest:
    """A protocol-specific inference request, ready to be POSTed."""

    path: str
    payload: Dict[str, Any]
    method: str = "POST"


@dataclass
class ProtocolDecision:
    """How the effective protocol was determined, for diagnostics."""

    protocol: str
    source: str
    evidence: Dict[str, Any] = field(default_factory=dict)


class ProtocolAdapter(abc.ABC):
    """Translates canonical payloads to one OVMS wire protocol."""

    #: Stable identifier used in configuration, logs and error details.
    name: str = ""

    @abc.abstractmethod
    def build_request(self, model_name: str, payload: dict) -> BackendRequest:
        """Build the inference request for ``model_name``.

        Must raise :class:`ProtocolPayloadError` when the canonical payload
        cannot be represented in this protocol.
        """

    @abc.abstractmethod
    def normalize_response(self, model_name: str, body: Any, payload: dict) -> dict:
        """Normalise a backend response into the canonical ``predictions`` shape."""

    @abc.abstractmethod
    def model_available(self, model_name: str, http: HttpClient) -> bool:
        """Report whether ``model_name`` is currently loaded and AVAILABLE.

        Used by the DeviceBroker idle sweeper and the zero-resident cold-load
        wait loop, so it must work for models that are *not* loaded yet.
        """
