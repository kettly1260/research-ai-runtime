"""Capability probe backing ``OVMS_PROTOCOL=auto``.

Auto-detection must be based on what the server actually answers, never on a
Docker tag or image string -- ``latest-gpu`` has already drifted once from
OVMS 2026.1 to 2026.3 and a tag is not a compatibility contract.

Real-server evidence (captured from the pinned acceptance images with an empty
runtime config, i.e. the zero-resident state that production normally sits in)::

    endpoint                          OVMS 2026.1              OVMS 2026.3.1
    GET  /v2/health/ready             200                      200
    GET  /v1/config                   200 {}                   200 {}
    GET  /v1/models                   404                      200 {"data":[]}
    POST /v1/models/{m}:predict       404                      412
                                      "Model with requested     "The file is not valid json
                                       name is not found"       - model field is missing
                                                                in JSON body"

Two consequences drive this implementation:

* ``/v2/health/ready`` is **not** a discriminator -- 2026.1 serves it too.
* ``/v1/config`` is **not** a discriminator either.  It answers 200 on both
  versions, and with no model loaded its body is ``{}`` on both, so keying on
  the presence of ``model_config_list`` silently flips the answer depending on
  whether anything happens to be resident.  An earlier revision of this module
  did exactly that and mis-detected 2026.1 as ``kserve``.

The only reliable discriminator is how ``POST /v1/models/{name}:predict``
*rejects* a TFS body:

* HTTP 412 ``The file is not valid json - model field is missing in JSON body``
  (or HTTP 404 naming a MediaPipe graph) means the Classic Model REST API is
  gone and the path now routes to the MediaPipe handler -- **kserve**.
* HTTP 404 ``Model with requested name is not found`` is the Classic Model
  registry answering for an unknown model, i.e. the TFS path is alive and only
  the model name was unknown -- **tfs**.

The probe uses a model name that does not exist and an empty ``instances``
list, so it can never produce a successful inference and can never cold-load a
model.  It is therefore safe in a zero-resident deployment.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from .base import HttpClient, ProtocolDecision, ProtocolDetectionError

#: Model name used for the negative predict probe.  It is intentionally not a
#: real model: the probe only inspects how the server *rejects* the request.
PROBE_MODEL_NAME = "__ovms_protocol_probe__"

#: Endpoints probed by the ladder.
KSERVE_READY_PATH = "/v2/health/ready"
TFS_CONFIG_PATH = "/v1/config"

DEFAULT_PROBE_TIMEOUT = 5.0

#: Verbatim server strings the ladder keys on.  Kept as named constants so the
#: tests can assert against the exact text observed on the pinned images.
MEDIAPIPE_MODEL_FIELD_MARKER = "model field is missing"
CLASSIC_MODEL_LOOKUP_MARKER = "model with requested name is not found"


def _as_text(body: Any) -> str:
    if isinstance(body, str):
        return body
    try:
        return json.dumps(body, default=str)
    except (TypeError, ValueError):
        return str(body)


def _looks_like_mediapipe_rejection(status: Optional[int], body: Any) -> bool:
    """Detect the 2026.3 MediaPipe handler rejecting a TFS Classic body."""
    if status is None:
        return False
    lowered = _as_text(body).lower()
    if MEDIAPIPE_MODEL_FIELD_MARKER in lowered:
        return True
    return "mediapipe" in lowered and "graph" in lowered


def _looks_like_classic_model_lookup(status: Optional[int], body: Any) -> bool:
    """Detect the TFS Classic Model registry answering for an unknown model."""
    if status is None:
        return False
    return CLASSIC_MODEL_LOOKUP_MARKER in _as_text(body).lower()


def probe_protocol(http: HttpClient, timeout: float = DEFAULT_PROBE_TIMEOUT) -> ProtocolDecision:
    """Determine the backend protocol from live capability probes."""
    kserve_status, _ = http.get(KSERVE_READY_PATH, timeout)
    tfs_status, tfs_body = http.get(TFS_CONFIG_PATH, timeout)

    # Liveness of the backend at all.  2026.1 and 2026.3 both answer at least
    # one of these; if neither answers, the backend is down or not an OVMS.
    server_reachable = kserve_status == 200 or tfs_status == 200

    # Supporting signal only -- see the module docstring for why this cannot be
    # decisive in a zero-resident deployment.
    tfs_config_lists_models = (
        tfs_status == 200
        and isinstance(tfs_body, dict)
        and "model_config_list" in tfs_body
    )

    probe_status, probe_body = http.post(
        f"/v1/models/{PROBE_MODEL_NAME}:predict",
        {"instances": []},
        timeout,
    )
    mediapipe = _looks_like_mediapipe_rejection(probe_status, probe_body)
    classic_lookup = _looks_like_classic_model_lookup(probe_status, probe_body)

    evidence: Dict[str, Any] = {
        "kserve_health_ready_status": kserve_status,
        "tfs_config_status": tfs_status,
        "tfs_config_lists_models": tfs_config_lists_models,
        "tfs_predict_probe_status": probe_status,
        "tfs_predict_probe_mediapipe_rejection": mediapipe,
        "tfs_predict_probe_classic_model_lookup": classic_lookup,
    }

    if not server_reachable:
        raise ProtocolDetectionError(
            "backend is unreachable: neither /v2/health/ready nor /v1/config "
            f"answered with HTTP 200; evidence={evidence}"
        )

    if mediapipe:
        return ProtocolDecision("kserve", "auto_probe_predict_marker", evidence)
    if classic_lookup:
        return ProtocolDecision("tfs", "auto_probe_predict_marker", evidence)

    # The predict probe was inconclusive (unexpected status or body).  Fall back
    # to the config body, which is only informative when models are resident.
    if tfs_config_lists_models:
        return ProtocolDecision("tfs", "auto_probe_config_body", evidence)

    raise ProtocolDetectionError(
        "could not discriminate the OVMS protocol: the /v1/models/{name}:predict "
        "probe matched neither the MediaPipe rejection nor the Classic Model "
        f"lookup signature, and /v1/config did not list any models; evidence={evidence}"
    )
