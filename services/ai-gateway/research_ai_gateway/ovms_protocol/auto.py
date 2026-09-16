"""Capability probe backing ``OVMS_PROTOCOL=auto``.

Auto-detection must be based on what the server actually answers, never on a
Docker tag or image string -- ``latest-gpu`` has already drifted once from
OVMS 2026.1 to 2026.3 and a tag is not a compatibility contract.

The ladder below is intentionally explicit about its evidence so an operator can
see exactly *why* a protocol was chosen, and it fails closed when no branch is
decisive.

Signals, in order:

1. ``GET /v2/health/ready`` -- present on every KServe v2 server (2026.1 too).
2. ``GET /v1/config``       -- the TFS config API; returns ``model_config_list``.
3. If exactly one API is present, that decides it.
4. If *both* are present (the normal 2026.1 case) a negative predict probe is
   used: OVMS 2026.3 routes ``/v1/models/{name}:predict`` to the MediaPipe graph
   handler, which rejects a TFS body with HTTP 412 "model field is missing in
   JSON body" (or HTTP 404 "Mediapipe graph definition with requested name is
   not found").  Either marker means the Classic Model REST API is gone.
5. If neither API is present, auto-detection fails closed.

The probe is deliberately model-independent so it also works in a zero-resident
deployment where no model is loaded yet.

NOTE: step 4 depends on real-server behaviour.  Per task item 25 the ladder must
be re-validated against a pinned OVMS 2026.3.1 image before ``auto`` is enabled
anywhere other than development, acceptance and CI.  Production must use an
explicit ``OVMS_PROTOCOL`` value.
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


def _looks_like_mediapipe_rejection(status: Optional[int], body: Any) -> bool:
    """Detect the 2026.3 MediaPipe handler rejecting a TFS Classic body."""
    if status is None:
        return False
    if isinstance(body, str):
        text = body
    else:
        try:
            text = json.dumps(body, default=str)
        except (TypeError, ValueError):
            text = str(body)
    lowered = (text or "").lower()
    if "model field is missing" in lowered:
        return True
    return "mediapipe" in lowered and "graph" in lowered


def probe_protocol(http: HttpClient, timeout: float = DEFAULT_PROBE_TIMEOUT) -> ProtocolDecision:
    """Determine the backend protocol from live capability probes."""
    kserve_status, _ = http.get(KSERVE_READY_PATH, timeout)
    tfs_status, tfs_body = http.get(TFS_CONFIG_PATH, timeout)

    kserve_api = kserve_status == 200
    tfs_api = (
        tfs_status == 200
        and isinstance(tfs_body, dict)
        and "model_config_list" in tfs_body
    )

    evidence: Dict[str, Any] = {
        "kserve_health_ready_status": kserve_status,
        "tfs_config_status": tfs_status,
        "kserve_api_present": kserve_api,
        "tfs_api_present": tfs_api,
    }

    if tfs_api and not kserve_api:
        return ProtocolDecision("tfs", "auto_probe", evidence)
    if kserve_api and not tfs_api:
        return ProtocolDecision("kserve", "auto_probe", evidence)
    if not tfs_api and not kserve_api:
        raise ProtocolDetectionError(
            "neither the TFS config API nor the KServe v2 health API responded; "
            f"evidence={evidence}"
        )

    # Both APIs answered -- discriminate with a negative predict probe.
    probe_status, probe_body = http.post(
        f"/v1/models/{PROBE_MODEL_NAME}:predict",
        {"instances": []},
        timeout,
    )
    mediapipe = _looks_like_mediapipe_rejection(probe_status, probe_body)
    evidence["tfs_predict_probe_status"] = probe_status
    evidence["tfs_predict_probe_mediapipe_rejection"] = mediapipe

    return ProtocolDecision("kserve" if mediapipe else "tfs", "auto_probe", evidence)
