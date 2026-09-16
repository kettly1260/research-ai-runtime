#!/usr/bin/env python
"""Capture a live OVMS REST capability surface, endpoint by endpoint.

Why this exists
---------------
The task requires the 2026.3.1 API surface to be *recorded from a real server*
rather than inferred from KServe documentation, and the `OVMS_PROTOCOL=auto`
ladder is built on three specific signals.  This script captures those signals
and, crucially, reuses the gateway's own probe implementation so the reported
decision is exactly the decision the gateway would make.

Safety
------
* GET probes only, by default.  A GET cannot load or unload a model.
* `POST /v1/models/{m}:predict` with an empty `instances` list is a negative
  probe: it cannot produce a successful inference, so it cannot cold-load a
  model.  It is on by default because the MediaPipe 412/404 marker is the
  tie-breaker the auto ladder depends on.
* `POST /v3/embeddings` is a **real inference** and can cold-load a model, so it
  is off unless `--allow-inference-probes` is passed.  Only use it against an
  isolated acceptance stack, never against production.

Usage
-----
    python scripts/ovms_protocol_capability_probe.py \
        --ovms-base http://127.0.0.1:28342 \
        --label ovms-2026.3.1 \
        --model qwen-reranker \
        --out .workbuddy-ai/artifacts/capability-20263.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
GATEWAY_DIR = REPO_ROOT / "services" / "ai-gateway"
if str(GATEWAY_DIR) not in sys.path:
    sys.path.insert(0, str(GATEWAY_DIR))

BODY_LIMIT = 600

#: ``requests`` honours ``HTTP_PROXY``/``HTTPS_PROXY`` from the environment.
#: Dev shells and CI runners frequently export them (this machine exports
#: ``HTTP_PROXY=http://127.0.0.1:<port>``), and the proxy then intercepts calls
#: to ``127.0.0.1``.  The symptom is a *ReadTimeout naming the proxy's port*
#: instead of a ConnectionError naming the target -- i.e. every row in the
#: capture would describe the proxy rather than OVMS.  Env trust is therefore
#: off by default and must be requested explicitly with ``--use-env-proxy``.
_TRUST_ENV_PROXY = False
_SESSION: Optional[requests.Session] = None


def set_env_proxy(enabled: bool) -> None:
    """Opt in to honouring proxy environment variables."""
    global _TRUST_ENV_PROXY, _SESSION
    _TRUST_ENV_PROXY = bool(enabled)
    _SESSION = None


def _session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        session = requests.Session()
        session.trust_env = _TRUST_ENV_PROXY
        _SESSION = session
    return _SESSION


def _truncate(text: Optional[str], limit: int = BODY_LIMIT) -> str:
    value = (text or "").strip()
    return value if len(value) <= limit else value[:limit] + "..."


def _load_gateway_protocol():
    """Import the gateway's protocol package, or return None if unavailable."""
    try:
        import research_ai_gateway.ovms_protocol as protocol_mod  # noqa: PLC0415

        return protocol_mod
    except Exception as exc:  # pragma: no cover - depends on the environment
        print(f"[warn] gateway protocol package unavailable: {exc}", file=sys.stderr)
        return None


def build_endpoint_plan(model: Optional[str]) -> List[Dict[str, Any]]:
    """The endpoints whose real responses must be recorded (task item 25)."""
    plan: List[Dict[str, Any]] = [
        {"method": "GET", "path": "/v2/health/live", "note": "KServe liveness"},
        {"method": "GET", "path": "/v2/health/ready", "note": "KServe readiness (auto signal 1)"},
        {"method": "GET", "path": "/v1/config", "note": "TFS config API (auto signal 2)"},
        {"method": "GET", "path": "/v1/models", "note": "TFS model listing"},
        {"method": "GET", "path": "/v3/embeddings", "note": "GenAI v3 surface presence"},
    ]
    if model:
        plan.extend(
            [
                {"method": "GET", "path": f"/v1/models/{model}", "note": "TFS model metadata"},
                {"method": "GET", "path": f"/v2/models/{model}", "note": "KServe model metadata"},
                {"method": "GET", "path": f"/v2/models/{model}/ready", "note": "KServe per-model readiness"},
                {
                    "method": "POST",
                    "path": f"/v1/models/{model}:predict",
                    "body": {"instances": []},
                    "note": "negative probe: MediaPipe marker tie-breaker (auto signal 3)",
                },
                {
                    "method": "POST",
                    "path": f"/v2/models/{model}/infer",
                    "body": {"inputs": []},
                    "note": "KServe infer presence",
                },
            ]
        )
    return plan


def run_probe(
    base_url: str,
    model: Optional[str],
    timeout: float,
    allow_inference_probes: bool,
    use_env_proxy: bool = False,
) -> Dict[str, Any]:
    set_env_proxy(use_env_proxy)
    base = base_url.rstrip("/")
    protocol_mod = _load_gateway_protocol()
    calls: List[Dict[str, Any]] = []
    errors: Dict[str, str] = {}

    def _do(method: str, path: str, body: Optional[dict], timeout_value: float) -> Tuple[Optional[int], Any]:
        started = time.monotonic()
        try:
            if method == "GET":
                response = _session().get(f"{base}{path}", timeout=timeout_value)
            else:
                response = _session().post(f"{base}{path}", json=body, timeout=timeout_value)
        except requests.RequestException as exc:
            detail = f"{type(exc).__name__}: {exc}"
            errors[f"{method} {path}"] = detail
            calls.append(
                {
                    "method": method,
                    "path": path,
                    "status": None,
                    "elapsed_s": round(time.monotonic() - started, 4),
                    "error": detail,
                }
            )
            return None, None

        elapsed = round(time.monotonic() - started, 4)
        content_type = response.headers.get("Content-Type", "")
        parsed: Any = None
        try:
            parsed = response.json()
        except ValueError:
            parsed = None
        calls.append(
            {
                "method": method,
                "path": path,
                "status": response.status_code,
                "content_type": content_type,
                "elapsed_s": elapsed,
                "json": parsed,
                "body": None if parsed is not None else _truncate(response.text),
            }
        )
        return response.status_code, parsed if parsed is not None else response.text

    results: List[Dict[str, Any]] = []

    def _record(entry: Dict[str, Any], method: str, path: str, status: Optional[int], body: Any) -> None:
        results.append(
            {
                **entry,
                "status": status,
                "response": body,
                "error": errors.get(f"{method} {path}") if status is None else None,
            }
        )

    for entry in build_endpoint_plan(model):
        if entry.get("path") == "/v3/embeddings" and entry["method"] == "GET":
            # Presence probe only; the POST form is a real inference.
            status, body = _do("GET", entry["path"], None, timeout)
            _record(entry, "GET", entry["path"], status, body)
            continue

        status, body = _do(entry["method"], entry["path"], entry.get("body"), timeout)
        _record(entry, entry["method"], entry["path"], status, body)

    inference_probe = None
    if allow_inference_probes:
        status, body = _do(
            "POST",
            "/v3/embeddings",
            {"model": model or "qwen3-embedding-0.6b", "input": ["probe"], "encoding_format": "float"},
            max(timeout, 120.0),
        )
        inference_probe = {"status": status, "response": body}

    decision: Optional[Dict[str, Any]] = None
    decision_error: Optional[str] = None
    if protocol_mod is not None:
        class _Http(protocol_mod.HttpClient):
            def get(self, path: str, timeout_value: float):
                return _do("GET", path, None, timeout_value)

            def post(self, path: str, payload: dict, timeout_value: float):
                return _do("POST", path, payload, timeout_value)

        try:
            decided = protocol_mod.probe_protocol(_Http(), timeout)
            decision = {
                "protocol": decided.protocol,
                "source": decided.source,
                "evidence": decided.evidence,
            }
        except protocol_mod.ProtocolDetectionError as exc:
            decision_error = str(exc)

    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "ovms_base": base,
        "model": model,
        "endpoints": results,
        "auto_decision": decision,
        "auto_decision_error": decision_error,
        "inference_probe": inference_probe,
        "raw_calls": calls,
    }


def render_markdown(capture: Dict[str, Any]) -> str:
    lines = [
        f"### Capability capture — {capture.get('label') or capture['ovms_base']}",
        "",
        f"Captured at `{capture['captured_at']}`, base `{capture['ovms_base']}`.",
        "",
        "| Endpoint | Status | Response |",
        "| --- | --- | --- |",
    ]
    for entry in capture["endpoints"]:
        status = entry["status"] if entry["status"] is not None else "transport error"
        payload = entry.get("response")
        if payload is None and entry.get("error"):
            rendered = _truncate(entry["error"], 200)
        else:
            rendered = _truncate(json.dumps(payload, ensure_ascii=False) if payload is not None else "", 200)
        lines.append(f"| `{entry['method']} {entry['path']}` | {status} | `{rendered}` |")

    lines.append("")
    if capture.get("auto_decision"):
        decision = capture["auto_decision"]
        lines.append(
            f"`OVMS_PROTOCOL=auto` would resolve to **{decision['protocol']}** "
            f"(source `{decision['source']}`), evidence: `{json.dumps(decision['evidence'])}`."
        )
    else:
        lines.append(
            "`OVMS_PROTOCOL=auto` **failed closed**: "
            f"{capture.get('auto_decision_error')}"
        )
    if capture.get("inference_probe"):
        probe = capture["inference_probe"]
        lines.append("")
        lines.append(f"`POST /v3/embeddings` probe: status {probe['status']}.")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ovms-base", required=True, help="e.g. http://127.0.0.1:28342")
    parser.add_argument("--label", default=None, help="human label for the report")
    parser.add_argument("--model", default=None, help="model name for model-scoped probes")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--out", default=None, help="JSON output path")
    parser.add_argument(
        "--allow-inference-probes",
        action="store_true",
        help="also POST /v3/embeddings (real inference; acceptance stacks only)",
    )
    parser.add_argument(
        "--use-env-proxy",
        action="store_true",
        help="honour HTTP_PROXY/HTTPS_PROXY (off by default: loopback runs must be direct)",
    )
    args = parser.parse_args(argv)

    if not args.use_env_proxy:
        ambient = [name for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY") if os.getenv(name)]
        if ambient:
            print(
                f"[info] ignoring ambient proxy vars {ambient}; pass --use-env-proxy to honour them",
                file=sys.stderr,
            )

    capture = run_probe(
        args.ovms_base,
        args.model,
        args.timeout,
        args.allow_inference_probes,
        use_env_proxy=args.use_env_proxy,
    )
    capture["label"] = args.label
    markdown = render_markdown(capture)
    print(markdown)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(capture, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\n[ok] wrote {out_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
