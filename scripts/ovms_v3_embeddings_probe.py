#!/usr/bin/env python
"""Verify the OVMS GenAI v3 ``/v3/embeddings`` surface on both OVMS versions.

Why this is a separate gate
---------------------------
``/v3/embeddings`` is the OpenAI-compatible GenAI v3 surface.  It is *not* part
of the TFS Classic-Model / KServe-v2 protocol split that the adapter covers, so
the adapter's own protocol tests say nothing about it.  The migration risk is
therefore independent: 2026.3.1 could pass every KServe assertion and still
remove or change ``/v3/embeddings``, which would break the
``qwen3-embedding-0.6b-int8`` production path.

"Not covered by the adapter" is not "safe to skip".  This script is the gate.

What it records, per version
----------------------------
* endpoint presence, via a real POST (a GET cannot establish this -- both
  versions answer ``400 Invalid request URL`` to ``GET /v3/embeddings``, which
  says nothing about whether POST is served)
* HTTP status, vector dimension, all-finite, L2 norm
* whether the call was cold (model not resident) or warm, taken from the
  gateway's own broker metrics rather than assumed
* the negative case: a POST with no model, which must not perform an inference

Safety
------
A successful ``/v3/embeddings`` POST is a **real inference** and can cold-load a
model, so this script refuses to run without ``--allow-inference-probes``.  Use
it against an isolated acceptance stack only, never against production.

Usage
-----
    python scripts/ovms_v3_embeddings_probe.py \
        --gateway-base http://127.0.0.1:28012 \
        --ovms-base    http://127.0.0.1:28342 \
        --label        ovms-2026.3.1 \
        --gateway-model qwen3-embedding-0.6b-int8 \
        --allow-inference-probes \
        --out .workbuddy-ai/artifacts/v3-embeddings-20263.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

BODY_LIMIT = 600

#: ``requests`` honours ``HTTP_PROXY``/``HTTPS_PROXY`` from the environment.
#: Dev shells frequently export them (this machine exports
#: ``HTTP_PROXY=http://127.0.0.1:<port>``) and the proxy then intercepts calls
#: to LAN/loopback addresses, so every measurement would describe the proxy
#: rather than OVMS.  Env trust is off unless ``--use-env-proxy`` is passed.
_TRUST_ENV_PROXY = False
_SESSION: Optional[requests.Session] = None


def set_env_proxy(enabled: bool) -> None:
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


def _truncate(value: Any) -> Any:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= BODY_LIMIT else text[:BODY_LIMIT] + "...<truncated>"


def _vectors(payload: Any) -> List[List[float]]:
    """Pull embedding vectors out of whichever shape the server returned."""
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        if "embedding" in data[0]:
            return [row["embedding"] for row in data]
    embeddings = payload.get("embeddings")
    if isinstance(embeddings, list) and embeddings:
        return embeddings
    predictions = payload.get("predictions")
    if isinstance(predictions, list) and predictions:
        if isinstance(predictions[0], dict):
            keys = list(predictions[0].keys())
            if len(keys) == 1:
                return [row.get(keys[0], []) for row in predictions]
            return [list(row.values()) for row in predictions]
        return predictions
    return []


def _vector_stats(vectors: List[List[float]]) -> Dict[str, Any]:
    if not vectors:
        return {"count": 0, "dim": 0, "all_finite": None, "norm": None}
    first = vectors[0]
    try:
        values = [float(v) for v in first]
    except (TypeError, ValueError):
        return {"count": len(vectors), "dim": 0, "all_finite": False, "norm": None}
    norm = math.sqrt(sum(v * v for v in values))
    return {
        "count": len(vectors),
        "dim": len(values),
        "all_finite": all(math.isfinite(v) for v in values),
        "norm": round(norm, 6),
    }


def _post(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    started = time.time()
    try:
        response = _session().post(url, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        return {
            "http": None,
            "seconds": round(time.time() - started, 3),
            "transport_error": f"{type(exc).__name__}: {exc}",
            "body": None,
        }
    elapsed = round(time.time() - started, 3)
    try:
        body = response.json()
    except ValueError:
        body = response.text
    return {
        "http": response.status_code,
        "seconds": elapsed,
        "transport_error": None,
        "body": body,
    }


def _get(url: str, timeout: float) -> Dict[str, Any]:
    try:
        response = _session().get(url, timeout=timeout)
    except requests.RequestException as exc:
        return {"http": None, "transport_error": f"{type(exc).__name__}: {exc}", "body": None}
    try:
        body = response.json()
    except ValueError:
        body = response.text
    return {"http": response.status_code, "transport_error": None, "body": body}


def _resident_aliases(gateway_base: str, timeout: float) -> Dict[str, Any]:
    """Read the broker's own view of which aliases are loaded."""
    result = _get(f"{gateway_base}/v1/broker/metrics", timeout)
    body = result.get("body")
    if not isinstance(body, dict):
        return {"aliases": [], "raw": body, "http": result.get("http")}
    loaded = body.get("loaded_models") or {}
    aliases = [name for name, state in loaded.items() if isinstance(state, dict) and state.get("loaded")]
    return {"aliases": aliases, "raw": body, "http": result.get("http")}


def _alias_base(alias: str) -> str:
    return alias.rsplit("__", 1)[0] if "__" in alias else alias


def _prefer_device(items: List[str]) -> str:
    """Deterministic pick when both ``__gpu`` and ``__cpu`` aliases exist."""
    gpu = [a for a in items if a.endswith("__gpu")]
    return sorted(gpu or items)[0]


def _pick_alias(aliases: List[str], base: str, exclude: Tuple[str, ...] = ()) -> Optional[str]:
    """Identify the runtime alias the gateway used for the logical model ``base``.

    Exact name matching is not enough.  The registry maps the logical model
    ``qwen3-embedding-0.6b-int8`` to ``ovms_model: qwen3-embedding-0.6b``, so the
    runtime alias is ``qwen3-embedding-0.6b__cpu`` and shares no exact name with
    the logical id.  The alias that *appeared* after the gateway loaded the model
    is therefore the strongest signal, tried first; prefix matching in either
    direction is the fallback.
    """
    pool = [a for a in aliases if a not in exclude] or list(aliases)
    exact = [a for a in pool if _alias_base(a) == base]
    if exact:
        return _prefer_device(exact)
    related = [
        a
        for a in pool
        if _alias_base(a).startswith(base) or base.startswith(_alias_base(a))
    ]
    if related:
        return _prefer_device(related)
    return None


def _wait_until_not_resident(
    gateway_base: str,
    alias: str,
    timeout: float,
    wait_seconds: float = 300.0,
    poll_seconds: float = 5.0,
) -> Dict[str, Any]:
    started = time.monotonic()
    while time.monotonic() - started < wait_seconds:
        if alias not in _resident_aliases(gateway_base, min(timeout, 15))["aliases"]:
            return {"unloaded": True, "waited_seconds": round(time.monotonic() - started, 3)}
        time.sleep(poll_seconds)
    return {"unloaded": False, "waited_seconds": None}


def _gateway_cold_warm(
    gateway_base: str,
    gateway_model: str,
    alias: str,
    timeout: float,
) -> Dict[str, Any]:
    """Measure cold vs warm where they actually happen in production.

    A genuinely cold *OVMS-level* call is not reachable through the supported
    path: the broker owns the runtime config, so an idle alias is removed from
    ``config.json`` and OVMS would answer 404.  Cold start therefore has to be
    measured at the gateway, which is also where production experiences it --
    the first request cold-loads the alias, the second is warm.
    """
    wait = _wait_until_not_resident(gateway_base, alias, timeout)
    result: Dict[str, Any] = {
        "waited_for_unload": wait,
        "cold": None,
        "warm": None,
    }
    if not wait["unloaded"]:
        result["error"] = (
            f"alias {alias} was still resident after the wait window; "
            "a cold measurement was not possible"
        )
        return result

    for index, key in enumerate(("cold", "warm")):
        call = _post(
            f"{gateway_base}/v1/embeddings",
            {"model": gateway_model, "input": ["cold vs warm probe"]},
            timeout,
        )
        call.update(_vector_stats(_vectors(call.pop("body", None))))
        call["kind"] = key
        call["resident_before_call"] = index > 0
        result[key] = call
    return result


def run_probe(
    gateway_base: str,
    ovms_base: str,
    gateway_model: str,
    timeout: float,
    allow_inference_probes: bool,
) -> Dict[str, Any]:
    if not allow_inference_probes:
        raise SystemExit(
            "refusing to run: a successful POST /v3/embeddings is a real inference "
            "and can cold-load a model.  Pass --allow-inference-probes to confirm "
            "you are targeting an isolated acceptance stack."
        )

    record: Dict[str, Any] = {
        "gateway_base": gateway_base,
        "ovms_base": ovms_base,
        "gateway_model": gateway_model,
        "trust_env_proxy": _TRUST_ENV_PROXY,
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    # 1. Endpoint presence.  A GET cannot establish this: both versions answer
    #    400 Invalid request URL, which is indistinguishable from "no such
    #    route" in this API.  Recorded for completeness only.
    record["get_endpoint"] = _get(f"{ovms_base}/v3/embeddings", timeout)

    # 2. Negative probe: no model named, so no inference can happen.
    record["post_without_model"] = _post(f"{ovms_base}/v3/embeddings", {"input": ["hello"]}, timeout)

    # 3. Make sure the alias is resident, via the gateway (the supported path).
    before = _resident_aliases(gateway_base, timeout)
    alias = _pick_alias(before["aliases"], gateway_model)
    warm_via_gateway = None
    if alias is None:
        warm_via_gateway = _post(
            f"{gateway_base}/v1/embeddings",
            {"model": gateway_model, "input": "warmup"},
            timeout,
        )
        after = _resident_aliases(gateway_base, timeout)
        # The alias that just appeared is the one the gateway used; fall back to
        # the whole set if it was already resident.
        alias = _pick_alias(
            after["aliases"], gateway_model, exclude=tuple(before["aliases"])
        ) or _pick_alias(after["aliases"], gateway_model)
        before = after

    record["broker_aliases_before"] = before["aliases"]
    record["alias_used"] = alias
    record["warmup_via_gateway"] = (
        None
        if warm_via_gateway is None
        else {
            "http": warm_via_gateway["http"],
            "seconds": warm_via_gateway["seconds"],
            "transport_error": warm_via_gateway["transport_error"],
        }
    )

    if alias is None:
        record["first_call"] = None
        record["second_call"] = None
        record["verdict"] = "FAIL"
        record["reason"] = "no resident alias for the model; cannot exercise /v3/embeddings"
        return record

    # 4. First call.  Whether it is genuinely cold is decided by the broker's
    #    own state, not by assumption, so the report can say which it was.
    first_resident = alias in before["aliases"]
    first = _post(
        f"{ovms_base}/v3/embeddings",
        {"model": alias, "input": ["hello world"]},
        timeout,
    )
    first.update(_vector_stats(_vectors(first.pop("body", None))))
    first["resident_before_call"] = first_resident
    first["kind"] = "warm" if first_resident else "cold"
    record["first_call"] = first

    # 5. Second call -- a warm measurement by construction.
    second = _post(
        f"{ovms_base}/v3/embeddings",
        {"model": alias, "input": ["hello world"]},
        timeout,
    )
    second.update(_vector_stats(_vectors(second.pop("body", None))))
    second["resident_before_call"] = True
    second["kind"] = "warm"
    record["second_call"] = second

    record["broker_aliases_after"] = _resident_aliases(gateway_base, timeout)["aliases"]

    # 6. Cold vs warm, measured at the gateway (see _gateway_cold_warm).
    record["gateway_cold_warm"] = _gateway_cold_warm(
        gateway_base, gateway_model, alias, timeout
    )

    cold_warm = record["gateway_cold_warm"]
    cold_warm_ok = (
        isinstance(cold_warm.get("cold"), dict)
        and isinstance(cold_warm.get("warm"), dict)
        and cold_warm["cold"].get("http") == 200
        and cold_warm["warm"].get("http") == 200
        and (cold_warm["cold"].get("dim") or 0) > 0
        and cold_warm["cold"].get("all_finite") is True
    )

    ok = (
        first["http"] == 200
        and second["http"] == 200
        and first["dim"] > 0
        and second["dim"] > 0
        and first["all_finite"] is True
        and second["all_finite"] is True
        and cold_warm_ok
    )
    record["verdict"] = "PASS" if ok else "FAIL"
    if not ok:
        record["reason"] = (
            f"first http={first['http']} dim={first['dim']} finite={first['all_finite']}; "
            f"second http={second['http']} dim={second['dim']} finite={second['all_finite']}"
        )
    return record


def render_markdown(record: Dict[str, Any]) -> str:
    lines = [
        f"### `/v3/embeddings` (GenAI v3, independent gate) — {record.get('label', '')}".rstrip(),
        "",
        f"- gateway: `{record['gateway_base']}`  ·  OVMS: `{record['ovms_base']}`",
        f"- gateway model: `{record['gateway_model']}`  ·  OVMS alias exercised: `{record.get('alias_used')}`",
        f"- ambient proxy env trusted: `{record['trust_env_proxy']}`",
        f"- captured: {record['captured_at']}",
        "",
        "| probe | HTTP | seconds | dim | finite | norm | note |",
        "|---|---|---|---|---|---|---|",
    ]
    rows = [
        ("GET /v3/embeddings", record.get("get_endpoint"), None),
        ("POST /v3/embeddings (no model)", record.get("post_without_model"), None),
        (f"POST /v3/embeddings model={record.get('alias_used')} #1", record.get("first_call"), "cold" if not (record.get("first_call") or {}).get("resident_before_call") else "warm"),
        (f"POST /v3/embeddings model={record.get('alias_used')} #2", record.get("second_call"), "warm"),
    ]
    for name, row, note in rows:
        if not row:
            lines.append(f"| {name} | - | - | - | - | - | not run |")
            continue
        http = row.get("http")
        http_cell = "-" if http is None else str(http)
        error = row.get("transport_error")
        if error:
            note_cell = f"transport error: {error}"
        else:
            note_cell = note or ""
        lines.append(
            "| {n} | {h} | {s} | {d} | {f} | {norm} | {note} |".format(
                n=name,
                h=http_cell,
                s=row.get("seconds", "-"),
                d=row.get("dim", "-") if row.get("dim") is not None else "-",
                f=row.get("all_finite", "-"),
                norm=row.get("norm", "-"),
                note=note_cell,
            )
        )
    lines += [
        "",
        f"**Verdict: {record['verdict']}**" + (f" — {record['reason']}" if record.get("reason") else ""),
        "",
    ]

    cold_warm = record.get("gateway_cold_warm") or {}
    wait = cold_warm.get("waited_for_unload") or {}
    lines += [
        "#### Cold vs warm (measured at the gateway, where production sees it)",
        "",
        "A genuinely cold *OVMS-level* `/v3/embeddings` call is not reachable through",
        "the supported path: the broker owns the runtime config, so an idle alias is",
        "removed from `config.json` and OVMS would answer 404.  Cold start is",
        "therefore measured at the gateway, which cold-loads the alias on the first",
        "request.",
        "",
        f"- waited for idle unload: `{wait.get('unloaded')}` ({wait.get('waited_seconds')}s)",
        "",
        "| call | HTTP | seconds | dim | finite | norm |",
        "|---|---|---|---|---|---|",
    ]
    for key in ("cold", "warm"):
        row = cold_warm.get(key)
        if not row:
            lines.append(f"| {key} | - | - | - | - | - |")
            continue
        lines.append(
            "| {k} | {h} | {s} | {d} | {f} | {n} |".format(
                k=key,
                h=row.get("http", "-"),
                s=row.get("seconds", "-"),
                d=row.get("dim", "-"),
                f=row.get("all_finite", "-"),
                n=row.get("norm", "-"),
            )
        )
    if cold_warm.get("error"):
        lines += ["", f"> cold/warm measurement note: {cold_warm['error']}"]
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify OVMS /v3/embeddings (GenAI v3) on an acceptance stack."
    )
    parser.add_argument("--gateway-base", required=True, help="ai-gateway base URL")
    parser.add_argument("--ovms-base", required=True, help="OVMS REST base URL")
    parser.add_argument(
        "--gateway-model",
        default="qwen3-embedding-0.6b-int8",
        help="gateway registry model whose GenAI v3 backend uses /v3/embeddings",
    )
    parser.add_argument("--label", default="", help="label for the report")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--out", help="write the JSON capture here")
    parser.add_argument("--report", help="write the markdown fragment here")
    parser.add_argument(
        "--allow-inference-probes",
        action="store_true",
        help="required: a successful POST /v3/embeddings is a real inference",
    )
    parser.add_argument(
        "--use-env-proxy",
        action="store_true",
        help="honour HTTP_PROXY/HTTPS_PROXY from the environment (off by default)",
    )
    args = parser.parse_args(argv)

    set_env_proxy(args.use_env_proxy)
    if not args.use_env_proxy:
        ambient = [k for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY") if os.environ.get(k)]
        if ambient:
            print(f"[info] ignoring ambient proxy vars {ambient}; pass --use-env-proxy to honour them", flush=True)

    record = run_probe(
        gateway_base=args.gateway_base.rstrip("/"),
        ovms_base=args.ovms_base.rstrip("/"),
        gateway_model=args.gateway_model,
        timeout=args.timeout,
        allow_inference_probes=args.allow_inference_probes,
    )
    record["label"] = args.label

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    markdown = render_markdown(record)
    print(markdown)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(markdown, encoding="utf-8")

    return 0 if record["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
