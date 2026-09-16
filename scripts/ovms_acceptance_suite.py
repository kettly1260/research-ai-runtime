#!/usr/bin/env python
"""Gate 2 / Gate 3 acceptance harness for the OVMS protocol migration.

Runs the same script against a 2026.1 (TFS) stack and a 2026.3.1 (KServe) stack
so the comparison is like-for-like, then emits machine-readable JSON plus a
markdown fragment that can be pasted into
`docs/OVMS_2026_1_VS_2026_3_ACCEPTANCE.md`.

What it measures
----------------
* capability surface (delegated to `ovms_protocol_capability_probe`)
* model matrix: every registry model, per protocol
* long text: 256 / 512 / 1024 / ~4591 tokens, single and concurrency=2
* the liveness gate: a short-text embedding immediately after every long round
* reranker correctness: ordering and 0..1 relevance probability
* zero-resident lifecycle: cold load, AVAILABLE, idle unload

What it deliberately does NOT do
--------------------------------
* It never writes to a runtime config directory; it only reads the gateway's
  own `/v1/broker/metrics`.
* It never restarts or reconfigures OVMS.  It is a client, nothing more.

Usage
-----
    python scripts/ovms_acceptance_suite.py \
        --gateway-base http://127.0.0.1:28012 \
        --ovms-base    http://127.0.0.1:28342 \
        --protocol kserve --label ovms-2026.3.1 \
        --out .workbuddy-ai/artifacts/acceptance-20263.json \
        --report .workbuddy-ai/artifacts/acceptance-20263.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

# --------------------------------------------------------------------------- #
# Transport
# --------------------------------------------------------------------------- #

#: ``requests`` honours ``HTTP_PROXY``/``HTTPS_PROXY`` from the environment.
#: Dev shells and CI runners frequently export them (this machine exports
#: ``HTTP_PROXY=http://127.0.0.1:<port>``), and the proxy then intercepts calls
#: to ``127.0.0.1``.  The symptom is a *ReadTimeout against the proxy's port*
#: instead of a ConnectionError against the target -- i.e. the harness would be
#: benchmarking the proxy.  Env trust is therefore off by default and must be
#: requested explicitly with ``--use-env-proxy``.
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

from ovms_protocol_capability_probe import render_markdown as render_capability_markdown  # noqa: E402
from ovms_protocol_capability_probe import run_probe  # noqa: E402

# --------------------------------------------------------------------------- #
# Matrix definition
# --------------------------------------------------------------------------- #

#: Models the gateway can address, with the request kind that exercises them.
#: `required` mirrors task item 20: a model that exists in production is a hard
#: gate, a model with no deployed production path is recorded as not applicable.
MODEL_MATRIX: List[Dict[str, Any]] = [
    {"model": "qwen3-embedding-0.6b-int4", "kind": "embedding", "required": True},
    {"model": "qwen3-embedding-0.6b-int8", "kind": "embedding", "required": True},
    {"model": "bge-m3-i8", "kind": "embedding", "required": True},
    {"model": "bge-m3", "kind": "embedding", "required": True},
    {"model": "arctic-embed-m-v2-int8", "kind": "embedding", "required": True},
    {"model": "qwen-reranker", "kind": "rerank", "required": True},
    {
        "model": "DINO",
        "kind": "image_embedding",
        "required": False,
        "note": "no production OVMS deployment (official DINOv3 blocked on Meta gated weights)",
    },
    {
        "model": "multimodal Classic IR",
        "kind": "image_embedding",
        "required": False,
        "note": "export-only track; not present in the production registry",
    },
]

LONG_TEXT_LENGTHS = [256, 512, 1024, 4591]
LONG_TEXT_MODEL = "qwen3-embedding-0.6b-int4"

RERANK_QUERY = "what is biology?"
RERANK_DOCS = [
    "Biology is the study of living organisms.",
    "The Eiffel Tower is in Paris.",
]

SHORT_TEXT = "liveness probe"

FILLER = "biology is the study of living organisms and their interactions "


class Counters:
    def __init__(self) -> None:
        self.http_5xx = 0
        self.exceptions = 0

    def note_status(self, status: Optional[int]) -> None:
        if status is not None and status >= 500:
            self.http_5xx += 1

    def note_exception(self) -> None:
        self.exceptions += 1

    def as_dict(self) -> Dict[str, int]:
        return {"http_5xx": self.http_5xx, "exceptions": self.exceptions}


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #

def _timed_post(url: str, payload: dict, timeout: float, counters: Counters) -> Tuple[Optional[int], Any, float]:
    started = time.monotonic()
    try:
        response = _session().post(url, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        counters.note_exception()
        return None, f"{type(exc).__name__}: {exc}", time.monotonic() - started
    elapsed = time.monotonic() - started
    counters.note_status(response.status_code)
    try:
        return response.status_code, response.json(), elapsed
    except ValueError:
        return response.status_code, response.text, elapsed


def _timed_get(url: str, timeout: float, counters: Counters) -> Tuple[Optional[int], Any, float]:
    started = time.monotonic()
    try:
        response = _session().get(url, timeout=timeout)
    except requests.RequestException as exc:
        counters.note_exception()
        return None, f"{type(exc).__name__}: {exc}", time.monotonic() - started
    elapsed = time.monotonic() - started
    counters.note_status(response.status_code)
    try:
        return response.status_code, response.json(), elapsed
    except ValueError:
        return response.status_code, response.text, elapsed


def _first_vector(payload: Any) -> List[float]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return []
    entry = data[0]
    if isinstance(entry, dict):
        vector = entry.get("embedding")
        return list(vector) if isinstance(vector, list) else []
    return []


def _vector_stats(vector: List[float]) -> Tuple[int, Optional[float]]:
    if not vector:
        return 0, None
    norm = sum(float(value) ** 2 for value in vector) ** 0.5
    return len(vector), round(norm, 6)


def embed(gateway_base: str, model: str, text: str, timeout: float, counters: Counters) -> Dict[str, Any]:
    status, payload, elapsed = _timed_post(
        f"{gateway_base}/v1/embeddings",
        {"model": model, "input": text},
        timeout,
        counters,
    )
    vector = _first_vector(payload)
    dim, norm = _vector_stats(vector)
    usage = payload.get("usage") if isinstance(payload, dict) else None
    prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    return {
        "http": status,
        "seconds": round(elapsed, 3),
        "dim": dim,
        "norm": norm,
        "prompt_tokens": prompt_tokens,
        "error": None if status == 200 and dim > 0 else _summarise_error(payload),
    }


def _summarise_error(payload: Any) -> Optional[str]:
    if isinstance(payload, dict) and payload.get("detail"):
        return str(payload["detail"])[:300]
    if isinstance(payload, str):
        return payload[:300]
    return None


def calibrate_text(gateway_base: str, model: str, target_tokens: int, timeout: float, counters: Counters) -> Tuple[str, Optional[int]]:
    """Grow a filler text until the gateway reports about `target_tokens` tokens.

    The gateway returns real `usage.prompt_tokens`, so the harness measures the
    token count instead of assuming a chars-per-token ratio.
    """
    repeats = max(4, target_tokens // 10)
    text = FILLER * repeats
    actual: Optional[int] = None
    for _ in range(3):
        probe = embed(gateway_base, model, text, timeout, counters)
        actual = probe.get("prompt_tokens")
        if not actual:
            return text, None
        if abs(actual - target_tokens) <= max(16, target_tokens * 0.05):
            return text, actual
        scale = target_tokens / actual
        repeats = max(1, int(round(repeats * scale)))
        text = FILLER * repeats
    return text, actual


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #

def run_model_matrix(gateway_base: str, timeout: float, counters: Counters) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for entry in MODEL_MATRIX:
        model = entry["model"]
        if not entry["required"]:
            rows.append(
                {
                    "model": model,
                    "kind": entry["kind"],
                    "required": False,
                    "status": "N/A",
                    "note": entry.get("note", ""),
                }
            )
            continue

        if entry["kind"] == "rerank":
            status, payload, elapsed = _timed_post(
                f"{gateway_base}/v1/rerank",
                {"model": model, "query": RERANK_QUERY, "documents": RERANK_DOCS},
                timeout,
                counters,
            )
            results = payload.get("results") if isinstance(payload, dict) else None
            ok = status == 200 and isinstance(results, list) and len(results) == len(RERANK_DOCS)
            rows.append(
                {
                    "model": model,
                    "kind": "rerank",
                    "required": True,
                    "status": "PASS" if ok else "FAIL",
                    "http": status,
                    "seconds": round(elapsed, 3),
                    "dim": None,
                    "norm": None,
                    "error": None if ok else _summarise_error(payload),
                }
            )
            continue

        outcome = embed(gateway_base, model, "acceptance matrix probe", timeout, counters)
        ok = outcome["http"] == 200 and outcome["dim"] > 0 and outcome["norm"] is not None
        rows.append(
            {
                "model": model,
                "kind": entry["kind"],
                "required": True,
                "status": "PASS" if ok else "FAIL",
                **outcome,
            }
        )
    return rows


def run_long_text(
    gateway_base: str,
    timeout: float,
    counters: Counters,
    lengths: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for target in lengths or LONG_TEXT_LENGTHS:
        text, actual_tokens = calibrate_text(gateway_base, LONG_TEXT_MODEL, target, timeout, counters)

        single = embed(gateway_base, LONG_TEXT_MODEL, text, timeout, counters)
        single_liveness = embed(gateway_base, LONG_TEXT_MODEL, SHORT_TEXT, timeout, counters)

        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(embed, gateway_base, LONG_TEXT_MODEL, text, timeout, counters)
                for _ in range(2)
            ]
            concurrent = [future.result() for future in futures]
        wall = time.monotonic() - started
        concurrent_liveness = embed(gateway_base, LONG_TEXT_MODEL, SHORT_TEXT, timeout, counters)

        rows.append(
            {
                "target_tokens": target,
                "actual_tokens": actual_tokens,
                "single": single,
                "single_liveness": single_liveness,
                "concurrency": 2,
                "concurrent": concurrent,
                "wall_seconds": round(wall, 3),
                "concurrent_liveness": concurrent_liveness,
            }
        )
    return rows


def run_reranker(gateway_base: str, timeout: float, counters: Counters) -> Dict[str, Any]:
    status, payload, elapsed = _timed_post(
        f"{gateway_base}/v1/rerank",
        {"model": "qwen-reranker", "query": RERANK_QUERY, "documents": RERANK_DOCS},
        timeout,
        counters,
    )
    result: Dict[str, Any] = {
        "http": status,
        "seconds": round(elapsed, 3),
        "scores": {},
        "ordering_ok": False,
        "range_ok": False,
        "error": None,
    }
    if status != 200 or not isinstance(payload, dict):
        result["error"] = _summarise_error(payload)
        return result

    scored: Dict[str, float] = {}
    for item in payload.get("results", []):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or item.get("document") or "")
        try:
            scored[text] = float(item.get("score"))
        except (TypeError, ValueError):
            continue
    result["scores"] = scored

    biology = next((v for k, v in scored.items() if "Biology is the study" in k), None)
    eiffel = next((v for k, v in scored.items() if "Eiffel" in k), None)
    if biology is not None and eiffel is not None:
        result["ordering_ok"] = biology > eiffel
        result["range_ok"] = all(0.0 <= value <= 1.0 for value in scored.values())
        result["biology_score"] = biology
        result["eiffel_score"] = eiffel
    else:
        result["error"] = "reranker response did not contain both expected documents"
    return result


def run_zero_resident(
    gateway_base: str,
    timeout: float,
    counters: Counters,
    unload_timeout: float = 900.0,
    poll_seconds: float = 5.0,
) -> Dict[str, Any]:
    """Observe cold load -> AVAILABLE -> idle unload through the broker metrics."""
    metrics_url = f"{gateway_base}/v1/broker/metrics"

    def loaded_models() -> List[str]:
        status, payload, _ = _timed_get(metrics_url, min(timeout, 15), counters)
        if status != 200 or not isinstance(payload, dict):
            return []
        loaded = payload.get("loaded_models")
        if not isinstance(loaded, dict):
            return []
        return sorted(name for name, entry in loaded.items() if isinstance(entry, dict) and entry.get("loaded"))

    result: Dict[str, Any] = {
        "initial_loaded": loaded_models(),
        "after_request": [],
        "unloaded": False,
        "unload_seconds": None,
        "error": None,
    }

    embed(gateway_base, LONG_TEXT_MODEL, SHORT_TEXT, timeout, counters)
    result["after_request"] = loaded_models()
    if not result["after_request"]:
        result["error"] = "broker reported no loaded model after a successful embedding"
        return result

    deadline = time.monotonic() + unload_timeout
    started = time.monotonic()
    while time.monotonic() < deadline:
        if not loaded_models():
            result["unloaded"] = True
            result["unload_seconds"] = round(time.monotonic() - started, 3)
            break
        time.sleep(poll_seconds)
    if not result["unloaded"] and result["error"] is None:
        result["error"] = f"model still resident after {unload_timeout}s"
    return result


# --------------------------------------------------------------------------- #
# Gate evaluation and reporting (pure)
# --------------------------------------------------------------------------- #

def _gate(applicable: bool, ok: bool) -> str:
    if not applicable:
        return "N/A"
    return "PASS" if ok else "FAIL"


def _fmt(value: Any, dash: str = "-") -> Any:
    """Render a missing measurement as a dash instead of ``None``."""
    return dash if value is None else value


def _embedding_rows(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = [row for row in results.get("model_matrix", []) if row.get("dim") is not None]
    for entry in results.get("long_text", []):
        for candidate in [entry.get("single"), *entry.get("concurrent", [])]:
            if isinstance(candidate, dict) and candidate.get("dim") is not None:
                rows.append(candidate)
    return rows


def evaluate_gates(results: Dict[str, Any]) -> Dict[str, str]:
    """Turn raw measurements into the verdict table required by the task."""
    matrix = results.get("model_matrix", [])
    required_rows = [row for row in matrix if row.get("required")]

    long_text = results.get("long_text", [])
    liveness_ok = bool(long_text) and all(
        (entry.get("single_liveness") or {}).get("http") == 200
        and (entry.get("concurrent_liveness") or {}).get("http") == 200
        for entry in long_text
    )
    concurrency_ok = bool(long_text) and all(
        all((item or {}).get("http") == 200 for item in entry.get("concurrent", []))
        and len(entry.get("concurrent", [])) == 2
        for entry in long_text
    )

    embedding_rows = _embedding_rows(results)
    correctness_ok = bool(embedding_rows) and all(
        row.get("dim", 0) > 0 and row.get("norm") is not None and abs(float(row["norm"]) - 1.0) < 1e-3
        for row in embedding_rows
    )

    reranker = results.get("reranker") or {}
    zero_resident = results.get("zero_resident") or {}
    counters = results.get("counters") or {}

    return {
        "Compatibility": _gate(bool(required_rows), bool(required_rows) and all(row.get("status") == "PASS" for row in required_rows)),
        "Correctness": _gate(bool(embedding_rows), correctness_ok),
        "Long-text liveness": _gate(bool(long_text), liveness_ok),
        "Concurrency": _gate(bool(long_text), concurrency_ok),
        "Reranker": _gate(bool(reranker), bool(reranker.get("ordering_ok")) and bool(reranker.get("range_ok"))),
        "Zero-resident": _gate(bool(zero_resident), bool(zero_resident.get("after_request")) and bool(zero_resident.get("unloaded"))),
        "Failure counters": _gate(True, counters.get("http_5xx", 0) == 0 and counters.get("exceptions", 0) == 0),
    }


def render_report(results: Dict[str, Any]) -> str:
    label = results.get("label") or "unlabelled"
    lines = [
        f"## Acceptance run — {label}",
        "",
        f"* gateway: `{results.get('gateway_base')}`",
        f"* OVMS: `{results.get('ovms_base')}`",
        f"* protocol: `{results.get('protocol')}`",
        f"* captured at: `{results.get('captured_at')}`",
        "",
        "### Verdict",
        "",
        "| Dimension | Result |",
        "| --- | --- |",
    ]
    for dimension, verdict in evaluate_gates(results).items():
        lines.append(f"| {dimension} | **{verdict}** |")

    lines += ["", "### Model matrix", "", "| Model | Kind | Required | Status | HTTP | Dim | Norm | Note |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in results.get("model_matrix", []):
        lines.append(
            "| {model} | {kind} | {required} | **{status}** | {http} | {dim} | {norm} | {note} |".format(
                model=row.get("model"),
                kind=row.get("kind"),
                required="yes" if row.get("required") else "no",
                status=row.get("status"),
                http=_fmt(row.get("http")),
                dim=_fmt(row.get("dim")),
                norm=_fmt(row.get("norm")),
                note=(row.get("note") or row.get("error") or "")[:80],
            )
        )

    lines += ["", "### Long text", "", "| Target tok | Actual tok | Single | Dim | Norm | Live | Concurrency=2 wall | Live |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for entry in results.get("long_text", []):
        single = entry.get("single") or {}
        lines.append(
            "| {target} | {actual} | {http} / {seconds}s | {dim} | {norm} | {live} | {wall}s | {clive} |".format(
                target=entry.get("target_tokens"),
                actual=_fmt(entry.get("actual_tokens")),
                http=_fmt(single.get("http")),
                seconds=_fmt(single.get("seconds")),
                dim=_fmt(single.get("dim")),
                norm=_fmt(single.get("norm")),
                live=_fmt((entry.get("single_liveness") or {}).get("http")),
                wall=_fmt(entry.get("wall_seconds")),
                clive=_fmt((entry.get("concurrent_liveness") or {}).get("http")),
            )
        )

    reranker = results.get("reranker") or {}
    lines += [
        "",
        "### Reranker",
        "",
        f"* biology score: `{_fmt(reranker.get('biology_score'))}`",
        f"* Eiffel score: `{_fmt(reranker.get('eiffel_score'))}`",
        f"* ordering (biology > Eiffel): `{reranker.get('ordering_ok')}`",
        f"* all scores within 0..1: `{reranker.get('range_ok')}`",
        "",
        "### Zero-resident",
        "",
        f"* loaded before request: `{_fmt(results.get('zero_resident', {}).get('initial_loaded'))}`",
        f"* loaded after request: `{_fmt(results.get('zero_resident', {}).get('after_request'))}`",
        f"* idle unload observed: `{results.get('zero_resident', {}).get('unloaded')}` "
        f"({_fmt(results.get('zero_resident', {}).get('unload_seconds'))}s)",
        "",
        "### Counters",
        "",
        f"* HTTP 5xx: `{_fmt(results.get('counters', {}).get('http_5xx'))}`",
        f"* client exceptions: `{_fmt(results.get('counters', {}).get('exceptions'))}`",
    ]

    if results.get("capability"):
        lines += ["", render_capability_markdown(results["capability"])]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

ALL_SECTIONS = ["capability", "matrix", "long_text", "reranker", "zero_resident"]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gateway-base", required=True)
    parser.add_argument("--ovms-base", required=True)
    parser.add_argument("--protocol", required=True, choices=["tfs", "kserve"])
    parser.add_argument("--label", default=None)
    parser.add_argument("--model", default="qwen-reranker", help="model used for capability probes")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--sections", default=",".join(ALL_SECTIONS))
    parser.add_argument("--out", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--allow-inference-probes", action="store_true")
    parser.add_argument(
        "--use-env-proxy",
        action="store_true",
        help="honour HTTP_PROXY/HTTPS_PROXY (off by default: loopback runs must be direct)",
    )
    args = parser.parse_args(argv)

    set_env_proxy(args.use_env_proxy)
    if not args.use_env_proxy:
        ambient = [name for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY") if os.getenv(name)]
        if ambient:
            print(
                f"[info] ignoring ambient proxy vars {ambient}; pass --use-env-proxy to honour them",
                file=sys.stderr,
            )

    sections = [item.strip() for item in args.sections.split(",") if item.strip()]
    unknown = [item for item in sections if item not in ALL_SECTIONS]
    if unknown:
        parser.error(f"unknown sections: {unknown}")

    counters = Counters()
    results: Dict[str, Any] = {
        "label": args.label,
        "protocol": args.protocol,
        "gateway_base": args.gateway_base,
        "ovms_base": args.ovms_base,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }

    if "capability" in sections:
        capture = run_probe(
            args.ovms_base,
            args.model,
            min(args.timeout, 15.0),
            args.allow_inference_probes,
            use_env_proxy=args.use_env_proxy,
        )
        capture["label"] = args.label
        results["capability"] = capture
    if "matrix" in sections:
        results["model_matrix"] = run_model_matrix(args.gateway_base, args.timeout, counters)
    if "long_text" in sections:
        results["long_text"] = run_long_text(args.gateway_base, args.timeout, counters)
    if "reranker" in sections:
        results["reranker"] = run_reranker(args.gateway_base, args.timeout, counters)
    if "zero_resident" in sections:
        results["zero_resident"] = run_zero_resident(args.gateway_base, args.timeout, counters)

    results["counters"] = counters.as_dict()
    results["gates"] = evaluate_gates(results)

    report = render_report(results)
    print(report)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[ok] wrote {out_path}", file=sys.stderr)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        print(f"[ok] wrote {report_path}", file=sys.stderr)

    return 0 if all(verdict != "FAIL" for verdict in results["gates"].values()) else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
