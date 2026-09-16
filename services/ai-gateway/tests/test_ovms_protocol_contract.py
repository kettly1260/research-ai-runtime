"""CPU contract tests: the gateway against a real HTTP OVMS-shaped server.

The unit tests in ``test_ovms_protocol.py`` assert the adapter's in-process
behaviour.  These tests instead put a real HTTP server in front of
``ovms_predict()`` so the actual wire format is exercised end to end:

* the request really lands on ``/v1/models/{m}:predict`` or ``/v2/models/{m}/infer``
* the JSON on the socket really carries ``instances`` or shaped ``inputs``
* the response really is parsed back into one canonical vector

This is the CPU/protocol gate of the isolated-VM stage.  It proves protocol
correctness, not GPU acceptance -- the two are tracked separately.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

import research_ai_gateway.ovms_client as ovms_client
import research_ai_gateway.ovms_protocol as protocol_mod
from research_ai_gateway.inference import embeddings as embeddings_mod
from research_ai_gateway.ovms_protocol import reset_protocol_cache

MODEL = "contract-embed"


def _embed_rows(rows):
    """Deterministic pseudo-embedding so both protocols can be compared."""
    out = []
    for row in rows:
        length = len(row)
        mean = float(sum(row)) / length if length else 0.0
        out.append([mean, float(length)])
    return out


def _flatten(rows):
    flat = []
    for row in rows:
        flat.extend(row)
    return flat


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *args):  # keep the test output clean
        pass

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, status, text):
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _record(self, body):
        self.server.recorded.append({"method": self.command, "path": self.path, "body": body})

    def do_GET(self):
        self._record(None)
        if self.path == "/v1/config":
            if not self.server.tfs_config_present:
                return self._send_text(404, "not found")
            if self.server.config_empty:
                # Both OVMS 2026.1 and 2026.3.1 return {} here when no model is
                # resident.  The body therefore cannot discriminate a protocol.
                return self._send_json(200, {})
            return self._send_json(200, {"model_config_list": []})
        if self.path == "/v2/health/ready":
            return self._send_json(200, {"status": "ready"})
        if self.path.startswith("/v2/models/") and self.path.endswith("/ready"):
            if self.server.model_loaded:
                return self._send_json(200, {})
            return self._send_json(503, {"error": "model not loaded"})
        if self.path.startswith("/v2/models/"):
            if self.server.model_loaded:
                return self._send_json(200, {"name": MODEL})
            return self._send_text(404, "not found")
        if self.path.startswith("/v1/models/"):
            state = "AVAILABLE" if self.server.model_loaded else "LOADING"
            return self._send_json(200, {"model_version_status": [{"state": state}]})
        return self._send_text(404, "not found")

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode("utf-8")
        body = json.loads(raw) if raw else None
        self._record(body)

        if self.path.startswith("/v1/models/") and self.path.endswith(":predict"):
            if self.server.mediapipe_v1:
                # Reproduce the OVMS 2026.3 MediaPipe rejection verbatim.
                return self._send_text(
                    412,
                    "The file is not valid json - model field is missing in JSON body",
                )
            if not self.path.startswith(f"/v1/models/{MODEL}:predict"):
                # OVMS 2026.1 answers an unknown model with the Classic Model
                # registry error verbatim.  This is what the auto ladder keys on.
                return self._send_json(404, {"error": "Model with requested name is not found"})
            instances = body.get("instances") or []
            rows = [item["input_ids"] for item in instances]
            self.server.observed_lengths.extend(len(row) for row in rows)
            return self._send_json(200, {"predictions": _embed_rows(rows)})

        if self.path.startswith("/v2/models/") and self.path.endswith("/infer"):
            inputs = body.get("inputs") or []
            ids = next(item for item in inputs if item["name"] == "input_ids")
            batch, seq = ids["shape"]
            flat = ids["data"]
            rows = [flat[index * seq : (index + 1) * seq] for index in range(batch)]
            self.server.observed_lengths.extend(len(row) for row in rows)
            embeddings = _embed_rows(rows)
            return self._send_json(
                200,
                {
                    "model_name": MODEL,
                    "outputs": [
                        {
                            "name": "embeddings",
                            "shape": [batch, 2],
                            "datatype": "FP32",
                            "data": _flatten(embeddings),
                        }
                    ],
                },
            )

        return self._send_text(404, "not found")


def _start_server(**flags):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.recorded = []
    server.observed_lengths = []
    server.model_loaded = flags.get("model_loaded", True)
    server.tfs_config_present = flags.get("tfs_config_present", True)
    server.config_empty = flags.get("config_empty", False)
    server.mediapipe_v1 = flags.get("mediapipe_v1", False)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


@pytest.fixture
def fake_ovms(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    reset_protocol_cache()
    server = _start_server()
    monkeypatch.setattr(ovms_client, "OVMS_BASE", f"http://127.0.0.1:{server.server_port}")
    yield server
    server.shutdown()
    server.server_close()
    reset_protocol_cache()


def _payload(batch=2, length=3):
    return {
        "instances": [
            {"input_ids": list(range(index * length, (index + 1) * length))}
            for index in range(batch)
        ]
    }


# --------------------------------------------------------------------------- #
# Wire-format contract
# --------------------------------------------------------------------------- #

def test_tfs_contract_hits_v1_predict_and_returns_canonical_predictions(fake_ovms, monkeypatch):
    monkeypatch.setattr(protocol_mod, "OVMS_PROTOCOL", "tfs")
    payload = _payload()

    result = ovms_client.ovms_predict(MODEL, payload, timeout=10)

    assert [entry["path"] for entry in fake_ovms.recorded] == [f"/v1/models/{MODEL}:predict"]
    assert fake_ovms.recorded[0]["body"] == payload
    assert result["predictions"] == _embed_rows([row["input_ids"] for row in payload["instances"]])


def test_kserve_contract_hits_v2_infer_with_shaped_inputs(fake_ovms, monkeypatch):
    monkeypatch.setattr(protocol_mod, "OVMS_PROTOCOL", "kserve")

    result = ovms_client.ovms_predict(MODEL, _payload(), timeout=10)

    assert [entry["path"] for entry in fake_ovms.recorded] == [f"/v2/models/{MODEL}/infer"]
    sent = fake_ovms.recorded[0]["body"]
    assert sent == {
        "inputs": [
            {
                "name": "input_ids",
                "shape": [2, 3],
                "datatype": "INT64",
                "data": [0, 1, 2, 3, 4, 5],
            }
        ]
    }
    assert result["predictions"] == _embed_rows([[0, 1, 2], [3, 4, 5]])


def test_both_protocols_produce_identical_vectors_over_real_http(fake_ovms, monkeypatch):
    payload = _payload(batch=3, length=4)

    monkeypatch.setattr(protocol_mod, "OVMS_PROTOCOL", "tfs")
    tfs_result = ovms_client.ovms_predict(MODEL, payload, timeout=10)

    monkeypatch.setattr(protocol_mod, "OVMS_PROTOCOL", "kserve")
    kserve_result = ovms_client.ovms_predict(MODEL, payload, timeout=10)

    assert tfs_result["predictions"] == kserve_result["predictions"]


def test_auto_contract_detects_a_kserve_only_backend(monkeypatch):
    """Reproduce the real OVMS 2026.3.1 surface: /v1/config answers 200 {} and
    /v1/models/{m}:predict answers 412 with the MediaPipe marker."""
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    reset_protocol_cache()
    server = _start_server(config_empty=True, mediapipe_v1=True)
    monkeypatch.setattr(ovms_client, "OVMS_BASE", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setattr(protocol_mod, "OVMS_PROTOCOL", "auto")
    try:
        result = ovms_client.ovms_predict(MODEL, _payload(), timeout=10)
        assert result["predictions"]
        assert protocol_mod.protocol_diagnostics()["ovms_protocol_effective"] == "kserve"
        assert server.recorded[-1]["path"] == f"/v2/models/{MODEL}/infer"
    finally:
        server.shutdown()
        server.server_close()
        reset_protocol_cache()


def test_auto_contract_detects_a_2026_1_backend_in_the_zero_resident_state(monkeypatch):
    """The regression Gate 2 caught: an empty /v1/config body on a server that
    still serves the Classic Model REST API must resolve to tfs, not kserve."""
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    reset_protocol_cache()
    server = _start_server(config_empty=True, mediapipe_v1=False)
    monkeypatch.setattr(ovms_client, "OVMS_BASE", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setattr(protocol_mod, "OVMS_PROTOCOL", "auto")
    try:
        result = ovms_client.ovms_predict(MODEL, _payload(), timeout=10)
        assert result["predictions"]
        assert protocol_mod.protocol_diagnostics()["ovms_protocol_effective"] == "tfs"
        assert server.recorded[-1]["path"] == f"/v1/models/{MODEL}:predict"
    finally:
        server.shutdown()
        server.server_close()
        reset_protocol_cache()


def test_auto_contract_detects_a_mediapipe_backed_2026_3_server(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    reset_protocol_cache()
    server = _start_server(mediapipe_v1=True)
    monkeypatch.setattr(ovms_client, "OVMS_BASE", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setattr(protocol_mod, "OVMS_PROTOCOL", "auto")
    try:
        ovms_client.ovms_predict(MODEL, _payload(), timeout=10)
        assert protocol_mod.protocol_diagnostics()["ovms_protocol_effective"] == "kserve"
    finally:
        server.shutdown()
        server.server_close()
        reset_protocol_cache()


def test_kserve_availability_probe_tracks_load_state(fake_ovms, monkeypatch):
    monkeypatch.setattr(protocol_mod, "OVMS_PROTOCOL", "kserve")
    assert ovms_client.is_model_available(MODEL) is True

    fake_ovms.model_loaded = False
    assert ovms_client.is_model_available(MODEL) is False


# --------------------------------------------------------------------------- #
# Long text must survive the protocol switch (hard gate)
# --------------------------------------------------------------------------- #

class _FakeQwenTokenizer:
    """Minimal stand-in for Qwen2Tokenizer used by the pooled-IR long-text path."""

    pad_token_id = 0
    eos_token_id = 2

    def __init__(self, content_tokens):
        self._content_tokens = content_tokens

    def build_inputs_with_special_tokens(self, ids):
        return [1, *ids, 2]

    def __call__(
        self,
        text,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=True,
        **kwargs,
    ):
        ids = list(self._content_tokens)
        if add_special_tokens:
            ids = self.build_inputs_with_special_tokens(ids)
        payload = {"input_ids": ids}
        if return_attention_mask:
            payload["attention_mask"] = [1] * len(ids)
        return payload


def test_long_text_chunking_is_protocol_independent(fake_ovms, monkeypatch):
    """4591-token-class inputs must chunk identically under TFS and KServe.

    Asserts the three things the acceptance gate cares about at protocol level:
    one vector comes back, it is L2-normalised, and no individual backend
    request ever exceeded the configured pooled-IR window.
    """
    content_tokens = 2500
    tokenizer = _FakeQwenTokenizer(list(range(10, 10 + content_tokens)))
    monkeypatch.setattr(embeddings_mod, "get_embedding_tokenizer", lambda path=None: tokenizer)
    monkeypatch.setattr(embeddings_mod, "POOLED_CHUNK_BATCH_SIZE", 1)

    long_text = "x" * content_tokens
    window = embeddings_mod.POOLED_CHUNK_TOKENS

    monkeypatch.setattr(protocol_mod, "OVMS_PROTOCOL", "tfs")
    tfs_vectors = embeddings_mod.pooled_ir_response(MODEL, [long_text], MODEL, "/tokenizers/qwen")

    tfs_lengths = list(fake_ovms.observed_lengths)
    fake_ovms.observed_lengths.clear()

    monkeypatch.setattr(protocol_mod, "OVMS_PROTOCOL", "kserve")
    kserve_vectors = embeddings_mod.pooled_ir_response(MODEL, [long_text], MODEL, "/tokenizers/qwen")

    kserve_lengths = list(fake_ovms.observed_lengths)

    assert len(tfs_vectors["data"]) == 1
    assert len(kserve_vectors["data"]) == 1

    tfs_vector = np.asarray(tfs_vectors["data"][0]["embedding"], dtype=np.float32)
    kserve_vector = np.asarray(kserve_vectors["data"][0]["embedding"], dtype=np.float32)

    assert tfs_vector.shape == kserve_vector.shape
    np.testing.assert_allclose(tfs_vector, kserve_vector, rtol=1e-6)

    assert abs(float(np.linalg.norm(tfs_vector)) - 1.0) < 1e-6

    assert len(tfs_lengths) > 1, "the long text must have been chunked"
    assert max(tfs_lengths) <= window
    assert max(kserve_lengths) <= window
    assert tfs_lengths == kserve_lengths


def test_short_text_is_not_chunked_and_stays_a_single_request(fake_ovms, monkeypatch):
    tokenizer = _FakeQwenTokenizer(list(range(10, 40)))
    monkeypatch.setattr(embeddings_mod, "get_embedding_tokenizer", lambda path=None: tokenizer)
    monkeypatch.setattr(protocol_mod, "OVMS_PROTOCOL", "kserve")

    result = embeddings_mod.pooled_ir_response(MODEL, ["short"], MODEL, "/tokenizers/qwen")

    assert len(result["data"]) == 1
    assert len([entry for entry in fake_ovms.recorded if entry["method"] == "POST"]) == 1
    # 30 content tokens plus the tokenizer's [bos, eos] wrapper.
    assert fake_ovms.observed_lengths == [32]
