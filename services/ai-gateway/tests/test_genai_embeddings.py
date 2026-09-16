"""GenAI v3 (graph-backed) embedding parity tests.

Covers the ported legacy behaviour:

* OVMS ``/v3/embeddings`` payload construction and error mapping;
* response normalization into the gateway's OpenAI-compatible shape;
* the adaptive GenAI batcher (policy, lease usage, cross-request merging);
* ``/v1/embeddings`` dispatch for ``embedding_backend == "genai_v3"``;
* ``/v1/embedding-batch-stats`` following the configured backend.

A live test against a real OVMS instance is included and is skipped unless
``OVMS_GENAI_TEST_BASE`` is set, e.g.

    OVMS_GENAI_TEST_BASE=http://127.0.0.1:29191 \
    OVMS_GENAI_TEST_MODEL=qwen3-embedding-0.6b \
    pytest services/ai-gateway/tests/test_genai_embeddings.py -v
"""

import os
import threading

import numpy as np
import pytest
import requests
from fastapi import HTTPException
from fastapi.testclient import TestClient

import research_ai_gateway.ovms_client as ovms_mod
from research_ai_gateway.inference import embeddings as embeddings_mod


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

class _FakeLease:
    def __init__(self, owner, alias):
        self.owner = owner
        self.alias = alias

    def __enter__(self):
        self.owner.lease_active = True
        return self.alias

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.owner.lease_active = False


class _FakeBroker:
    def __init__(self, alias="qwen3-embedding-0.6b__gpu"):
        self.alias = alias
        self.embedding_inference_lock = threading.RLock()
        self.lease_active = False
        self.lease_calls = []

    def lease(self, model_name, preferred_device="GPU"):
        self.lease_calls.append((model_name, preferred_device))
        return _FakeLease(self, self.alias)


class _FakeGenAITokenizer:
    """Deterministic stand-in: one token per whitespace-separated word."""

    def __call__(self, text, add_special_tokens=True, truncation=False, return_attention_mask=False):
        words = [w for w in str(text or "").split() if w]
        return {"input_ids": [0] * max(1, len(words))}


class _FakePooledTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(
        self,
        text,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=False,
    ):
        words = [w for w in str(text or "").split() if w]
        ids = list(range(10, 10 + len(words)))
        if add_special_tokens:
            ids = self.build_inputs_with_special_tokens(ids)
        payload = {"input_ids": ids}
        if return_attention_mask:
            payload["attention_mask"] = [1] * len(ids)
        return payload

    def build_inputs_with_special_tokens(self, token_ids):
        return [1, *list(token_ids), 2]


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            error = requests.exceptions.HTTPError(f"HTTP {self.status_code}")
            error.response = self
            raise error

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def _ovms_genai_payload(model, texts, dim=3):
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": idx, "embedding": [float(idx + 1)] * dim}
            for idx in range(len(texts))
        ],
        "model": model,
        "usage": {"prompt_tokens": len(texts), "total_tokens": len(texts)},
    }


# --------------------------------------------------------------------------
# response normalization
# --------------------------------------------------------------------------

def test_normalize_genai_response_replaces_model_and_shape():
    raw = _ovms_genai_payload("qwen3-embedding-0.6b", ["a", "b"])
    result = embeddings_mod.normalize_genai_embeddings_response(raw, "qwen3-embedding-0.6b-int8")

    assert result["object"] == "list"
    assert result["model"] == "qwen3-embedding-0.6b-int8"
    assert [item["index"] for item in result["data"]] == [0, 1]
    assert all(item["object"] == "embedding" for item in result["data"])
    assert result["usage"] == {"prompt_tokens": 2, "total_tokens": 2}


def test_normalize_genai_response_sorts_by_index():
    raw = {
        "data": [
            {"index": 2, "embedding": [2.0]},
            {"index": 0, "embedding": [0.0]},
            {"index": 1, "embedding": [1.0]},
        ]
    }
    result = embeddings_mod.normalize_genai_embeddings_response(raw, "logical")
    assert [item["index"] for item in result["data"]] == [0, 1, 2]
    assert [item["embedding"] for item in result["data"]] == [[0.0], [1.0], [2.0]]


def test_normalize_genai_response_usage_fallback():
    result = embeddings_mod.normalize_genai_embeddings_response({"data": [{"embedding": [1.0]}]}, "m")
    assert result["usage"] == {"prompt_tokens": 0, "total_tokens": 0}
    assert result["data"][0]["index"] == 0


@pytest.mark.parametrize(
    "payload, detail_fragment",
    [
        (["not", "an", "object"], "non-object JSON"),
        ({}, "empty genai embedding response"),
        ({"data": []}, "empty genai embedding response"),
        ({"data": ["nope"]}, "unexpected genai embedding item type"),
        ({"data": [{"index": 0}]}, "missing a usable embedding vector"),
        ({"data": [{"index": 0, "embedding": []}]}, "missing a usable embedding vector"),
        ({"data": [{"index": 0, "embedding": ["x"]}]}, "non-numeric values"),
    ],
)
def test_normalize_genai_response_rejects_malformed_payloads(payload, detail_fragment):
    with pytest.raises(HTTPException) as excinfo:
        embeddings_mod.normalize_genai_embeddings_response(payload, "m")
    assert excinfo.value.status_code == 502
    assert detail_fragment in excinfo.value.detail


# --------------------------------------------------------------------------
# client: payload + error mapping
# --------------------------------------------------------------------------

def test_genai_embeddings_payload_and_url(monkeypatch):
    seen = {}

    def fake_post(url, json=None, timeout=None):
        seen["url"] = url
        seen["json"] = json
        seen["timeout"] = timeout
        return _FakeResponse(payload=_ovms_genai_payload(json["model"], json["input"]))

    monkeypatch.setattr(ovms_mod, "OVMS_BASE", "http://ovms-server:8001")
    monkeypatch.setattr(ovms_mod.requests, "post", fake_post)

    result = ovms_mod.ovms_genai_embeddings("qwen3-embedding-0.6b", ["hello", "world"])

    assert seen["url"] == "http://ovms-server:8001/v3/embeddings"
    assert seen["json"] == {
        "model": "qwen3-embedding-0.6b",
        "input": ["hello", "world"],
        "encoding_format": "float",
    }
    assert len(result["data"]) == 2


def test_genai_embeddings_timeout_maps_to_504(monkeypatch):
    def fake_post(url, json=None, timeout=None):
        raise requests.exceptions.Timeout("too slow")

    monkeypatch.setattr(ovms_mod.requests, "post", fake_post)
    with pytest.raises(HTTPException) as excinfo:
        ovms_mod.ovms_genai_embeddings("m", ["a"])
    assert excinfo.value.status_code == 504


def test_genai_embeddings_http_error_maps_to_502(monkeypatch):
    def fake_post(url, json=None, timeout=None):
        return _FakeResponse(status_code=500, text="boom")

    monkeypatch.setattr(ovms_mod.requests, "post", fake_post)
    with pytest.raises(HTTPException) as excinfo:
        ovms_mod.ovms_genai_embeddings("m", ["a"])
    assert excinfo.value.status_code == 502
    assert "boom" in excinfo.value.detail


def test_genai_embeddings_invalid_json_maps_to_502(monkeypatch):
    def fake_post(url, json=None, timeout=None):
        return _FakeResponse(payload=None)

    monkeypatch.setattr(ovms_mod.requests, "post", fake_post)
    with pytest.raises(HTTPException) as excinfo:
        ovms_mod.ovms_genai_embeddings("m", ["a"])
    assert excinfo.value.status_code == 502
    assert "invalid JSON" in excinfo.value.detail


# --------------------------------------------------------------------------
# adaptive GenAI batcher
# --------------------------------------------------------------------------

def test_genai_batcher_policy_boundaries():
    policy = embeddings_mod.AdaptiveGenAIBatcher._policy
    assert policy(1)[0] == "le256" and policy(1)[1] == 4
    assert policy(256)[0] == "le256"
    assert policy(257)[0] == "257_384" and policy(257)[1] == 2
    assert policy(384)[0] == "257_384"
    assert policy(385)[0] == "385_512" and policy(385)[2] == 0.0
    assert policy(512)[0] == "385_512"
    assert policy(513)[0] == "gt512" and policy(513)[1] == 1


def test_genai_batcher_merges_concurrent_short_texts(monkeypatch):
    monkeypatch.setattr(embeddings_mod, "get_embedding_tokenizer", lambda path: _FakeGenAITokenizer())
    # Widen the coalescing window so the assertion is deterministic.
    monkeypatch.setattr(embeddings_mod, "ADAPTIVE_BATCH_WAIT_MS", 200.0)

    fake_broker = _FakeBroker()
    batches = []
    lock = threading.Lock()

    def fake_call(model_name, raw_input, response_model_name):
        texts = raw_input if isinstance(raw_input, list) else [raw_input]
        with lock:
            batches.append(list(texts))
        return {
            "object": "list",
            "data": [
                {"object": "embedding", "index": idx, "embedding": [float(len(t))]}
                for idx, t in enumerate(texts)
            ],
            "model": response_model_name,
            "usage": {"prompt_tokens": len(texts), "total_tokens": len(texts)},
        }

    monkeypatch.setattr(embeddings_mod, "call_genai_embedding", fake_call)

    batcher = embeddings_mod.AdaptiveGenAIBatcher(broker=fake_broker)
    results = {}

    def submit(key, text):
        results[key] = batcher.submit("qwen3-embedding-0.6b", "qwen3-embedding-0.6b-int8", text, "/tok")

    threads = [threading.Thread(target=submit, args=(i, f"text number {i}")) for i in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert len(results) == 3
    assert max(len(batch) for batch in batches) >= 2, batches
    # Every item keeps its own single-vector response shape.
    for result in results.values():
        assert result["object"] == "list"
        assert len(result["data"]) == 1
        assert result["data"][0]["index"] == 0
        assert result["model"] == "qwen3-embedding-0.6b-int8"
    # The lease was taken for the concrete alias while the backend ran.
    assert fake_broker.lease_calls
    assert fake_broker.lease_active is False

    stats = batcher.stats()
    assert stats["backend"] == "genai_v3"
    assert stats["submitted_items"] == 3
    assert stats["backend_items"] == 3
    assert stats["errors"] == 0
    assert set(stats["policy"]) == {"<=256", "257-384", "385-512", ">512"}


def test_genai_batcher_reports_backend_errors(monkeypatch):
    monkeypatch.setattr(embeddings_mod, "get_embedding_tokenizer", lambda path: _FakeGenAITokenizer())

    def failing_call(model_name, raw_input, response_model_name):
        raise HTTPException(status_code=502, detail="backend exploded")

    monkeypatch.setattr(embeddings_mod, "call_genai_embedding", failing_call)

    batcher = embeddings_mod.AdaptiveGenAIBatcher(broker=None)
    with pytest.raises(HTTPException) as excinfo:
        batcher.submit("m", "logical", "some text", "/tok")
    assert excinfo.value.status_code == 502
    assert batcher.stats()["errors"] == 1


def test_pooled_ir_long_text_is_windowed_and_merged(monkeypatch):
    monkeypatch.setattr(embeddings_mod, "get_embedding_tokenizer", lambda path: _FakePooledTokenizer())
    monkeypatch.setattr(embeddings_mod, "POOLED_CHUNK_TOKENS", 8)
    monkeypatch.setattr(embeddings_mod, "POOLED_CHUNK_OVERLAP", 2)
    monkeypatch.setattr(embeddings_mod, "POOLED_MAX_TOTAL_TOKENS", 100)
    monkeypatch.setattr(embeddings_mod, "POOLED_CHUNK_BATCH_SIZE", 1)

    calls = []

    def fake_predict(model_name, prepared, tokenizer_path):
        assert len(prepared) == 1
        ids, mask = prepared[0]
        calls.append((len(ids), int(np.asarray(mask).sum())))
        idx = float(len(calls))
        return np.asarray([[idx, 1.0]], dtype=np.float32), int(np.asarray(mask).sum())

    monkeypatch.setattr(embeddings_mod, "predict_pooled_ir", fake_predict)

    text = "one two three four five six seven eight nine ten"
    result = embeddings_mod.pooled_ir_response(
        "qwen3-embedding-0.6b-int4-pooled",
        [text],
        "qwen3-embedding-0.6b-int4",
        "/tok",
    )

    assert len(calls) == 2
    assert all(length <= 8 for length, _ in calls)
    assert result["usage"] == {"prompt_tokens": 12, "total_tokens": 12}
    vector = np.asarray(result["data"][0]["embedding"], dtype=np.float32)
    assert vector.shape == (2,)
    assert np.isclose(np.linalg.norm(vector), 1.0)
    expected = np.asarray([1.5, 1.0], dtype=np.float32)
    expected /= np.linalg.norm(expected)
    assert np.allclose(vector, expected)


def test_adaptive_pooled_long_text_bypasses_worker(monkeypatch):
    monkeypatch.setattr(embeddings_mod, "POOLED_CHUNK_TOKENS", 8)
    monkeypatch.setattr(
        embeddings_mod,
        "_tokenize_qwen_text",
        lambda text, path: (
            np.arange(9, dtype=np.int64),
            np.ones(9, dtype=np.int64),
        ),
    )
    seen = {}

    def fake_long_response(model_name, texts, response_model_name, tokenizer_path, broker=None):
        seen.update(
            model_name=model_name,
            texts=texts,
            response_model_name=response_model_name,
            tokenizer_path=tokenizer_path,
            broker=broker,
        )
        return {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [1.0, 0.0]}],
            "model": response_model_name,
            "usage": {"prompt_tokens": 9, "total_tokens": 9},
        }

    monkeypatch.setattr(embeddings_mod, "pooled_ir_response", fake_long_response)
    fake_broker = _FakeBroker(alias="qwen3-embedding-0.6b-int4-pooled__gpu")
    batcher = embeddings_mod.AdaptivePooledIRBatcher(broker=fake_broker)

    result = batcher.submit("pooled", "logical", "a b c d e f g h i", "/tok")

    assert result["data"][0]["embedding"] == [1.0, 0.0]
    assert seen["texts"] == ["a b c d e f g h i"]
    assert seen["broker"] is fake_broker
    stats = batcher.stats()
    assert stats["long_text_requests"] == 1
    assert stats["submitted_items"] == 0
    assert stats["backend_batches"] == 0


# --------------------------------------------------------------------------
# route dispatch
# --------------------------------------------------------------------------

def _genai_registry(extra=None):
    entry = {
        "type": "embedding",
        "ovms_model": "qwen3-embedding-0.6b",
        "embedding_backend": "genai_v3",
        "preferred_device": "GPU",
        "owned_by": "openvino",
    }
    if extra:
        entry.update(extra)
    return {"qwen3-embedding-0.6b-int8": entry}


def test_embeddings_genai_v3_dispatch_uses_lease_and_logical_model(monkeypatch):
    import research_ai_gateway.main as main_mod

    monkeypatch.setattr(main_mod, "MODEL_REGISTRY", _genai_registry())
    fake_broker = _FakeBroker(alias="qwen3-embedding-0.6b__gpu")
    monkeypatch.setattr(main_mod, "DEVICE_BROKER", fake_broker)

    seen = {}

    def fake_call(model_name, raw_input, response_model_name):
        seen["model_name"] = model_name
        seen["raw_input"] = raw_input
        seen["response_model_name"] = response_model_name
        seen["lease_active"] = fake_broker.lease_active
        return {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.5, 0.25]}],
            "model": response_model_name,
            "usage": {"prompt_tokens": 2, "total_tokens": 2},
        }

    monkeypatch.setattr(main_mod, "call_genai_embedding", fake_call)

    client = TestClient(main_mod.app)
    response = client.post(
        "/v1/embeddings",
        json={"model": "qwen3-embedding-0.6b-int8", "input": "what is biology?"},
    )

    assert response.status_code == 200
    assert seen == {
        "model_name": "qwen3-embedding-0.6b__gpu",
        # The route normalizes input to a list before dispatching, exactly like
        # the legacy gateway did.
        "raw_input": ["what is biology?"],
        "response_model_name": "qwen3-embedding-0.6b-int8",
        "lease_active": True,
    }
    assert fake_broker.lease_calls == [("qwen3-embedding-0.6b", "GPU")]
    body = response.json()
    assert body["model"] == "qwen3-embedding-0.6b-int8"
    assert body["data"][0]["embedding"] == [0.5, 0.25]


def test_embeddings_genai_v3_without_adaptive_batching_bypasses_batcher(monkeypatch):
    """Production's int8 entry sets no adaptive_batching/tokenizer_path.

    Parity requirement: that request must go straight to the graph backend
    through the broker lease, exactly like the legacy gateway.
    """
    import research_ai_gateway.main as main_mod

    monkeypatch.setattr(main_mod, "MODEL_REGISTRY", _genai_registry())
    fake_broker = _FakeBroker(alias="qwen3-embedding-0.6b__gpu")
    monkeypatch.setattr(main_mod, "DEVICE_BROKER", fake_broker)

    def forbidden_submit(*args, **kwargs):
        raise AssertionError("genai batcher must not engage without adaptive_batching")

    monkeypatch.setattr(main_mod.adaptive_genai_batcher, "submit", forbidden_submit)
    monkeypatch.setattr(
        main_mod,
        "call_genai_embedding",
        lambda model_name, raw_input, response_model_name: {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [1.0]}],
            "model": response_model_name,
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        },
    )

    client = TestClient(main_mod.app)
    response = client.post(
        "/v1/embeddings",
        json={"model": "qwen3-embedding-0.6b-int8", "input": "hello"},
    )
    assert response.status_code == 200
    assert response.json()["data"][0]["embedding"] == [1.0]


def test_embedding_batch_stats_follows_configured_backend(monkeypatch):
    import research_ai_gateway.main as main_mod

    monkeypatch.setattr(
        main_mod,
        "MODEL_REGISTRY",
        {"qwen3-embedding-0.6b-int4": {"embedding_backend": "pooled_ir"}},
    )
    client = TestClient(main_mod.app)
    pooled = client.get("/v1/embedding-batch-stats").json()
    assert pooled["backend"] == "pooled_ir"

    monkeypatch.setattr(
        main_mod,
        "MODEL_REGISTRY",
        {"qwen3-embedding-0.6b-int4": {"embedding_backend": "genai_v3"}},
    )
    genai = client.get("/v1/embedding-batch-stats").json()
    assert genai["backend"] == "genai_v3"


def test_embeddings_genai_v3_unknown_model_returns_404(monkeypatch):
    import research_ai_gateway.main as main_mod

    monkeypatch.setattr(main_mod, "MODEL_REGISTRY", _genai_registry())
    client = TestClient(main_mod.app)
    response = client.post("/v1/embeddings", json={"model": "nope", "input": "x"})
    assert response.status_code == 404


# --------------------------------------------------------------------------
# live OVMS integration (opt-in)
# --------------------------------------------------------------------------

LIVE_BASE = os.getenv("OVMS_GENAI_TEST_BASE")
LIVE_MODEL = os.getenv("OVMS_GENAI_TEST_MODEL", "qwen3-embedding-0.6b")

live_required = pytest.mark.skipif(
    not LIVE_BASE,
    reason="set OVMS_GENAI_TEST_BASE to run the live OVMS genai embeddings test",
)


@live_required
def test_genai_live_ovms_matches_reference_payload(monkeypatch):
    """The ported client must reproduce the legacy payload byte-for-byte.

    We post the legacy payload directly to OVMS and compare the resulting
    vectors with what the gateway path produces. Exact equality (no tolerance)
    is required because both paths must hit the same graph model with the same
    request body.
    """
    base = LIVE_BASE.rstrip("/")
    texts = ["hello world", "what is biology?"]

    reference = requests.post(
        f"{base}/v3/embeddings",
        json={"model": LIVE_MODEL, "input": texts, "encoding_format": "float"},
        timeout=180,
    )
    reference.raise_for_status()
    ref = reference.json()

    monkeypatch.setattr(ovms_mod, "OVMS_BASE", base)
    monkeypatch.setattr(ovms_mod, "OVMS_TIMEOUT", 180)

    raw = ovms_mod.ovms_genai_embeddings(LIVE_MODEL, texts)
    normalized = embeddings_mod.normalize_genai_embeddings_response(raw, "qwen3-embedding-0.6b-int8")

    assert normalized["model"] == "qwen3-embedding-0.6b-int8"
    assert len(normalized["data"]) == len(texts)

    ref_data = sorted(ref["data"], key=lambda item: item["index"])
    for produced, expected in zip(normalized["data"], ref_data):
        produced_vec = np.asarray(produced["embedding"], dtype=np.float64)
        expected_vec = np.asarray(expected["embedding"], dtype=np.float64)
        assert produced_vec.shape == expected_vec.shape
        assert produced_vec.shape[0] > 0
        assert np.array_equal(produced_vec, expected_vec)

    # Same input must be deterministic across calls on this backend.
    raw_again = ovms_mod.ovms_genai_embeddings(LIVE_MODEL, texts)
    again = embeddings_mod.normalize_genai_embeddings_response(raw_again, "qwen3-embedding-0.6b-int8")
    first_vec = np.asarray(normalized["data"][0]["embedding"], dtype=np.float64)
    again_vec = np.asarray(again["data"][0]["embedding"], dtype=np.float64)
    assert np.array_equal(first_vec, again_vec)


@live_required
def test_genai_live_ovms_route_end_to_end(monkeypatch):
    """Full gateway path against live OVMS using the real registry entry."""
    import research_ai_gateway.main as main_mod

    base = LIVE_BASE.rstrip("/")
    monkeypatch.setattr(ovms_mod, "OVMS_BASE", base)
    monkeypatch.setattr(ovms_mod, "OVMS_TIMEOUT", 180)
    monkeypatch.setattr(main_mod, "MODEL_REGISTRY", _genai_registry())

    fake_broker = _FakeBroker(alias=LIVE_MODEL)
    monkeypatch.setattr(main_mod, "DEVICE_BROKER", fake_broker)

    client = TestClient(main_mod.app)
    response = client.post(
        "/v1/embeddings",
        json={"model": "qwen3-embedding-0.6b-int8", "input": "hello world"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] == "qwen3-embedding-0.6b-int8"
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    vector = body["data"][0]["embedding"]
    assert len(vector) > 0
    assert all(isinstance(value, float) for value in vector)
