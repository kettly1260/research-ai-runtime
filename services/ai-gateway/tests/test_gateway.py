import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath("packages/contracts/src"))
sys.path.insert(0, os.path.abspath("services/ai-gateway"))

from research_ai_gateway.main import app
from research_ai_gateway.broker import DEVICE_BROKER

import research_ai_gateway.main as main_mod

# The production logical registry (`config/models.yaml`) intentionally excludes
# the export-only jina-clip-v2 / dinov3 tracks, because listing a model that is
# not deployed in OVMS would advertise it as available. The DINO code path is
# still a supported feature of the modular gateway, so these tests inject a
# DINO entry explicitly instead of relying on the global registry.
DINO_TEST_MODELS = {
    "dinov3": {
        "type": "dino_embedding",
        "ovms_model": "dinov3",
        "model_variant": "vit-s-16",
        "preferred_device": "GPU",
        "fallback_device": "CPU",
        "policy": "GPU_PREFERRED",
        "input_size": [3, 224, 224],
        "output_dimension": 384,
        "normalize": True,
    }
}


def _use_dino_registry(monkeypatch):
    registry = dict(main_mod.MODEL_REGISTRY.items())
    registry.update(DINO_TEST_MODELS)
    monkeypatch.setattr(main_mod, "MODEL_REGISTRY", registry)


@pytest.fixture
def client():
    return TestClient(app)


def test_models_list(client):
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) > 0
    ids = [m["id"] for m in data["data"]]
    assert "bge-m3" in ids
    assert "qwen-reranker" in ids


def test_models_compat(client):
    response = client.get("/models")
    assert response.status_code == 200
    assert response.json()["object"] == "list"


def test_model_detail(client):
    response = client.get("/v1/models/bge-m3")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "bge-m3"

    response_missing = client.get("/v1/models/non-existent-model")
    assert response_missing.status_code == 404


def test_broker_metrics(client):
    response = client.get("/v1/broker/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert "loaded_models" in data


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@patch.object(DEVICE_BROKER, "_set_model_enabled", return_value=True)
@patch("research_ai_gateway.ovms_client.ovms_predict")
def test_rerank_flow(mock_predict, mock_set, client):
    class FakeRerankTokenizer:
        padding_side = "left"
        pad_token_id = 0
        eos_token = "<eos>"

        def encode(self, text, add_special_tokens=False):
            return [7]

        def __call__(
            self,
            texts,
            padding=False,
            truncation=None,
            max_length=None,
            return_attention_mask=False,
        ):
            return {"input_ids": [[1, 2, 3] for _ in texts]}

        def pad(self, encoded, padding=True, return_attention_mask=True, return_tensors="np"):
            rows = encoded["input_ids"]
            max_len = max(len(row) for row in rows)
            ids = []
            masks = []
            for row in rows:
                pad_count = max_len - len(row)
                ids.append(([0] * pad_count) + row)
                masks.append(([0] * pad_count) + ([1] * len(row)))
            return {
                "input_ids": np.asarray(ids, dtype=np.int64),
                "attention_mask": np.asarray(masks, dtype=np.int64),
            }

        def convert_tokens_to_ids(self, token):
            return {"yes": 4, "no": 5}[token]

    # Mock final-token vocabulary logits. First doc is relevant, second is not.
    mock_predict.return_value = {
        "predictions": [
            [[0.0, 0.0, 0.0, 0.0, 5.0, -5.0]],
            [[0.0, 0.0, 0.0, 0.0, -4.0, 4.0]],
        ]
    }

    with patch("research_ai_gateway.inference.rerank.get_rerank_tokenizer") as mock_tok:
        mock_tok.return_value = FakeRerankTokenizer()

        payload = {
            "model": "qwen-reranker",
            "query": "what is biology?",
            "documents": ["Biology is the study of living organisms.", "Chemistry is the study of matter."],
            "top_n": 1
        }
        res = client.post("/v1/rerank", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert len(data["results"]) == 1
        assert "score" in data["results"][0]
        assert data["results"][0]["text"] == "Biology is the study of living organisms."
        assert data["results"][0]["score"] > 0.99
        assert data["meta"]["total_documents"] == 2
        assert data["meta"]["applied_top_n"] == 1


def test_dino_rejects_text(client, monkeypatch):
    _use_dino_registry(monkeypatch)
    payload = {
        "model": "dinov3",
        "modality": "text",
        "input": "This should fail because DINO is image-only"
    }
    res = client.post("/v1/embeddings", json=payload)
    assert res.status_code == 400
    assert "DINO only supports image-to-image" in res.json()["detail"]


@patch.object(DEVICE_BROKER, "_set_model_enabled", return_value=True)
@patch("research_ai_gateway.ovms_client.ovms_predict")
def test_dino_image_embedding(mock_predict, mock_set, client, monkeypatch):
    _use_dino_registry(monkeypatch)
    from PIL import Image
    import io
    import base64

    # Create dummy 10x10 image base64
    img = Image.new("RGB", (10, 10), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

    mock_predict.return_value = {
        "predictions": [
            [0.1] * 384
        ]
    }

    payload = {
        "model": "dinov3",
        "modality": "image",
        "input": [b64_str]
    }
    res = client.post("/v1/embeddings", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert len(data["data"][0]["embedding"]) == 384


def test_device_broker_concurrent_gpu_model_a_and_cpu_model_b():
    """Verify concurrent execution:
    GPU runs Model A while CPU simultaneously runs Model B without model name collision.
    """
    import threading
    import time

    results = {}
    barrier = threading.Barrier(2)

    with patch.object(DEVICE_BROKER, "_set_model_enabled", return_value=True), \
         patch("research_ai_gateway.broker.is_model_available", return_value=True):
        def worker_a():
            with DEVICE_BROKER.lease("qwen-reranker", preferred_device="GPU") as alias:
                results["a_alias"] = alias
                results["a_active"] = DEVICE_BROKER.active_inferences.get(alias, 0)
                barrier.wait(timeout=5.0)
                time.sleep(0.05)

        def worker_b():
            time.sleep(0.01)
            with DEVICE_BROKER.lease("bge-m3-i8", preferred_device="GPU") as alias:
                results["b_alias"] = alias
                results["b_active"] = DEVICE_BROKER.active_inferences.get(alias, 0)
                barrier.wait(timeout=5.0)
                time.sleep(0.05)

        t1 = threading.Thread(target=worker_a)
        t2 = threading.Thread(target=worker_b)
        t1.start()
        t2.start()
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)

    # Model A ran on GPU alias
    assert results["a_alias"] == "qwen-reranker__gpu"
    # Model B ran concurrently on CPU alias due to coexistence policy
    assert results["b_alias"] == "bge-m3-i8__cpu"
    assert DEVICE_BROKER.metrics["cpu_coexistence_inferences"] >= 1


def test_device_broker_high_load_gpu_spillover_to_temporary_cpu_replica():
    """Verify high-load spillover:
    When Model A is busy on GPU, another request for Model A spills over to temporary CPU replica A__cpu.
    Once the temporary CPU request completes, A__cpu is automatically evicted.
    """
    import threading
    import time

    results = {}
    barrier = threading.Barrier(2)

    with patch.object(DEVICE_BROKER, "_set_model_enabled", return_value=True), \
         patch("research_ai_gateway.broker.is_model_available", return_value=True):
        def primary_request():
            with DEVICE_BROKER.lease("qwen-reranker", preferred_device="GPU") as alias:
                results["req1_alias"] = alias
                results["req1_device"] = "GPU"
                barrier.wait(timeout=5.0)
                time.sleep(0.05)

        def spillover_request():
            time.sleep(0.01)
            with DEVICE_BROKER.lease("qwen-reranker", preferred_device="GPU") as alias:
                results["req2_alias"] = alias
                results["req2_device"] = "CPU"
                results["was_temporary"] = "qwen-reranker__cpu" in DEVICE_BROKER.temporary_cpu_replicas
                barrier.wait(timeout=5.0)
                time.sleep(0.02)

        t1 = threading.Thread(target=primary_request)
        t2 = threading.Thread(target=spillover_request)
        t1.start()
        t2.start()
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)

    assert results["req1_alias"] == "qwen-reranker__gpu"
    assert results["req2_alias"] == "qwen-reranker__cpu"
    assert results["was_temporary"] is True
    # Verify temporary CPU replica was marked for eviction / unloaded after completion
    assert "qwen-reranker__cpu" not in DEVICE_BROKER.temporary_cpu_replicas
    assert DEVICE_BROKER.active_inferences.get("qwen-reranker__cpu", 0) == 0
    assert DEVICE_BROKER.metrics["cpu_spillover_inferences"] >= 1
