import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath("packages/contracts/src"))
sys.path.insert(0, os.path.abspath("services/ai-gateway"))

from app.main import app
from app.broker import DEVICE_BROKER


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


@patch.object(DEVICE_BROKER, "ensure_model_available", return_value="GPU")
@patch("app.ovms_client.ovms_predict")
def test_rerank_flow(mock_predict, mock_ensure, client):
    # Mock OVMS predict response for rerank
    mock_predict.return_value = {
        "predictions": [
            [[0.1, 0.9], [0.2, 0.8]]
        ]
    }

    with patch("app.inference.rerank.get_rerank_tokenizer") as mock_tok:
        tok_inst = MagicMock()
        tok_inst.return_value = {
            "input_ids": np.array([[101, 2054, 102], [101, 2055, 102]], dtype=np.int64),
            "attention_mask": np.array([[1, 1, 1], [1, 1, 1]], dtype=np.int64),
        }
        mock_tok.return_value = tok_inst

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
        assert data["meta"]["total_documents"] == 2
        assert data["meta"]["applied_top_n"] == 1


def test_dino_rejects_text(client):
    payload = {
        "model": "dinov3",
        "modality": "text",
        "input": "This should fail because DINO is image-only"
    }
    res = client.post("/v1/embeddings", json=payload)
    assert res.status_code == 400
    assert "DINO only supports image-to-image" in res.json()["detail"]


@patch.object(DEVICE_BROKER, "ensure_model_available", return_value="GPU")
@patch("app.ovms_client.ovms_predict")
def test_dino_image_embedding(mock_predict, mock_ensure, client):
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
