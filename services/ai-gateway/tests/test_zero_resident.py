import json
import threading
from unittest.mock import patch

import numpy as np

import research_ai_gateway.broker as broker_mod
import research_ai_gateway.ovms_client as ovms_mod
from research_ai_gateway.broker import DeviceBroker
from research_ai_gateway.inference import embeddings as embeddings_mod


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
    def __init__(self, alias="qwen3-embedding-0.6b-int4-pooled__gpu"):
        self.alias = alias
        self.embedding_inference_lock = threading.RLock()
        self.lease_active = False
        self.lease_calls = []

    def lease(self, model_name, preferred_device="GPU"):
        self.lease_calls.append((model_name, preferred_device))
        return _FakeLease(self, self.alias)


def test_pooled_ir_response_holds_lease_and_uses_selected_alias(monkeypatch):
    fake_broker = _FakeBroker()
    prepared = [(np.asarray([1, 2], dtype=np.int64), np.asarray([1, 1], dtype=np.int64))]
    monkeypatch.setattr(embeddings_mod, "tokenize_qwen_texts", lambda texts, path: prepared)

    seen = {}

    def fake_predict(model_name, prepared_arg, tokenizer_path):
        seen["model_name"] = model_name
        seen["lease_active"] = fake_broker.lease_active
        assert len(prepared_arg) == 1
        np.testing.assert_array_equal(prepared_arg[0][0], prepared[0][0])
        np.testing.assert_array_equal(prepared_arg[0][1], prepared[0][1])
        return np.asarray([[0.25, 0.75]], dtype=np.float32), 2

    monkeypatch.setattr(embeddings_mod, "predict_pooled_ir", fake_predict)

    result = embeddings_mod.pooled_ir_response(
        "qwen3-embedding-0.6b-int4-pooled",
        ["hello"],
        "qwen3-embedding-0.6b-int4",
        "/tokenizers/qwen",
        broker=fake_broker,
    )

    assert seen == {
        "model_name": "qwen3-embedding-0.6b-int4-pooled__gpu",
        "lease_active": True,
    }
    assert fake_broker.lease_calls == [("qwen3-embedding-0.6b-int4-pooled", "GPU")]
    assert result["data"][0]["embedding"] == [0.25, 0.75]
    assert fake_broker.lease_active is False


def test_adaptive_batcher_holds_lease_and_uses_selected_alias(monkeypatch):
    fake_broker = _FakeBroker()
    batcher = embeddings_mod.AdaptivePooledIRBatcher(broker=fake_broker)
    prepared = [(np.asarray([1], dtype=np.int64), np.asarray([1], dtype=np.int64))]

    seen = {}

    def fake_predict(model_name, prepared_arg, tokenizer_path):
        seen["model_name"] = model_name
        seen["lease_active"] = fake_broker.lease_active
        assert prepared_arg is prepared
        return np.asarray([[1.0, 0.0]], dtype=np.float32), 1

    monkeypatch.setattr(embeddings_mod, "predict_pooled_ir", fake_predict)

    vectors, token_count = batcher._predict_batch(
        "qwen3-embedding-0.6b-int4-pooled",
        prepared,
        "/tokenizers/qwen",
    )

    assert vectors.shape == (1, 2)
    assert token_count == 1
    assert seen == {
        "model_name": "qwen3-embedding-0.6b-int4-pooled__gpu",
        "lease_active": True,
    }
    assert fake_broker.lease_active is False


def test_empty_runtime_config_can_cold_load_alias_via_polling(tmp_path, monkeypatch):
    runtime_path = tmp_path / "config.json"
    catalog_path = tmp_path / "model_catalog.json"
    runtime_path.write_text('{"model_config_list": []}\n', encoding="utf-8")
    catalog_path.write_text(
        json.dumps(
            {
                "model_config_list": [
                    {
                        "config": {
                            "name": "cold-model__gpu",
                            "base_path": "/models/cold-model",
                            "target_device": "GPU",
                        }
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(broker_mod, "OVMS_CONFIG_PATH", str(runtime_path))
    monkeypatch.setattr(ovms_mod, "OVMS_CONFIG_PATH", str(runtime_path))
    monkeypatch.setattr(ovms_mod, "OVMS_MODEL_CATALOG_PATH", str(catalog_path))
    monkeypatch.setattr(broker_mod, "OVMS_CONFIG_UPDATE_MODE", "poll")

    broker = DeviceBroker()

    def fake_wait(alias, available, timeout):
        payload = json.loads(runtime_path.read_text(encoding="utf-8"))
        names = {
            item["config"]["name"]
            for item in payload["model_config_list"]
        }
        present = alias in names
        return present if available else not present

    monkeypatch.setattr(broker, "_wait_for_model_state", fake_wait)

    with patch.object(broker_mod, "reload_ovms_config") as reload_mock:
        assert broker._set_model_enabled("cold-model__gpu", True, target_device="GPU") is True
        reload_mock.assert_not_called()

    payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    configs = [entry["config"] for entry in payload["model_config_list"]]
    assert configs == [
        {
            "name": "cold-model__gpu",
            "base_path": "/models/cold-model",
            "target_device": "GPU",
        }
    ]


def test_api_update_mode_remains_available_for_polling_disabled_deployments(tmp_path, monkeypatch):
    runtime_path = tmp_path / "config.json"
    catalog_path = tmp_path / "model_catalog.json"
    runtime_path.write_text('{"model_config_list": []}\n', encoding="utf-8")
    catalog_path.write_text(
        json.dumps(
            {
                "model_config_list": [
                    {
                        "config": {
                            "name": "cold-model__gpu",
                            "base_path": "/models/cold-model",
                            "target_device": "GPU",
                        }
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(broker_mod, "OVMS_CONFIG_PATH", str(runtime_path))
    monkeypatch.setattr(ovms_mod, "OVMS_CONFIG_PATH", str(runtime_path))
    monkeypatch.setattr(ovms_mod, "OVMS_MODEL_CATALOG_PATH", str(catalog_path))
    monkeypatch.setattr(broker_mod, "OVMS_CONFIG_UPDATE_MODE", "api")

    broker = DeviceBroker()
    monkeypatch.setattr(broker, "_wait_for_model_state", lambda alias, available, timeout: True)

    with patch.object(broker_mod, "reload_ovms_config", return_value=True) as reload_mock:
        assert broker._set_model_enabled("cold-model__gpu", True, target_device="GPU") is True
        reload_mock.assert_called_once_with()
