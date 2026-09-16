"""Regression tests for the DeviceBroker device-preference contract.

Background (found by the OVMS 2026.1 / 2026.3.1 acceptance Gate 2, CPU-only
host):

``main.py`` reads ``preferred_device`` from the model registry for every
inference route it owns, but the three embedding call sites in
``inference/embeddings.py`` called ``broker.lease(model_name)`` with no
preference at all.  ``lease()`` therefore fell back to its ``"GPU"`` default on
a host that has no GPU.  The consequences were all real and all measured:

* every embedding request paid a full ``_wait_for_model_state`` timeout
  (``OVMS_TIMEOUT``, 120 s in the acceptance stack) before falling back to CPU,
* the failed alias was written into ``config.json`` and never removed, because
  ``_set_model_enabled`` only rolls back aliases it tracks in ``model_loaded``,
  which a failed load never reaches,
* OVMS then re-attempted that failing compile once per filesystem-poll interval
  for the lifetime of the container -- 579 occurrences in one acceptance run --
  and the deployment could never reach a true zero-resident state.

These tests pin the fix: the registry is the single source of truth for device
placement, an explicit override still wins, and a definitively failed alias is
rolled back out of ``config.json`` instead of leaking.
"""

import json
import threading

import numpy as np

import research_ai_gateway.broker as broker_mod
import research_ai_gateway.ovms_client as ovms_mod
from research_ai_gateway.broker import DeviceBroker
from research_ai_gateway.inference import embeddings as embeddings_mod

POOLED_BASE = "qwen3-embedding-0.6b-int4-pooled"
POOLED_LOGICAL = "qwen3-embedding-0.6b-int4"


def _prepare(tmp_path, monkeypatch, registry):
    """Point the broker at throwaway runtime/catalog files and a fake registry."""
    runtime_path = tmp_path / "config.json"
    catalog_path = tmp_path / "model_catalog.json"
    runtime_path.write_text('{"model_config_list": []}\n', encoding="utf-8")
    catalog_path.write_text(
        json.dumps(
            {
                "model_config_list": [
                    {
                        "config": {
                            "name": POOLED_BASE,
                            "base_path": f"/models/{POOLED_BASE}",
                            "target_device": "GPU",
                            "plugin_config": {"PERFORMANCE_HINT": "LATENCY"},
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
    monkeypatch.setattr(broker_mod, "MODEL_REGISTRY", registry)
    return runtime_path


def _entries(runtime_path):
    payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    return [entry["config"] for entry in payload["model_config_list"]]


def _names(runtime_path):
    return [cfg["name"] for cfg in _entries(runtime_path)]


def test_lease_honours_the_registry_cpu_preference(tmp_path, monkeypatch):
    """A CPU-declared model must never even attempt the GPU alias."""
    runtime_path = _prepare(
        tmp_path,
        monkeypatch,
        {POOLED_LOGICAL: {"ovms_model": POOLED_BASE, "preferred_device": "CPU"}},
    )
    broker = DeviceBroker()
    waited = []
    monkeypatch.setattr(
        broker,
        "_wait_for_model_state",
        lambda alias, available, timeout: waited.append((alias, available)) or True,
    )

    with broker.lease(POOLED_BASE) as alias:
        assert alias == f"{POOLED_BASE}__cpu"

    # The GPU alias was never registered, so it was never waited on.
    assert waited == [(f"{POOLED_BASE}__cpu", True)]
    assert _names(runtime_path) == [f"{POOLED_BASE}__cpu"]
    assert _entries(runtime_path)[0]["target_device"] == "CPU"


def test_failed_gpu_alias_is_rolled_back_and_never_leaks(tmp_path, monkeypatch):
    """A GPU-declared model on a GPU-less host must fall back AND clean up."""
    runtime_path = _prepare(
        tmp_path,
        monkeypatch,
        {POOLED_LOGICAL: {"ovms_model": POOLED_BASE, "preferred_device": "GPU"}},
    )
    broker = DeviceBroker()

    def fake_wait(alias, available, timeout):
        # The __gpu alias registers in OVMS but never compiles on a CPU-only
        # host; the __cpu alias loads fine.
        return not alias.endswith("__gpu")

    monkeypatch.setattr(broker, "_wait_for_model_state", fake_wait)

    with broker.lease(POOLED_BASE) as alias:
        assert alias == f"{POOLED_BASE}__cpu"

    names = _names(runtime_path)
    assert f"{POOLED_BASE}__gpu" not in names, (
        "a definitively failed alias must not be left in config.json; OVMS would "
        "retry the failing compile once per poll interval forever"
    )
    assert names == [f"{POOLED_BASE}__cpu"]


def test_set_model_enabled_wait_false_skips_the_poll(tmp_path, monkeypatch):
    """The rollback path must not block for another full timeout."""
    runtime_path = _prepare(tmp_path, monkeypatch, {})
    broker = DeviceBroker()
    calls = []
    monkeypatch.setattr(
        broker, "_wait_for_model_state", lambda *args: calls.append(args) or True
    )

    assert broker._set_model_enabled(
        f"{POOLED_BASE}__gpu", True, target_device="GPU", wait=False
    )
    assert calls == []
    assert _names(runtime_path) == [f"{POOLED_BASE}__gpu"]


def test_explicit_preferred_device_overrides_the_registry(tmp_path, monkeypatch):
    """A deliberate caller override must still win over the registry."""
    runtime_path = _prepare(
        tmp_path,
        monkeypatch,
        {POOLED_LOGICAL: {"ovms_model": POOLED_BASE, "preferred_device": "CPU"}},
    )
    broker = DeviceBroker()
    monkeypatch.setattr(
        broker, "_wait_for_model_state", lambda alias, available, timeout: True
    )

    with broker.lease(POOLED_BASE, preferred_device="GPU") as alias:
        assert alias == f"{POOLED_BASE}__gpu"

    assert _names(runtime_path) == [f"{POOLED_BASE}__gpu"]


def test_registry_lookup_accepts_logical_id_and_defaults_to_gpu(tmp_path, monkeypatch):
    _prepare(
        tmp_path,
        monkeypatch,
        {"logical-id": {"ovms_model": "some-ovms-model", "preferred_device": "cpu"}},
    )
    broker = DeviceBroker()

    assert broker._registry_device_preference("logical-id") == "CPU"
    assert broker._registry_device_preference("some-ovms-model") == "CPU"
    # Unknown models keep the historical GPU-first behaviour.
    assert broker._registry_device_preference("not-registered") == "GPU"


class _RecordingLease:
    def __init__(self, alias):
        self.alias = alias

    def __enter__(self):
        return self.alias

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class _RecordingBroker:
    """Records the device preference each embedding path passes to ``lease``."""

    def __init__(self):
        self.embedding_inference_lock = threading.RLock()
        self.devices = []

    def lease(self, model_name, preferred_device=None):
        self.devices.append(preferred_device)
        return _RecordingLease(f"{model_name}__cpu")


def test_embedding_paths_defer_device_choice_to_the_broker(monkeypatch):
    broker = _RecordingBroker()
    prepared = [
        (np.asarray([1, 2], dtype=np.int64), np.asarray([1, 1], dtype=np.int64))
    ]
    monkeypatch.setattr(
        embeddings_mod, "prepare_qwen_chunked_texts", lambda texts, path: (prepared, [(0, 1)], [1], 2)
    )
    monkeypatch.setattr(
        embeddings_mod,
        "predict_pooled_ir",
        lambda model_name, prepared_arg, tokenizer_path: (
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            2,
        ),
    )

    embeddings_mod.pooled_ir_response(
        POOLED_BASE, ["hi"], POOLED_LOGICAL, "/tokenizers/qwen", broker=broker
    )
    embeddings_mod.AdaptivePooledIRBatcher(broker=broker)._predict_batch(
        POOLED_BASE, prepared, "/tokenizers/qwen"
    )

    assert broker.devices == [None, None], (
        "embedding paths must not hardcode a device; passing None lets the "
        "broker resolve preferred_device from the registry"
    )
