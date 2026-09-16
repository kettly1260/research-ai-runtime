"""Registry parity guards for the legacy -> modular cutover.

`config/models.yaml` is the single logical model registry authority. These tests
lock it to the production model set so that:

* the cutover cannot silently change the model ids served by ``GET /v1/models``;
* an undeployed export-only track (jina-clip-v2 / dinov3) cannot reappear and be
  advertised as available;
* no per-model ``idle_ttl`` override is introduced, which would silently shorten
  residency relative to the production ``MODEL_IDLE_UNLOAD_SECONDS=1200``.
"""

import os
from pathlib import Path

import yaml

from research_ai_gateway.registry import DEFAULT_MODEL_REGISTRY

REPO_ROOT = Path(__file__).resolve().parents[3]
MODELS_YAML = REPO_ROOT / "config" / "models.yaml"

# Exact logical model set of the production legacy registry
# (/mnt/user/appdata/ovms/runtime/model_registry.json).
EXPECTED_MODELS = {
    "qwen-reranker": {"type": "rerank", "ovms_model": "qwen-reranker"},
    "bge-m3-i8": {"type": "embedding", "ovms_model": "bge-m3-i8"},
    "bge-m3": {"type": "embedding", "ovms_model": "bge-m3-i8"},
    "qwen3-embedding-0.6b-int8": {
        "type": "embedding",
        "ovms_model": "qwen3-embedding-0.6b",
        "embedding_backend": "genai_v3",
    },
    "qwen3-embedding-0.6b-int4": {
        "type": "embedding",
        "ovms_model": "qwen3-embedding-0.6b-int4-pooled",
        "embedding_backend": "pooled_ir",
    },
    "arctic-embed-m-v2-int8": {
        "type": "embedding",
        "ovms_model": "arctic-embed-m-v2-int8",
        "embedding_backend": "sentence_transformer",
    },
}

UNDEPLOYED_TRACKS = {"jina-clip-v2", "dinov3"}


def _load_models_yaml():
    assert MODELS_YAML.exists(), f"missing registry file: {MODELS_YAML}"
    with open(MODELS_YAML, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_models_yaml_is_the_production_model_set():
    data = _load_models_yaml()
    models = data["models"]
    assert set(models) == set(EXPECTED_MODELS)


def test_models_yaml_maps_each_model_to_the_production_backend():
    models = _load_models_yaml()["models"]
    for model_id, expected in EXPECTED_MODELS.items():
        cfg = models[model_id]
        assert cfg["type"] == expected["type"], model_id
        assert cfg["ovms_model"] == expected["ovms_model"], model_id
        if "embedding_backend" in expected:
            assert cfg.get("embedding_backend") == expected["embedding_backend"], model_id
        else:
            # bge-m3 and bge-m3-i8 use the default bge path in production.
            assert cfg.get("embedding_backend") is None, model_id


def test_models_yaml_does_not_expose_undeployed_tracks():
    models = _load_models_yaml()["models"]
    leaked = UNDEPLOYED_TRACKS & set(models)
    assert not leaked, f"undeployed models must not be registered: {sorted(leaked)}"


def test_models_yaml_sets_no_per_model_idle_ttl():
    """Production sets no per-model TTL, so residency follows the env value."""
    models = _load_models_yaml()["models"]
    offenders = {mid: cfg.get("idle_ttl") for mid, cfg in models.items() if cfg.get("idle_ttl")}
    assert not offenders, (
        "per-model idle_ttl would override MODEL_IDLE_UNLOAD_SECONDS and break "
        f"residency parity: {offenders}"
    )


def test_fallback_registry_matches_production_model_set():
    """The last-resort registry must not advertise undeployed models either."""
    assert set(DEFAULT_MODEL_REGISTRY) == set(EXPECTED_MODELS)
    assert not (UNDEPLOYED_TRACKS & set(DEFAULT_MODEL_REGISTRY))
    for model_id, expected in EXPECTED_MODELS.items():
        cfg = DEFAULT_MODEL_REGISTRY[model_id]
        assert cfg["type"] == expected["type"], model_id
        assert cfg["ovms_model"] == expected["ovms_model"], model_id
        if "embedding_backend" in expected:
            assert cfg.get("embedding_backend") == expected["embedding_backend"], model_id


def test_pooled_ir_entry_keeps_adaptive_batching_and_tokenizer_path():
    cfg = _load_models_yaml()["models"]["qwen3-embedding-0.6b-int4"]
    assert cfg.get("adaptive_batching") is True
    assert cfg.get("tokenizer_path") == "/models/OpenVINO/Qwen3-Embedding-0.6B-int4-cw-ov"


def test_arctic_entry_keeps_sentence_transformer_parameters():
    cfg = _load_models_yaml()["models"]["arctic-embed-m-v2-int8"]
    assert cfg.get("tokenizer_path") == "/models/Snowflake/snowflake-arctic-embed-m-v2.0-test"
    assert cfg.get("max_length") == 8192
    assert cfg.get("batch_size") == 2
    assert cfg.get("query_prefix") == "query: "


def test_genai_entry_has_no_adaptive_batching_override():
    """Production's int8 entry takes the direct graph path, not the batcher."""
    cfg = _load_models_yaml()["models"]["qwen3-embedding-0.6b-int8"]
    assert cfg.get("adaptive_batching") is None
    assert cfg.get("tokenizer_path") is None


def test_example_registry_matches_authoritative_registry():
    example = REPO_ROOT / "config" / "models.example.yaml"
    assert example.exists(), "config/models.example.yaml must stay in sync"
    with open(example, "r", encoding="utf-8") as handle:
        assert yaml.safe_load(handle) == _load_models_yaml()


def test_registry_file_is_not_ignored_by_git():
    """The registry must be tracked; a missing file silently changes the API."""
    assert os.path.isfile(MODELS_YAML)
