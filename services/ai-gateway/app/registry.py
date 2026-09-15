import json
import os
import threading
from typing import Any, Dict, Optional
from .config import MODEL_REGISTRY_PATH, MODELS_CONFIG_PATH, load_yaml_models_config

DEFAULT_MODEL_REGISTRY = {
    "qwen-reranker": {
        "type": "rerank",
        "ovms_model": "qwen-reranker",
        "owned_by": "openvino",
        "preferred_device": "GPU",
        "fallback_device": "CPU",
        "policy": "GPU_PREFERRED",
    },
    "bge-m3-i8": {
        "type": "embedding",
        "ovms_model": "bge-m3-i8",
        "owned_by": "openvino",
        "preferred_device": "GPU",
        "fallback_device": "CPU",
        "policy": "GPU_PREFERRED",
    },
    "bge-m3": {
        "type": "embedding",
        "ovms_model": "bge-m3-i8",
        "owned_by": "openvino",
        "preferred_device": "GPU",
        "fallback_device": "CPU",
        "policy": "GPU_PREFERRED",
    },
    "jina-clip-v2": {
        "type": "multimodal_embedding",
        "ovms_model": "jina-clip-v2",
        "owned_by": "openvino",
        "preferred_device": "GPU",
        "fallback_device": "CPU",
        "policy": "GPU_PREFERRED",
    },
    "dinov3": {
        "type": "dino_embedding",
        "ovms_model": "dinov3",
        "owned_by": "openvino",
        "preferred_device": "GPU",
        "fallback_device": "CPU",
        "policy": "GPU_PREFERRED",
    },
}


class HotModelRegistry:
    """Hot-reloading model registry supporting both JSON and YAML configurations."""

    def __init__(self, json_path: str, yaml_path: str, fallback: dict):
        self.json_path = json_path
        self.yaml_path = yaml_path
        self._lock = threading.Lock()
        self._last_good = dict(fallback)
        self._last_error = None

    def _load(self) -> dict:
        with self._lock:
            # 1. Try loading from external YAML first if present
            yaml_data = load_yaml_models_config(self.yaml_path)
            if yaml_data and isinstance(yaml_data.get("models"), dict):
                models_dict = yaml_data["models"]
                validated = {}
                for m_id, cfg in models_dict.items():
                    if isinstance(cfg, dict):
                        m_cfg = dict(cfg)
                        m_cfg.setdefault("ovms_model", m_cfg.get("model_id", m_id))
                        m_cfg.setdefault("preferred_device", "GPU")
                        m_cfg.setdefault("fallback_device", "CPU")
                        m_cfg.setdefault("policy", "GPU_PREFERRED")
                        validated[m_id] = m_cfg
                if validated:
                    self._last_good = validated
                    self._last_error = None
                    return dict(self._last_good)

            # 2. Fall back to JSON model registry
            try:
                if os.path.exists(self.json_path):
                    with open(self.json_path, "r", encoding="utf-8") as handle:
                        payload = json.load(handle)
                    if isinstance(payload, dict) and isinstance(payload.get("models"), dict):
                        payload = payload["models"]
                    if isinstance(payload, dict) and payload:
                        validated = {}
                        for model_id, cfg in payload.items():
                            if isinstance(cfg, dict):
                                m_cfg = dict(cfg)
                                m_cfg.setdefault("preferred_device", "GPU")
                                m_cfg.setdefault("fallback_device", "CPU")
                                m_cfg.setdefault("policy", "GPU_PREFERRED")
                                validated[model_id] = m_cfg
                        self._last_good = validated
                        self._last_error = None
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                if msg != self._last_error:
                    print(f"model registry reload failed; keeping last good: {msg}", flush=True)
                    self._last_error = msg

            return dict(self._last_good)

    def get(self, key, default=None):
        return self._load().get(key, default)

    def __getitem__(self, key):
        return self._load()[key]

    def items(self):
        return self._load().items()

    def values(self):
        return self._load().values()

    def keys(self):
        return self._load().keys()

    def __contains__(self, key):
        return key in self._load()

    def __iter__(self):
        return iter(self._load())

    def __len__(self):
        return len(self._load())


MODEL_REGISTRY = HotModelRegistry(MODEL_REGISTRY_PATH, MODELS_CONFIG_PATH, DEFAULT_MODEL_REGISTRY)
