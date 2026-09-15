import time
import threading
from typing import Dict, Optional, Tuple
from fastapi import HTTPException
from .config import (
    MODEL_IDLE_UNLOAD_SECONDS,
    MAX_LOADED_MODELS,
    MODEL_IDLE_SWEEP_SECONDS,
    OVMS_TIMEOUT,
    PINNED_MODELS,
    CPU_SEMAPHORE_LIMIT,
    MAX_QUEUE_DEPTH,
)
from .registry import MODEL_REGISTRY
from .ovms_client import (
    is_model_available,
    read_model_config_file,
    catalog_model_config,
    write_runtime_config,
    reload_ovms_config,
    OVMS_CONFIG_PATH,
)
from .tokenizers import release_tokenizers_for_model


class DeviceBroker:
    """Device broker implementing GPU First + CPU Spillover + Zero Resident."""

    def __init__(self):
        self.model_state_lock = threading.Lock()
        self.config_file_lock = threading.Lock()
        self.embedding_inference_lock = threading.RLock()
        self.cpu_semaphore = threading.Semaphore(CPU_SEMAPHORE_LIMIT)

        self.model_last_used: Dict[str, float] = {}
        self.model_loaded: Dict[str, bool] = {}
        self.model_device: Dict[str, str] = {}  # "GPU" or "CPU"

        self._idle_sweeper_started = False
        self._sweeper_thread = None

        # Metrics
        self.metrics = {
            "gpu_inferences": 0,
            "cpu_spillover_inferences": 0,
            "evictions": 0,
            "active_queue_depth": 0,
        }

    def start_sweeper(self):
        with self.model_state_lock:
            if not self._idle_sweeper_started:
                self._idle_sweeper_started = True
                self._sweeper_thread = threading.Thread(
                    target=self._idle_sweep_loop,
                    name="device-broker-idle-sweeper",
                    daemon=True,
                )
                self._sweeper_thread.start()

    def _idle_sweep_loop(self):
        while True:
            time.sleep(max(1, MODEL_IDLE_SWEEP_SECONDS))
            try:
                with self.model_state_lock:
                    self._sync_model_loaded_state()
                    self._evict_idle_models()
            except Exception as exc:
                print(f"[DeviceBroker] idle sweep error: {exc}", flush=True)

    def _sync_model_loaded_state(self):
        for model_name in list(self.model_loaded):
            self.model_loaded[model_name] = is_model_available(model_name)

    def _registry_pinned_models(self) -> set:
        pinned = set(PINNED_MODELS)
        for cfg in MODEL_REGISTRY.values():
            if cfg.get("pinned") is True and cfg.get("ovms_model"):
                pinned.add(cfg["ovms_model"])
        return pinned

    def _model_type_for_ovms(self, model_name: str) -> Optional[str]:
        for cfg in MODEL_REGISTRY.values():
            if cfg.get("ovms_model") == model_name:
                return cfg.get("type")
        lowered = str(model_name).lower()
        if model_name == "bge-m3-i8" or "embedding" in lowered or "arctic-embed" in lowered:
            return "embedding"
        return None

    def _wait_for_model_state(self, model_name: str, available: bool, timeout: int) -> bool:
        deadline = time.time() + max(1, timeout)
        while time.time() < deadline:
            if is_model_available(model_name) == available:
                return True
            time.sleep(0.5)
        return is_model_available(model_name) == available

    def _set_model_enabled(self, model_name: str, enabled: bool, target_device: str = "GPU") -> bool:
        with self.config_file_lock:
            try:
                payload = read_model_config_file(OVMS_CONFIG_PATH)
                entries = list(payload.get("model_config_list", []))
                names = {
                    item.get("config", {}).get("name")
                    for item in entries
                    if isinstance(item, dict)
                }

                changed = False
                if enabled:
                    if model_name not in names:
                        cfg = dict(catalog_model_config(model_name))
                        # Device routing override if CPU requested
                        if target_device == "CPU":
                            cfg["target_device"] = "CPU"
                        entries.append({"config": cfg})
                        changed = True
                else:
                    if model_name not in names:
                        return True
                    entries = [
                        item for item in entries
                        if item.get("config", {}).get("name") != model_name
                    ]
                    changed = True

                if changed:
                    payload["model_config_list"] = entries
                    write_runtime_config(payload)
                    if not reload_ovms_config():
                        return False
            except (OSError, ValueError, KeyError):
                return False

        if enabled:
            return self._wait_for_model_state(model_name, True, max(OVMS_TIMEOUT, 30))
        return self._wait_for_model_state(model_name, False, min(max(OVMS_TIMEOUT, 30), 60))

    def _release_resources(self, model_name: str):
        tokenizer_paths = {
            cfg.get("tokenizer_path")
            for cfg in MODEL_REGISTRY.values()
            if cfg.get("ovms_model") == model_name and cfg.get("tokenizer_path")
        }
        release_tokenizers_for_model(model_name, tokenizer_paths)

    def _evict_idle_models(self, exclude: Optional[str] = None):
        now = time.time()
        pinned = self._registry_pinned_models()
        candidates = []
        for model_name, loaded in list(self.model_loaded.items()):
            if not loaded or model_name in pinned or (exclude and model_name == exclude):
                continue
            last_used = self.model_last_used.get(model_name, now)
            # Check model specific idle ttl
            ttl = MODEL_IDLE_UNLOAD_SECONDS
            for cfg in MODEL_REGISTRY.values():
                if cfg.get("ovms_model") == model_name and cfg.get("idle_ttl"):
                    ttl = int(cfg["idle_ttl"])
                    break
            if now - last_used >= ttl:
                candidates.append(model_name)

        for model_name in candidates:
            if self._set_model_enabled(model_name, False):
                self.model_loaded[model_name] = False
                self._release_resources(model_name)
                self.metrics["evictions"] += 1

    def _evict_other_embedding_models(self, exclude: str):
        for model_name, loaded in list(self.model_loaded.items()):
            if not loaded or model_name == exclude:
                continue
            if self._model_type_for_ovms(model_name) != "embedding":
                continue
            if self._set_model_enabled(model_name, False):
                self.model_loaded[model_name] = False
                self._release_resources(model_name)
                self.metrics["evictions"] += 1

    def _evict_to_capacity(self, exclude: str):
        pinned = self._registry_pinned_models()
        while sum(1 for v in self.model_loaded.values() if v) >= max(1, MAX_LOADED_MODELS):
            candidates = [
                m for m, is_loaded in self.model_loaded.items()
                if is_loaded and m != exclude and m not in pinned
            ]
            if not candidates:
                break
            victim = min(candidates, key=lambda m: self.model_last_used.get(m, 0.0))
            if self._set_model_enabled(victim, False):
                self.model_loaded[victim] = False
                self._release_resources(victim)
                self.metrics["evictions"] += 1
            else:
                break

    def ensure_model_available(self, model_name: str, preferred_device: str = "GPU") -> str:
        """Ensures the target model is loaded in OVMS according to Device Broker policy."""
        with self.model_state_lock:
            if model_name not in self.model_last_used:
                self.model_last_used[model_name] = time.time()
                self.model_loaded[model_name] = False

            self._sync_model_loaded_state()
            self._evict_idle_models(exclude=model_name)

            if self._model_type_for_ovms(model_name) == "embedding":
                self._evict_other_embedding_models(exclude=model_name)

            if is_model_available(model_name):
                self.model_loaded[model_name] = True
                self.model_last_used[model_name] = time.time()
                return self.model_device.get(model_name, "GPU")

            self._evict_to_capacity(exclude=model_name)

            target_device = preferred_device
            if not self._set_model_enabled(model_name, True, target_device=target_device):
                # If GPU load failed and CPU is an option, try CPU fallback
                if target_device == "GPU":
                    print(f"[DeviceBroker] GPU load for {model_name} failed, attempting CPU fallback", flush=True)
                    if self._set_model_enabled(model_name, True, target_device="CPU"):
                        target_device = "CPU"
                        self.metrics["cpu_spillover_inferences"] += 1
                    else:
                        raise HTTPException(
                            status_code=503,
                            detail=f"failed to load model {model_name} in backend",
                        )
                else:
                    raise HTTPException(
                        status_code=503,
                        detail=f"failed to load model {model_name} in backend",
                    )

            self.model_loaded[model_name] = True
            self.model_last_used[model_name] = time.time()
            self.model_device[model_name] = target_device
            if target_device == "GPU":
                self.metrics["gpu_inferences"] += 1
            return target_device


DEVICE_BROKER = DeviceBroker()
