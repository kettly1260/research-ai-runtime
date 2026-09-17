from __future__ import annotations

import time
import threading
from typing import Dict, Optional, Set, Tuple
from fastapi import HTTPException
from .config import (
    MODEL_IDLE_UNLOAD_SECONDS,
    MAX_LOADED_MODELS,
    MODEL_IDLE_SWEEP_SECONDS,
    OVMS_TIMEOUT,
    PINNED_MODELS,
    CPU_SEMAPHORE_LIMIT,
    CPU_THREAD_BUDGET,
    MAX_QUEUE_DEPTH,
    OVMS_CONFIG_UPDATE_MODE,
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


def get_model_alias(logical_name: str, device: str) -> str:
    """Constructs internal OVMS runtime model alias (e.g. bge-m3-i8__gpu)."""
    return f"{logical_name}__{device.lower()}"


def parse_model_alias(alias: str) -> Tuple[str, str]:
    if "__" in alias:
        base, dev = alias.rsplit("__", 1)
        return base, dev.upper()
    return alias, "GPU"


class ModelLease:
    """RAII lease for an allocated model device instance."""

    def __init__(
        self,
        broker: DeviceBroker,
        alias: str,
        logical_name: str,
        device: str,
        is_spillover: bool = False,
    ):
        self.broker = broker
        self.alias = alias
        self.logical_name = logical_name
        self.device = device
        self.is_spillover = is_spillover

    def __enter__(self) -> str:
        return self.alias

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.broker.release_lease(self)


class DeviceBroker:
    """Device broker implementing true GPU First + CPU Spillover + Multi-device Coexistence.

    Capabilities:
    1. GPU runs Model A while CPU concurrently runs Model B.
    2. High-load GPU Model A spills over to temporary CPU Model A replica (A__cpu).
    3. Temporary CPU replicas are safely unloaded after active inferences drain.
    4. Hardware runtime aliases (<model>__gpu, <model>__cpu) prevent OVMS model collisions.
    5. Full accounting: CPU semaphores, active inferences, queue-depth accounting, RAM/thread budget.
    """

    def __init__(self):
        self.model_state_lock = threading.Lock()
        self.config_file_lock = threading.Lock()
        self.embedding_inference_lock = threading.RLock()

        # Semaphores and budgets
        self.cpu_semaphore = threading.BoundedSemaphore(max(1, CPU_SEMAPHORE_LIMIT))
        self.cpu_thread_budget = CPU_THREAD_BUDGET
        self.active_cpu_threads = 0
        self.max_queue_depth = MAX_QUEUE_DEPTH

        # Accounting maps (keyed by alias: e.g. 'bge-m3-i8__gpu')
        self.model_last_used: Dict[str, float] = {}
        self.model_loaded: Dict[str, bool] = {}
        self.model_device: Dict[str, str] = {}
        self.active_inferences: Dict[str, int] = {}
        self.queue_depth: Dict[str, int] = {}

        # Tracking for temporary CPU spillover replicas
        self.temporary_cpu_replicas: Set[str] = set()

        self._idle_sweeper_started = False
        self._sweeper_thread = None

        # Metrics
        self.metrics = {
            "gpu_inferences": 0,
            "cpu_spillover_inferences": 0,
            "cpu_coexistence_inferences": 0,
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
        for alias in list(self.model_loaded):
            self.model_loaded[alias] = is_model_available(alias)

    def _registry_pinned_models(self) -> set:
        pinned = set(PINNED_MODELS)
        for cfg in MODEL_REGISTRY.values():
            if cfg.get("pinned") is True and cfg.get("ovms_model"):
                pinned.add(cfg["ovms_model"])
                pinned.add(get_model_alias(cfg["ovms_model"], "GPU"))
                pinned.add(get_model_alias(cfg["ovms_model"], "CPU"))
        return pinned

    def _wait_for_model_state(self, alias: str, available: bool, timeout: int) -> bool:
        deadline = time.time() + max(1, timeout)
        while time.time() < deadline:
            if is_model_available(alias) == available:
                return True
            time.sleep(0.1)
        return is_model_available(alias) == available

    def _registry_device_preference(self, logical_model: str) -> str:
        """Resolves the device the model registry declares for ``logical_model``.

        ``main.py`` already reads ``preferred_device`` from the registry for every
        inference route it owns, but the embedding paths used to call
        ``lease(model_name)`` with no preference at all.  That silently reverted
        them to the GPU default, so on a CPU-only deployment every embedding
        request attempted a GPU alias first: the load could never succeed, the
        caller paid a full ``_wait_for_model_state`` timeout, and the failed
        alias was left in ``config.json``.  Resolving the preference here keeps a
        single source of truth and stays GPU-first wherever the registry says so.

        Accepts either the logical registry id or its ``ovms_model`` name, since
        the embedding call sites hold the latter.
        """
        try:
            for model_id, cfg in MODEL_REGISTRY.items():
                if not isinstance(cfg, dict):
                    continue
                if model_id == logical_model or cfg.get("ovms_model") == logical_model:
                    return str(cfg.get("preferred_device") or "GPU").upper()
        except Exception:  # pragma: no cover - registry read failures fall back to GPU
            pass
        return "GPU"

    def _set_model_enabled(
        self,
        alias: str,
        enabled: bool,
        target_device: Optional[str] = None,
        wait: bool = True,
    ) -> bool:
        """Enables or disables an internal model alias in OVMS runtime config.

        ``wait=False`` writes the config and returns immediately without waiting
        for OVMS to reach the requested state.  It exists for the rollback path:
        removing an alias that already failed to load must not block for another
        full timeout.
        """
        base_name, inferred_dev = parse_model_alias(alias)
        dev = target_device or inferred_dev

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
                    if alias not in names:
                        cfg = dict(catalog_model_config(alias))
                        cfg["name"] = alias
                        cfg["target_device"] = dev
                        entries.append({"config": cfg})
                        changed = True
                else:
                    if alias not in names:
                        return True
                    entries = [
                        item for item in entries
                        if item.get("config", {}).get("name") != alias
                    ]
                    changed = True

                if changed:
                    payload["model_config_list"] = entries
                    write_runtime_config(payload)
                    # OVMS monitors config_path automatically when filesystem
                    # polling is enabled.  Prefer that path in production so a
                    # broker load/unload does not force the global config reload
                    # endpoint, which can transiently reload unrelated models.
                    # Deployments that explicitly disable polling can opt back
                    # into the API path with OVMS_CONFIG_UPDATE_MODE=api.
                    if OVMS_CONFIG_UPDATE_MODE == "api" and not reload_ovms_config():
                        return False
            except (OSError, ValueError, KeyError):
                return False

        if not wait:
            return True
        if enabled:
            return self._wait_for_model_state(alias, True, max(OVMS_TIMEOUT, 30))
        return self._wait_for_model_state(alias, False, min(max(OVMS_TIMEOUT, 30), 60))

    def _release_resources(self, alias: str):
        base_name, _ = parse_model_alias(alias)
        tokenizer_paths = {
            cfg.get("tokenizer_path")
            for cfg in MODEL_REGISTRY.values()
            if cfg.get("ovms_model") in (alias, base_name) and cfg.get("tokenizer_path")
        }
        release_tokenizers_for_model(base_name, tokenizer_paths)

    def _evict_idle_models(self, exclude: Optional[str] = None):
        now = time.time()
        pinned = self._registry_pinned_models()
        candidates = []

        for alias, loaded in list(self.model_loaded.items()):
            if not loaded or alias in pinned or (exclude and alias == exclude):
                continue
            if self.active_inferences.get(alias, 0) > 0:
                continue

            last_used = self.model_last_used.get(alias, now)
            base_name, _ = parse_model_alias(alias)
            ttl = MODEL_IDLE_UNLOAD_SECONDS
            for cfg in MODEL_REGISTRY.values():
                if cfg.get("ovms_model") in (alias, base_name) and cfg.get("idle_ttl"):
                    ttl = int(cfg["idle_ttl"])
                    break

            if now - last_used >= ttl:
                candidates.append(alias)

        for alias in candidates:
            if self._set_model_enabled(alias, False):
                self.model_loaded[alias] = False
                self._release_resources(alias)
                self.metrics["evictions"] += 1

    def _evict_to_capacity(self, exclude: str):
        pinned = self._registry_pinned_models()
        while sum(1 for v in self.model_loaded.values() if v) >= max(1, MAX_LOADED_MODELS):
            candidates = [
                m for m, is_loaded in self.model_loaded.items()
                if is_loaded and m != exclude and m not in pinned and self.active_inferences.get(m, 0) == 0
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

    def lease(self, logical_model: str, preferred_device: Optional[str] = None) -> ModelLease:
        """Acquires a model execution lease with true GPU-first, CPU spillover, and multi-model coexistence.

        ``preferred_device=None`` (the default) resolves the preference from the
        model registry, which is the same source ``main.py`` uses.  Pass an
        explicit ``"GPU"``/``"CPU"`` only to override the registry deliberately.
        """
        if preferred_device is None:
            preferred_device = self._registry_device_preference(logical_model)
        else:
            preferred_device = str(preferred_device).upper()

        with self.model_state_lock:
            total_in_flight = sum(self.active_inferences.values()) + sum(self.queue_depth.values())
            if total_in_flight >= self.max_queue_depth:
                raise HTTPException(status_code=503, detail="AI Gateway inference queue is full")

            self.queue_depth[logical_model] = self.queue_depth.get(logical_model, 0) + 1
            self.metrics["active_queue_depth"] = sum(self.queue_depth.values())

        gpu_alias = get_model_alias(logical_model, "GPU")
        cpu_alias = get_model_alias(logical_model, "CPU")

        try:
            # Determine routing decision
            with self.model_state_lock:
                self._sync_model_loaded_state()

                # Check if another model is currently running on GPU
                active_gpu_models = {
                    parse_model_alias(a)[0]
                    for a, count in self.active_inferences.items()
                    if count > 0 and parse_model_alias(a)[1] == "GPU"
                }
                gpu_busy_with_other = any(m != logical_model for m in active_gpu_models)
                gpu_self_active = self.active_inferences.get(gpu_alias, 0)

                should_spillover = (
                    preferred_device == "CPU"
                    or gpu_busy_with_other
                    or gpu_self_active > 0
                )

            if not should_spillover:
                # Route to GPU
                with self.model_state_lock:
                    self._evict_to_capacity(exclude=gpu_alias)
                    if not self.model_loaded.get(gpu_alias, False):
                        if not self._set_model_enabled(gpu_alias, True, target_device="GPU"):
                            # The alias is registered in OVMS but never became
                            # available.  Leaving it behind makes OVMS retry the
                            # failing compile once per filesystem-poll interval
                            # for the lifetime of the container, which both
                            # floods the OVMS log and keeps the deployment out of
                            # the zero-resident state.  Roll it back out.
                            self._set_model_enabled(gpu_alias, False, wait=False)
                            should_spillover = True
                        else:
                            self.model_loaded[gpu_alias] = True

                if not should_spillover:
                    with self.model_state_lock:
                        self.active_inferences[gpu_alias] = self.active_inferences.get(gpu_alias, 0) + 1
                        self.model_last_used[gpu_alias] = time.time()
                        self.model_device[gpu_alias] = "GPU"
                        self.metrics["gpu_inferences"] += 1
                        return ModelLease(self, gpu_alias, logical_model, "GPU", is_spillover=False)

            # Route to CPU (either coexistence or high-load spillover)
            # Atomic reservation: verify budget and reserve slot under state lock to eliminate TOCTOU window
            with self.model_state_lock:
                if self.active_cpu_threads >= self.cpu_thread_budget:
                    raise HTTPException(status_code=503, detail="CPU thread budget exhausted")
                self.active_cpu_threads += 1

            acquired = self.cpu_semaphore.acquire(blocking=True, timeout=10.0)
            if not acquired:
                with self.model_state_lock:
                    self.active_cpu_threads = max(0, self.active_cpu_threads - 1)
                raise HTTPException(status_code=503, detail="CPU semaphore budget exhausted")

            with self.model_state_lock:
                self._evict_to_capacity(exclude=cpu_alias)
                if not self.model_loaded.get(cpu_alias, False):
                    if not self._set_model_enabled(cpu_alias, True, target_device="CPU"):
                        self._set_model_enabled(cpu_alias, False, wait=False)
                        self.active_cpu_threads = max(0, self.active_cpu_threads - 1)
                        self.cpu_semaphore.release()
                        raise HTTPException(status_code=503, detail=f"Failed to load {cpu_alias} on CPU")
                    self.model_loaded[cpu_alias] = True

                is_temp_spillover = (gpu_self_active > 0)
                if is_temp_spillover:
                    self.temporary_cpu_replicas.add(cpu_alias)
                    self.metrics["cpu_spillover_inferences"] += 1
                else:
                    self.metrics["cpu_coexistence_inferences"] += 1

                self.active_inferences[cpu_alias] = self.active_inferences.get(cpu_alias, 0) + 1
                self.model_last_used[cpu_alias] = time.time()
                self.model_device[cpu_alias] = "CPU"

                return ModelLease(
                    self, cpu_alias, logical_model, "CPU", is_spillover=is_temp_spillover
                )

        finally:
            with self.model_state_lock:
                if logical_model in self.queue_depth:
                    self.queue_depth[logical_model] = max(0, self.queue_depth[logical_model] - 1)
                self.metrics["active_queue_depth"] = sum(self.queue_depth.values())

    def release_lease(self, lease: ModelLease):
        """Releases the execution lease, frees CPU semaphore/thread budget, and tears down temporary replicas."""
        with self.model_state_lock:
            alias = lease.alias
            if alias in self.active_inferences:
                self.active_inferences[alias] = max(0, self.active_inferences[alias] - 1)
            self.model_last_used[alias] = time.time()

            if lease.device == "CPU":
                self.active_cpu_threads = max(0, self.active_cpu_threads - 1)
                try:
                    self.cpu_semaphore.release()
                except ValueError:
                    pass

            # If this was a temporary CPU replica and active inferences reached 0, unload it
            if alias in self.temporary_cpu_replicas and self.active_inferences.get(alias, 0) == 0:
                if self._set_model_enabled(alias, False):
                    self.model_loaded[alias] = False
                    self.temporary_cpu_replicas.discard(alias)
                    self.metrics["evictions"] += 1
                    self._release_resources(alias)

    def ensure_model_available(
        self,
        model_name: str,
        preferred_device: Optional[str] = None,
    ) -> str:
        """Backward-compatible method returning the selected target device.

        When no explicit override is supplied, defer to ``lease()`` so the
        model registry remains the single source of truth for device
        preference.  This avoids reintroducing the historical GPU-first trap
        on CPU-only deployments.
        """
        lease = self.lease(model_name, preferred_device=preferred_device)
        dev = lease.device
        self.release_lease(lease)
        return dev


DEVICE_BROKER = DeviceBroker()
