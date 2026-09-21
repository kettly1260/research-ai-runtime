from __future__ import annotations

import os
from typing import Dict, List, Optional
import yaml

from .base import BaseParserDriver
from .drivers.http import GenericHttpDriver
from .drivers.command import GenericCommandDriver
from .models import ProviderDefinition


class ProviderRegistry:
    """Hot-reloading registry for declarative parser providers.

    Reloads are atomic: a malformed edit never clears the last known-good
    driver set.  The file is checked lazily when callers access the registry,
    so editing the mounted providers.yaml does not require a service restart.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.explicit_config = config_path is not None or "PARSER_CONFIG_PATH" in os.environ
        self.config_path = config_path or os.getenv("PARSER_CONFIG_PATH", "config/providers.yaml")
        self.drivers: Dict[str, BaseParserDriver] = {}
        self.resolved_config_path: Optional[str] = None
        self._observed_signature: Optional[tuple[int, int]] = None
        self.last_reload_error: Optional[str] = None
        self.load_from_yaml(self.config_path)

    def register_driver(self, name: str, driver: BaseParserDriver):
        self.drivers[name] = driver

    def get_driver(self, name: str) -> Optional[BaseParserDriver]:
        self.reload_if_changed()
        return self.drivers.get(name)

    def list_drivers(self) -> List[BaseParserDriver]:
        self.reload_if_changed()
        return list(self.drivers.values())

    @staticmethod
    def _signature(path: str) -> tuple[int, int]:
        stat = os.stat(path)
        return (stat.st_mtime_ns, stat.st_size)

    def _resolve_path(self, path: str) -> str:
        resolved_path = path

        if not os.path.exists(resolved_path):
            if not os.path.isabs(resolved_path):
                candidates = [
                    os.path.abspath(path),
                    os.path.join(os.getcwd(), path),
                    os.path.join(os.getcwd(), "config/providers.example.yaml"),
                    "config/providers.example.yaml",
                    "/config/providers.yaml",
                    "/config/providers.example.yaml",
                    "/app/config/providers.yaml",
                    "/app/config/providers.example.yaml",
                ]
                for cand in candidates:
                    if os.path.exists(cand):
                        resolved_path = cand
                        break

        if not os.path.exists(resolved_path):
            if self.explicit_config:
                raise FileNotFoundError(
                    f"Configured PARSER_CONFIG_PATH '{path}' does not exist (resolved: '{resolved_path}')"
                )
            return resolved_path
        return os.path.abspath(resolved_path)

    def load_from_yaml(self, path: str):
        resolved_path = self._resolve_path(path)
        if not os.path.exists(resolved_path):
            return

        with open(resolved_path, "r", encoding="utf-8") as f:
            try:
                data = yaml.safe_load(f)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to parse provider configuration YAML at '{resolved_path}': {exc}"
                ) from exc

        if not isinstance(data, dict):
            raise ValueError(f"Provider configuration at '{resolved_path}' must be a YAML dictionary")

        raw_providers = data.get("providers", {})
        if not isinstance(raw_providers, dict) or len(raw_providers) == 0:
            if self.explicit_config:
                raise ValueError(
                    f"Provider configuration at '{resolved_path}' contains 0 providers; fail-fast triggered"
                )

        new_drivers: Dict[str, BaseParserDriver] = {}
        for name, p_data in raw_providers.items():
            if not isinstance(p_data, dict):
                continue
            provider_data = dict(p_data)
            provider_data["name"] = name
            driver_type = provider_data.get("driver", "http")

            definition = ProviderDefinition(**provider_data)

            # Factory: instantiate generic driver based on driver protocol
            if driver_type == "http":
                driver_inst = GenericHttpDriver(definition)
            elif driver_type in ("command", "cli"):
                driver_inst = GenericCommandDriver(definition)
            else:
                raise ValueError(f"Unknown driver type '{driver_type}' for provider '{name}'")
            new_drivers[name] = driver_inst

        # Atomic swap only after the entire file validates.
        self.drivers = new_drivers
        self.resolved_config_path = resolved_path
        self._observed_signature = self._signature(resolved_path)
        self.last_reload_error = None

    def reload_if_changed(self) -> bool:
        """Reload the provider file if its mtime/size changed.

        A bad hot edit is recorded but the last known-good drivers remain
        active.  The next file change is retried automatically.
        """
        try:
            resolved_path = self._resolve_path(self.config_path)
            if not os.path.exists(resolved_path):
                return False
            signature = self._signature(resolved_path)
            if signature == self._observed_signature:
                return False
            try:
                self.load_from_yaml(self.config_path)
                return True
            except Exception as exc:
                self.last_reload_error = str(exc)
                self.resolved_config_path = resolved_path
                self._observed_signature = signature
                return False
        except Exception as exc:
            self.last_reload_error = str(exc)
            return False


PROVIDER_REGISTRY = ProviderRegistry()
