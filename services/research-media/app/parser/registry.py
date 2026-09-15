import os
from typing import Dict, List, Optional
import yaml

from .base import BaseParserDriver
from .drivers.http import GenericHttpDriver
from .models import ProviderDefinition


class ProviderRegistry:
    """Registry that loads declarative provider definitions from YAML and instantiates generic drivers."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.getenv("PROVIDERS_CONFIG_PATH", "config/providers.yaml")
        self.drivers: Dict[str, BaseParserDriver] = {}
        self.load_from_yaml(self.config_path)

    def register_driver(self, name: str, driver: BaseParserDriver):
        self.drivers[name] = driver

    def get_driver(self, name: str) -> Optional[BaseParserDriver]:
        return self.drivers.get(name)

    def list_drivers(self) -> List[BaseParserDriver]:
        return list(self.drivers.values())

    def load_from_yaml(self, path: str):
        self.drivers.clear()
        resolved_path = path
        if not os.path.isabs(resolved_path) and not os.path.exists(resolved_path):
            # Try searching up from current working directory or app directory
            candidates = [
                os.path.abspath(path),
                os.path.join(os.getcwd(), path),
                os.path.join(os.getcwd(), "config/providers.example.yaml"),
                "/app/config/providers.yaml",
                "/app/config/providers.example.yaml",
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    resolved_path = cand
                    break

        if not os.path.exists(resolved_path):
            return

        try:
            with open(resolved_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            raw_providers = data.get("providers", {})
            for name, p_data in raw_providers.items():
                if not isinstance(p_data, dict):
                    continue
                p_data["name"] = name
                driver_type = p_data.get("driver", "http")

                definition = ProviderDefinition(**p_data)

                # Factory: instantiate generic driver based on driver protocol
                if driver_type == "http":
                    driver_inst = GenericHttpDriver(definition)
                    self.register_driver(name, driver_inst)
                else:
                    raise ValueError(f"Unknown driver type '{driver_type}' for provider '{name}'")
        except Exception as e:
            print(f"Warning: Failed to load providers configuration from {resolved_path}: {e}")


PROVIDER_REGISTRY = ProviderRegistry()
