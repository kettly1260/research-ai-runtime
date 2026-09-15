import os
import yaml
from typing import Any, Dict, Optional

PARSER_CONFIG_PATH = os.getenv("PARSER_CONFIG_PATH", "config/providers.yaml")
AI_GATEWAY_ENDPOINT = os.getenv("AI_GATEWAY_ENDPOINT", "http://ai-gateway:8000")
MEDIA_DATA_DIR = os.getenv("MEDIA_DATA_DIR", "/data/media.lance")
SUPERVISOR_ENDPOINT = os.getenv("SUPERVISOR_ENDPOINT", "http://supervisor:9001")


def load_providers_config(path: Optional[str] = None) -> Dict[str, Any]:
    target_path = path or PARSER_CONFIG_PATH
    if not os.path.exists(target_path):
        candidates = [
            os.path.abspath(target_path),
            "config/providers.example.yaml",
            "../../config/providers.example.yaml",
            "/app/config/providers.yaml",
            "/app/config/providers.example.yaml",
        ]
        for cand in candidates:
            if os.path.exists(cand):
                target_path = cand
                break
        else:
            return {"providers": {}, "routing": {}}
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict):
                return data
    except Exception as exc:
        print(f"Warning: Failed to load parser providers config from {target_path}: {exc}", flush=True)
    return {"providers": {}, "routing": {}}
