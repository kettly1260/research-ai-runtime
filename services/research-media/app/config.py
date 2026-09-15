import os
import yaml
from typing import Any, Dict, Optional

PARSER_CONFIG_PATH = os.getenv("PARSER_CONFIG_PATH", "/config/providers.yaml")
AI_GATEWAY_ENDPOINT = os.getenv("AI_GATEWAY_ENDPOINT", "http://ai-gateway:8000")
MEDIA_DATA_DIR = os.getenv("MEDIA_DATA_DIR", "/data/media.lance")

# API Keys & Endpoints (defaults or overrides)
MINERU_API_KEY = os.getenv("MINERU_API_KEY", "")
MINERU_ENDPOINT = os.getenv("MINERU_ENDPOINT", "https://mineru.net/api/v4")

PADDLEOCR_API_KEY = os.getenv("PADDLEOCR_API_KEY", "")
PADDLEOCR_ENDPOINT = os.getenv("PADDLEOCR_ENDPOINT", "")

LOCAL_MINERU_ENDPOINT = os.getenv("LOCAL_MINERU_ENDPOINT", "http://mineru-local:8080")
LOCAL_PADDLE_ENDPOINT = os.getenv("LOCAL_PADDLE_ENDPOINT", "http://paddle-local:8080")


def load_providers_config(path: Optional[str] = None) -> Dict[str, Any]:
    target_path = path or PARSER_CONFIG_PATH
    if not os.path.exists(target_path):
        # Fallback to config/providers.example.yaml if running locally
        if os.path.exists("config/providers.example.yaml"):
            target_path = "config/providers.example.yaml"
        elif os.path.exists("../../config/providers.example.yaml"):
            target_path = "../../config/providers.example.yaml"
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
