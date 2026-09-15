import os
import time
from typing import Dict, List, Optional
from fastapi import HTTPException

from ..config import (
    load_providers_config,
    MINERU_API_KEY,
    MINERU_ENDPOINT,
    PADDLEOCR_API_KEY,
    PADDLEOCR_ENDPOINT,
    LOCAL_MINERU_ENDPOINT,
    LOCAL_PADDLE_ENDPOINT,
    AI_GATEWAY_ENDPOINT,
)
from .base import BaseParserProvider
from .providers import MinerUProvider, PaddleOCRProvider, OpenVINOOCRProvider
from contracts import (
    ParseRequest,
    ParsedDocument,
    ProviderConfig,
    ProviderExecutionInfo,
)


class ParserManager:
    """Parser Manager orchestrating capability routing, privacy filtering, and fallback."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.providers: Dict[str, BaseParserProvider] = {}
        self.routing_rules: Dict[str, Any] = {}
        self.reload_providers()

    def reload_providers(self):
        raw_config = load_providers_config(self.config_path)
        providers_dict = raw_config.get("providers", {})
        self.routing_rules = raw_config.get("routing", {})

        # Default fallback providers if config is empty
        if not providers_dict:
            providers_dict = {
                "mineru_cloud": {
                    "driver": "mineru",
                    "transport": "http",
                    "location": "remote",
                    "endpoint": MINERU_ENDPOINT,
                    "api_key": MINERU_API_KEY,
                    "enabled": True,
                    "capabilities": {"pdf": True, "image": True, "ocr": True, "layout": True, "figure": True, "table": True, "formula": True},
                    "priority": 100,
                },
                "paddle_cloud": {
                    "driver": "paddleocr",
                    "transport": "http",
                    "location": "remote",
                    "endpoint": PADDLEOCR_ENDPOINT,
                    "api_key": PADDLEOCR_API_KEY,
                    "enabled": bool(PADDLEOCR_ENDPOINT),
                    "capabilities": {"pdf": True, "image": True, "ocr": True, "layout": True, "figure": True, "table": True, "formula": True},
                    "priority": 90,
                },
                "mineru_local": {
                    "driver": "mineru",
                    "transport": "http",
                    "location": "local",
                    "endpoint": LOCAL_MINERU_ENDPOINT,
                    "enabled": True,
                    "capabilities": {"pdf": True, "image": True, "ocr": True, "layout": True, "figure": True, "table": True, "formula": True},
                    "priority": 80,
                },
                "openvino_ocr": {
                    "driver": "generic_ocr",
                    "transport": "http",
                    "location": "local",
                    "endpoint": AI_GATEWAY_ENDPOINT,
                    "model": "openvino-ocr",
                    "enabled": True,
                    "capabilities": {"image": True, "ocr": True, "pdf": False, "layout": False, "figure": False, "table": False, "formula": False},
                    "priority": 50,
                },
            }

        loaded = {}
        for name, p_cfg in providers_dict.items():
            cfg_obj = ProviderConfig(**p_cfg)
            # Resolve environment variable mappings
            if cfg_obj.endpoint_env and os.getenv(cfg_obj.endpoint_env):
                cfg_obj.endpoint = os.getenv(cfg_obj.endpoint_env)
            elif name == "mineru_cloud" and MINERU_ENDPOINT:
                cfg_obj.endpoint = MINERU_ENDPOINT
            elif name == "mineru_local" and LOCAL_MINERU_ENDPOINT:
                cfg_obj.endpoint = LOCAL_MINERU_ENDPOINT
            elif name == "paddle_local" and LOCAL_PADDLE_ENDPOINT:
                cfg_obj.endpoint = LOCAL_PADDLE_ENDPOINT
            elif name == "openvino_ocr" and AI_GATEWAY_ENDPOINT:
                cfg_obj.endpoint = AI_GATEWAY_ENDPOINT

            if cfg_obj.api_key_env and os.getenv(cfg_obj.api_key_env):
                cfg_obj.api_key = os.getenv(cfg_obj.api_key_env)
            elif name == "mineru_cloud" and MINERU_API_KEY:
                cfg_obj.api_key = MINERU_API_KEY
            elif name == "paddle_cloud" and PADDLEOCR_API_KEY:
                cfg_obj.api_key = PADDLEOCR_API_KEY

            driver = cfg_obj.driver.lower()
            if driver == "mineru":
                provider = MinerUProvider(name, cfg_obj)
            elif driver == "paddleocr":
                provider = PaddleOCRProvider(name, cfg_obj)
            elif driver == "generic_ocr":
                provider = OpenVINOOCRProvider(name, cfg_obj)
            else:
                continue
            loaded[name] = provider

        self.providers = loaded

    def select_candidates(self, request: ParseRequest) -> List[BaseParserProvider]:
        """Selects and orders suitable providers based on privacy, capabilities, and health."""
        candidates = []

        # If user explicitly preferred a provider
        if request.preferred_provider and request.preferred_provider in self.providers:
            p = self.providers[request.preferred_provider]
            if p.is_available():
                # Privacy check: if private, cannot use remote
                if request.privacy == "private" and p.config.location == "remote":
                    pass
                else:
                    return [p]

        for p in self.providers.values():
            if not p.is_available():
                continue

            # 1. Privacy Routing (Section 16)
            if request.privacy == "private" and p.config.location == "remote":
                # Remote forbidden for private documents!
                continue

            # 2. Capability Routing (Section 15)
            if request.needs and not p.satisfies_capabilities(request.needs):
                continue

            candidates.append(p)

        # Sort candidates by preference and priority
        pref_order = self.routing_rules.get(request.privacy, {}).get("prefer", [])

        def sort_key(prov: BaseParserProvider):
            pref_score = 0
            if prov.name in pref_order:
                # Earlier in pref list = higher score
                pref_score = (len(pref_order) - pref_order.index(prov.name)) * 1000
            return pref_score + prov.config.priority

        candidates.sort(key=sort_key, reverse=True)
        return candidates

    async def parse(self, request: ParseRequest) -> ParsedDocument:
        """Parses a document with automatic multi-provider fallback."""
        candidates = self.select_candidates(request)
        if not candidates:
            raise HTTPException(
                status_code=503,
                detail=f"No available document parser provider matching requirements (privacy={request.privacy}, needs={request.needs})"
            )

        attempted = []
        last_error = None
        t_start = time.monotonic()

        for provider in candidates:
            attempted.append(provider.name)
            try:
                doc = await provider.parse(request)
                latency_ms = (time.monotonic() - t_start) * 1000.0

                doc.provider_info = ProviderExecutionInfo(
                    provider_name=provider.name,
                    latency_ms=latency_ms,
                    status="success" if len(attempted) == 1 else "fallback",
                    fallback_reason=str(last_error) if last_error else None,
                    attempted_providers=attempted,
                )
                return doc
            except Exception as exc:
                last_error = exc
                print(f"[ParserManager] Provider {provider.name} failed: {exc}, falling back to next provider...", flush=True)

        total_latency = (time.monotonic() - t_start) * 1000.0
        raise HTTPException(
            status_code=502,
            detail=f"All parser providers failed. Attempted: {attempted}. Last error: {last_error}"
        )


PARSER_MANAGER = ParserManager()
