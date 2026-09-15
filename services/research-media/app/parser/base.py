from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import time

import sys
import os
sys.path.insert(0, os.path.abspath("packages/contracts/src"))

from contracts import ParseRequest, ParsedDocument, ProviderConfig, ProviderStatus


class BaseParserProvider(ABC):
    """Abstract Base Class for Document Parser Providers."""

    def __init__(self, name: str, config: ProviderConfig):
        self.name = name
        self.config = config
        self.status = ProviderStatus(
            name=name,
            available=config.enabled,
        )

    def is_available(self) -> bool:
        lifecycle_mode = getattr(self.config.lifecycle, "mode", "external")
        if not self.config.enabled and lifecycle_mode not in ("on_demand", "model_on_demand"):
            return False
        if self.status.quota_state == "exhausted":
            return False
        if self.status.cooldown_until and time.time() < self.status.cooldown_until:
            return False
        return True

    def record_success(self, latency_ms: float):
        self.status.available = True
        self.status.last_success = time.time()
        self.status.latency = latency_ms
        self.status.failure_count = 0
        self.status.cooldown_until = None

    def record_failure(self, error: Exception, quota_exhausted: bool = False):
        now = time.time()
        self.status.last_failure = now
        self.status.failure_count += 1
        if quota_exhausted:
            self.status.quota_state = "exhausted"
            self.status.cooldown_until = now + 3600.0  # 1 hour cooldown
        else:
            # Exponential backoff: 30s, 60s, 120s, up to 300s
            backoff = min(30.0 * (2 ** (self.status.failure_count - 1)), 300.0)
            self.status.cooldown_until = now + backoff

    def satisfies_capabilities(self, needs: list) -> bool:
        for need in needs:
            key = need.lower()
            if key == "figures":
                key = "figure"
            elif key == "tables":
                key = "table"
            if not self.config.capabilities.get(key, False) and not self.config.capabilities.get(need, False):
                return False
        return True

    @abstractmethod
    async def parse(self, request: ParseRequest) -> ParsedDocument:
        """Parses the document into a unified ParsedDocument."""
        pass
