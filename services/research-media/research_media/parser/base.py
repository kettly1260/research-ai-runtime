from abc import ABC, abstractmethod
import time
from contracts import ParseRequest, ParsedDocument, ProviderStatus
from .models import ProviderDefinition


class BaseParserDriver(ABC):
    """Abstract Base Class for generic parser drivers."""

    def __init__(self, definition: ProviderDefinition):
        self.name = definition.name or "unnamed_provider"
        self.definition = definition
        self.status = ProviderStatus(
            name=self.name,
            available=definition.enabled,
        )

    @property
    def config(self) -> ProviderDefinition:
        return self.definition

    def is_available(self) -> bool:
        if not self.definition.enabled:
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
            backoff = min(30.0 * (2 ** (self.status.failure_count - 1)), 300.0)
            self.status.cooldown_until = now + backoff

    def satisfies_capabilities(self, needs: list) -> bool:
        # Convert definition capabilities list to set
        caps = {c.lower() for c in self.definition.capabilities}
        for need in needs:
            key = need.lower()
            if key == "figures":
                key = "figure"
            elif key == "tables":
                key = "table"
            if key not in caps and need.lower() not in caps:
                return False
        return True

    @abstractmethod
    async def parse(self, request: ParseRequest) -> ParsedDocument:
        """Parses the document into a unified ParsedDocument."""
        pass
