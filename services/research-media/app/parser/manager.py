import time
from typing import Dict, List, Optional
from fastapi import HTTPException

from contracts import (
    ParseRequest,
    ParsedDocument,
    ProviderExecutionInfo,
)
from .base import BaseParserDriver
from .registry import PROVIDER_REGISTRY, ProviderRegistry
from .lifecycle import LIFECYCLE_MANAGER, LifecycleManager


class ParserManager:
    """Parser Manager orchestrating capability routing, privacy filtering, and multi-provider fallback.

    Completely decoupled from any specific vendor: manages generic BaseParserDrivers loaded by ProviderRegistry.
    """

    def __init__(self, registry: Optional[ProviderRegistry] = None, lifecycle: Optional[LifecycleManager] = None):
        self.registry = registry or PROVIDER_REGISTRY
        self.lifecycle = lifecycle or LIFECYCLE_MANAGER

    def select_candidates(self, request: ParseRequest) -> List[BaseParserDriver]:
        """Selects and prioritizes drivers according to privacy, capabilities, and health."""
        candidates = []

        for driver in self.registry.list_drivers():
            # 1. Health / cooldown check
            if not driver.is_available():
                continue

            # 2. Strict Privacy Hard Boundary: Private requests MUST NOT route to remote providers
            if request.privacy == "private" and driver.definition.location == "remote":
                continue

            # 3. Capability Matching: Must satisfy all requested needs
            if request.needs and not driver.satisfies_capabilities(request.needs):
                continue

            candidates.append(driver)

        # Sort by priority descending (higher number = higher priority)
        candidates.sort(key=lambda d: d.definition.priority, reverse=True)
        return candidates

    async def parse(self, request: ParseRequest) -> ParsedDocument:
        candidates = self.select_candidates(request)
        if not candidates:
            raise HTTPException(
                status_code=503,
                detail=f"No available parser provider matches request: privacy={request.privacy}, needs={request.needs}"
            )

        attempted_providers = []
        last_error = None

        for idx, driver in enumerate(candidates):
            attempted_providers.append(driver.name)
            t0 = time.monotonic()

            try:
                # Prepare lifecycle (e.g. spin up on-demand local process via Supervisor)
                await self.lifecycle.prepare_provider(driver.definition.lifecycle)

                # Execute parsing
                doc = await driver.parse(request)

                latency_ms = (time.monotonic() - t0) * 1000.0

                # Release lifecycle resource
                await self.lifecycle.release_provider(driver.definition.lifecycle)

                # Attach execution metadata
                status_val = "primary" if idx == 0 else "fallback"
                fallback_reason = str(last_error) if idx > 0 else None

                doc.provider_info = ProviderExecutionInfo(
                    provider_name=driver.name,
                    status=status_val,
                    latency_ms=latency_ms,
                    attempted_providers=attempted_providers,
                    fallback_reason=fallback_reason,
                )
                return doc

            except Exception as exc:
                last_error = exc
                await self.lifecycle.release_provider(driver.definition.lifecycle)
                continue

        # If all candidates failed
        raise HTTPException(
            status_code=502,
            detail=f"All parser providers failed. Attempted: {attempted_providers}. Last error: {str(last_error)}"
        )


PARSER_MANAGER = ParserManager()
