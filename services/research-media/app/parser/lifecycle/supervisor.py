import asyncio
import os
import time
from typing import Dict, Optional
import httpx
from ..models import LifecycleConfig


class SupervisorClient:
    """Independent supervisor client for managing on-demand local parser processes/containers.

    Decoupled from HTTP parser drivers: HTTP drivers never interact with container runtimes.
    """

    def __init__(self, supervisor_url: Optional[str] = None):
        self.supervisor_url = supervisor_url or os.getenv("SUPERVISOR_ENDPOINT", "http://supervisor:9001")

    async def start_resource(self, resource_name: str, timeout_seconds: float = 60.0) -> bool:
        """Requests supervisor to spin up an on-demand resource."""
        if not resource_name:
            return True
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.post(f"{self.supervisor_url}/resources/{resource_name}/start")
                return res.status_code in (200, 204)
        except Exception:
            # If supervisor is not deployed, on_demand acts gracefully
            return False

    async def stop_resource(self, resource_name: str) -> bool:
        """Requests supervisor to tear down or pause an idle resource."""
        if not resource_name:
            return True
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.post(f"{self.supervisor_url}/resources/{resource_name}/stop")
                return res.status_code in (200, 204)
        except Exception:
            return False


class LifecycleManager:
    """Manages lifecycle transitions for configured parser providers."""

    def __init__(self, supervisor: Optional[SupervisorClient] = None):
        self.supervisor = supervisor or SupervisorClient()
        self.active_resources: Dict[str, float] = {}

    async def prepare_provider(self, lifecycle: LifecycleConfig) -> bool:
        """Prepares provider before request execution based on lifecycle mode."""
        if lifecycle.mode in ("external", "persistent"):
            return True
        elif lifecycle.mode == "on_demand" and lifecycle.resource:
            ok = await self.supervisor.start_resource(
                lifecycle.resource, timeout_seconds=lifecycle.startup_timeout_seconds
            )
            if ok:
                self.active_resources[lifecycle.resource] = time.time()
            return ok
        return True

    async def release_provider(self, lifecycle: LifecycleConfig) -> bool:
        if lifecycle.mode == "on_demand" and lifecycle.resource:
            self.active_resources[lifecycle.resource] = time.time()
        return True


LIFECYCLE_MANAGER = LifecycleManager()
