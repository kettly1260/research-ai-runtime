from typing import Any, List, Optional
import httpx
from fastapi import HTTPException
from ..config import AI_GATEWAY_ENDPOINT


class GatewayClient:
    """Client for AI Gateway model inference."""

    def __init__(self, endpoint: Optional[str] = None):
        self.endpoint = (endpoint or AI_GATEWAY_ENDPOINT).rstrip("/")

    async def get_text_embedding(self, texts: List[str], model: str = "jina-clip-v2") -> List[List[float]]:
        url = f"{self.endpoint}/v1/embeddings"
        payload = {
            "model": model,
            "modality": "text",
            "input": texts,
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=f"Gateway embedding request failed: {resp.text}"
                )
            data = resp.json()
            return [item["embedding"] for item in data.get("data", [])]

    async def get_image_embedding(self, images: List[Any], model: str = "jina-clip-v2") -> List[List[float]]:
        url = f"{self.endpoint}/v1/embeddings"
        payload = {
            "model": model,
            "modality": "image",
            "input": images,
        }
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=f"Gateway image embedding request failed: {resp.text}"
                )
            data = resp.json()
            return [item["embedding"] for item in data.get("data", [])]

    async def get_dino_embedding(self, images: List[Any], model: str = "dinov2-small") -> List[List[float]]:
        url = f"{self.endpoint}/v1/embeddings"
        payload = {
            "model": model,
            "modality": "image",
            "input": images,
        }
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=f"Gateway DINO embedding request failed: {resp.text}"
                )
            data = resp.json()
            return [item["embedding"] for item in data.get("data", [])]
