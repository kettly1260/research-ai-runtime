from typing import Any, Dict, List, Optional
from .store import LanceMediaStore, MEDIA_STORE
from ..client.gateway import GatewayClient


class MediaSearchService:
    """Search service supporting text->image (multimodal) and image->image (DINO) search."""

    def __init__(self, store: Optional[LanceMediaStore] = None, gateway: Optional[GatewayClient] = None):
        self.store = store or MEDIA_STORE
        self.gateway = gateway or GatewayClient()

    async def search_by_text(
        self,
        query: str,
        top_k: int = 10,
        filter_expr: Optional[str] = None,
        model: str = "jina-clip-v2",
    ) -> List[Dict[str, Any]]:
        """Text to Image search using Multimodal Embeddings."""
        vecs = await self.gateway.get_text_embedding([query], model=model)
        if not vecs:
            return []
        query_vec = vecs[0]
        results = self.store.search_multimodal(query_vec, top_k=top_k, filter_expr=filter_expr)
        # Drop large vector arrays from output for readability
        for r in results:
            r.pop("multimodal_vector", None)
            r.pop("dino_vector", None)
        return results

    async def search_by_image(
        self,
        image_input: Any,
        top_k: int = 10,
        filter_expr: Optional[str] = None,
        model: str = "dinov3",
    ) -> List[Dict[str, Any]]:
        """Image to Image search using DINO visual representations."""
        vecs = await self.gateway.get_dino_embedding([image_input], model=model)
        if not vecs:
            return []
        query_vec = vecs[0]
        results = self.store.search_dino(query_vec, top_k=top_k, filter_expr=filter_expr)
        for r in results:
            r.pop("multimodal_vector", None)
            r.pop("dino_vector", None)
        return results


MEDIA_SEARCH = MediaSearchService()
